from __future__ import annotations

"""
realistic_hand_control.py

Neural control and MuJoCo bridge for realistic 5-finger hands.
Both hands: 44 DOF, 42 tactile touch sensors.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))


import math
import numpy as np
import torch
import torch.nn as nn
import mujoco

from src.platform.scene_builder.realistic_hand_mjcf import both_hand_control_names, both_hand_sensor_names


class RealisticHandControlHead(nn.Module):
    def __init__(self, workspace_dim: int, body_dim: int, tactile_latent_dim: int = 128, object_repr_dim: int = 128, out_dim: int = 44) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + body_dim + tactile_latent_dim + object_repr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, workspace, body_self, tactile_latent, object_repr):
        return self.net(torch.cat([workspace, body_self, tactile_latent, object_repr], dim=-1))


class RealisticHandTactileEncoder(nn.Module):
    def __init__(self, tactile_dim: int = 42, latent_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tactile_dim, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, tactile):
        return self.net(tactile)


class RealisticHandBridge:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.control_names = both_hand_control_names()
        self.sensor_names = both_hand_sensor_names()
        self.act_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}") for name in self.control_names}
        self.sensor_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in self.sensor_names}
        self.prev_ctrl = np.zeros(len(self.control_names), dtype=np.float64)

    def _map_control_to_angle(self, name: str, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if name.endswith("palm_roll") or name.endswith("palm_pitch"):
            return (u * 2.0 - 1.0) * math.radians(28)
        if name.endswith("_mcp_yaw"):
            # Finger spread/abduction. Neutral is 0.5.
            return (u * 2.0 - 1.0) * math.radians(28)
        if name.endswith("_mcp"):
            return math.radians(-10) + u * math.radians(95)
        if name.endswith("_pip"):
            return u * math.radians(100)
        if name.endswith("_dip"):
            return u * math.radians(80)
        return 0.0

    def apply(self, data: mujoco.MjData, controls: np.ndarray, smoothing: float = 0.25) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float64).reshape(-1)
        if controls.size != len(self.control_names):
            raise ValueError(
                f"Expected {len(self.control_names)} hand controls from both_hand_control_names(), "
                f"got {controls.size}. If you added xxx_mcp_yaw joints, update "
                f"scene_builder.realistic_hand_mjcf.hand_control_names() and actuators to 44."
            )
        smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.prev_ctrl = (1.0 - smoothing) * self.prev_ctrl + smoothing * controls
        angles = np.array([self._map_control_to_angle(name, u) for name, u in zip(self.control_names, self.prev_ctrl)], dtype=np.float64)
        for name, angle in zip(self.control_names, angles):
            aid = self.act_ids.get(name, -1)
            if aid is not None and aid >= 0:
                data.ctrl[aid] = float(angle)
        return angles

    def read_tactile(self, data: mujoco.MjData) -> np.ndarray:
        vals = []
        for name in self.sensor_names:
            sid = self.sensor_ids.get(name, -1)
            if sid is None or sid < 0:
                vals.append(0.0)
                continue
            adr = int(self.model.sensor_adr[sid])
            dim = int(self.model.sensor_dim[sid])
            raw = data.sensordata[adr:adr + dim]
            vals.append(float(np.asarray(raw).reshape(-1)[0]) if raw.size else 0.0)
        arr = np.asarray(vals, dtype=np.float32)
        return np.clip(arr, 0.0, 100.0) / 100.0
