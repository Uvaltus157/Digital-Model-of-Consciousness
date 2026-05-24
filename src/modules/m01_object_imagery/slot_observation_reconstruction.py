
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class SlotObservation:
    slot_id: int
    target_name: str
    live_step: int
    rgb: np.ndarray | None
    depth: np.ndarray | None
    camera_pose: np.ndarray | None
    formed_conf: float
    z_dynamic_norm: float


class SlotObservationBuffer:
    def __init__(self, max_frames_per_slot: int = 96) -> None:
        self.max_frames_per_slot = max(1, int(max_frames_per_slot))
        self.frames: dict[int, deque[SlotObservation]] = defaultdict(
            lambda: deque(maxlen=self.max_frames_per_slot)
        )

    def add(self, obs: SlotObservation) -> int:
        self.frames[int(obs.slot_id)].append(obs)
        return len(self.frames[int(obs.slot_id)])

    def count(self, slot_id: int) -> int:
        return len(self.frames.get(int(slot_id), ()))

    def latest(self, slot_id: int) -> SlotObservation | None:
        q = self.frames.get(int(slot_id))
        if not q:
            return None
        return q[-1]


class SlotPointCloudReconstructor:
    def __init__(self, max_points_per_slot: int = 24000, stride: int = 6) -> None:
        self.max_points_per_slot = max(128, int(max_points_per_slot))
        self.stride = max(1, int(stride))
        self.points: dict[int, np.ndarray] = {}
        self.colors: dict[int, np.ndarray] = {}

    @staticmethod
    def _to_numpy_image(x: Any) -> np.ndarray | None:
        if x is None:
            return None
        try:
            if torch.is_tensor(x):
                x = x.detach().float().cpu()
                if x.ndim == 4:
                    x = x[0]
                if x.ndim == 3 and x.shape[0] in (1, 3, 4):
                    x = x.permute(1, 2, 0)
                arr = x.numpy()
            else:
                arr = np.asarray(x)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        except Exception:
            return None

    def observation_from_runtime(
        self,
        *,
        slot_id: int,
        target_name: str,
        runtime_obs: dict,
        live_step: int,
        formed_conf: float,
        z_dynamic_norm: float,
    ) -> SlotObservation:
        rgb = self._to_numpy_image(runtime_obs.get("left"))
        depth = self._to_numpy_image(runtime_obs.get("depth"))
        if depth is not None and depth.ndim == 3:
            depth = depth[..., 0]
        camera_pose = None
        for key in ("camera_pose", "cam_pose", "camera_matrix", "extrinsics"):
            if key in runtime_obs:
                camera_pose = self._to_numpy_image(runtime_obs.get(key))
                break
        return SlotObservation(
            slot_id=int(slot_id),
            target_name=str(target_name),
            live_step=int(live_step),
            rgb=rgb,
            depth=depth,
            camera_pose=camera_pose,
            formed_conf=float(formed_conf),
            z_dynamic_norm=float(z_dynamic_norm),
        )

    def integrate(self, obs: SlotObservation) -> dict[str, Any]:
        if obs.depth is None or obs.depth.size == 0:
            return {"slot_id": int(obs.slot_id), "target_name": obs.target_name, "depth_valid": False, "points_added": 0, "points_total": int(len(self.points.get(int(obs.slot_id), ())))}

        depth = np.asarray(obs.depth, dtype=np.float32)
        if depth.ndim != 2:
            depth = depth.reshape(depth.shape[-2], depth.shape[-1])

        h, w = depth.shape
        s = max(1, int(self.stride))
        yy, xx = np.mgrid[0:h:s, 0:w:s]
        z = depth[0:h:s, 0:w:s].reshape(-1)
        valid = np.isfinite(z) & (z > 1.0e-6)
        if not np.any(valid):
            return {"slot_id": int(obs.slot_id), "target_name": obs.target_name, "depth_valid": False, "points_added": 0, "points_total": int(len(self.points.get(int(obs.slot_id), ())))}

        x = xx.reshape(-1)[valid].astype(np.float32)
        y = yy.reshape(-1)[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        fx = max(float(w), 1.0)
        fy = max(float(h), 1.0)
        cx = float(w - 1) * 0.5
        cy = float(h - 1) * 0.5
        pts = np.stack([(x - cx) * z / fx, -(y - cy) * z / fy, z], axis=-1)

        if obs.rgb is not None and obs.rgb.ndim >= 2:
            rgb = obs.rgb
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=-1)
            rgb_s = rgb[0:h:s, 0:w:s]
            if rgb_s.shape[-1] > 3:
                rgb_s = rgb_s[..., :3]
            col = rgb_s.reshape(-1, rgb_s.shape[-1])[valid]
            if col.shape[-1] == 1:
                col = np.repeat(col, 3, axis=-1)
            col = np.clip(col.astype(np.float32), 0.0, 1.0)
        else:
            col = np.ones((pts.shape[0], 3), dtype=np.float32) * 0.5

        sid = int(obs.slot_id)
        old_pts = self.points.get(sid)
        old_col = self.colors.get(sid)
        pts_all = pts if old_pts is None else np.concatenate([old_pts, pts], axis=0)
        col_all = col if old_col is None else np.concatenate([old_col, col], axis=0)

        if pts_all.shape[0] > self.max_points_per_slot:
            pts_all = pts_all[-self.max_points_per_slot:]
            col_all = col_all[-self.max_points_per_slot:]

        self.points[sid] = pts_all.astype(np.float32, copy=False)
        self.colors[sid] = col_all.astype(np.float32, copy=False)
        return {
            "slot_id": sid,
            "target_name": obs.target_name,
            "depth_valid": True,
            "points_added": int(pts.shape[0]),
            "points_total": int(pts_all.shape[0]),
            "formed_conf": float(obs.formed_conf),
            "z_dynamic_norm": float(obs.z_dynamic_norm),
        }
