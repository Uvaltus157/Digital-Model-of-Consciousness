
from __future__ import annotations

"""
mujoco_live_world_mocap_contacts_limb_damping.py

Mocap-controlled agent rig with physical contacts preserved.

The central rig/cameras are stabilized by the existing mocap pose path,
while arms/hands/fingers/legs/objects remain MuJoCo geoms/joints/sensors.
This is for stable object exploration without the freejoint body spinning.
"""

import numpy as np
import torch

from src.apps.unified_conscious_viewer import MujocoLiveWorldV57


class MujocoLiveWorldMocapContacts(MujocoLiveWorldV57):
    def __init__(
        self,
        *args,
        add_vestibular_to_body_state: bool = True,
        balance_reward_weight: float = 0.04,
        balance_gyro_penalty: float = 0.015,
        balance_diff_penalty: float = 0.010,
        min_flight_z: float = 0.35,
        max_flight_z: float = 10.0,
        **kwargs,
    ):
        self.add_vestibular_to_body_state = bool(add_vestibular_to_body_state)
        self.balance_reward_weight = float(balance_reward_weight)
        self.balance_gyro_penalty = float(balance_gyro_penalty)
        self.balance_diff_penalty = float(balance_diff_penalty)
        self.min_flight_z = float(min_flight_z)
        self.max_flight_z = float(max_flight_z)
        super().__init__(*args, **kwargs)
        self.roll_deg = 0.0
        self._init_vestibular_sensor_ids()
        self._init_head_actuators()

    def _init_vestibular_sensor_ids(self):
        import mujoco
        self.vestibular_sensor_names = [
            "vestibular_left_gyro",
            "vestibular_left_accel",
            "vestibular_right_gyro",
            "vestibular_right_accel",
        ]
        self.vestibular_sensor_ids = {}
        for name in self.vestibular_sensor_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.vestibular_sensor_ids[name] = int(sid)
        self.latest_vestibular = np.zeros(24, dtype=np.float32)
        found = sum(1 for v in self.vestibular_sensor_ids.values() if v >= 0)
        print(f"[mocap_contacts][vestibular] IMU sensors found: {found}/4")


    def _init_head_actuators(self):
        import mujoco
        self.head_act_ids = {}
        self.head_ctrl = np.zeros(3, dtype=np.float64)
        for key, name in [
            ("yaw", "act_head_yaw"),
            ("pitch", "act_head_pitch"),
            ("roll", "act_head_roll"),
        ]:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.head_act_ids[key] = int(aid)
        print(f"[agent_head] actuators: {self.head_act_ids}")

    def _apply_head_control(self, embodied_targets: np.ndarray):
        emb = np.asarray(embodied_targets, dtype=np.float64).reshape(-1)

        # New neural-body outputs:
        #   index 12 -> head_yaw
        #   index 13 -> head_pitch
        #   index 14 -> head_roll
        if emb.shape[0] < 15:
            return

        raw = np.clip(emb[12:15], -1.0, 1.0)
        ranges = np.array([
            [-0.785398, 0.785398],   # yaw -45..45
            [-1.22173, 0.872665],    # pitch -70..50
            [-1.22173, 1.22173],     # roll -70..70
        ], dtype=np.float64)
        target = ranges[:, 0] + (raw + 1.0) * 0.5 * (ranges[:, 1] - ranges[:, 0])
        self.head_ctrl = 0.82 * self.head_ctrl + 0.18 * target

        for i, key in enumerate(["yaw", "pitch", "roll"]):
            aid = self.head_act_ids.get(key, -1)
            if aid >= 0:
                self.data.ctrl[aid] = float(self.head_ctrl[i])


    def read_vestibular(self) -> np.ndarray:
        def read3(name: str) -> np.ndarray:
            sid = self.vestibular_sensor_ids.get(name, -1)
            if sid is None or sid < 0:
                return np.zeros(3, dtype=np.float32)
            try:
                adr = int(self.model.sensor_adr[sid])
                dim = int(self.model.sensor_dim[sid])
                data = self.data.sensordata[adr:adr + dim]
                out = np.zeros(3, dtype=np.float32)
                out[:min(3, dim)] = np.asarray(data[:min(3, dim)], dtype=np.float32)
                return out
            except Exception:
                return np.zeros(3, dtype=np.float32)

        lg = read3("vestibular_left_gyro")
        la = read3("vestibular_left_accel")
        rg = read3("vestibular_right_gyro")
        ra = read3("vestibular_right_accel")
        gyro_common = 0.5 * (lg + rg)
        gyro_diff = 0.5 * (lg - rg)
        accel_common = 0.5 * (la + ra)
        accel_diff = 0.5 * (la - ra)
        self.latest_vestibular = np.concatenate([
            lg, la, rg, ra,
            gyro_common, gyro_diff,
            accel_common, accel_diff,
        ]).astype(np.float32)
        return self.latest_vestibular




    @staticmethod
    def _quat_mul(q1, q2):
        # wxyz quaternion multiplication
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)

    @staticmethod
    def _axis_angle_quat(axis, angle_rad: float):
        axis = np.asarray(axis, dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = axis / n
        s = np.sin(0.5 * angle_rad)
        return np.array([np.cos(0.5 * angle_rad), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)

    def _quat_from_yaw_pitch_roll(self):
        """
        Standard aircraft-like orientation, MuJoCo wxyz quaternion:

            yaw   / azimuth  -> rotation around Z
            pitch / tangage  -> rotation around Y
            roll  / kren     -> rotation around X

        Composition is ZYX:
            q = q_yaw(Z) * q_pitch(Y) * q_roll(X)

        This avoids the previous forward/right/up construction where roll/pitch
        could look swapped depending on body axes.
        """
        yaw_deg = float(getattr(self, "yaw_deg", 0.0))
        pitch_deg = float(getattr(self, "pitch_deg", 0.0))
        roll_deg = float(getattr(self, "roll_deg", 0.0))
        if not np.isfinite(yaw_deg):
            yaw_deg = 0.0
            self.yaw_deg = yaw_deg
        if not np.isfinite(pitch_deg):
            pitch_deg = 0.0
            self.pitch_deg = pitch_deg
        if not np.isfinite(roll_deg):
            roll_deg = 0.0
            self.roll_deg = roll_deg

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)

        q_yaw = self._axis_angle_quat([0.0, 0.0, 1.0], yaw)
        q_pitch = self._axis_angle_quat([0.0, 1.0, 0.0], pitch)
        q_roll = self._axis_angle_quat([1.0, 0.0, 0.0], roll)

        q = self._quat_mul(self._quat_mul(q_yaw, q_pitch), q_roll)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        return q
    def _clamp_flight_zone(self):
        # Keep mocap-controlled body inside a safe vertical flight corridor.
        try:
            if not np.all(np.isfinite(np.asarray(self.cam_pos, dtype=np.float64))):
                self.cam_pos[:] = np.asarray(getattr(self.cfg, "start_pos", [-3.0, -3.0, 2.2]), dtype=np.float64)
            self.cam_pos[2] = float(np.clip(float(self.cam_pos[2]), self.min_flight_z, self.max_flight_z))
        except Exception:
            pass

    def _apply_embodied_base_and_arm(self, embodied_targets: np.ndarray):
        # Reuse parent mocap movement for vx/vy/vz/yaw/pitch and arms.
        super()._apply_embodied_base_and_arm(embodied_targets)

        # Add body roll control from embodied_targets[11].
        # Head yaw/pitch/roll are controlled separately by embodied_targets[12:15].
        emb = np.asarray(embodied_targets, dtype=np.float32).reshape(-1)
        if emb.shape[0] > 11:
            dt = float(getattr(self.cfg, "dt", 0.02))
            # If cfg has no dt, approximate from MuJoCo timestep/substeps.
            try:
                dt = float(self.model.opt.timestep) * int(getattr(self.cfg, "mj_substeps", 4))
            except Exception:
                pass
            roll_rate_cmd = float(np.clip(emb[11], -1.0, 1.0))
            roll_gain_deg_s = float(getattr(self.cfg, "mocap_roll_gain_deg_s", 70.0))
            self.roll_deg += roll_rate_cmd * roll_gain_deg_s * dt
            max_roll = float(getattr(self.cfg, "mocap_max_roll_deg", 70.0))
            self.roll_deg = float(np.clip(self.roll_deg, -max_roll, max_roll))

        self._clamp_flight_zone()
        self._apply_head_control(embodied_targets)



    def _update_rig_pose(self):
        # Parent V57 writes mocap position/quaternion. Here we keep the same
        # position but use yaw+pitch+roll quaternion.
        try:
            mid = self._mocap_id_for_body("agent_rig", fallback=0)
            self.data.mocap_pos[mid] = self.cam_pos
            self.data.mocap_quat[mid] = self._quat_from_yaw_pitch_roll()
        except Exception:
            # Fallback to parent if mocap arrays are unavailable.
            try:
                super()._update_rig_pose()
            except Exception:
                pass


    def _balance_reward_from_vestibular(self, vest: np.ndarray) -> float:
        # Layout:
        # 0:3 Lgyro, 3:6 Lacc, 6:9 Rgyro, 9:12 Racc,
        # 12:15 gyro_common, 15:18 gyro_diff,
        # 18:21 accel_common, 21:24 accel_diff
        gyro_common = vest[12:15]
        gyro_diff = vest[15:18]
        accel_diff = vest[21:24]

        gyro_penalty = float(np.linalg.norm(gyro_common))
        diff_penalty = float(np.linalg.norm(gyro_diff) + 0.5 * np.linalg.norm(accel_diff))

        reward = self.balance_reward_weight
        reward -= self.balance_gyro_penalty * gyro_penalty
        reward -= self.balance_diff_penalty * diff_penalty
        return float(np.clip(reward, -0.25, 0.25))


    def observe(self, action_id: int, embodied_targets: np.ndarray, hand_controls: np.ndarray):
        obs = super().observe(action_id, embodied_targets, hand_controls)
        vest = self.read_vestibular()
        vest_tensor = torch.from_numpy(vest).float().unsqueeze(0).to(self.device)
        obs["vestibular"] = vest_tensor

        if self.add_vestibular_to_body_state and "body_state" in obs:
            obs["body_state"] = torch.cat([obs["body_state"], vest_tensor], dim=-1)

        if "reward" in obs:
            bal = self._balance_reward_from_vestibular(vest)
            obs["balance_reward"] = torch.tensor([bal], dtype=torch.float32, device=self.device)
            obs["reward"] = obs["reward"] + obs["balance_reward"]

        return obs
