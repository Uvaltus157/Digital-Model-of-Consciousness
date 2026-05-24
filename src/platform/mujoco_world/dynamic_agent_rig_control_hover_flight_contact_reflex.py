from __future__ import annotations

"""
dynamic_agent_rig_control.py

Dynamic/freejoint control for agent_rig.

Goal:
    Remove mocap control from the embodied agent.
    agent_rig becomes a normal MuJoCo dynamic body with a freejoint.

Control:
    neural embodied_action_head
        -> target velocities [vx, vy, vz, yaw_rate, pitch_rate, roll_rate]
        -> PD velocity controller
        -> xfrc_applied force/torque on agent_rig

This keeps MuJoCo physics authoritative:
    - contacts can stop motion
    - ground can block penetration
    - object collisions have normal contact velocities
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class DynamicAgentRigControlConfig:
    enabled: bool = True

    body_name: str = "agent_rig"
    freejoint_name: str = "agent_rig_free"

    # embodied action vector indices
    vx_idx: int = 0
    vy_idx: int = 1
    vz_idx: int = 2
    yaw_idx: int = 3
    pitch_idx: int = 4
    roll_idx: int = 5

    max_linear_speed: float = 0.75
    max_vertical_speed: float = 0.35
    max_angular_speed: float = 1.1

    # PD velocity controller gains
    linear_kv: float = 35.0
    angular_kv: float = 8.0

    max_force: float = 800.0
    max_torque: float = 14.0

    # flight / hover stabilization.
    # Without mocap the rig is a real physical body, so it needs lift.
    gravity_compensation: bool = True
    hover_enabled: bool = True
    hover_height: float = 1.65
    dynamic_hover_target: bool = True
    min_hover_height: float = 0.75
    max_hover_height: float = 3.0
    vertical_command_gain: float = 0.55
    hover_kp: float = 220.0
    hover_kd: float = 45.0
    emergency_lift_enabled: bool = True
    emergency_z: float = 0.85
    emergency_vz: float = 1.2

    # Keep the body from tumbling while still allowing yaw.
    upright_enabled: bool = True
    upright_kp: float = 18.0
    upright_kd: float = 5.0

    # Contact/vestibular reflex: damp angular velocity caused by limb contacts.
    contact_angular_damping_enabled: bool = True
    contact_roll_pitch_damping: float = 6.0
    contact_yaw_damping: float = 2.0
    contact_spin_limit: float = 2.0
    contact_spin_deadzone: float = 0.25

    # Strong damping only while limb/object contact is detected.
    contact_active_angular_damping: float = 65.0
    contact_active_yaw_damping: float = 18.0
    contact_active_upright_kp: float = 45.0
    contact_active_upright_kd: float = 16.0
    contact_torque_limit: float = 55.0

    # safety
    min_z: float = 0.55
    max_z: float = 2.2
    ground_push_k: float = 100.0
    clear_xfrc_each_step: bool = True

    # if true, map local forward/right commands to world frame using current yaw
    local_frame_linear: bool = True

    # prevent dead policy
    deadzone: float = 0.015


class DynamicAgentRigController:
    def __init__(self, model, data, cfg: Optional[DynamicAgentRigControlConfig] = None):
        self.model = model
        self.data = data
        self.cfg = cfg or DynamicAgentRigControlConfig()

        import mujoco

        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.body_name)
        if self.body_id < 0:
            raise ValueError(f"Body not found: {self.cfg.body_name}")

        self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.cfg.freejoint_name)
        if self.joint_id < 0:
            raise ValueError(f"Freejoint not found: {self.cfg.freejoint_name}")

        self.qpos_adr = int(model.jnt_qposadr[self.joint_id])
        self.qvel_adr = int(model.jnt_dofadr[self.joint_id])
        self.hover_target_z = float(self.cfg.hover_height)
        self.external_contact_level = 0.0

    def clear_forces(self):
        if self.cfg.clear_xfrc_each_step:
            self.data.xfrc_applied[:, :] = 0.0

    def _extract_target_velocity(self, embodied_action: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(embodied_action, dtype=np.float64).reshape(-1)
        if a.size == 0:
            return np.zeros(3), np.zeros(3)

        def get(idx: int, default: float = 0.0) -> float:
            return float(a[idx]) if 0 <= idx < a.size else default

        vx = get(self.cfg.vx_idx)
        vy = get(self.cfg.vy_idx)
        vz = get(self.cfg.vz_idx)
        yaw = get(self.cfg.yaw_idx)
        pitch = get(self.cfg.pitch_idx)
        roll = get(self.cfg.roll_idx)

        # tanh bounds neural output safely
        lin = np.array([
            np.tanh(vx) * self.cfg.max_linear_speed,
            np.tanh(vy) * self.cfg.max_linear_speed,
            np.tanh(vz) * self.cfg.max_vertical_speed,
        ], dtype=np.float64)

        ang = np.array([
            np.tanh(roll) * self.cfg.max_angular_speed,
            np.tanh(pitch) * self.cfg.max_angular_speed,
            np.tanh(yaw) * self.cfg.max_angular_speed,
        ], dtype=np.float64)

        lin[np.abs(lin) < self.cfg.deadzone] = 0.0
        ang[np.abs(ang) < self.cfg.deadzone] = 0.0
        return lin, ang

    def _roll_pitch_from_quat(self) -> tuple[float, float]:
        # freejoint quaternion qpos order: x y z qw qx qy qz
        q = self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7].copy()
        qw, qx, qy, qz = q

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return float(roll), float(pitch)


    def _rotate_local_xy_to_world(self, lin: np.ndarray) -> np.ndarray:
        if not self.cfg.local_frame_linear:
            return lin

        # freejoint quaternion qpos order: x y z qw qx qy qz
        q = self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7].copy()
        qw, qx, qy, qz = q
        # yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        c, s = np.cos(yaw), np.sin(yaw)
        x, y, z = lin
        return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)

    def apply(self, embodied_action: Sequence[float]) -> dict:
        self.clear_forces()

        target_lin, target_ang = self._extract_target_velocity(embodied_action)
        target_lin = self._rotate_local_xy_to_world(target_lin)

        qpos = self.data.qpos[self.qpos_adr:self.qpos_adr + 7]
        qvel = self.data.qvel[self.qvel_adr:self.qvel_adr + 6]

        cur_lin = qvel[0:3].copy()
        cur_ang = qvel[3:6].copy()

        # Neural vertical command controls target altitude.
        # target_lin[2] comes from embodied_action[vz_idx].
        if self.cfg.dynamic_hover_target:
            dt = float(getattr(self.model.opt, "timestep", 0.01))
            self.hover_target_z += float(target_lin[2]) * float(self.cfg.vertical_command_gain) * dt
            self.hover_target_z = float(np.clip(self.hover_target_z, self.cfg.min_hover_height, self.cfg.max_hover_height))
        else:
            self.hover_target_z = float(self.cfg.hover_height)

        force = self.cfg.linear_kv * (target_lin - cur_lin)
        torque = self.cfg.angular_kv * (target_ang - cur_ang)

        # z safety: do not keep pushing down through ground; push up if too low.
        z = float(qpos[2])

        rig_mass = 1.0
        if self.cfg.gravity_compensation:
            try:
                rig_mass = float(self.model.body_subtreemass[self.body_id])
                if rig_mass <= 0.0:
                    rig_mass = float(self.model.body_mass[self.body_id])
            except Exception:
                rig_mass = float(self.model.body_mass[self.body_id])
            # Use a small safety factor because contacts/children/joints can add load.
            force[2] += rig_mass * 9.81 * 1.25

        if self.cfg.hover_enabled:
            z_err = float(self.hover_target_z - z)
            vz_current = float(cur_lin[2])
            force[2] += self.cfg.hover_kp * z_err - self.cfg.hover_kd * vz_current

        if z < self.cfg.min_z:
            force[2] += self.cfg.ground_push_k * (self.cfg.min_z - z)
            target_lin[2] = max(0.0, target_lin[2])
        elif z > self.cfg.max_z:
            force[2] -= self.cfg.ground_push_k * (z - self.cfg.max_z)
            target_lin[2] = min(0.0, target_lin[2])

        if self.cfg.upright_enabled:
            roll, pitch = self._roll_pitch_from_quat()
            torque[0] += -self.cfg.upright_kp * roll - self.cfg.upright_kd * float(cur_ang[0])
            torque[1] += -self.cfg.upright_kp * pitch - self.cfg.upright_kd * float(cur_ang[1])

        # Contact anti-spin damping.
        # When limbs touch objects, contact impulses create angular velocity.
        # This acts like a vestibular reflex: oppose roll/pitch/yaw spin.
        contact_level = float(getattr(self, "external_contact_level", 0.0))
        if self.cfg.contact_angular_damping_enabled:
            spin = np.clip(cur_ang, -self.cfg.contact_spin_limit, self.cfg.contact_spin_limit)

            # Weak free-flight damping with deadzone.
            for ax in (0, 1):
                if abs(float(spin[ax])) > getattr(self.cfg, 'contact_spin_deadzone', 0.25):
                    torque[ax] += -self.cfg.contact_roll_pitch_damping * float(spin[ax])
            if abs(float(spin[2])) > getattr(self.cfg, 'contact_spin_deadzone', 0.25):
                torque[2] += -self.cfg.contact_yaw_damping * float(spin[2])

            # Strong reflex only during actual limb/object contact.
            if contact_level > 0.0:
                torque[0] += -contact_level * self.cfg.contact_active_angular_damping * float(cur_ang[0])
                torque[1] += -contact_level * self.cfg.contact_active_angular_damping * float(cur_ang[1])
                torque[2] += -contact_level * self.cfg.contact_active_yaw_damping * float(cur_ang[2])

                # During contact, strongly resist roll/pitch tilt, but still allow height/x/y motion.
                roll, pitch = self._roll_pitch_from_quat()
                torque[0] += -contact_level * self.cfg.contact_active_upright_kp * roll - contact_level * self.cfg.contact_active_upright_kd * float(cur_ang[0])
                torque[1] += -contact_level * self.cfg.contact_active_upright_kp * pitch - contact_level * self.cfg.contact_active_upright_kd * float(cur_ang[1])

        # Emergency anti-fall layer. It is still dynamic: we do not set qpos,
        # only kill downward velocity and add lift if the body falls too low.
        if self.cfg.emergency_lift_enabled and z < self.cfg.emergency_z:
            force[2] += self.cfg.max_force * 0.85
            if self.data.qvel[self.qvel_adr + 2] < 0.0:
                self.data.qvel[self.qvel_adr + 2] = max(float(self.data.qvel[self.qvel_adr + 2]), float(self.cfg.emergency_vz))

        force = np.clip(force, -self.cfg.max_force, self.cfg.max_force)
        if float(getattr(self, "external_contact_level", 0.0)) > 0.0:
            contact_limit = min(float(self.cfg.max_torque), float(self.cfg.contact_torque_limit))
            torque = np.clip(torque, -contact_limit, contact_limit)
        else:
            torque = np.clip(torque, -self.cfg.max_torque, self.cfg.max_torque)

        self.data.xfrc_applied[self.body_id, 0:3] = force
        self.data.xfrc_applied[self.body_id, 3:6] = torque

        return {
            "target_lin": target_lin.astype(np.float32),
            "target_ang": target_ang.astype(np.float32),
            "cur_lin": cur_lin.astype(np.float32),
            "cur_ang": cur_ang.astype(np.float32),
            "force": force.astype(np.float32),
            "torque": torque.astype(np.float32),
            "z": z,
            "hover_height": float(self.cfg.hover_height),
            "hover_target_z": float(self.hover_target_z),
            "rig_mass": float(rig_mass),
            "force_z": float(force[2]),
            "gravity_compensation": bool(self.cfg.gravity_compensation),
            "hover_enabled": bool(self.cfg.hover_enabled),
            "upright_enabled": bool(self.cfg.upright_enabled),
            "contact_angular_damping_enabled": bool(self.cfg.contact_angular_damping_enabled),
            "contact_level": float(getattr(self, "external_contact_level", 0.0)),
        }


def patch_mocap_agent_rig_xml_to_dynamic(xml: str) -> str:
    """
    Best-effort XML text patch:
      <body name="agent_rig" mocap="true" ...>
    becomes:
      <body name="agent_rig" ...>
        <freejoint name="agent_rig_free"/>
        <geom name="agent_rig_collision" .../>

    If the XML already contains agent_rig_free, it is left unchanged.
    """
    if "agent_rig_free" in xml:
        return xml

    # remove mocap attr only from agent_rig body opening tag
    def repl_body(m):
        tag = m.group(0)
        tag = re.sub(r'\s+mocap="true"', "", tag)
        if tag.endswith(">"):
            return tag + '\n      <freejoint name="agent_rig_free"/>\n      <geom name="agent_rig_collision" type="box" size="0.18 0.12 0.10" mass="1.0" contype="1" conaffinity="1" condim="3" rgba="0.2 0.4 1.0 0.12"/>'
        return tag

    import re
    xml = re.sub(r'<body\b(?=[^>]*name="agent_rig")(?=[^>]*mocap="true")[^>]*>', repl_body, xml, count=1)

    # If the tag did not have mocap=true but no freejoint exists, still insert after agent_rig body tag.
    if "agent_rig_free" not in xml:
        xml = re.sub(
            r'(<body\b(?=[^>]*name="agent_rig")[^>]*>)',
            r'\1\n      <freejoint name="agent_rig_free"/>\n      <geom name="agent_rig_collision" type="box" size="0.18 0.12 0.10" mass="1.0" contype="1" conaffinity="1" condim="3" rgba="0.2 0.4 1.0 0.12"/>',
            xml,
            count=1,
        )
    return xml
