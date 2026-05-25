from __future__ import annotations

"""Config-to-kwargs helper for dynamic rig control."""

from typing import Any, Dict


def dynamic_rig_kwargs(cfg: Any) -> Dict[str, Any]:
    rig = cfg.dynamic_agent_rig
    return {
        "enabled": rig.enabled,
        "body_name": rig.body_name,
        "freejoint_name": rig.freejoint_name,
        "max_linear_speed": rig.max_linear_speed,
        "max_vertical_speed": rig.max_vertical_speed,
        "max_angular_speed": rig.max_angular_speed,
        "linear_kv": rig.linear_kv,
        "angular_kv": rig.angular_kv,
        "max_force": rig.max_force,
        "max_torque": rig.max_torque,
        "min_z": rig.min_z,
        "max_z": rig.max_z,
        "ground_push_k": rig.ground_push_k,
        "local_frame_linear": rig.local_frame_linear,
        "gravity_compensation": rig.gravity_compensation,
        "hover_enabled": rig.hover_enabled,
        "hover_height": rig.hover_height,
        "dynamic_hover_target": rig.dynamic_hover_target,
        "min_hover_height": rig.min_hover_height,
        "max_hover_height": rig.max_hover_height,
        "vertical_command_gain": rig.vertical_command_gain,
        "hover_kp": rig.hover_kp,
        "hover_kd": rig.hover_kd,
        "upright_enabled": rig.upright_enabled,
        "upright_kp": rig.upright_kp,
        "upright_kd": rig.upright_kd,
        "emergency_lift_enabled": rig.emergency_lift_enabled,
        "emergency_z": rig.emergency_z,
        "emergency_vz": rig.emergency_vz,
        "contact_angular_damping_enabled": rig.contact_angular_damping_enabled,
        "contact_roll_pitch_damping": rig.contact_roll_pitch_damping,
        "contact_yaw_damping": rig.contact_yaw_damping,
        "contact_spin_limit": rig.contact_spin_limit,
        "contact_active_angular_damping": rig.contact_active_angular_damping,
        "contact_active_yaw_damping": rig.contact_active_yaw_damping,
        "contact_active_upright_kp": rig.contact_active_upright_kp,
        "contact_active_upright_kd": rig.contact_active_upright_kd,
        "contact_torque_limit": rig.contact_torque_limit,
    }
