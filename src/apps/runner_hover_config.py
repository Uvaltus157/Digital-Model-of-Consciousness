from __future__ import annotations

"""Flight-safe hover configuration patch for the V5.10 runner.

This extracts `UnifiedSystem._force_hover_flight_runtime_config()` from the
large runner into a small app-level helper. The values are intentionally the
same safety clamps used by the previous runner implementation.
"""

from typing import Any


def apply_hover_flight_runtime_config(cfg: Any) -> None:
    """Apply flight-safe runtime clamps to `cfg.dynamic_agent_rig` in place."""
    rig = cfg.dynamic_agent_rig
    rig.max_force = max(float(rig.max_force), 800.0)
    rig.max_torque = max(float(rig.max_torque), 120.0)
    rig.max_vertical_speed = max(float(rig.max_vertical_speed), 0.45)
    rig.max_angular_speed = max(float(rig.max_angular_speed), 1.2)
    rig.angular_kv = max(float(rig.angular_kv), 14.0)
    rig.upright_kp = min(float(rig.upright_kp), 8.0)
    rig.upright_kd = min(float(rig.upright_kd), 2.0)
    rig.contact_angular_damping_enabled = True
    rig.contact_roll_pitch_damping = float(rig.contact_roll_pitch_damping)
    rig.contact_yaw_damping = float(rig.contact_yaw_damping)
    rig.contact_spin_limit = float(rig.contact_spin_limit)
    rig.hover_kp = max(float(rig.hover_kp), 260.0)
    rig.hover_kd = max(float(rig.hover_kd), 55.0)
    rig.gravity_compensation = True
    rig.hover_enabled = True
    rig.dynamic_hover_target = True
    rig.upright_enabled = True


def force_hover_flight_runtime_config_for_system(system: Any) -> None:
    """Method-compatible wrapper for `UnifiedSystem` instances."""
    try:
        apply_hover_flight_runtime_config(system.cfg)
    except Exception as e:
        print(f"[hover_config] force patch skipped: {e}")
