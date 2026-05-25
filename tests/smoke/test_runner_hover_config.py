from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_hover_config import apply_hover_flight_runtime_config


def test_apply_hover_flight_runtime_config_clamps_values() -> None:
    rig = SimpleNamespace(
        max_force=18.0,
        max_torque=10.0,
        max_vertical_speed=0.1,
        max_angular_speed=0.3,
        angular_kv=2.0,
        upright_kp=99.0,
        upright_kd=99.0,
        contact_angular_damping_enabled=False,
        contact_roll_pitch_damping=1,
        contact_yaw_damping=2,
        contact_spin_limit=3,
        hover_kp=10.0,
        hover_kd=10.0,
        gravity_compensation=False,
        hover_enabled=False,
        dynamic_hover_target=False,
        upright_enabled=False,
    )
    cfg = SimpleNamespace(dynamic_agent_rig=rig)

    apply_hover_flight_runtime_config(cfg)

    assert rig.max_force == 800.0
    assert rig.max_torque == 120.0
    assert rig.max_vertical_speed == 0.45
    assert rig.max_angular_speed == 1.2
    assert rig.angular_kv == 14.0
    assert rig.upright_kp == 8.0
    assert rig.upright_kd == 2.0
    assert rig.contact_angular_damping_enabled is True
    assert rig.hover_kp == 260.0
    assert rig.hover_kd == 55.0
    assert rig.gravity_compensation is True
    assert rig.hover_enabled is True
    assert rig.dynamic_hover_target is True
    assert rig.upright_enabled is True


def test_apply_hover_flight_runtime_config_preserves_safe_higher_values() -> None:
    rig = SimpleNamespace(
        max_force=900.0,
        max_torque=130.0,
        max_vertical_speed=0.8,
        max_angular_speed=2.0,
        angular_kv=20.0,
        upright_kp=4.0,
        upright_kd=1.0,
        contact_angular_damping_enabled=False,
        contact_roll_pitch_damping=1,
        contact_yaw_damping=2,
        contact_spin_limit=3,
        hover_kp=300.0,
        hover_kd=60.0,
        gravity_compensation=False,
        hover_enabled=False,
        dynamic_hover_target=False,
        upright_enabled=False,
    )
    cfg = SimpleNamespace(dynamic_agent_rig=rig)

    apply_hover_flight_runtime_config(cfg)

    assert rig.max_force == 900.0
    assert rig.max_torque == 130.0
    assert rig.max_vertical_speed == 0.8
    assert rig.max_angular_speed == 2.0
    assert rig.angular_kv == 20.0
    assert rig.upright_kp == 4.0
    assert rig.upright_kd == 1.0
    assert rig.hover_kp == 300.0
    assert rig.hover_kd == 60.0
