from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_dynamic_rig_config import dynamic_rig_kwargs
from src.apps.runner_world_factories import simulation_world_kwargs, simulation_world_snapshot


def _cfg():
    return SimpleNamespace(
        embodied_dim=64,
        hand_motor_dim=32,
        vestibular=SimpleNamespace(
            add_to_body_state=True,
            balance_reward_weight=1.2,
            balance_gyro_penalty=0.3,
            balance_diff_penalty=0.4,
        ),
        dynamic_agent_rig=SimpleNamespace(
            enabled=True,
            body_name="agent_rig",
            freejoint_name="agent_rig_free",
            max_linear_speed=1.0,
            max_vertical_speed=0.5,
            max_angular_speed=1.2,
            linear_kv=10.0,
            angular_kv=14.0,
            max_force=800.0,
            max_torque=120.0,
            min_z=0.2,
            max_z=5.0,
            ground_push_k=10.0,
            local_frame_linear=True,
            gravity_compensation=True,
            hover_enabled=True,
            hover_height=1.2,
            dynamic_hover_target=True,
            min_hover_height=0.6,
            max_hover_height=3.0,
            vertical_command_gain=1.0,
            hover_kp=260.0,
            hover_kd=55.0,
            upright_enabled=True,
            upright_kp=8.0,
            upright_kd=2.0,
            emergency_lift_enabled=True,
            emergency_z=0.4,
            emergency_vz=0.6,
            contact_angular_damping_enabled=True,
            contact_roll_pitch_damping=1.0,
            contact_yaw_damping=1.0,
            contact_spin_limit=2.0,
            contact_active_angular_damping=1.0,
            contact_active_yaw_damping=1.0,
            contact_active_upright_kp=1.0,
            contact_active_upright_kd=1.0,
            contact_torque_limit=100.0,
        ),
    )


def test_simulation_world_kwargs() -> None:
    kwargs = simulation_world_kwargs(_cfg())
    assert kwargs["embodied_dim"] == 64
    assert kwargs["hand_motor_dim"] == 32
    assert kwargs["add_vestibular_to_body_state"] is True
    assert kwargs["balance_reward_weight"] == 1.2


def test_simulation_world_snapshot_is_json_friendly() -> None:
    snap = simulation_world_snapshot(_cfg())
    data = snap.to_dict()
    assert data["kwargs"]["embodied_dim"] == 64


def test_dynamic_rig_kwargs() -> None:
    kwargs = dynamic_rig_kwargs(_cfg())
    assert kwargs["enabled"] is True
    assert kwargs["body_name"] == "agent_rig"
    assert kwargs["freejoint_name"] == "agent_rig_free"
    assert kwargs["max_force"] == 800.0
    assert kwargs["hover_enabled"] is True
