from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_startup_state import (
    apply_startup_state,
    apply_startup_state_snapshot,
    build_startup_state_snapshot,
    startup_flag,
)


def _cfg(**startup_values):
    startup_defaults = {
        "training": False,
        "mujoco_next_run": False,
        "inner_world": False,
        "cameras": False,
        "latent_semantic": False,
        "actions": False,
        "manual_actions": False,
        "module_debug": False,
        "object_image": False,
        "event_code_visualizer": False,
        "object_image_open3d": False,
        "static_dynamic_code": False,
    }
    startup_defaults.update(startup_values)
    return SimpleNamespace(
        control_startup=SimpleNamespace(**startup_defaults),
        sleep_sensors=SimpleNamespace(
            video_sensor_enabled=True,
            contact_sensor_enabled=False,
            imu_sensor_enabled=True,
        ),
        viewer=SimpleNamespace(allow_mujoco_window=False),
    )


def test_startup_flag_reads_bool_with_default() -> None:
    startup = SimpleNamespace(cameras=True)
    assert startup_flag(startup, "cameras") is True
    assert startup_flag(startup, "missing", True) is True
    assert startup_flag(None, "missing", False) is False


def test_build_startup_state_snapshot_from_config() -> None:
    cfg = _cfg(
        training=True,
        mujoco_next_run=True,
        inner_world=True,
        cameras=True,
        manual_actions=True,
        object_image=True,
    )
    snap = build_startup_state_snapshot(cfg)

    assert snap.training_enabled is True
    assert snap.mujoco_next_run is True
    assert snap.video_sensor_enabled is True
    assert snap.contact_sensor_enabled is False
    assert snap.imu_sensor_enabled is True
    assert snap.show_inner_world_window is True
    assert snap.show_camera_preview_window is True
    assert snap.camera_preview_armed is True
    assert snap.ipc_manual_actions_enabled is True
    assert snap.show_inner_object_window is True
    assert snap.show_manual_action_override_window is False


def test_apply_startup_state_snapshot_to_system() -> None:
    cfg = _cfg(training=True, mujoco_next_run=True, cameras=True, object_image_open3d=True)
    snap = build_startup_state_snapshot(cfg)
    system = SimpleNamespace(cfg=cfg)

    apply_startup_state_snapshot(system, snap)

    assert system.training_enabled is True
    assert system.cfg.viewer.allow_mujoco_window is True
    assert system.show_camera_preview_window is True
    assert system.camera_preview_armed is True
    assert system.show_inner_object_open3d_window is True
    assert system.contact_sensor_enabled is False


def test_apply_startup_state_returns_snapshot() -> None:
    cfg = _cfg(static_dynamic_code=True)
    system = SimpleNamespace(cfg=cfg)

    snap = apply_startup_state(system, cfg)

    assert snap.show_static_dynamic_code_window is True
    assert system.show_static_dynamic_code_window is True
