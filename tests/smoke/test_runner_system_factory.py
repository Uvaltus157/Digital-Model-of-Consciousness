from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_system_factory import BuiltSystemContext, build_unified_system


class DummySystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.training_enabled = False
        self.module_status_server = "already-running"
        self.ipc_server = "already-running"
        self.external_calls = 0
        self.sensor_calls = 0

    def _write_initial_external_control_flags(self):
        self.external_calls += 1

    def _init_sensor_preview_metadata(self):
        self.sensor_calls += 1


def _cfg(tmp_path, mode="run", training=False):
    return SimpleNamespace(
        mode=mode,
        runtime=SimpleNamespace(out_dir=str(tmp_path / "out")),
        viewer=SimpleNamespace(allow_mujoco_window=False),
        train=SimpleNamespace(enabled=False),
        control_startup=SimpleNamespace(
            training=training,
            mujoco_next_run=True,
            inner_world=True,
            cameras=True,
            latent_semantic=False,
            actions=False,
            manual_actions=False,
            module_debug=False,
            object_image=False,
            event_code_visualizer=False,
            object_image_open3d=False,
            static_dynamic_code=False,
        ),
        sleep_sensors=SimpleNamespace(
            video_sensor_enabled=True,
            contact_sensor_enabled=True,
            imu_sensor_enabled=True,
        ),
        module_status_ipc=SimpleNamespace(enabled=True, host="127.0.0.1", port=8766),
        ipc_control=SimpleNamespace(enabled=True, host="127.0.0.1", port=8765),
    )


def test_build_unified_system_returns_system(tmp_path) -> None:
    system = build_unified_system(_cfg(tmp_path), DummySystem)
    assert isinstance(system, DummySystem)
    assert system.out_dir.exists()
    assert system.show_inner_world_window is True
    assert system.show_camera_preview_window is True
    assert system.cfg.viewer.allow_mujoco_window is True
    assert system.external_calls == 1
    assert system.sensor_calls == 1


def test_build_unified_system_can_return_context(tmp_path) -> None:
    system, context = build_unified_system(_cfg(tmp_path, mode="train", training=True), DummySystem, return_context=True)
    assert isinstance(system, DummySystem)
    assert isinstance(context, BuiltSystemContext)
    assert context.mode == "train"
    assert context.startup_state.training_enabled is True
    assert context.services.module_status_running is True
    assert system.training_enabled is True
    assert system.cfg.train.enabled is True
