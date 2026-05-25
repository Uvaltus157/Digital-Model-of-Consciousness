from __future__ import annotations

"""Startup state helpers for the V5.10 runner.

This module extracts the simple startup flag/sensor/window-state assignment
logic from the heavy runner into a dependency-light helper.

For now the heavy `UnifiedSystemV510.__init__` still performs the original
assignments. The slim entrypoint reapplies the same state after construction so
this helper becomes the tested source of truth before a later direct edit of the
large runner file.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StartupStateSnapshot:
    training_enabled: bool
    mujoco_next_run: bool
    video_sensor_enabled: bool
    contact_sensor_enabled: bool
    imu_sensor_enabled: bool
    show_inner_world_window: bool
    show_camera_preview_window: bool
    show_latent_semantic_window: bool
    show_action_outputs_window: bool
    show_manual_action_override_window: bool
    ipc_manual_actions_enabled: bool
    ipc_manual_actions_prev_enabled: bool
    show_module_debug_window: bool
    show_inner_object_window: bool
    show_event_code_visualizer_window: bool
    show_inner_object_open3d_window: bool
    camera_preview_armed: bool
    show_static_dynamic_code_window: bool

    def to_dict(self) -> Dict[str, bool]:
        return self.__dict__.copy()


def startup_flag(startup: Any, name: str, default: bool = False) -> bool:
    """Read a boolean startup flag from `cfg.control_startup`."""
    return bool(getattr(startup, name, default))


def build_startup_state_snapshot(cfg: Any) -> StartupStateSnapshot:
    """Build the startup state snapshot from a config object."""
    startup = getattr(cfg, "control_startup", None)
    sleep_sensors = getattr(cfg, "sleep_sensors", None)

    show_camera_preview_window = startup_flag(startup, "cameras", False)

    return StartupStateSnapshot(
        training_enabled=startup_flag(startup, "training", False),
        mujoco_next_run=startup_flag(startup, "mujoco_next_run", False),
        video_sensor_enabled=bool(getattr(sleep_sensors, "video_sensor_enabled", True)),
        contact_sensor_enabled=bool(getattr(sleep_sensors, "contact_sensor_enabled", True)),
        imu_sensor_enabled=bool(getattr(sleep_sensors, "imu_sensor_enabled", True)),
        show_inner_world_window=startup_flag(startup, "inner_world", False),
        show_camera_preview_window=show_camera_preview_window,
        show_latent_semantic_window=startup_flag(startup, "latent_semantic", False),
        show_action_outputs_window=startup_flag(startup, "actions", False),
        show_manual_action_override_window=False,
        ipc_manual_actions_enabled=startup_flag(startup, "manual_actions", False),
        ipc_manual_actions_prev_enabled=False,
        show_module_debug_window=startup_flag(startup, "module_debug", False),
        show_inner_object_window=startup_flag(startup, "object_image", False),
        show_event_code_visualizer_window=startup_flag(startup, "event_code_visualizer", False),
        show_inner_object_open3d_window=startup_flag(startup, "object_image_open3d", False),
        camera_preview_armed=bool(show_camera_preview_window),
        show_static_dynamic_code_window=startup_flag(startup, "static_dynamic_code", False),
    )


def apply_startup_state_snapshot(system: Any, snapshot: StartupStateSnapshot) -> None:
    """Apply a startup state snapshot to an already constructed system."""
    system.training_enabled = snapshot.training_enabled
    try:
        system.cfg.viewer.allow_mujoco_window = snapshot.mujoco_next_run
    except Exception:
        pass

    system.video_sensor_enabled = snapshot.video_sensor_enabled
    system.contact_sensor_enabled = snapshot.contact_sensor_enabled
    system.imu_sensor_enabled = snapshot.imu_sensor_enabled

    system.show_inner_world_window = snapshot.show_inner_world_window
    system.show_camera_preview_window = snapshot.show_camera_preview_window
    system.show_latent_semantic_window = snapshot.show_latent_semantic_window
    system.show_action_outputs_window = snapshot.show_action_outputs_window
    system.show_manual_action_override_window = snapshot.show_manual_action_override_window
    system._ipc_manual_actions_enabled = snapshot.ipc_manual_actions_enabled
    system._ipc_manual_actions_prev_enabled = snapshot.ipc_manual_actions_prev_enabled
    system.show_module_debug_window = snapshot.show_module_debug_window
    system.show_inner_object_window = snapshot.show_inner_object_window
    system.show_event_code_visualizer_window = snapshot.show_event_code_visualizer_window
    system.show_inner_object_open3d_window = snapshot.show_inner_object_open3d_window
    system.camera_preview_armed = snapshot.camera_preview_armed
    system.show_static_dynamic_code_window = snapshot.show_static_dynamic_code_window


def apply_startup_state(system: Any, cfg: Any) -> StartupStateSnapshot:
    """Build and apply startup state; return the snapshot for diagnostics."""
    snapshot = build_startup_state_snapshot(cfg)
    apply_startup_state_snapshot(system, snapshot)
    return snapshot
