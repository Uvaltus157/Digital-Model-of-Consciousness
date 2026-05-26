from __future__ import annotations

"""Extracted initializer for `UnifiedSystemV510`.

This is the first direct replacement of heavy runner construction logic. The
class itself still lives in `src/apps/runner.py`, but the slim launch path
patches `UnifiedSystemV510.__init__` to this function so construction starts
using the extracted factories.
"""

from pathlib import Path
from typing import Any


def initialize_unified_system_v510(self: Any, cfg: Any) -> None:
    """Initialize a V5.10 system instance using extracted app-level factories."""
    import torch

    from src.apps.runner_components import (
        create_emotional_drive,
        create_inner_object_open3d_viewer,
        create_inner_object_system,
        create_inner_object_visualizer,
        create_latent_semantic_map_visualizer,
        create_self_core,
    )
    from src.apps.runner_dynamic_rig_config import dynamic_rig_kwargs
    from src.apps.runner_memory_factory import create_novelty_detector, create_quality_meter, create_replay_buffer, seed_python_random
    from src.apps.runner_model_factory import (
        create_base_optimizer,
        create_conscious_dreamer,
        create_conscious_dreamer_config,
        create_torch_device,
        seed_torch,
    )
    from src.apps.runner_services import ensure_runner_services
    from src.apps.runner_startup_state import startup_flag
    from src.apps.runner_visualizer_factory import create_inner_world_visualizer
    from src.apps.runner_world_factories import create_simulation_world
    from src.modules.m06_learning_sleep_consolidation.module_training_gate import ModuleTrainingGate
    from src.modules.m08_debug_visual_control.module_debug_status_ipc import ModuleDebugStatusServer
    from src.platform.ipc.ipc_control_bus import IPCControlServer
    from src.platform.mujoco_world.dynamic_agent_rig_control_hover_flight_contact_reflex import (
        DynamicAgentRigControlConfig,
        DynamicAgentRigController,
    )
    from src.shared.config import (
        ModuleDebugStatusIPCRuntimeConfig,
        ModuleTrainingDebugRuntimeConfig,
        SelfCoreRuntimeConfig,
    )

    self.cfg = cfg

    # Safety fallback for older filtered configs/checkpoints.
    if not hasattr(cfg, "self_core"):
        cfg.self_core = SelfCoreRuntimeConfig()
    if not hasattr(cfg, "module_debug"):
        cfg.module_debug = ModuleTrainingDebugRuntimeConfig()
    if not hasattr(cfg, "module_status_ipc"):
        cfg.module_status_ipc = ModuleDebugStatusIPCRuntimeConfig()

    self._force_hover_flight_runtime_config()
    self.device = create_torch_device(cfg)
    seed_torch(cfg)

    self.speech_vocab, self.speech_teacher = load_inner_speech_teacher_from_config(cfg)
    self.model_cfg = create_conscious_dreamer_config(cfg, speech_vocab_size=self.speech_vocab.size)
    # Compatibility attribute for older code/checkpoints that still look for the old name.
    self.v23_cfg = self.model_cfg
    self.model = create_conscious_dreamer(cfg, self.device, speech_vocab_size=self.speech_vocab.size)
    self.optimizer = create_base_optimizer(self.model, cfg)

    self.init_leg_control_head()

    self.inner_object_system = create_inner_object_system(cfg, self.device)
    try:
        self.optimizer.add_param_group({"params": self.inner_object_system.parameters()})
    except Exception as e:
        print(f"[inner_object] optimizer add skipped: {e}")
    self.inner_object_state = self.inner_object_system.initial_state(batch_size=1, device=self.device)
    self.latest_object_decoder_stats = {}

    self.self_core = create_self_core(cfg, self.device)
    try:
        self.optimizer.add_param_group({"params": self.self_core.parameters()})
    except Exception as e:
        print(f"[self_core] optimizer add skipped: {e}")
    self.self_core_state = self.self_core.initial_state(batch_size=1, device=self.device)
    print(
        f"[self_core] initialized | enabled={cfg.self_core.enabled} "
        f"self_dim={cfg.self_core.self_dim} action_dim={cfg.embodied_dim} "
        f"params={sum(p.numel() for p in self.self_core.parameters()):,}"
    )

    self.inner_object_viz = create_inner_object_visualizer(cfg)
    self.inner_object_open3d_viz = create_inner_object_open3d_viewer(cfg)
    self.inner_object_slot_snapshots = []
    self.last_inner_object_snapshot_step = -10**9

    seed_python_random(cfg)
    self.replay = create_replay_buffer(cfg)
    self.quality = create_quality_meter(ema_decay=0.98)
    self.novelty = create_novelty_detector(cfg)
    self.emotional_drive = create_emotional_drive(cfg)

    self.world = create_simulation_world(cfg, self.device)

    self.dynamic_agent_rig_controller = None
    if cfg.dynamic_agent_rig.enabled:
        try:
            self.dynamic_agent_rig_controller = DynamicAgentRigController(
                self.world.model,
                self.world.data,
                DynamicAgentRigControlConfig(**dynamic_rig_kwargs(cfg)),
            )
            print("[dynamic_agent_rig] enabled: freejoint force/velocity control")
        except Exception as e:
            print(f"[dynamic_agent_rig] disabled: {e}")
            self.dynamic_agent_rig_controller = None

    # Always create visualizer objects; display is controlled by startup/window flags.
    self.inner_viz = create_inner_world_visualizer(cfg, speech_vocab=self.speech_vocab)
    self.latent_semantic_viz = create_latent_semantic_map_visualizer(cfg)
    if cfg.inner_world.save_frames:
        Path(cfg.inner_world.out_dir).mkdir(parents=True, exist_ok=True)

    self.out_dir = Path(cfg.runtime.out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.log_path = self.out_dir / "live_log.jsonl"

    self.shutdown = False
    self._mujoco_runtime_viewer = None
    startup = getattr(cfg, "control_startup", None)

    self.training_enabled = startup_flag(startup, "training", False)
    self.cfg.viewer.allow_mujoco_window = startup_flag(startup, "mujoco_next_run", False)
    self.cfg.camera_preview.show_depth = True
    self.video_sensor_enabled = bool(getattr(cfg.sleep_sensors, "video_sensor_enabled", True))
    self.contact_sensor_enabled = bool(getattr(cfg.sleep_sensors, "contact_sensor_enabled", True))
    self.imu_sensor_enabled = bool(getattr(cfg.sleep_sensors, "imu_sensor_enabled", True))
    self.last_module_training_seq = 0
    self.global_step = 0
    self.train_steps = 0
    self.last_train_reason = "not_started"
    self.last_train_loss = None
    self.last_train_error = ""

    self.state = self.model.initial_state(batch_size=1, device=self.device)
    self.prev_embodied_action = torch.zeros(1, cfg.embodied_dim, device=self.device)
    self.prev_hand_motor = torch.zeros(1, cfg.hand_motor_dim, device=self.device)

    self.maybe_load_checkpoint()

    self.module_training_gate = ModuleTrainingGate(self)
    self.module_training_gate.set_flags(self.resolve_module_training_flags_from_config())

    self.module_status_server = None
    if cfg.module_status_ipc.enabled:
        self.module_status_server = ModuleDebugStatusServer(cfg.module_status_ipc.host, cfg.module_status_ipc.port)
        self.module_status_server.start()
    self.rebuild_optimizer_from_trainable_modules()
    self.write_module_debug_status()

    self.latest_stats = None
    self.latest_out = None
    self.last_print_time = 0.0
    self._action_trace = {}

    self.show_inner_world_window = startup_flag(startup, "inner_world", False)
    self.show_camera_preview_window = startup_flag(startup, "cameras", False)
    self.show_latent_semantic_window = startup_flag(startup, "latent_semantic", False)
    self.show_action_outputs_window = startup_flag(startup, "actions", False)
    self.show_manual_action_override_window = False
    self._ipc_manual_actions_enabled = startup_flag(startup, "manual_actions", False)
    self._ipc_manual_actions_prev_enabled = False
    self._ipc_manual_body_action = None
    self._ipc_manual_arm_action = None
    self._ipc_manual_hand_action = None
    self._ipc_manual_leg_action = None
    self.show_module_debug_window = startup_flag(startup, "module_debug", False)
    self.show_inner_object_window = startup_flag(startup, "object_image", False)
    self.show_event_code_visualizer_window = startup_flag(startup, "event_code_visualizer", False)
    self.show_inner_object_open3d_window = startup_flag(startup, "object_image_open3d", False)
    self.start_slot_4d_jsonrpc_streamer_if_enabled()

    self.camera_preview_armed = bool(self.show_camera_preview_window)
    self.ipc_server = None
    self.ipc_close_counter = 0
    if cfg.ipc_control.enabled:
        self.ipc_server = IPCControlServer(cfg.ipc_control.host, cfg.ipc_control.port)
        self.ipc_server.start()

    self.show_static_dynamic_code_window = startup_flag(startup, "static_dynamic_code", False)
    self.external_control_last_mtime = 0.0
    self.external_control_last_close_counter = 0
    self._write_initial_external_control_flags()
    self._init_sensor_preview_metadata()

    # Idempotent service boundary for the slim path. If the old code above
    # already started services, this does not create duplicates.
    ensure_runner_services(self, cfg)
