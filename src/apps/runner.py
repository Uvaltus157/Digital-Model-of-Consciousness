from __future__ import annotations

"""
unified_conscious_viewer_v5_10.py

V5.10:
- upgrades V5.7 from ConsciousDreamerV2.2 -> ConsciousDreamerV2.3
- upgrades inner world window to show internal object image panel
- keeps MuJoCo life-loop, realistic hands, tactile sensors, memory, thought loop,
  symbolic report and live visualization

Dependencies expected in the same project folder:
- conscious_dreamer_object_imagery.py
- inner_world_visualizer_object_image.py
- unified_conscious_viewer.py
"""

import json
import os
import importlib

import numpy as np
import time
from pathlib import Path
from threading import Thread

import mujoco
import mujoco.viewer

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
import random

from src.shared.console_colors import install_colored_errors

install_colored_errors()

# models
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
)
from src.modules.m09_self_core.models.self_core import SelfCore, SelfCoreConfig, build_self_experience_text, pad_or_trim_selfcore
from src.modules.m01_object_imagery.models.object_inner_imagery_3d import InnerObjectRepresentation3DSystem, ObjectInnerImagery3DConfig, summarize_vision_tensors, pad_or_trim
# visualizer
from src.modules.m07_inner_speech_thoughts.inner_world_visualizer_text_thought_hybrid import DreamerInnerWorldVisualizerV3
from src.modules.m07_inner_speech_thoughts.inner_world_visualizer import InnerWorldVizConfig
from src.modules.m01_object_imagery.visualizer_inner_object_3d import InnerObject3DVisualizer
from src.modules.m01_object_imagery.visualizer_inner_object import InnerObjectVisualizerV2, InnerObjectVisualizerV2Config
from src.modules.m01_object_imagery.inner_object_open3d_viewer import InnerObjectOpen3DViewerV2, InnerObjectOpen3DViewerV2Config
from src.modules.m14_semantic_grounding.latent_semantic_map import LatentSemanticMapConfig, LatentSemanticMapVisualizer
# ctrl_panel
from src.platform.ipc.ipc_control_bus import IPCControlServer
from src.modules.m08_debug_visual_control.module_debug_status_ipc import ModuleDebugStatusServer, ModuleDebugStatusIPCConfig

from src.modules.m06_learning_sleep_consolidation.module_training_gate import ModuleTrainingGate, ModuleTrainingDebugConfig, DEFAULT_MODULE_FLAGS

from src.platform.mujoco_world.dynamic_agent_rig_control_hover_flight_contact_reflex import DynamicAgentRigControlConfig, DynamicAgentRigController
#from mujoco_live_world_dynamic_rig_vestibular_body import MujocoLiveWorldDynamicRigVestibularBody as MujocoLiveWorldDynamicRig
from src.platform.mujoco_world.mujoco_live_world_mocap_contacts import MujocoLiveWorldMocapContacts as MujocoLiveWorldDynamicRig

from src.modules.m03_self_action_causality.manual_action_override_window import ManualActionOverrideWindow, ManualActionOverrideConfig
from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive, EmotionalDriveConfig


        
# Reuse the stable MuJoCo world / configs / helpers from V5.7
from src.apps.unified_conscious_viewer import (
    UnifiedV57Config,
    UnifiedSystemV57,
    MujocoLiveWorldV57,
    InnerWorldWindowConfig,
    NoveltyConfig,
    ReplayConfig,
    LifeConfig,
    MujocoWorldConfig,
    ViewerConfig,
    RuntimeConfig,
    ReplayBuffer, 
    QualityMeter, 
    NoveltyDetector
)

from src.shared.config import (
    make_v23_config_from_unified,
    TrainLoopV510Config,
    IPCControlConfig,
    ExternalControlConfig,
    ActionOutputsWindowConfig,
    InnerObjectImageConfig,
    ModuleDebugStatusIPCRuntimeConfig,
    ControlStartupConfig,
    SleepSensorGateRuntimeConfig,
    ModuleTrainingDebugRuntimeConfig,
    ActionSignalTraceConfig,
    SelfCoreRuntimeConfig,
    MocapFlightBoundsConfig,
    VestibularRuntimeConfig,
    ManualActionOverrideRuntimeConfig,
    InnerObjectOpen3DConfig,
    CameraPreviewConfig,
    CheckpointLoadConfig,
    EmotionalDriveRuntimeConfig,
    TetraDynamicSlotDiagnosticConfig,
    LatentSemanticPanelConfig,
    ExplorationMotorConfig,
    DynamicAgentRigRuntimeConfig,
    BirdBodyRuntimeConfig,
    LegControlHeadConfig,
    UnifiedV510Config,
)
from src.platform.mujoco_world.camera_preview_window import CameraPreviewMixin
from src.modules.m03_self_action_causality.action_outputs_window import ActionOutputsMixin
from src.modules.m06_learning_sleep_consolidation.sleep_sensors import SleepSensorsMixin
from src.platform.ipc.external_control import ExternalControlMixin
from src.platform.ipc.ipc_runtime import IPCRuntimeMixin
from src.modules.m01_object_imagery.runtime import ObjectImageryRuntimeMixin
from src.shared.checkpointing import CheckpointingMixin
from src.modules.m08_debug_visual_control.module_status_runtime import ModuleStatusRuntimeMixin
from src.modules.m03_self_action_causality.action_runtime import ActionRuntimeMixin
from src.platform.mujoco_world.leg_bird_runtime import LegBirdRuntimeMixin
from src.modules.m09_self_core.self_core_runtime import SelfCoreRuntimeMixin
from src.modules.m01_object_imagery.inner_visual_runtime import InnerVisualRuntimeMixin
from src.modules.m06_learning_sleep_consolidation.training_runtime import TrainingRuntimeMixin
from src.apps.life_runtime import LifeRuntimeMixin
from src.modules.m05_world_model_attention_workspace.tetra_dynamic_slot_diagnostic import TetraDynamicSlotDiagnosticMixin
from src.platform.mujoco_world.mujoco_viewer_thread import MujocoViewerThread
from src.platform.gui.opencv_gui_thread import shutdown_cv2_gui_thread


def load_inner_speech_teacher_from_config(cfg):
    train_cfg = getattr(cfg, "train", None)
    teacher_file = str(getattr(train_cfg, "inner_speech_teacher_file", "english_inner_speech_teacher.py"))
    if bool(getattr(train_cfg, "russian_inner_speech_teacher_enabled", False)):
        teacher_file = "russian_inner_speech_teacher.py"
    elif not bool(getattr(train_cfg, "english_inner_speech_teacher_enabled", True)):
        print("[inner_speech_teacher] english teacher disabled; falling back to russian teacher")
        teacher_file = "russian_inner_speech_teacher.py"

    module_name = Path(teacher_file).stem if teacher_file.endswith(".py") else teacher_file
    if "." not in module_name:
        module_name = f"src.modules.m07_inner_speech_thoughts.{module_name}"
    module = importlib.import_module(module_name)

    vocab_cls = (
        getattr(module, "InnerSpeechVocab", None)
        or getattr(module, "AnglishInnerSpeechVocab", None)
        or getattr(module, "EnglishInnerSpeechVocab", None)
        or getattr(module, "RussianInnerSpeechVocab", None)
    )
    teacher_cls = (
        getattr(module, "InnerSpeechTeacher", None)
        or getattr(module, "AnglishInnerSpeechTeacher", None)
        or getattr(module, "EnglishInnerSpeechTeacher", None)
        or getattr(module, "RussianInnerSpeechTeacher", None)
    )
    if vocab_cls is None or teacher_cls is None:
        raise RuntimeError(f"inner speech teacher module {module_name!r} must expose vocab and teacher classes")

    vocab = vocab_cls()
    teacher = teacher_cls(vocab)
    print(f"[inner_speech_teacher] loaded {module_name} | vocab={vocab.size}")
    return vocab, teacher


class UnifiedSystemV510(
                        CameraPreviewMixin, 
                        ActionOutputsMixin, 
                        SleepSensorsMixin, 
                        ExternalControlMixin, 
                        IPCRuntimeMixin, 
                        ObjectImageryRuntimeMixin, 
                        CheckpointingMixin, 
                        ModuleStatusRuntimeMixin, 
                        ActionRuntimeMixin, 
                        LegBirdRuntimeMixin, 
                        SelfCoreRuntimeMixin, 
                        InnerVisualRuntimeMixin, 
                        TrainingRuntimeMixin, 
                        LifeRuntimeMixin, 
                        TetraDynamicSlotDiagnosticMixin,
                        UnifiedSystemV57):
    """
    Minimal upgrade over V5.7:
    - model = ConsciousDreamerV23
    - inner_viz = DreamerInnerWorldVisualizerV3
    - loss gets a small term for object imagery stability
    """

    def resolve_module_training_flags_from_config(self) -> dict:
        """
        Resolve startup module train/passive flags from config.

        YAML:
            module_debug:
              module_modes:
                world_model: train
                object_imagery: passive

        Values:
            train   -> module participates in optimizer/train step
            passive -> module is used in life/forward, but parameters are frozen

        The old train_core_model/train_world_model/... booleans were removed.
        """
        md = getattr(self.cfg, "module_debug", None)
        modes = getattr(md, "module_modes", None) if md is not None else None

        defaults = {
            "world_model": True,
            "object_imagery": True,
            "core_model": True,
            "action_heads": True,
            "leg_control": True,
            "self_core": True,
            "inner_speech": True,
            "long_dynamic_memory": True,
        }

        aliases = {
            "train": True,
            "training": True,
            "on": True,
            "enabled": True,
            "active": True,
            "true": True,
            "1": True,

            "passive": False,
            "frozen": False,
            "freeze": False,
            "off": False,
            "disabled": False,
            "false": False,
            "0": False,
        }

        if not isinstance(modes, dict) or len(modes) == 0:
            print("[module_debug] module_modes missing/empty; using all modules in train mode")
            return defaults

        flags = dict(defaults)
        for key, value in modes.items():
            key = str(key).strip()
            if key not in flags:
                print(f"[module_debug] unknown module_modes key ignored: {key!r}")
                continue
            if isinstance(value, bool):
                flags[key] = bool(value)
                continue
            mode = str(value).lower().strip()
            if mode not in aliases:
                print(f"[module_debug] unknown mode for {key}: {value!r}; using default=train")
                continue
            flags[key] = bool(aliases[mode])

        print("[module_debug] startup module modes -> " + ", ".join(
            f"{k}={'train' if v else 'passive'}" for k, v in flags.items()
        ))
        return flags

    def __init__(self, cfg: UnifiedV510Config) -> None:
        # build core fields without calling V57.__init__, because we swap the model and visualizer
        self.cfg = cfg
        # Safety fallback for older filtered configs/checkpoints:
        # if self_core section was removed before OmegaConf.to_object(), restore defaults.
        if not hasattr(cfg, "self_core"):
            cfg.self_core = SelfCoreRuntimeConfig()
        if not hasattr(cfg, "module_debug"):
            cfg.module_debug = ModuleTrainingDebugRuntimeConfig()
        if not hasattr(cfg, "module_status_ipc"):
            cfg.module_status_ipc = ModuleDebugStatusIPCRuntimeConfig()
        self._force_hover_flight_runtime_config()
        self.device = torch.device(cfg.runtime.device)
        torch.manual_seed(cfg.runtime.seed)

        self.v23_cfg = make_v23_config_from_unified(cfg)
        self.speech_vocab, self.speech_teacher = load_inner_speech_teacher_from_config(cfg)
        self.v23_cfg.symbolic_report.text_vocab_size = self.speech_vocab.size
        try:
            self.inner_viz.speech_vocab = self.speech_vocab
        except Exception:
            pass
        self.model = ConsciousDreamerV23(self.v23_cfg).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.init_leg_control_head()

        self.inner_object_system = InnerObjectRepresentation3DSystem(ObjectInnerImagery3DConfig(
            enabled=cfg.object_image.enabled,
            latent_dim=cfg.object_image.latent_dim,
            hidden_dim=cfg.object_image.hidden_dim,
            image_size=cfg.object_image.image_size,
            tactile_dim=cfg.object_image.tactile_dim,
            body_dim=cfg.embodied_dim,
            hand_dim=cfg.hand_motor_dim,
            leg_dim=cfg.leg_control.leg_motor_dim,
            point_count=cfg.object_image.point_count,
            voxel_res=cfg.object_image.voxel_res,
            num_slots=getattr(cfg.object_image, "num_slots", 4),
            max_object_proposals=getattr(cfg.object_image, "max_object_proposals", 4),
            proposal_slot_lock=getattr(cfg.object_image, "proposal_slot_lock", True),
            sleep_freeze_memory_update=getattr(cfg.object_image, "sleep_freeze_object_slots", False),
            dream_latent_dynamics=getattr(cfg.object_image, "dream_latent_dynamics", True),
            dream_strength=getattr(cfg.object_image, "dream_strength", 0.025),
            dream_cycle_slots=getattr(cfg.object_image, "dream_cycle_slots", False),
            dream_slot_cycle_steps=getattr(cfg.object_image, "dream_slot_cycle_steps", 90),
            dream_empty_confidence_threshold=getattr(cfg.object_image, "dream_empty_confidence_threshold", 0.05),
        )).to(self.device)
        try:
            self.optimizer.add_param_group({"params": self.inner_object_system.parameters()})
        except Exception as e:
            print(f"[inner_object] optimizer add skipped: {e}")
        self.inner_object_state = self.inner_object_system.initial_state(batch_size=1, device=self.device)
        self.latest_object_decoder_stats = {}
        self.self_core = SelfCore(SelfCoreConfig(
            enabled=cfg.self_core.enabled,
            body_state_dim=cfg.body_state_dim,
            action_dim=cfg.embodied_dim,
            tactile_dim=cfg.tactile_dim,
            vestibular_dim=24,
            object_latent_dim=cfg.self_core.object_latent_dim,
            workspace_dim=cfg.self_core.workspace_dim,
            hidden_dim=cfg.self_core.hidden_dim,
            self_dim=cfg.self_core.self_dim,
        )).to(self.device)
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
        """
        self.inner_object_viz = InnerObject3DVisualizer(
            window_name=cfg.object_image.window_name,
            width=cfg.object_image.width,
            height=cfg.object_image.height,
        )
        """
        self.inner_object_viz = InnerObjectVisualizerV2(InnerObjectVisualizerV2Config(
            window_name=cfg.object_image.window_name,
            width=max(int(cfg.object_image.width), 1520),
            height=max(int(cfg.object_image.height), 1260),
            max_slots=getattr(cfg.object_image, "num_slots", 10),
        ))
        
        self.inner_object_open3d_viz = InnerObjectOpen3DViewerV2(InnerObjectOpen3DViewerV2Config(
            enabled=cfg.object_image_open3d.enabled,
            window_name=cfg.object_image_open3d.window_name,
            width=cfg.object_image_open3d.width,
            height=cfg.object_image_open3d.height,
            update_every_steps=cfg.object_image_open3d.update_every_steps,
            point_size=cfg.object_image_open3d.point_size,
            voxel_threshold=cfg.object_image_open3d.voxel_threshold,
            max_voxel_points=cfg.object_image_open3d.max_voxel_points,
            show_voxels=cfg.object_image_open3d.show_voxels,
            use_internal_color=cfg.object_image_open3d.use_internal_color,
            max_slots=cfg.object_image_open3d.max_slots,
            slot_spacing=cfg.object_image_open3d.slot_spacing,
            export_dir=cfg.object_image_open3d.export_dir,
        ))
        self.inner_object_slot_snapshots = []
        self.last_inner_object_snapshot_step = -10**9

        # Reuse parent initialization logic by importing the same helper classes indirectly:

        random.seed(cfg.runtime.seed)

        self.replay = ReplayBuffer(cfg.replay.capacity)
        self.quality = QualityMeter(ema_decay=0.98)
        self.novelty = NoveltyDetector(cfg.novelty)

        self.emotional_drive = EmotionalDrive(EmotionalDriveConfig(
            enabled=cfg.emotional_drive.enabled,
            ema_decay=cfg.emotional_drive.ema_decay,
            reward_scale=cfg.emotional_drive.reward_scale,
            w_gap_fill=cfg.emotional_drive.w_gap_fill,
            w_coherence_gain=cfg.emotional_drive.w_coherence_gain,
            w_object_conf_gain=cfg.emotional_drive.w_object_conf_gain,
            w_multimodal_alignment=cfg.emotional_drive.w_multimodal_alignment,
            w_contact_pleasure=cfg.emotional_drive.w_contact_pleasure,
            w_curiosity=cfg.emotional_drive.w_curiosity,
            w_inner_speech_conf=cfg.emotional_drive.w_inner_speech_conf,
            w_uncertainty_increase=cfg.emotional_drive.w_uncertainty_increase,
            w_coherence_loss=cfg.emotional_drive.w_coherence_loss,
            w_object_conf_loss=cfg.emotional_drive.w_object_conf_loss,
            w_speech_conf_loss=cfg.emotional_drive.w_speech_conf_loss,
            w_alignment_loss=cfg.emotional_drive.w_alignment_loss,
            w_chaotic_touch=cfg.emotional_drive.w_chaotic_touch,
            w_instability=cfg.emotional_drive.w_instability,
        ))

        self.world = MujocoLiveWorldDynamicRig(
            self.device,
            cfg.mujoco_world,
            embodied_dim=cfg.embodied_dim,
            hand_motor_dim=cfg.hand_motor_dim,
            #use_dynamic_agent_rig=cfg.dynamic_agent_rig.enabled,
            add_vestibular_to_body_state=cfg.vestibular.add_to_body_state,
            balance_reward_weight=cfg.vestibular.balance_reward_weight,
            balance_gyro_penalty=cfg.vestibular.balance_gyro_penalty,
            balance_diff_penalty=cfg.vestibular.balance_diff_penalty,
        )

        # Dynamic physical control for agent_rig. If the scene contains
        # agent_rig_free, this replaces mocap-style motion with force/velocity control.
        self.dynamic_agent_rig_controller = None
        if cfg.dynamic_agent_rig.enabled:
            try:
                self.dynamic_agent_rig_controller = DynamicAgentRigController(
                    self.world.model,
                    self.world.data,
                    DynamicAgentRigControlConfig(
                        enabled=cfg.dynamic_agent_rig.enabled,
                        body_name=cfg.dynamic_agent_rig.body_name,
                        freejoint_name=cfg.dynamic_agent_rig.freejoint_name,
                        max_linear_speed=cfg.dynamic_agent_rig.max_linear_speed,
                        max_vertical_speed=cfg.dynamic_agent_rig.max_vertical_speed,
                        max_angular_speed=cfg.dynamic_agent_rig.max_angular_speed,
                        linear_kv=cfg.dynamic_agent_rig.linear_kv,
                        angular_kv=cfg.dynamic_agent_rig.angular_kv,
                        max_force=cfg.dynamic_agent_rig.max_force,
                        max_torque=cfg.dynamic_agent_rig.max_torque,
                        min_z=cfg.dynamic_agent_rig.min_z,
                        max_z=cfg.dynamic_agent_rig.max_z,
                        ground_push_k=cfg.dynamic_agent_rig.ground_push_k,
                        local_frame_linear=cfg.dynamic_agent_rig.local_frame_linear,
                        gravity_compensation=cfg.dynamic_agent_rig.gravity_compensation,
                        hover_enabled=cfg.dynamic_agent_rig.hover_enabled,
                        hover_height=cfg.dynamic_agent_rig.hover_height,
                        dynamic_hover_target=cfg.dynamic_agent_rig.dynamic_hover_target,
                        min_hover_height=cfg.dynamic_agent_rig.min_hover_height,
                        max_hover_height=cfg.dynamic_agent_rig.max_hover_height,
                        vertical_command_gain=cfg.dynamic_agent_rig.vertical_command_gain,
                        hover_kp=cfg.dynamic_agent_rig.hover_kp,
                        hover_kd=cfg.dynamic_agent_rig.hover_kd,
                        upright_enabled=cfg.dynamic_agent_rig.upright_enabled,
                        upright_kp=cfg.dynamic_agent_rig.upright_kp,
                        upright_kd=cfg.dynamic_agent_rig.upright_kd,
                        emergency_lift_enabled=cfg.dynamic_agent_rig.emergency_lift_enabled,
                        emergency_z=cfg.dynamic_agent_rig.emergency_z,
                        emergency_vz=cfg.dynamic_agent_rig.emergency_vz,
                        contact_angular_damping_enabled=cfg.dynamic_agent_rig.contact_angular_damping_enabled,
                        contact_roll_pitch_damping=cfg.dynamic_agent_rig.contact_roll_pitch_damping,
                        contact_yaw_damping=cfg.dynamic_agent_rig.contact_yaw_damping,
                        contact_spin_limit=cfg.dynamic_agent_rig.contact_spin_limit,
                        contact_active_angular_damping=cfg.dynamic_agent_rig.contact_active_angular_damping,
                        contact_active_yaw_damping=cfg.dynamic_agent_rig.contact_active_yaw_damping,
                        contact_active_upright_kp=cfg.dynamic_agent_rig.contact_active_upright_kp,
                        contact_active_upright_kd=cfg.dynamic_agent_rig.contact_active_upright_kd,
                        contact_torque_limit=cfg.dynamic_agent_rig.contact_torque_limit,
                    ),
                )
                print("[dynamic_agent_rig] enabled: freejoint force/velocity control")
            except Exception as e:
                print(f"[dynamic_agent_rig] disabled: {e}")
                self.dynamic_agent_rig_controller = None

        viz_cfg = InnerWorldVizConfig(width=cfg.inner_world.width, height=cfg.inner_world.height)
        # Always create visualizer object; display is controlled by show_inner_world_window.
        self.inner_viz = DreamerInnerWorldVisualizerV3(viz_cfg)
        self.latent_semantic_viz = LatentSemanticMapVisualizer(
            LatentSemanticMapConfig(
                enabled=cfg.latent_semantic_map.enabled,
                window_name=cfg.latent_semantic_map.window_name,
                width=cfg.latent_semantic_map.width,
                height=cfg.latent_semantic_map.height,
                max_history=cfg.latent_semantic_map.max_history,
                show_every_steps=cfg.latent_semantic_map.show_every_steps,
                delay_ms=cfg.latent_semantic_map.delay_ms,
                thumbnail_size=cfg.latent_semantic_map.thumbnail_size,
                max_thumbnails=cfg.latent_semantic_map.max_thumbnails,
                point_radius=cfg.latent_semantic_map.point_radius,
                draw_grid=cfg.latent_semantic_map.draw_grid,
                follow_inner_world_toggle=cfg.latent_semantic_map.follow_inner_world_toggle,
            )
        )
        if cfg.inner_world.save_frames:
            Path(cfg.inner_world.out_dir).mkdir(parents=True, exist_ok=True)

        self.out_dir = Path(cfg.runtime.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "live_log.jsonl"

        self.shutdown = False
        self._mujoco_runtime_viewer = None
        startup = getattr(cfg, "control_startup", ControlStartupConfig())

        def startup_flag(name: str, default: bool = False) -> bool:
            return bool(getattr(startup, name, default))

        self.training_enabled = startup_flag("training", False)
        self.cfg.viewer.allow_mujoco_window = startup_flag("mujoco_next_run", False)
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

        # Load model checkpoint if it exists.
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

        self.show_inner_world_window = startup_flag("inner_world", False)
        self.show_camera_preview_window = startup_flag("cameras", False)
        self.show_latent_semantic_window = startup_flag("latent_semantic", False)
        self.show_action_outputs_window = startup_flag("actions", False)
        self.show_manual_action_override_window = False
        self._ipc_manual_actions_enabled = startup_flag("manual_actions", False)
        self._ipc_manual_actions_prev_enabled = False
        self._ipc_manual_body_action = None
        self._ipc_manual_arm_action = None
        self._ipc_manual_hand_action = None
        self._ipc_manual_leg_action = None
        self.show_module_debug_window = startup_flag("module_debug", False)
        self.show_inner_object_window = startup_flag("object_image", False)
        self.show_event_code_visualizer_window = startup_flag("event_code_visualizer", False)
        self.show_inner_object_open3d_window = startup_flag("object_image_open3d", False)
        self.start_slot_4d_jsonrpc_streamer_if_enabled()

        # Hard gate: prevents camera preview from opening at startup because of
        # unrelated inherited code paths. Config/startup or IPC may arm it.
        self.camera_preview_armed = bool(self.show_camera_preview_window)
        self.ipc_server = None
        self.ipc_close_counter = 0
        if cfg.ipc_control.enabled:
            self.ipc_server = IPCControlServer(cfg.ipc_control.host, cfg.ipc_control.port)
            self.ipc_server.start()

        self.show_static_dynamic_code_window = startup_flag("static_dynamic_code", False)
        self.external_control_last_mtime = 0.0
        self.external_control_last_close_counter = 0
        self._write_initial_external_control_flags()
        self._init_sensor_preview_metadata()

    def _force_hover_flight_runtime_config(self):
        """
        Safety patch: old YAML/checkpoints often keep max_force=18.
        That clips lift and makes the physical body fall. Force flight-safe
        runtime values before DynamicAgentRigController is created.
        """
        try:
            self.cfg.dynamic_agent_rig.max_force = max(float(self.cfg.dynamic_agent_rig.max_force), 800.0)
            self.cfg.dynamic_agent_rig.max_torque = max(float(self.cfg.dynamic_agent_rig.max_torque), 120.0)
            self.cfg.dynamic_agent_rig.max_vertical_speed = max(float(self.cfg.dynamic_agent_rig.max_vertical_speed), 0.45)
            self.cfg.dynamic_agent_rig.max_angular_speed = max(float(self.cfg.dynamic_agent_rig.max_angular_speed), 1.2)
            self.cfg.dynamic_agent_rig.angular_kv = max(float(self.cfg.dynamic_agent_rig.angular_kv), 14.0)
            self.cfg.dynamic_agent_rig.upright_kp = min(float(self.cfg.dynamic_agent_rig.upright_kp), 8.0)
            self.cfg.dynamic_agent_rig.upright_kd = min(float(self.cfg.dynamic_agent_rig.upright_kd), 2.0)
            self.cfg.dynamic_agent_rig.contact_angular_damping_enabled = True
            self.cfg.dynamic_agent_rig.contact_roll_pitch_damping = float(self.cfg.dynamic_agent_rig.contact_roll_pitch_damping)
            self.cfg.dynamic_agent_rig.contact_yaw_damping = float(self.cfg.dynamic_agent_rig.contact_yaw_damping)
            self.cfg.dynamic_agent_rig.contact_spin_limit = float(self.cfg.dynamic_agent_rig.contact_spin_limit)
            self.cfg.dynamic_agent_rig.hover_kp = max(float(self.cfg.dynamic_agent_rig.hover_kp), 260.0)
            self.cfg.dynamic_agent_rig.hover_kd = max(float(self.cfg.dynamic_agent_rig.hover_kd), 55.0)
            self.cfg.dynamic_agent_rig.gravity_compensation = True
            self.cfg.dynamic_agent_rig.hover_enabled = True
            self.cfg.dynamic_agent_rig.dynamic_hover_target = True
            self.cfg.dynamic_agent_rig.upright_enabled = True
        except Exception as e:
            print(f"[hover_config] force patch skipped: {e}")


    def run(self) -> None:
        self.world.reset()
        self.log_tetra_runner_started()

        train_thread = Thread(target=self.train_loop, daemon=True)
        train_thread.start()

        period = 1.0 / max(self.cfg.life.fps, 1e-6)

        viewer = None
        last_viewer_sync_time = 0.0
        threaded_viewer = None
        if bool(getattr(self.cfg.viewer, "mujoco_threaded", False)):
            threaded_viewer = MujocoViewerThread(
                self.world.model,
                sync_fps=float(getattr(self.cfg.viewer, "mujoco_sync_fps", 8.0)),
                show_left_ui=False,
                show_right_ui=False,
            )
            threaded_viewer.start()
      
        while not self.shutdown and self.global_step < self.cfg.life.max_steps:
            t0 = time.time()
            self.life_step()
            self.tick_slot_4d_jsonrpc_streamer()
            # --------- mujoco.viewer --------------- 
            if threaded_viewer is not None:
              threaded_viewer.set_enabled(bool(self.cfg.viewer.allow_mujoco_window))
              sync_every_steps = max(1, int(getattr(self.cfg.viewer, "mujoco_sync_every_steps", 1)))
              if self.cfg.viewer.allow_mujoco_window and (self.global_step % sync_every_steps) == 0:
                threaded_viewer.update_from(self.world.data)
            else:
              if self.cfg.viewer.allow_mujoco_window==True and viewer is None:
                viewer = mujoco.viewer.launch_passive(
                    self.world.model,
                    self.world.data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
              elif self.cfg.viewer.allow_mujoco_window==False and viewer is not None:
                viewer.close()
                viewer = None

              if viewer is not None:
                if not viewer.is_running():
                  try:
                    viewer.close()
                  except Exception:
                    pass
                  viewer = None
                  self.cfg.viewer.allow_mujoco_window = False
                else:
                  sync_fps = float(getattr(self.cfg.viewer, "mujoco_sync_fps", 8.0))
                  sync_every_steps = max(1, int(getattr(self.cfg.viewer, "mujoco_sync_every_steps", 1)))
                  sync_interval = 0.0 if sync_fps <= 0.0 else 1.0 / max(sync_fps, 1e-6)
                  now = time.time()
                  if (self.global_step % sync_every_steps) == 0 and (now - last_viewer_sync_time) >= sync_interval:
                    try:
                      viewer.sync()
                      last_viewer_sync_time = now
                    except Exception as e:
                      print(f"[mujoco.viewer] sync failed, closing viewer: {e}")
                      try:
                        viewer.close()
                      except Exception:
                        pass
                      viewer = None
                      self.cfg.viewer.allow_mujoco_window = False
               
            self.maybe_print_status()

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

        self.shutdown = True
        train_thread.join(timeout=2.0)
        if threaded_viewer is not None:
            threaded_viewer.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        try:
            shutdown_cv2_gui_thread(timeout=3.0)
        except Exception as e:
            print(f"[opencv_gui_thread] shutdown skipped: {e}")
        self.shutdown_slot_4d_jsonrpc_streamer()
        #self.save_checkpoint("last.pt")
        self.maybe_save_checkpoint(force=True, owner="life")
        self.world.close()

os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))


@hydra.main(version_base=None, config_path="../../config", config_name="runner")
def main(cfg_raw) -> None:
    raw = OmegaConf.create(OmegaConf.to_container(cfg_raw, resolve=False))
    base = OmegaConf.structured(UnifiedV510Config())

    allowed_top_keys = {
        "mode",
        "novelty",
        "replay",
        "life",
        "train",
        "mujoco_world",
        "viewer",
        "runtime",
        "checkpoint_load",
        "emotional_drive",
        "inner_world",
        "ipc_control",
        "external_control",
        "camera_preview",
        "action_outputs",
        "latent_semantic_map",
        "action_dim",
        "embodied_dim",
        "hand_motor_dim",
        "tactile_dim",
        "body_state_dim",
        "exploration",
        "dynamic_agent_rig",
        "bird_body",
        "leg_control",
        "module_debug",
        "module_status_ipc",
        "control_startup",
        "sleep_sensors",
        "object_image",
        "object_image_open3d",
        "vestibular",
        "mocap_flight_bounds",
        "self_core",
        "manual_action_override",
        "action_trace",
        "adaptive_scenario_controller",
        "tetra_dynamic_slot_diagnostic",
        "inner_speech_loss_weight",
    }

    allowed_nested_keys = {
        "novelty": set(OmegaConf.structured(NoveltyConfig()).keys()),
        "replay": set(OmegaConf.structured(ReplayConfig()).keys()),
        "life": set(OmegaConf.structured(LifeConfig()).keys()),
        "train": set(OmegaConf.structured(TrainLoopV510Config()).keys()),
        "mujoco_world": set(OmegaConf.structured(MujocoWorldConfig()).keys()),
        "viewer": set(OmegaConf.structured(ViewerConfig()).keys()),
        "runtime": set(OmegaConf.structured(RuntimeConfig()).keys()),
        "checkpoint_load": set(OmegaConf.structured(CheckpointLoadConfig()).keys()),
        "tetra_dynamic_slot_diagnostic": set(OmegaConf.structured(TetraDynamicSlotDiagnosticConfig()).keys()),
        "emotional_drive": set(OmegaConf.structured(EmotionalDriveRuntimeConfig()).keys()),
        "exploration": set(OmegaConf.structured(ExplorationMotorConfig()).keys()),
        "dynamic_agent_rig": set(OmegaConf.structured(DynamicAgentRigRuntimeConfig()).keys()),
        "bird_body": set(OmegaConf.structured(BirdBodyRuntimeConfig()).keys()),
        "leg_control": set(OmegaConf.structured(LegControlHeadConfig()).keys()),
        "inner_world": set(OmegaConf.structured(InnerWorldWindowConfig()).keys()),
        "ipc_control": set(OmegaConf.structured(IPCControlConfig()).keys()),
        "external_control": set(OmegaConf.structured(ExternalControlConfig()).keys()),
        "camera_preview": set(OmegaConf.structured(CameraPreviewConfig()).keys()),
        "action_outputs": set(OmegaConf.structured(ActionOutputsWindowConfig()).keys()),
        "module_debug": set(OmegaConf.structured(ModuleTrainingDebugRuntimeConfig()).keys()),
        "module_status_ipc": set(OmegaConf.structured(ModuleDebugStatusIPCRuntimeConfig()).keys()),
        "control_startup": set(OmegaConf.structured(ControlStartupConfig()).keys()),
        "sleep_sensors": set(OmegaConf.structured(SleepSensorGateRuntimeConfig()).keys()),
        "object_image": set(OmegaConf.structured(InnerObjectImageConfig()).keys()),
        "object_image_open3d": set(OmegaConf.structured(InnerObjectOpen3DConfig()).keys()),
        "latent_semantic_map": set(OmegaConf.structured(LatentSemanticPanelConfig()).keys()),
        "manual_action_override": set(OmegaConf.structured(ManualActionOverrideRuntimeConfig()).keys()),
        "action_trace": set(OmegaConf.structured(ActionSignalTraceConfig()).keys()),
        "vestibular": set(OmegaConf.structured(VestibularRuntimeConfig()).keys()),
        "mocap_flight_bounds": set(OmegaConf.structured(MocapFlightBoundsConfig()).keys()),
        "self_core": set(OmegaConf.structured(SelfCoreRuntimeConfig()).keys()),
    }

    clean_dict = {}
    for key in raw.keys():
        if key not in allowed_top_keys:
            continue
        if key in allowed_nested_keys:
            if OmegaConf.is_config(raw[key]) or isinstance(raw[key], dict):
                clean_dict[key] = {
                    sub_key: raw[key][sub_key]
                    for sub_key in raw[key].keys()
                    if sub_key in allowed_nested_keys[key]
                }
            continue
        clean_dict[key] = raw[key]

    clean = OmegaConf.create(clean_dict)
    cfg = OmegaConf.merge(base, clean)

    print("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    cfg_obj: UnifiedV510Config = OmegaConf.to_object(cfg)

    system = UnifiedSystemV510(cfg_obj)

    mode = str(getattr(cfg_obj, "mode", "run")).lower().strip()
    if mode in ("train", "training"):
        # Training is a parallel thread, not a separate sequential loop.
        # The inherited run() owns life_step and starts train_loop in background.
        system.training_enabled = bool(getattr(cfg_obj.control_startup, "training", False))
        if system.training_enabled:
            try:
                system.cfg.train.enabled = True
            except Exception:
                pass
            print("[mode=train] parallel train thread enabled; starting normal life run")
        else:
            print("[mode=train] training stays OFF because control_startup.training=false")

    system.run()


if __name__ == "__main__":
    main()
