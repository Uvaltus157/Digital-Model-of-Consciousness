from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

from omegaconf import MISSING

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import ConsciousDreamerV23Config

from src.apps.unified_conscious_viewer import (
    InnerWorldWindowConfig,
    NoveltyConfig,
    ReplayConfig,
    LifeConfig,
    TrainLoopConfig,
    MujocoWorldConfig,
    ViewerConfig,
    RuntimeConfig,
)


@dataclass
class TrainLoopV510Config(TrainLoopConfig):
    inner_speech_teacher_file: str = "english_inner_speech_teacher.py"
    english_inner_speech_teacher_enabled: bool = True
    russian_inner_speech_teacher_enabled: bool = False


MODEL_DIM_KEYS = (
    "action_dim",
    "embodied_dim",
    "hand_motor_dim",
    "tactile_dim",
    "body_state_dim",
)


def _cfg_get(cfg, key: str):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg[key]
    raise KeyError(key)


def model_dimensions_from_runner_cfg(cfg) -> dict:
    dims = {key: int(_cfg_get(cfg, key)) for key in MODEL_DIM_KEYS}
    return dims


def validate_runner_model_dimensions(cfg) -> dict:
    dims = model_dimensions_from_runner_cfg(cfg)
    expected = {
        "action_dim": 24,
        "embodied_dim": 15,
        "hand_motor_dim": 44,
        "tactile_dim": 42,
        "body_state_dim": 83,
    }
    bad = {k: (dims[k], expected[k]) for k in expected if dims[k] != expected[k]}
    if bad:
        raise ValueError(
            "Invalid model dimensions in runner config: "
            + ", ".join(f"{k}={got} expected {exp}" for k, (got, exp) in bad.items())
        )
    return dims

def make_v23_config_from_unified(cfg: "UnifiedV510Config") -> ConsciousDreamerV23Config:
    dims = validate_runner_model_dimensions(cfg)

    model_cfg = ConsciousDreamerV23Config()
    model_cfg.data.image_height = cfg.mujoco_world.height
    model_cfg.data.image_width = cfg.mujoco_world.width

    # runner.yaml is the single source of truth for model dimensions.
    model_cfg.data.body_state_dim = dims["body_state_dim"]
    model_cfg.data.tactile_dim = dims["tactile_dim"]
    model_cfg.data.hand_motor_dim = dims["hand_motor_dim"]
    model_cfg.data.embodied_dim = dims["embodied_dim"]
    model_cfg.data.action_dim = dims["action_dim"]

    model_cfg.object_imagery.image_size = 96
    return model_cfg


@dataclass
class IPCControlConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    poll_every_steps: int = 1


@dataclass
class ExternalControlConfig:
    enabled: bool = True
    state_file: str = "/tmp/conscious_viewer_control.json"
    poll_every_steps: int = 1


@dataclass
class ActionOutputsWindowConfig:
    enabled: bool = True
    window_name: str = "action outputs visualizer"
    width: int = 1780
    height: int = 820
    show_every_steps: int = 2
    delay_ms: int = 1
    scale: float = 1.0


@dataclass
class InnerObjectImageConfig:
    enabled: bool = True
    window_name: str = "inner object imagery 3D"
    width: int = 1420
    height: int = 1260
    show_every_steps: int = 2
    latent_dim: int = 128
    hidden_dim: int = 256
    image_size: int = 64
    tactile_dim: int = MISSING
    # Self-supervised object decoder training.
    decoder_loss_enabled: bool = True
    decoder_loss_weight: float = 0.20
    decoder_rgb_weight: float = 1.00
    decoder_depth_weight: float = 0.75
    decoder_mask_weight: float = 0.50
    decoder_conf_weight: float = 0.10
    decoder_use_geometry_objectness: bool = True
    # 3D
    point_count: int = 128
    voxel_res: int = 16
    # Multi-slot object memory / proposals
    num_slots: int = 10
    max_object_proposals: int = 10
    include_ground_proposal: bool = True
    proposal_slot_lock: bool = True
    # Sleep / dream mode
    sleep_decode_from_memory: bool = True
    sleep_freeze_object_slots: bool = False  # deprecated: memory is live; dream_mode handles full sleep
    sleep_skip_decoder_loss: bool = True
    dream_latent_dynamics: bool = True
    dream_strength: float = 0.025
    dream_cycle_slots: bool = False
    dream_slot_cycle_steps: int = 90
    dream_empty_confidence_threshold: float = 0.05

    # Step 2B: configurable Gaussian renderer backend / preview.
    # Values: "torch_lowres" | "cuda_3dgs" | "auto"
    slot_gaussian_renderer_backend: str = "auto"
    slot_gaussian_cuda_allow_fallback: bool = True
    slot_gaussian_image_size: int = 64
    slot_gaussian_max_gaussians: int = 768
    slot_gaussian_max_render_gaussians: int = 256
    slot_gaussian_lr: float = 0.003
    slot_gaussian_train_steps_per_update: int = 1
    slot_gaussian_depth_weight: float = 0.35
    slot_gaussian_preview_every_steps: int = 1

    # Step 3A: per-slot 4D timeline over Gaussian states.
    slot_4d_timeline_enabled: bool = True
    slot_4d_timeline_max_frames: int = 256
    slot_4d_sample_points: int = 128

    # Step 3B: neural deformation field over Step-3A timelines.
    slot_4d_deformation_enabled: bool = True
    slot_4d_deformation_hidden_dim: int = 96
    slot_4d_deformation_lr: float = 0.002
    slot_4d_deformation_train_steps_per_update: int = 1
    slot_4d_deformation_min_frames: int = 2
    slot_4d_deformation_delta_reg_weight: float = 0.0001

    # Step 3C: deformation-aware 4D playback preview.
    slot_4d_playback_enabled: bool = True
    slot_4d_playback_period_steps: int = 120
    slot_4d_playback_strength: float = 1.0

    # Step 3F: separate Open3D Slot Viewer export.
    slot_4d_open3d_export_enabled: bool = True
    slot_4d_open3d_export_path: str = "./checkpoint/slot_viewer/slot_4d_open3d_latest.npz"
    slot_4d_open3d_sample_points: int = 4096
    slot_4d_open3d_min_interval_sec: float = 0.05

    # Step 3H: JSON-RPC live stream for Inner Object Open3D.
    slot_4d_open3d_transport: str = "jsonrpc"
    slot_4d_jsonrpc_enabled: bool = True
    slot_4d_jsonrpc_host: str = "127.0.0.1"
    slot_4d_jsonrpc_port: int = 8771
    slot_4d_jsonrpc_sample_points: int = 4096

    # Step 4D/4E: persistent object memory / recall diagnostics.
    slot_object_memory_enabled: bool = True
    slot_object_memory_root: str = "./checkpoint/slot_memory"
    slot_object_memory_match_threshold: float = 0.30


@dataclass
class ModuleDebugStatusIPCRuntimeConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8766


@dataclass
class ControlStartupConfig:
    mujoco_next_run: bool = False
    inner_world: bool = False
    cameras: bool = False
    depth: bool = False
    actions: bool = False
    manual_actions: bool = False
    object_image: bool = False
    event_code_visualizer: bool = False
    object_image_open3d: bool = False
    static_dynamic_code: bool = False
    latent_semantic: bool = False
    module_debug: bool = False
    training: bool = False


@dataclass
class SleepSensorGateRuntimeConfig:
    enabled: bool = True
    # Startup mode:
    #   "config"   -> use explicit video/contact/imu booleans below
    #   "active"   -> video/contact/imu ON
    #   "sleep"    -> video/contact/imu OFF
    #   "blind"    -> video OFF, contact/imu ON
    #   "body_only"-> video/contact OFF, imu ON
    startup_state: str = "config"
    video_sensor_enabled: bool = True
    contact_sensor_enabled: bool = True
    imu_sensor_enabled: bool = True


@dataclass
class ModuleTrainingDebugRuntimeConfig:
    enabled: bool = True
    window_name: str = "module debug / отладка модулей"
    width: int = 760
    height: int = 520
    show_every_steps: int = 2

    # Startup training mode for each subsystem.
    #
    # Values:
    #   "train"   -> module participates in optimizer/train step
    #   "passive" -> module is used in life/forward, but parameters are frozen
    #
    # This replaces the old train_core_model/train_world_model/... booleans.
    module_modes: Dict[str, str] = field(default_factory=lambda: {
        "world_model": "train",
        "object_imagery": "train",
        "long_dynamic_memory": "train",
        "core_model": "train",
        "action_heads": "train",
        "leg_control": "train",
        "self_core": "train",
        "inner_speech": "train",
    })


@dataclass
class ActionSignalTraceConfig:
    enabled: bool = True
    print_every_steps: int = 15
    force_when_manual: bool = False
    force_gain: float = 1.0


@dataclass
class SelfCoreRuntimeConfig:
    enabled: bool = True
    self_dim: int = 128
    hidden_dim: int = 256
    workspace_dim: int = 256
    object_latent_dim: int = 128
    loss_weight: float = 0.02
    print_every_steps: int = 30


@dataclass
class MocapFlightBoundsConfig:
    enabled: bool = True
    min_z: float = 0.1
    max_z: float = 10.0


@dataclass
class VestibularRuntimeConfig:
    enabled: bool = True
    add_to_body_state: bool = True
    print_every_steps: int = 30
    balance_reward_weight: float = 0.04
    balance_gyro_penalty: float = 0.015
    balance_diff_penalty: float = 0.010
    balance_loss_weight: float = 0.03


@dataclass
class ManualActionOverrideRuntimeConfig:
    enabled: bool = True
    window_name: str = "manual body action override"
    max_linear: float = 2.0
    max_angular: float = 1.5
    width: int = 760
    height: int = 480
    show_every_steps: int = 1


@dataclass
class InnerObjectOpen3DConfig:
    enabled: bool = True
    window_name: str = "inner object Open3D"
    width: int = 960
    height: int = 760
    update_every_steps: int = 2
    point_size: float = 5.0
    voxel_threshold: float = 0.60
    max_voxel_points: int = 1200
    show_voxels: bool = True
    use_internal_color: bool = True
    max_slots: int = 10
    slot_spacing: float = 2.6
    export_dir: str = "exports/inner_object_3d"
    snapshot_conf_threshold: float = 0.56
    min_steps_between_snapshots: int = 24


@dataclass
class CameraPreviewConfig:
    enabled: bool = False
    window_name: str = "input sensors visualizer"
    scale: float = 1.25
    show_depth: bool = True
    show_actions_window: bool = False
    show_every_steps: int = 1
    delay_ms: int = 1



@dataclass
class EventLatentMemoryConfig:
    enabled: bool = True
    max_events: int = 512
    delta_threshold: float = 0.015
    action_threshold: float = 0.010
    contact_threshold: float = 0.010
    record_in_sleep: bool = True
    keep_z_snapshots: bool = True

    # Slot vocabulary gives each slot a stable internal word:
    # SLOT_1 -> OBJ_001, plus latent signature/passport.
    use_slot_vocabulary: bool = True
    slot_token_prefix: str = "OBJ"

    # Compose grammar-level sentences from tokens:
    #   tokens -> roles -> sentence
    compose_semantic_sentences: bool = True
    sentence_language: str = "code"  # code/en/ru

    # Level 3: sentence stream -> episodes / scenario memory.
    use_sentence_memory: bool = True
    max_sentences: int = 512
    max_episodes: int = 64
    episode_gap_steps: int = 25
    new_episode_on_slot_change: bool = False

    # Level 4: episode/sentence memory -> replayable latent scenario.
    use_scenario_decoder: bool = True
    scenario_max_replay_steps: int = 32
    scenario_interpolate_steps: int = 3
    scenario_loop: bool = True
    scenario_decode_in_sleep: bool = True

    # Level 5: trainable code/sentence -> latent dynamics decoder.
    neural_event_decoder_enabled: bool = True
    neural_event_decoder_hidden_dim: int = 256
    neural_event_decoder_loss_weight: float = 0.05
    neural_event_decoder_max_delta: float = 0.35



@dataclass
class EventCodeVisualizerConfig:
    enabled: bool = True
    window_name: str = "event code / slot vocabulary"
    width: int = 1500
    height: int = 980
    show_every_steps: int = 1
    delay_ms: int = 1
    max_slots: int = 10
    max_events: int = 14


@dataclass
class TetraDynamicSlotDiagnosticConfig:
    enabled: bool = False
    file_name: str = "tetra_dynamic_slot_diagnostic.log"
    reset_on_start: bool = True


@dataclass
class CheckpointLoadConfig:
    # Load and save are intentionally separate so a run can resume from a
    # checkpoint without overwriting it, or save fresh checkpoints without
    # restoring old state at startup.
    enabled_load: bool = True
    enabled_save: bool = True

    # Load path: which checkpoint to read on startup.
    load_path: str = "./checkpoint/last.pt"     # empty => runtime.out_dir / "last.pt"

    # Save path: where periodic/manual checkpoints write last.pt.
    # Empty => use `load_path`; if both are empty => runtime.out_dir / "last.pt".
    save_path: str = "./checkpoint/last.pt"

    strict: bool = False    # false is safer while architecture changes
    load_optimizer: bool = True
    load_counters: bool = True


@dataclass
class EmotionalDriveRuntimeConfig:
    enabled: bool = True
    reward_weight: float = 1.0
    inject_into_env_reward: bool = True
    log_every_steps: int = 25

    ema_decay: float = 0.985
    reward_scale: float = 0.15
    w_gap_fill: float = 1.25
    w_coherence_gain: float = 0.75
    w_object_conf_gain: float = 0.75
    w_multimodal_alignment: float = 0.55
    w_contact_pleasure: float = 0.35
    w_curiosity: float = 0.25
    w_inner_speech_conf: float = 0.35
    w_uncertainty_increase: float = 1.10
    w_coherence_loss: float = 0.85
    w_object_conf_loss: float = 0.75
    w_speech_conf_loss: float = 0.55
    w_alignment_loss: float = 0.70
    w_chaotic_touch: float = 0.45
    w_instability: float = 0.40


@dataclass
class LatentSemanticPanelConfig:
    enabled: bool = True
    window_name: str = "latent semantic map"
    width: int = 1600
    height: int = 980
    max_history: int = 320
    show_every_steps: int = 1
    delay_ms: int = 1
    thumbnail_size: int = 82
    max_thumbnails: int = 6
    point_radius: int = 4
    draw_grid: bool = True
    follow_inner_world_toggle: bool = True


@dataclass
class ExplorationMotorConfig:
    enabled: bool = True

    # During the beginning the model is often near-zero/deterministic.
    # This bootstrap motion lets it collect visual/tactile novelty.
    warmup_steps: int = 1500

    # If embodied/hand outputs are almost zero, inject exploratory motion.
    min_embodied_norm: float = 0.03
    min_hand_norm: float = 0.03

    # Strength of exploratory movement.
    base_amp: float = 0.65
    hand_amp: float = 0.18

    # Action cycling helps avoid one discrete action freezing the loop.
    cycle_action_when_stuck: bool = True
    action_cycle_period: int = 45

    # If novelty stays low, stronger search motion starts.
    low_novelty_threshold: float = 0.015
    stuck_boost: float = 1.75


@dataclass
class DynamicAgentRigRuntimeConfig:
    enabled: bool = False
    body_name: str = "agent_rig"
    freejoint_name: str = "agent_rig_free"

    max_linear_speed: float = 0.35
    max_vertical_speed: float = 0.45
    max_angular_speed: float = 0.6

    linear_kv: float = 10.0
    angular_kv: float = 6.0
    max_force: float = 800.0
    max_torque: float = 70.0

    min_z: float = 0.55
    max_z: float = 2.2
    ground_push_k: float = 100.0
    local_frame_linear: bool = True
    gravity_compensation: bool = True
    hover_enabled: bool = True
    hover_height: float = 1.65
    dynamic_hover_target: bool = True
    min_hover_height: float = 0.75
    max_hover_height: float = 3.0
    vertical_command_gain: float = 0.55
    hover_kp: float = 260.0
    hover_kd: float = 55.0
    emergency_lift_enabled: bool = True
    emergency_z: float = 0.85
    emergency_vz: float = 1.2
    upright_enabled: bool = True
    upright_kp: float = 8.0
    upright_kd: float = 2.0
    contact_angular_damping_enabled: bool = True
    contact_roll_pitch_damping: float = 6.0
    contact_yaw_damping: float = 2.0
    contact_spin_limit: float = 2.0
    contact_spin_deadzone: float = 0.25
    disable_pitch_roll_commands: bool = True
    contact_active_angular_damping: float = 90.0
    contact_active_yaw_damping: float = 18.0
    contact_active_upright_kp: float = 70.0
    contact_active_upright_kd: float = 16.0
    contact_torque_limit: float = 55.0


@dataclass
class BirdBodyRuntimeConfig:
    enabled: bool = True
    leg_smoothing: float = 0.20


@dataclass
class LegControlHeadConfig:
    enabled: bool = True
    leg_motor_dim: int = 18
    hidden_dim: int = 128
    smoothing: float = 0.05


@dataclass
class UnifiedV510Config:
    mode: str = "run"
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
    train: TrainLoopV510Config = field(default_factory=TrainLoopV510Config)
    mujoco_world: MujocoWorldConfig = field(default_factory=MujocoWorldConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(out_dir="runs/unified_conscious_viewer_v5_10"))
    checkpoint_load: CheckpointLoadConfig = field(default_factory=CheckpointLoadConfig)
    tetra_dynamic_slot_diagnostic: TetraDynamicSlotDiagnosticConfig = field(default_factory=TetraDynamicSlotDiagnosticConfig)
    event_code_visualizer: EventCodeVisualizerConfig = field(default_factory=EventCodeVisualizerConfig)
    event_memory: EventLatentMemoryConfig = field(default_factory=EventLatentMemoryConfig)
    emotional_drive: EmotionalDriveRuntimeConfig = field(default_factory=EmotionalDriveRuntimeConfig)
    exploration: ExplorationMotorConfig = field(default_factory=ExplorationMotorConfig)
    dynamic_agent_rig: DynamicAgentRigRuntimeConfig = field(default_factory=DynamicAgentRigRuntimeConfig)
    bird_body: BirdBodyRuntimeConfig = field(default_factory=BirdBodyRuntimeConfig)
    leg_control: LegControlHeadConfig = field(default_factory=LegControlHeadConfig)
    inner_world: InnerWorldWindowConfig = field(default_factory=InnerWorldWindowConfig)
    ipc_control: IPCControlConfig = field(default_factory=IPCControlConfig)
    external_control: ExternalControlConfig = field(default_factory=ExternalControlConfig)
    camera_preview: CameraPreviewConfig = field(default_factory=CameraPreviewConfig)
    action_outputs: ActionOutputsWindowConfig = field(default_factory=ActionOutputsWindowConfig)
    module_debug: ModuleTrainingDebugRuntimeConfig = field(default_factory=ModuleTrainingDebugRuntimeConfig)
    module_status_ipc: ModuleDebugStatusIPCRuntimeConfig = field(default_factory=ModuleDebugStatusIPCRuntimeConfig)
    control_startup: ControlStartupConfig = field(default_factory=ControlStartupConfig)
    sleep_sensors: SleepSensorGateRuntimeConfig = field(default_factory=SleepSensorGateRuntimeConfig)
    object_image: InnerObjectImageConfig = field(default_factory=InnerObjectImageConfig)
    object_image_open3d: InnerObjectOpen3DConfig = field(default_factory=InnerObjectOpen3DConfig)
    latent_semantic_map: LatentSemanticPanelConfig = field(default_factory=LatentSemanticPanelConfig)
    manual_action_override: ManualActionOverrideRuntimeConfig = field(default_factory=ManualActionOverrideRuntimeConfig)
    action_trace: ActionSignalTraceConfig = field(default_factory=ActionSignalTraceConfig)
    vestibular: VestibularRuntimeConfig = field(default_factory=VestibularRuntimeConfig)
    mocap_flight_bounds: MocapFlightBoundsConfig = field(default_factory=MocapFlightBoundsConfig)
    self_core: SelfCoreRuntimeConfig = field(default_factory=SelfCoreRuntimeConfig)
    adaptive_scenario_controller: Dict[str, Any] = field(default_factory=dict)

    action_dim: int = MISSING
    embodied_dim: int = MISSING
    hand_motor_dim: int = MISSING
    tactile_dim: int = MISSING
    body_state_dim: int = MISSING
    # supervised inner speech bootstrap
    inner_speech_loss_weight: float = 0.25
