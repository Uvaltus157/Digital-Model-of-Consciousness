from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

from omegaconf import MISSING

from src.shared.model_dimensions import (
    MODEL_DIM_KEYS,
    cfg_get as _cfg_get,
    model_dimensions_from_runner_cfg,
    validate_runner_model_dimensions,
)
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
    decoder_loss_enabled: bool = True
    decoder_loss_weight: float = 0.20
    decoder_rgb_weight: float = 1.00
    decoder_depth_weight: float = 0.75
    decoder_mask_weight: float = 0.50
    decoder_conf_weight: float = 0.10
    decoder_use_geometry_objectness: bool = True
    point_count: int = 128
    voxel_res: int = 16
    num_slots: int = 10
    max_object_proposals: int = 10
    include_ground_proposal: bool = True
    proposal_slot_lock: bool = True
    sleep_decode_from_memory: bool = True
    sleep_freeze_object_slots: bool = False
    sleep_skip_decoder_loss: bool = True
    dream_latent_dynamics: bool = True
    dream_strength: float = 0.025
    dream_cycle_slots: bool = False
    dream_slot_cycle_steps: int = 90
    dream_empty_confidence_threshold: float = 0.05
    slot_gaussian_renderer_backend: str = "auto"
    slot_gaussian_cuda_allow_fallback: bool = True
    slot_gaussian_image_size: int = 64
    slot_gaussian_max_gaussians: int = 768
    slot_gaussian_max_render_gaussians: int = 256
    slot_gaussian_lr: float = 0.003
    slot_gaussian_train_steps_per_update: int = 1
    slot_gaussian_depth_weight: float = 0.35
    slot_gaussian_preview_every_steps: int = 1
    slot_4d_timeline_enabled: bool = True
    slot_4d_timeline_max_frames: int = 256
    slot_4d_sample_points: int = 128
    slot_4d_deformation_enabled: bool = True
    slot_4d_deformation_hidden_dim: int = 96
    slot_4d_deformation_lr: float = 0.002
    slot_4d_deformation_train_steps_per_update: int = 1
    slot_4d_deformation_min_frames: int = 2
    slot_4d_deformation_delta_reg_weight: float = 0.0001
    slot_4d_playback_enabled: bool = True
    slot_4d_playback_period_steps: int = 120
    slot_4d_playback_strength: float = 1.0
    slot_4d_open3d_export_enabled: bool = True
    slot_4d_open3d_export_path: str = "./checkpoint/slot_viewer/slot_4d_open3d_latest.npz"
    slot_4d_open3d_sample_points: int = 4096
    slot_4d_open3d_min_interval_sec: float = 0.05
    slot_4d_open3d_transport: str = "jsonrpc"
    slot_4d_jsonrpc_enabled: bool = True
    slot_4d_jsonrpc_host: str = "127.0.0.1"
    slot_4d_jsonrpc_port: int = 8771
    slot_4d_jsonrpc_sample_points: int = 4096
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
    module_modes: Dict[str, str] = field(default_factory=lambda: {
        "world_model": "train",
        "object_imagery": "train",
        "long_dynamic_memory": "train",
        "core_model": "train",
        "action_heads": "train",
        "leg_control": "train",
        "self_core": "train",
        "inner_speech": "train",
        "thought_chain": "train",
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
    focus_context_dim: int = 256
    affect_latent_dim: int = 12
    loss_weight: float = 0.02
    print_every_steps: int = 30


@dataclass
class ThoughtChainRuntimeConfig:
    enabled: bool = True
    hidden_dim: int = 256
    thought_dim: int = 128
    plan_context_dim: int = 256
    chain_len: int = 4
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
class CameraPreviewConfig:
    enabled: bool = True
    window_name: str = "input sensors visualizer"
    scale: float = 1.25
    show_depth: bool = True
    show_actions_window: bool = False
    show_every_steps: int = 1
    delay_ms: int = 1


@dataclass
class LatentSemanticMapConfig:
    enabled: bool = True
    window_name: str = "latent semantic map"
    width: int = 1780
    height: int = 980
    max_history: int = 320
    show_every_steps: int = 1
    delay_ms: int = 1
    thumbnail_size: int = 82
    max_thumbnails: int = 6
    point_radius: int = 4
    draw_grid: bool = True
    follow_inner_world_toggle: bool = False


@dataclass
class CheckpointLoadConfig:
    enabled_load: bool = True
    enabled_save: bool = True
    load_path: str = "./checkpoint/last.pt"
    save_path: str = "./checkpoint/last.pt"
    strict: bool = False
    load_optimizer: bool = True
    load_counters: bool = True


@dataclass
class EmotionalDriveConfig:
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
    w_instability: float = 0.40
    w_uncertainty_increase: float = 1.10
    w_coherence_loss: float = 0.85
    w_object_conf_loss: float = 0.75
    w_speech_conf_loss: float = 0.55
    w_alignment_loss: float = 0.70
    w_chaotic_touch: float = 0.45


@dataclass
class ExplorationMotorConfig:
    enabled: bool = True
    warmup_steps: int = 1500
    min_embodied_norm: float = 0.03
    min_hand_norm: float = 0.03
    base_amp: float = 0.65
    hand_amp: float = 0.18
    cycle_action_when_stuck: bool = True
    action_cycle_period: int = 45
    low_novelty_threshold: float = 0.015
    stuck_boost: float = 1.75


@dataclass
class DynamicAgentRigRuntimeConfig:
    enabled: bool = True
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
    contact_active_angular_damping: float = 65.0
    contact_active_yaw_damping: float = 18.0
    contact_active_upright_kp: float = 45.0
    contact_active_upright_kd: float = 16.0
    contact_torque_limit: float = 55.0


@dataclass
class BirdBodyConfig:
    enabled: bool = True
    leg_smoothing: float = 0.05


@dataclass
class LegControlConfig:
    enabled: bool = True
    leg_motor_dim: int = 18
    hidden_dim: int = 128
    smoothing: float = 0.05


@dataclass
class MocapContactConfig:
    enabled: bool = True
    central_rig: str = "mocap"
    preserve_contacts: bool = True
    notes: str = "agent_rig/cameras are mocap-stabilized; arms/legs/objects keep contacts"


@dataclass
class AgentHeadConfig:
    enabled: bool = True
    control_indices: Dict[str, int] = field(default_factory=lambda: {
        "body_roll": 11,
        "head_yaw": 12,
        "head_pitch": 13,
        "head_roll": 14,
    })
    notes: str = "embodied_dim is 15: old body/arms preserved, body roll at 11, head controls at 12:15"


@dataclass
class UnifiedV510Config(RuntimeConfig):
    mode: str = "train"
    inner_speech_loss_weight: float = 0.25

    action_dim: int = MISSING
    embodied_dim: int = MISSING
    hand_motor_dim: int = MISSING
    tactile_dim: int = MISSING
    body_state_dim: int = MISSING

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mujoco_world: MujocoWorldConfig = field(default_factory=MujocoWorldConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
    train: TrainLoopV510Config = field(default_factory=TrainLoopV510Config)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    inner_world: InnerWorldWindowConfig = field(default_factory=InnerWorldWindowConfig)
    camera_preview: CameraPreviewConfig = field(default_factory=CameraPreviewConfig)
    action_outputs: ActionOutputsWindowConfig = field(default_factory=ActionOutputsWindowConfig)
    object_image: InnerObjectImageConfig = field(default_factory=InnerObjectImageConfig)
    ipc_control: IPCControlConfig = field(default_factory=IPCControlConfig)
    external_control: ExternalControlConfig = field(default_factory=ExternalControlConfig)
    control_startup: ControlStartupConfig = field(default_factory=ControlStartupConfig)
    checkpoint_load: CheckpointLoadConfig = field(default_factory=CheckpointLoadConfig)
    emotional_drive: EmotionalDriveConfig = field(default_factory=EmotionalDriveConfig)
    exploration: ExplorationMotorConfig = field(default_factory=ExplorationMotorConfig)
    action_trace: ActionSignalTraceConfig = field(default_factory=ActionSignalTraceConfig)
    self_core: SelfCoreRuntimeConfig = field(default_factory=SelfCoreRuntimeConfig)
    thought_chain: ThoughtChainRuntimeConfig = field(default_factory=ThoughtChainRuntimeConfig)
    vestibular: VestibularRuntimeConfig = field(default_factory=VestibularRuntimeConfig)
    sleep_sensors: SleepSensorGateRuntimeConfig = field(default_factory=SleepSensorGateRuntimeConfig)
    module_debug: ModuleTrainingDebugRuntimeConfig = field(default_factory=ModuleTrainingDebugRuntimeConfig)
    module_status_ipc: ModuleDebugStatusIPCRuntimeConfig = field(default_factory=ModuleDebugStatusIPCRuntimeConfig)
    dynamic_agent_rig: DynamicAgentRigRuntimeConfig = field(default_factory=DynamicAgentRigRuntimeConfig)
    bird_body: BirdBodyConfig = field(default_factory=BirdBodyConfig)
    leg_control: LegControlConfig = field(default_factory=LegControlConfig)
    mocap_contacts: MocapContactConfig = field(default_factory=MocapContactConfig)
    mocap_flight_bounds: MocapFlightBoundsConfig = field(default_factory=MocapFlightBoundsConfig)
    manual_action_override: ManualActionOverrideRuntimeConfig = field(default_factory=ManualActionOverrideRuntimeConfig)
    latent_semantic_map: LatentSemanticMapConfig = field(default_factory=LatentSemanticMapConfig)
    agent_head: AgentHeadConfig = field(default_factory=AgentHeadConfig)


__all__ = [
    "MODEL_DIM_KEYS",
    "_cfg_get",
    "model_dimensions_from_runner_cfg",
    "validate_runner_model_dimensions",
    "UnifiedV510Config",
    "TrainLoopV510Config",
    "IPCControlConfig",
    "ExternalControlConfig",
    "ActionOutputsWindowConfig",
    "InnerObjectImageConfig",
    "ModuleDebugStatusIPCRuntimeConfig",
    "ControlStartupConfig",
    "SleepSensorGateRuntimeConfig",
    "ModuleTrainingDebugRuntimeConfig",
    "ActionSignalTraceConfig",
    "SelfCoreRuntimeConfig",
    "ThoughtChainRuntimeConfig",
    "MocapFlightBoundsConfig",
    "VestibularRuntimeConfig",
    "ManualActionOverrideRuntimeConfig",
    "CameraPreviewConfig",
    "LatentSemanticMapConfig",
    "CheckpointLoadConfig",
    "EmotionalDriveConfig",
    "ExplorationMotorConfig",
    "DynamicAgentRigRuntimeConfig",
    "BirdBodyConfig",
    "LegControlConfig",
    "MocapContactConfig",
    "AgentHeadConfig",
]
