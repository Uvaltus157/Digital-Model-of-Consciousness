from __future__ import annotations

"""Hydra/OmegaConf normalization for the V5.10 runner.

This module is the first step in unloading `src/apps/runner.py`.
It extracts the config filtering/merging logic from the runner into a small,
testable helper without changing runtime behavior.
"""

from typing import Any, Dict, Mapping, Set

from omegaconf import DictConfig, OmegaConf

from src.apps.unified_conscious_viewer import (
    InnerWorldWindowConfig,
    LifeConfig,
    MujocoWorldConfig,
    NoveltyConfig,
    ReplayConfig,
    RuntimeConfig,
    ViewerConfig,
)
from src.shared.config import (
    AutobiographicalMemoryRuntimeConfig,
    ActionOutputsWindowConfig,
    ActionSignalTraceConfig,
    BirdBodyConfig,
    CameraPreviewConfig,
    CheckpointLoadConfig,
    ControlStartupConfig,
    DynamicAgentRigRuntimeConfig,
    EmotionalDriveConfig,
    EventCodeVisualizerRuntimeConfig,
    EventDreamReplayRuntimeConfig,
    ExplorationMotorConfig,
    ExternalControlConfig,
    GlobalBroadcastRuntimeConfig,
    IPCControlConfig,
    InnerSpeechRuntimeConfig,
    InnerObjectImageConfig,
    InnerObjectOpen3DConfig,
    LatentSemanticMapConfig,
    LegControlConfig,
    LongDynamicMemoryRuntimeConfig,
    ManualActionOverrideRuntimeConfig,
    MetacognitionRuntimeConfig,
    MocapFlightBoundsConfig,
    ModuleDebugStatusIPCRuntimeConfig,
    ModuleTrainingDebugRuntimeConfig,
    SemanticActionRuntimeConfig,
    SelfCoreRuntimeConfig,
    SleepSensorGateRuntimeConfig,
    TetraDynamicSlotDiagnosticConfig,
    ThreadAffinityConfig,
    ThoughtChainRuntimeConfig,
    RunnerTrainConfig,
    UnifiedConfig,
    VestibularRuntimeConfig,
)

ALLOWED_TOP_KEYS: Set[str] = {
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
    "event_dream_replay",
    "event_code_visualizer",
    "inner_speech",
    "long_dynamic_memory",
    "global_broadcast",
    "metacognition",
    "autobiographical_memory",
    "semantic_action",
    "thought_chain",
    "manual_action_override",
    "action_trace",
    "adaptive_scenario_controller",
    "tetra_dynamic_slot_diagnostic",
    "thread_affinity",
    "inner_speech_loss_weight",
}

DEFAULT_MODEL_DIMS: Dict[str, int] = {
    "action_dim": 24,
    "embodied_dim": 15,
    "hand_motor_dim": 44,
    "tactile_dim": 42,
    "body_state_dim": 83,
}

DEFAULT_OBJECT_IMAGE_TACTILE_DIM = 50


def _keys_of(config_type: type) -> Set[str]:
    return set(OmegaConf.structured(config_type()).keys())


def allowed_nested_keys() -> Dict[str, Set[str]]:
    """Return allowed nested keys for structured V5.10 config sections."""
    return {
        "novelty": _keys_of(NoveltyConfig),
        "replay": _keys_of(ReplayConfig),
        "life": _keys_of(LifeConfig),
        "train": _keys_of(RunnerTrainConfig),
        "mujoco_world": _keys_of(MujocoWorldConfig),
        "viewer": _keys_of(ViewerConfig),
        "runtime": _keys_of(RuntimeConfig),
        "checkpoint_load": _keys_of(CheckpointLoadConfig),
        "tetra_dynamic_slot_diagnostic": _keys_of(TetraDynamicSlotDiagnosticConfig),
        "thread_affinity": _keys_of(ThreadAffinityConfig),
        "emotional_drive": _keys_of(EmotionalDriveConfig),
        "exploration": _keys_of(ExplorationMotorConfig),
        "dynamic_agent_rig": _keys_of(DynamicAgentRigRuntimeConfig),
        "bird_body": _keys_of(BirdBodyConfig),
        "leg_control": _keys_of(LegControlConfig),
        "inner_world": _keys_of(InnerWorldWindowConfig),
        "ipc_control": _keys_of(IPCControlConfig),
        "external_control": _keys_of(ExternalControlConfig),
        "camera_preview": _keys_of(CameraPreviewConfig),
        "action_outputs": _keys_of(ActionOutputsWindowConfig),
        "module_debug": _keys_of(ModuleTrainingDebugRuntimeConfig),
        "module_status_ipc": _keys_of(ModuleDebugStatusIPCRuntimeConfig),
        "control_startup": _keys_of(ControlStartupConfig),
        "sleep_sensors": _keys_of(SleepSensorGateRuntimeConfig),
        "object_image": _keys_of(InnerObjectImageConfig),
        "object_image_open3d": _keys_of(InnerObjectOpen3DConfig),
        "latent_semantic_map": _keys_of(LatentSemanticMapConfig),
        "manual_action_override": _keys_of(ManualActionOverrideRuntimeConfig),
        "action_trace": _keys_of(ActionSignalTraceConfig),
        "vestibular": _keys_of(VestibularRuntimeConfig),
        "mocap_flight_bounds": _keys_of(MocapFlightBoundsConfig),
        "self_core": _keys_of(SelfCoreRuntimeConfig),
        "thought_chain": _keys_of(ThoughtChainRuntimeConfig),
        "event_dream_replay": _keys_of(EventDreamReplayRuntimeConfig),
        "event_code_visualizer": _keys_of(EventCodeVisualizerRuntimeConfig),
        "inner_speech": _keys_of(InnerSpeechRuntimeConfig),
        "long_dynamic_memory": _keys_of(LongDynamicMemoryRuntimeConfig),
        "global_broadcast": _keys_of(GlobalBroadcastRuntimeConfig),
        "metacognition": _keys_of(MetacognitionRuntimeConfig),
        "autobiographical_memory": _keys_of(AutobiographicalMemoryRuntimeConfig),
        "semantic_action": _keys_of(SemanticActionRuntimeConfig),
    }


def filter_runner_config_dict(raw: DictConfig | Mapping[str, Any]) -> Dict[str, Any]:
    """Drop unknown top-level and nested keys before structured merge."""
    raw_cfg = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    nested = allowed_nested_keys()

    clean_dict: Dict[str, Any] = {}
    for key in raw_cfg.keys():
        if key not in ALLOWED_TOP_KEYS:
            continue
        if key in nested:
            if OmegaConf.is_config(raw_cfg[key]) or isinstance(raw_cfg[key], dict):
                clean_dict[key] = {
                    sub_key: raw_cfg[key][sub_key]
                    for sub_key in raw_cfg[key].keys()
                    if sub_key in nested[key]
                }
            continue
        clean_dict[key] = raw_cfg[key]
    return clean_dict


def apply_required_runner_defaults(clean_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Fill mandatory runner dimensions before converting structured config."""
    clean = dict(clean_dict)
    for key, value in DEFAULT_MODEL_DIMS.items():
        clean.setdefault(key, value)

    object_image = clean.get("object_image")
    if object_image is None:
        object_image = {}
    else:
        object_image = dict(object_image)
    object_image.setdefault("tactile_dim", DEFAULT_OBJECT_IMAGE_TACTILE_DIM)
    clean["object_image"] = object_image
    return clean


def build_runner_config(cfg_raw: DictConfig | Mapping[str, Any]) -> UnifiedConfig:
    """Build the typed unified runner config from a Hydra raw config."""
    base = OmegaConf.structured(UnifiedConfig())
    clean = OmegaConf.create(apply_required_runner_defaults(filter_runner_config_dict(cfg_raw)))
    merged = OmegaConf.merge(base, clean)
    return OmegaConf.to_object(merged)


def render_resolved_runner_config(cfg_raw: DictConfig | Mapping[str, Any]) -> str:
    """Return the resolved YAML used for startup diagnostics."""
    base = OmegaConf.structured(UnifiedConfig())
    clean = OmegaConf.create(apply_required_runner_defaults(filter_runner_config_dict(cfg_raw)))
    merged = OmegaConf.merge(base, clean)
    return OmegaConf.to_yaml(merged, resolve=True)
