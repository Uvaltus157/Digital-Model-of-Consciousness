from __future__ import annotations

"""Slim compatibility module for the V5.10 unified system.

The previous version of this file contained the Hydra entrypoint, the full
`UnifiedSystemV510.__init__`, the outer run loop, config filtering, teacher
loading, optimizer rebuild logic, service startup and hover safety patching.

Those responsibilities now live in small app-level modules:

- `runner_entry.py`             — Hydra entrypoint
- `runner_config.py`            — config filtering / structured merge
- `runner_system_factory.py`    — construction and post-construction sequence
- `runner_unified_init.py`      — extracted `UnifiedSystemV510.__init__`
- `runner_loop.py`              — outer life loop
- `runner_teachers.py`          — inner-speech teacher loading
- `runner_training_flags.py`    — module train/passive flags
- `runner_hover_config.py`      — flight-safe hover clamps
- `runner_optimizer.py`         — optimizer rebuild
- `runner_services.py`          — IPC/status service boundary
- `runner_runtime_state.py`     — mutable runtime bookkeeping
- `runner_startup_state.py`     — startup/window/sensor flags

This file intentionally keeps the public import path
`src.apps.runner.UnifiedSystemV510` stable for legacy code.
"""

import os
from pathlib import Path

from src.shared.console_colors import install_colored_errors

try:
    install_colored_errors()
except Exception:
    pass

from src.apps.runner_hover_config import force_hover_flight_runtime_config_for_system
from src.apps.runner_loop import run_unified_life_loop
from src.apps.runner_optimizer import rebuild_optimizer_from_trainable_modules_for_system
from src.apps.runner_teachers import load_inner_speech_teacher_from_config
from src.apps.runner_training_flags import resolve_module_training_flags_for_system
from src.apps.runner_unified_init import initialize_unified_system_v510
from src.apps.unified_conscious_viewer import UnifiedSystemV57
from src.modules.m01_object_imagery.inner_visual_runtime import InnerVisualRuntimeMixin
from src.modules.m01_object_imagery.runtime import ObjectImageryRuntimeMixin
from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin
from src.modules.m03_self_action_causality.action_outputs_window import ActionOutputsMixin
from src.modules.m03_self_action_causality.action_runtime import ActionRuntimeMixin
from src.modules.m04_long_dynamic_memory.dynamic_object_passport_runtime import DynamicObjectPassportRuntimeMixin
from src.modules.m04_long_dynamic_memory.long_dynamic_memory_runtime import LongDynamicMemoryRuntimeMixin
from src.modules.m05_world_model_attention_workspace.tetra_dynamic_slot_diagnostic import TetraDynamicSlotDiagnosticMixin
from src.modules.m06_learning_sleep_consolidation.sleep_sensors import SleepSensorsMixin
from src.modules.m06_learning_sleep_consolidation.training_runtime import TrainingRuntimeMixin
from src.modules.m07_inner_speech_thoughts.inner_speech_runtime import InnerSpeechRuntimeMixin
from src.modules.m08_debug_visual_control.module_status_runtime import ModuleStatusRuntimeMixin
from src.modules.m09_self_core.self_core_runtime import SelfCoreRuntimeMixin
from src.modules.m10_global_conscious_broadcast.broadcast_runtime import GlobalBroadcastRuntimeMixin
from src.modules.m12_metacognition_monitor.metacognition_runtime import MetacognitionRuntimeMixin
from src.modules.m13_autobiographical_memory.autobiographical_memory_runtime import AutobiographicalMemoryRuntimeMixin
from src.modules.m14_semantic_grounding.semantic_action_runtime import SemanticActionRuntimeMixin
from src.modules.m15_counterfactual_imagination_planning.thought_chain_runtime import ThoughtChainRuntimeMixin
from src.platform.ipc.external_control import ExternalControlMixin
from src.platform.ipc.ipc_runtime import IPCRuntimeMixin
from src.platform.mujoco_world.camera_preview_window import CameraPreviewMixin
from src.platform.mujoco_world.leg_bird_runtime import LegBirdRuntimeMixin
from src.shared.checkpointing import CheckpointingMixin
from src.shared.config import UnifiedV510Config
from src.apps.life_runtime import LifeRuntimeMixin


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
    DynamicObjectPassportRuntimeMixin,
    LongDynamicMemoryRuntimeMixin,
    EventDreamReplayRuntimeMixin,
    ThoughtChainRuntimeMixin,
    GlobalBroadcastRuntimeMixin,
    InnerSpeechRuntimeMixin,
    MetacognitionRuntimeMixin,
    AutobiographicalMemoryRuntimeMixin,
    SemanticActionRuntimeMixin,
    InnerVisualRuntimeMixin,
    TrainingRuntimeMixin,
    LifeRuntimeMixin,
    TetraDynamicSlotDiagnosticMixin,
    UnifiedSystemV57,
):
    """V5.10 runtime assembled from mixins and extracted app helpers."""

    __init__ = initialize_unified_system_v510
    run = run_unified_life_loop
    resolve_module_training_flags_from_config = resolve_module_training_flags_for_system
    _force_hover_flight_runtime_config = force_hover_flight_runtime_config_for_system
    rebuild_optimizer_from_trainable_modules = rebuild_optimizer_from_trainable_modules_for_system


os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))


# Backward-compatible entrypoint for direct `python -m src.apps.runner` use.
def main() -> None:
    from src.apps.runner_entry import main as runner_entry_main

    runner_entry_main()


__all__ = [
    "UnifiedSystemV510",
    "UnifiedV510Config",
    "load_inner_speech_teacher_from_config",
    "main",
]


if __name__ == "__main__":
    main()
