from __future__ import annotations

"""Idempotent runtime patch records for the slim V5.10 entrypoint.

`src.apps.runner.UnifiedSystemV510` now defines its main hooks directly from the
extracted helper modules. This file remains as a compatibility/idempotency layer
for the slim entrypoint and for tests/docs that want to inspect what was moved
out of the former heavy runner.
"""

from dataclasses import dataclass
from typing import Any, List

from src.apps.runner_hover_config import force_hover_flight_runtime_config_for_system
from src.apps.runner_loop import run_unified_life_loop
from src.apps.runner_optimizer import rebuild_optimizer_from_trainable_modules_for_system
from src.apps.runner_teachers import load_inner_speech_teacher_from_config
from src.apps.runner_training_flags import resolve_module_training_flags_for_system
from src.apps.runner_unified_init import initialize_unified_system_v510


@dataclass(frozen=True)
class RunnerPatchRecord:
    target: str
    replacement: str
    reason: str


RUNNER_PATCHES: List[RunnerPatchRecord] = [
    RunnerPatchRecord(
        target="src.apps.runner.load_inner_speech_teacher_from_config",
        replacement="src.apps.runner_teachers.load_inner_speech_teacher_from_config",
        reason="teacher loading extracted from heavy runner",
    ),
    RunnerPatchRecord(
        target="UnifiedSystemV510.__init__",
        replacement="src.apps.runner_unified_init.initialize_unified_system_v510",
        reason="system construction extracted from heavy runner",
    ),
    RunnerPatchRecord(
        target="UnifiedSystemV510.run",
        replacement="src.apps.runner_loop.run_unified_life_loop",
        reason="outer life loop extracted from heavy runner",
    ),
    RunnerPatchRecord(
        target="UnifiedSystemV510.resolve_module_training_flags_from_config",
        replacement="src.apps.runner_training_flags.resolve_module_training_flags_for_system",
        reason="module train/passive mode resolution extracted from heavy runner",
    ),
    RunnerPatchRecord(
        target="UnifiedSystemV510._force_hover_flight_runtime_config",
        replacement="src.apps.runner_hover_config.force_hover_flight_runtime_config_for_system",
        reason="hover safety clamp extracted from heavy runner",
    ),
    RunnerPatchRecord(
        target="UnifiedSystemV510.rebuild_optimizer_from_trainable_modules",
        replacement="src.apps.runner_optimizer.rebuild_optimizer_from_trainable_modules_for_system",
        reason="optimizer rebuild extracted from heavy runner",
    ),
]


def apply_runner_patches(runner_runtime: Any, unified_system_cls: type) -> None:
    """Idempotently ensure runner module/class uses extracted helpers."""
    runner_runtime.load_inner_speech_teacher_from_config = load_inner_speech_teacher_from_config
    unified_system_cls.__init__ = initialize_unified_system_v510
    unified_system_cls.run = run_unified_life_loop
    unified_system_cls.resolve_module_training_flags_from_config = resolve_module_training_flags_for_system
    unified_system_cls._force_hover_flight_runtime_config = force_hover_flight_runtime_config_for_system
    unified_system_cls.rebuild_optimizer_from_trainable_modules = rebuild_optimizer_from_trainable_modules_for_system


def runner_patch_summary() -> list[dict[str, str]]:
    """Return a JSON-friendly summary for docs/debug/tests."""
    return [record.__dict__.copy() for record in RUNNER_PATCHES]
