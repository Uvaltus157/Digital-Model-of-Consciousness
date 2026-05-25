from __future__ import annotations

"""Behavior-preserving runtime patches for the slim V5.10 entrypoint.

During the runner unload, the large `src/apps/runner.py` file still owns the
heavy `UnifiedSystemV510` class. The slim entrypoint imports that class and then
applies these method/function replacements so extracted helpers are used without
risking a full edit of the large runtime file.
"""

from dataclasses import dataclass
from typing import Any, List

from src.apps.runner_hover_config import force_hover_flight_runtime_config_for_system
from src.apps.runner_loop import run_unified_life_loop
from src.apps.runner_optimizer import rebuild_optimizer_from_trainable_modules_for_system
from src.apps.runner_teachers import load_inner_speech_teacher_from_config
from src.apps.runner_training_flags import resolve_module_training_flags_for_system


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
    """Patch the heavy runner module/class to use extracted helpers."""
    runner_runtime.load_inner_speech_teacher_from_config = load_inner_speech_teacher_from_config
    unified_system_cls.run = run_unified_life_loop
    unified_system_cls.resolve_module_training_flags_from_config = resolve_module_training_flags_for_system
    unified_system_cls._force_hover_flight_runtime_config = force_hover_flight_runtime_config_for_system
    unified_system_cls.rebuild_optimizer_from_trainable_modules = rebuild_optimizer_from_trainable_modules_for_system


def runner_patch_summary() -> list[dict[str, str]]:
    """Return a JSON-friendly summary for docs/debug/tests."""
    return [record.__dict__.copy() for record in RUNNER_PATCHES]
