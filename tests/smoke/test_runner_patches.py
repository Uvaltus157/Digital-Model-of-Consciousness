from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_hover_config import force_hover_flight_runtime_config_for_system
from src.apps.runner_loop import run_unified_life_loop
from src.apps.runner_optimizer import rebuild_optimizer_from_trainable_modules_for_system
from src.apps.runner_patches import RUNNER_PATCHES, apply_runner_patches, runner_patch_summary
from src.apps.runner_teachers import load_inner_speech_teacher_from_config
from src.apps.runner_training_flags import resolve_module_training_flags_for_system
from src.apps.runner_unified_init import initialize_unified_system_v510


class DummyUnifiedSystem:
    def __init__(self):  # pragma: no cover - replaced by patch
        raise AssertionError("not patched")

    def run(self):  # pragma: no cover - replaced by patch
        raise AssertionError("not patched")

    def resolve_module_training_flags_from_config(self):  # pragma: no cover - replaced by patch
        raise AssertionError("not patched")

    def _force_hover_flight_runtime_config(self):  # pragma: no cover - replaced by patch
        raise AssertionError("not patched")

    def rebuild_optimizer_from_trainable_modules(self):  # pragma: no cover - replaced by patch
        raise AssertionError("not patched")


def test_apply_runner_patches_replaces_expected_hooks() -> None:
    runtime = SimpleNamespace(load_inner_speech_teacher_from_config=lambda cfg: None)

    apply_runner_patches(runtime, DummyUnifiedSystem)

    assert runtime.load_inner_speech_teacher_from_config is load_inner_speech_teacher_from_config
    assert DummyUnifiedSystem.__init__ is initialize_unified_system_v510
    assert DummyUnifiedSystem.run is run_unified_life_loop
    assert DummyUnifiedSystem.resolve_module_training_flags_from_config is resolve_module_training_flags_for_system
    assert DummyUnifiedSystem._force_hover_flight_runtime_config is force_hover_flight_runtime_config_for_system
    assert DummyUnifiedSystem.rebuild_optimizer_from_trainable_modules is rebuild_optimizer_from_trainable_modules_for_system


def test_runner_patch_summary_is_json_friendly() -> None:
    summary = runner_patch_summary()
    assert len(summary) == len(RUNNER_PATCHES)
    assert all("target" in item and "replacement" in item and "reason" in item for item in summary)
    assert any(item["target"] == "UnifiedSystemV510.__init__" for item in summary)
    assert any(item["target"] == "UnifiedSystemV510.run" for item in summary)
