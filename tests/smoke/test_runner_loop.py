from __future__ import annotations

import importlib


def test_runner_loop_imports_without_mujoco_or_display() -> None:
    module = importlib.import_module("src.apps.runner_loop")
    assert hasattr(module, "run_unified_life_loop")


def test_runner_entry_patches_life_loop() -> None:
    entry = importlib.import_module("src.apps.runner_entry")
    loop = importlib.import_module("src.apps.runner_loop")
    assert entry.UnifiedSystemV510.run is loop.run_unified_life_loop
