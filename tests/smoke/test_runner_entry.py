from __future__ import annotations

import importlib
from pathlib import Path


def test_slim_runner_entry_imports() -> None:
    module = importlib.import_module("src.apps.runner_entry")
    assert hasattr(module, "main")


def test_root_runner_target_exists() -> None:
    assert Path("runner.py").exists()
    assert Path("src/apps/runner.py").exists()
    assert Path("src/apps/runner_entry.py").exists()
