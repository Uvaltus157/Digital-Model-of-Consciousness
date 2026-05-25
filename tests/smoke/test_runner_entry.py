from __future__ import annotations

import importlib
import runpy
from pathlib import Path


def test_slim_runner_entry_imports() -> None:
    module = importlib.import_module("src.apps.runner_entry")
    assert hasattr(module, "main")


def test_root_runner_target_exists() -> None:
    assert Path("runner.py").exists()
    assert Path("src/apps/runner.py").exists()
    assert Path("src/apps/runner_entry.py").exists()


def test_root_runner_routes_to_slim_entry_by_default(monkeypatch) -> None:
    namespace = runpy.run_path("runner.py")
    monkeypatch.delenv("CWMS_LEGACY_RUNNER", raising=False)
    target = namespace["resolve_target"]()
    assert target.name == "runner_entry.py"


def test_root_runner_can_route_to_legacy_runner(monkeypatch) -> None:
    namespace = runpy.run_path("runner.py")
    monkeypatch.setenv("CWMS_LEGACY_RUNNER", "1")
    target = namespace["resolve_target"]()
    assert target.name == "runner.py"
    assert target.parent.name == "apps"
