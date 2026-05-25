from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.apps.runner_runtime_state import (
    apply_runtime_state,
    apply_runtime_state_snapshot,
    build_runtime_state_snapshot,
)


def _cfg(out_dir):
    return SimpleNamespace(runtime=SimpleNamespace(out_dir=str(out_dir)))


def test_build_runtime_state_snapshot(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "out")
    snap = build_runtime_state_snapshot(cfg)

    assert snap.out_dir.endswith("out")
    assert snap.log_path.endswith("live_log.jsonl")
    assert snap.shutdown is False
    assert snap.global_step == 0
    assert snap.train_steps == 0
    assert snap.last_train_reason == "not_started"
    assert snap.last_train_loss is None
    assert snap.last_train_error == ""
    assert snap.action_trace == {}
    assert snap.ipc_close_counter == 0


def test_apply_runtime_state_snapshot_creates_out_dir_and_sets_fields(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "out")
    snap = build_runtime_state_snapshot(cfg)
    system = SimpleNamespace()

    apply_runtime_state_snapshot(system, snap)

    assert system.out_dir.exists()
    assert system.out_dir == Path(snap.out_dir)
    assert system.log_path == Path(snap.log_path)
    assert system.shutdown is False
    assert system.global_step == 0
    assert system.train_steps == 0
    assert system.last_train_reason == "not_started"
    assert system.latest_stats is None
    assert system.latest_out is None
    assert system._action_trace == {}
    assert system.external_control_last_mtime == 0.0


def test_apply_runtime_state_returns_snapshot(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "out")
    system = SimpleNamespace()

    snap = apply_runtime_state(system, cfg)

    assert snap.out_dir == str(tmp_path / "out")
    assert system.out_dir == tmp_path / "out"
    assert system.log_path == tmp_path / "out" / "live_log.jsonl"
