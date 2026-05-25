from __future__ import annotations

"""Runtime-state initialization helpers for the V5.10 runner.

This module extracts simple counters, paths and mutable runtime bookkeeping from
the heavy runner into a small, dependency-light helper.

For now `UnifiedSystemV510.__init__` still performs the original assignments.
The slim entrypoint reapplies the same initialization after construction so this
helper becomes tested and explicit before a later direct edit of the large
runtime file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    out_dir: str
    log_path: str
    shutdown: bool
    global_step: int
    train_steps: int
    last_module_training_seq: int
    last_train_reason: str
    last_train_loss: Optional[float]
    last_train_error: str
    latest_stats: Any
    latest_out: Any
    last_print_time: float
    action_trace: Dict[str, Any]
    ipc_close_counter: int
    external_control_last_mtime: float
    external_control_last_close_counter: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "log_path": self.log_path,
            "shutdown": self.shutdown,
            "global_step": self.global_step,
            "train_steps": self.train_steps,
            "last_module_training_seq": self.last_module_training_seq,
            "last_train_reason": self.last_train_reason,
            "last_train_loss": self.last_train_loss,
            "last_train_error": self.last_train_error,
            "latest_stats": self.latest_stats,
            "latest_out": self.latest_out,
            "last_print_time": self.last_print_time,
            "action_trace": dict(self.action_trace),
            "ipc_close_counter": self.ipc_close_counter,
            "external_control_last_mtime": self.external_control_last_mtime,
            "external_control_last_close_counter": self.external_control_last_close_counter,
        }


def build_runtime_state_snapshot(cfg: Any) -> RuntimeStateSnapshot:
    """Build initial mutable runtime bookkeeping state from config."""
    out_dir = Path(getattr(cfg.runtime, "out_dir", "outputs")).expanduser()
    log_path = out_dir / "live_log.jsonl"
    return RuntimeStateSnapshot(
        out_dir=str(out_dir),
        log_path=str(log_path),
        shutdown=False,
        global_step=0,
        train_steps=0,
        last_module_training_seq=0,
        last_train_reason="not_started",
        last_train_loss=None,
        last_train_error="",
        latest_stats=None,
        latest_out=None,
        last_print_time=0.0,
        action_trace={},
        ipc_close_counter=0,
        external_control_last_mtime=0.0,
        external_control_last_close_counter=0,
    )


def apply_runtime_state_snapshot(system: Any, snapshot: RuntimeStateSnapshot) -> None:
    """Apply runtime-state bookkeeping to an already constructed system."""
    out_dir = Path(snapshot.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system.out_dir = out_dir
    system.log_path = Path(snapshot.log_path)
    system.shutdown = bool(snapshot.shutdown)
    system.last_module_training_seq = int(snapshot.last_module_training_seq)
    system.global_step = int(snapshot.global_step)
    system.train_steps = int(snapshot.train_steps)
    system.last_train_reason = snapshot.last_train_reason
    system.last_train_loss = snapshot.last_train_loss
    system.last_train_error = snapshot.last_train_error
    system.latest_stats = snapshot.latest_stats
    system.latest_out = snapshot.latest_out
    system.last_print_time = float(snapshot.last_print_time)
    system._action_trace = dict(snapshot.action_trace)
    system.ipc_close_counter = int(snapshot.ipc_close_counter)
    system.external_control_last_mtime = float(snapshot.external_control_last_mtime)
    system.external_control_last_close_counter = int(snapshot.external_control_last_close_counter)


def apply_runtime_state(system: Any, cfg: Any) -> RuntimeStateSnapshot:
    """Build and apply runtime state; return the snapshot for diagnostics."""
    snapshot = build_runtime_state_snapshot(cfg)
    apply_runtime_state_snapshot(system, snapshot)
    return snapshot
