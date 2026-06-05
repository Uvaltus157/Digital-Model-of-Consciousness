from __future__ import annotations

"""
Motor guard for sleep/replay mode.

During full sleep:
    - M5 may still imagine/propose actions internally.
    - M3/body must not execute those actions in the external MuJoCo world.

This guard zeros executable motor tensors while preserving proposed actions under
`out["sleep_motor_guard"]["imagined_*"]` for debugging.
"""

from typing import Any, Dict

import torch


MOTOR_KEYS = (
    "embodied_targets",
    "hand_ctrl",
    "leg_ctrl",
)


def _norm(x: Any) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().norm().cpu().item())
        return 0.0
    except Exception:
        return 0.0


def block_motor_outputs_for_sleep(out: Dict, *, sleep_mode: bool, stage: str = "") -> Dict:
    if not isinstance(out, dict):
        return out

    packet = {
        "blocked": bool(sleep_mode),
        "stage": str(stage),
        "blocked_motor_norm": 0.0,
        "blocked_keys": [],
    }

    if not bool(sleep_mode):
        packet["reason"] = "awake_or_partial_sensor_cut"
        out["sleep_motor_guard"] = packet
        return out

    total_norm = 0.0
    for key in MOTOR_KEYS:
        value = out.get(key)
        if torch.is_tensor(value):
            out[f"imagined_{key}"] = value.detach().clone()
            total_norm += _norm(value)
            out[key] = torch.zeros_like(value)
            packet["blocked_keys"].append(key)

    # Keep action proposal visible, but mark it non-executable.
    if torch.is_tensor(out.get("action_ids")):
        out["imagined_action_ids"] = out["action_ids"].detach().clone()

    packet["blocked_motor_norm"] = float(total_norm)
    packet["reason"] = "full_sleep_mode_blocks_external_motor_execution"
    out["sleep_motor_guard"] = packet
    return out


class SleepMotorGuardRuntimeMixin:
    def apply_sleep_motor_guard(self, out: Dict, *, stage: str = "") -> Dict:
        sleep_mode = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        return block_motor_outputs_for_sleep(out, sleep_mode=sleep_mode, stage=stage)


__all__ = [
    "MOTOR_KEYS",
    "SleepMotorGuardRuntimeMixin",
    "block_motor_outputs_for_sleep",
]
