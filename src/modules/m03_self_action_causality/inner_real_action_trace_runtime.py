from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import torch


def _scalar(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _norm(x: Any) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().norm(dim=-1).mean().cpu().item())
    except Exception:
        pass
    return 0.0


class InnerRealActionTraceRuntimeMixin:
    """
    Logs and exposes the real action path.

    Purpose:
        verify that inner coded-world intention really reaches the body only
        through the trust gate.

    Trace path:
        inner_action_body / inner_action_hand
        -> inner_trust_alpha
        -> out["embodied_targets"] / out["hand_ctrl"]
        -> prev_embodied_action / prev_hand_motor
        -> dynamic controller / world.observe on next step
    """

    def _inner_action_trace_enabled(self) -> bool:
        cfg = getattr(self.cfg, "inner_real_action_trace", None)
        return bool(getattr(cfg, "enabled", True))

    def _inner_action_trace_path(self) -> Path:
        cfg = getattr(self.cfg, "inner_real_action_trace", None)
        path = str(getattr(cfg, "path", "") or "")
        if path:
            return Path(path)
        root = getattr(self, "out_dir", Path("./runs"))
        return Path(root) / "inner_real_action_trace.jsonl"

    def _maybe_write_inner_action_trace(self, rec: Dict[str, Any]) -> None:
        cfg = getattr(self.cfg, "inner_real_action_trace", None)
        if not bool(getattr(cfg, "write_jsonl", True)):
            return

        every = max(1, int(getattr(cfg, "write_every_steps", 10)))
        step = int(getattr(self, "global_step", 0))
        if step % every != 0:
            return

        path = self._inner_action_trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    def update_inner_real_action_trace(self, obj: dict, out: dict | None = None) -> dict:
        if not self._inner_action_trace_enabled():
            return obj
        if not isinstance(obj, dict):
            return obj

        try:
            step = int(getattr(self, "global_step", 0))
            rec: Dict[str, Any] = {
                "time": time.time(),
                "step": step,
                "inner_action_active": bool(_scalar(obj.get("inner_action_active"), 0.0) > 0.5),
                "inner_action_confidence": _scalar(obj.get("inner_action_confidence"), 0.0),
                "inner_trust_value": _scalar(obj.get("inner_trust_value"), 0.0),
                "inner_trust_alpha": _scalar(obj.get("inner_trust_alpha"), 0.0),
                "inner_trust_allowed": bool(_scalar(obj.get("inner_trust_allowed"), 0.0) > 0.5),
                "inner_trust_applied_to_policy": bool(_scalar(obj.get("inner_trust_applied_to_policy"), 0.0) > 0.5),
                "inner_trust_reason": str(obj.get("inner_trust_reason", "")),
                "inner_mind_selected_sentence": str(obj.get("inner_mind_selected_sentence", "")),
                "inner_mind_selected_slot_token": str(obj.get("inner_mind_selected_slot_token", "")),
                "inner_action_slot_token": str(obj.get("inner_action_slot_token", "")),
                "inner_action_body_norm": _norm(obj.get("inner_action_body")),
                "inner_action_hand_norm": _norm(obj.get("inner_action_hand")),
            }

            if out is not None:
                rec["out_embodied_targets_norm"] = _norm(out.get("embodied_targets"))
                rec["out_hand_ctrl_norm"] = _norm(out.get("hand_ctrl"))
                rec["out_leg_ctrl_norm"] = _norm(out.get("leg_ctrl"))

            if hasattr(self, "prev_embodied_action"):
                rec["prev_embodied_action_norm"] = _norm(getattr(self, "prev_embodied_action"))
            if hasattr(self, "prev_hand_motor"):
                rec["prev_hand_motor_norm"] = _norm(getattr(self, "prev_hand_motor"))

            # If trust gate placed detailed before/after data, preserve it.
            extra = obj.get("inner_action_blend_trace", None)
            if isinstance(extra, dict):
                rec.update({f"blend_{k}": v for k, v in extra.items()})

            obj["inner_real_action_trace"] = rec
            obj["inner_real_action_trace_step"] = step
            obj["inner_real_action_trace_path"] = str(self._inner_action_trace_path())

            self._maybe_write_inner_action_trace(rec)
            self.latest_inner_real_action_trace = rec
            return obj

        except Exception as e:
            if not hasattr(self, "_inner_real_action_trace_warned"):
                print(f"[inner_real_action_trace] update failed: {e}")
                self._inner_real_action_trace_warned = True
            return obj
