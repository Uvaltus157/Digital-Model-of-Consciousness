from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class InnerTrustGateConfig:
    """
    Step 4 after InnerOutcomeEvaluator.

    We now have:
        inner_action_confidence       = current intention confidence
        inner_outcome_success_ema     = whether past scenario_z predictions worked
        inner_outcome_error_ema       = how wrong internal predictions are

    InnerTrustGate decides:
        can inner intention influence real policy?
        with what alpha?

    It is a safety/stability gate.
    """
    enabled: bool = True

    # Required trust level before any blend may be allowed.
    min_success_ema: float = 0.55
    max_error_ema: float = 0.18
    min_action_confidence: float = 0.25

    # Blend schedule.
    max_alpha: float = 0.08
    warmup_steps: int = 500
    allow_policy_blend: bool = False

    # If false, gate only reports recommended alpha.
    # If true, runtime may apply alpha to policy output.
    apply_to_policy: bool = False


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


class InnerTrustGate:
    """
    Converts outcome feedback into trust for internal action intentions.

    This prevents a newly initialized inner action decoder from affecting
    real behavior until it has a stable history.
    """

    def __init__(self, cfg: Optional[InnerTrustGateConfig] = None):
        self.cfg = cfg or InnerTrustGateConfig()
        self.step: int = 0
        self.last_gate: Dict[str, Any] = {}

    def compute(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        self.step += 1

        action_conf = _f(obj.get("inner_action_confidence"), 0.0)
        success_ema = _f(obj.get("inner_outcome_success_ema"), 0.0)
        error_ema = _f(obj.get("inner_outcome_error_ema"), 999.0)

        enough_warmup = self.step >= int(self.cfg.warmup_steps)
        pass_conf = action_conf >= float(self.cfg.min_action_confidence)
        pass_success = success_ema >= float(self.cfg.min_success_ema)
        pass_error = error_ema <= float(self.cfg.max_error_ema)

        allowed = (
            bool(self.cfg.allow_policy_blend)
            and enough_warmup
            and pass_conf
            and pass_success
            and pass_error
        )

        # Smooth trust estimate 0..1. Even if not allowed, it is useful for UI.
        conf_part = min(1.0, action_conf / max(1e-6, float(self.cfg.min_action_confidence)))
        success_part = min(1.0, success_ema / max(1e-6, float(self.cfg.min_success_ema)))
        error_part = max(0.0, 1.0 - error_ema / max(1e-6, float(self.cfg.max_error_ema)))
        warmup_part = min(1.0, self.step / max(1.0, float(self.cfg.warmup_steps)))
        trust = max(0.0, min(1.0, conf_part * success_part * error_part * warmup_part))

        alpha = float(self.cfg.max_alpha) * trust if allowed else 0.0

        reason = []
        if not bool(self.cfg.allow_policy_blend):
            reason.append("blend_disabled")
        if not enough_warmup:
            reason.append("warmup")
        if not pass_conf:
            reason.append("low_action_conf")
        if not pass_success:
            reason.append("low_success_ema")
        if not pass_error:
            reason.append("high_error_ema")
        if allowed:
            reason.append("allowed")

        out = {
            "inner_trust_active": True,
            "inner_trust_value": float(trust),
            "inner_trust_alpha": float(alpha),
            "inner_trust_allowed": bool(allowed),
            "inner_trust_reason": ",".join(reason),
            "inner_trust_step": int(self.step),
            "inner_trust_action_conf": float(action_conf),
            "inner_trust_success_ema": float(success_ema),
            "inner_trust_error_ema": float(error_ema),
        }
        self.last_gate = out
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "step": int(self.step),
            "last_gate": self.last_gate,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        cfg = state.get("cfg", {})
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if hasattr(self.cfg, k):
                    try:
                        setattr(self.cfg, k, v)
                    except Exception:
                        pass
        self.step = int(state.get("step", 0) or 0)
        self.last_gate = state.get("last_gate", {})
