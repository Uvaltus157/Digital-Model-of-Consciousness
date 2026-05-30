from __future__ import annotations

"""
M14 Semantic Action Grounding.

Architecture role:
    M14 translates the conscious branch state into action-level control hints.
    It does not replace low-level action heads. It can hold, soften or allow
    already proposed actions based on M12 metacognitive confidence and M11/M15
    danger/viability signals.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class SemanticActionGroundingConfig:
    enabled: bool = True
    hold_threshold: float = 0.70
    verify_threshold: float = 0.55
    high_doubt_threshold: float = 0.50
    min_action_scale: float = 0.05
    soft_hold_scale: float = 0.25


def _scalar_tensor(value, *, device: torch.device, default: float = 0.0) -> torch.Tensor:
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return torch.tensor([[default]], dtype=torch.float32, device=device)
            return value.detach().float().reshape(1, -1)[:, 0:1].to(device)
        if value is None:
            return torch.tensor([[default]], dtype=torch.float32, device=device)
        return torch.tensor([[float(value)]], dtype=torch.float32, device=device)
    except Exception:
        return torch.tensor([[default]], dtype=torch.float32, device=device)


def _device_from_out(out: Dict) -> torch.device:
    for value in out.values():
        if torch.is_tensor(value):
            return value.device
        if isinstance(value, dict):
            for nested in value.values():
                if torch.is_tensor(nested):
                    return nested.device
    return torch.device("cpu")


class SemanticActionGrounding:
    def __init__(self, cfg: Optional[SemanticActionGroundingConfig] = None) -> None:
        self.cfg = cfg or SemanticActionGroundingConfig()

    def compute(self, out: Dict, *, manual_override: bool = False) -> Dict[str, torch.Tensor | str | bool]:
        device = _device_from_out(out)
        c = self.cfg
        meta = out.get("metacognition", {}) if isinstance(out.get("metacognition"), dict) else {}
        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        bc = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}

        action_hold = _scalar_tensor(meta.get("action_hold"), device=device)
        verification_need = _scalar_tensor(meta.get("verification_need"), device=device)
        doubt = _scalar_tensor(meta.get("doubt"), device=device)
        should_hold = _scalar_tensor(meta.get("should_hold_action"), device=device)
        should_verify = _scalar_tensor(meta.get("should_verify"), device=device)
        high_doubt = _scalar_tensor(meta.get("high_doubt"), device=device)
        panic = _scalar_tensor(affect.get("panic_latent"), device=device)
        no_viable = _scalar_tensor(tc.get("no_viable_chain"), device=device)
        predicted_delta = _scalar_tensor(tc.get("predicted_affect_delta"), device=device)
        best_chain_score = _scalar_tensor(tc.get("best_chain_score"), device=device)
        broadcast_urgency = _scalar_tensor(bc.get("urgency"), device=device)

        verify_and_doubt = torch.clamp(verification_need * torch.maximum(doubt, high_doubt), 0.0, 1.0)
        emergency_mode = torch.clamp(torch.maximum(panic, no_viable), 0.0, 1.0)
        action_inhibition = torch.clamp(
            torch.maximum(should_hold, action_hold)
            + 0.55 * verify_and_doubt
            + 0.65 * emergency_mode,
            0.0,
            1.0,
        )
        action_scale = torch.clamp(1.0 - action_inhibition, min=float(c.min_action_scale), max=1.0)
        soft_hold = torch.clamp(torch.minimum(action_scale, torch.tensor([[float(c.soft_hold_scale)]], device=device)), 0.0, 1.0)
        allow_action = (action_inhibition < float(c.hold_threshold)).float()
        verify_before_action = torch.maximum(should_verify, (verification_need > float(c.verify_threshold)).float())

        if bool(manual_override):
            reason = "manual_override_preserved"
            apply_scale = torch.ones_like(action_scale)
        elif float(emergency_mode.reshape(-1)[0].item()) > 0.5:
            reason = "emergency_or_no_viable_chain"
            apply_scale = soft_hold
        elif float(verify_before_action.reshape(-1)[0].item()) > 0.5 and float(doubt.reshape(-1)[0].item()) > float(c.high_doubt_threshold):
            reason = "verify_before_action"
            apply_scale = soft_hold
        elif float(action_hold.reshape(-1)[0].item()) > float(c.hold_threshold):
            reason = "metacognitive_hold"
            apply_scale = soft_hold
        else:
            reason = "action_allowed"
            apply_scale = action_scale

        return {
            "action_scale": action_scale,
            "applied_action_scale": apply_scale,
            "action_inhibition": action_inhibition,
            "allow_action": allow_action,
            "verify_before_action": verify_before_action,
            "emergency_mode": emergency_mode,
            "manual_override_preserved": bool(manual_override),
            "predicted_affect_delta": predicted_delta,
            "best_chain_score": best_chain_score,
            "broadcast_urgency": broadcast_urgency,
            "selected_chain_id": torch.zeros(1, dtype=torch.long, device=device),
            "reason": reason,
        }


__all__ = [
    "SemanticActionGrounding",
    "SemanticActionGroundingConfig",
]
