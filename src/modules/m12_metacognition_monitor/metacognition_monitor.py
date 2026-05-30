from __future__ import annotations

"""
M12 Metacognition Monitor.

Architecture role:
    M12 runs after M9/M7. It does not create thoughts and does not select the
    broadcast. It monitors the conscious branch and estimates whether the agent
    should trust the current self-bound report, doubt it, verify it, or hold an
    action until more evidence arrives.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def pad_or_trim_metacog(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
    if x is None:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(batch_size, dim, device=device, dtype=dtype)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device, dtype=dtype or torch.float32)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    x = x.float()
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _match_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] == 1:
        return x.expand(batch_size, -1)
    return x[:batch_size]


@dataclass
class MetacognitionMonitorConfig:
    enabled: bool = True
    self_dim: int = 128
    focus_context_dim: int = 256
    affect_latent_dim: int = 12
    report_latent_dim: int = 128
    plan_context_dim: int = 256
    hidden_dim: int = 256
    verification_threshold: float = 0.55
    doubt_threshold: float = 0.50
    action_hold_threshold: float = 0.70


class MetacognitionMonitor(nn.Module):
    """Confidence/doubt/check monitor for the conscious branch."""

    def __init__(self, cfg: Optional[MetacognitionMonitorConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MetacognitionMonitorConfig()
        c = self.cfg
        input_dim = (
            c.self_dim
            + c.focus_context_dim
            + c.affect_latent_dim
            + c.report_latent_dim
            + c.plan_context_dim
            + 12
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, 8),
        )

    def forward(
        self,
        *,
        self_state: Optional[torch.Tensor] = None,
        focus_context: Optional[torch.Tensor] = None,
        affect_latents: Optional[torch.Tensor] = None,
        report_latent: Optional[torch.Tensor] = None,
        plan_context: Optional[torch.Tensor] = None,
        scalar_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        c = self.cfg

        # Pick batch size from the first available tensor.
        batch_size = 1
        for x in (self_state, focus_context, affect_latents, report_latent, plan_context, scalar_features):
            if torch.is_tensor(x) and x.ndim >= 1:
                batch_size = int(x.shape[0]) if x.ndim > 1 else 1
                break

        self_state = pad_or_trim_metacog(self_state, c.self_dim, device=device, batch_size=batch_size)
        focus_context = pad_or_trim_metacog(focus_context, c.focus_context_dim, device=device, batch_size=batch_size)
        affect_latents = pad_or_trim_metacog(affect_latents, c.affect_latent_dim, device=device, batch_size=batch_size)
        report_latent = pad_or_trim_metacog(report_latent, c.report_latent_dim, device=device, batch_size=batch_size)
        plan_context = pad_or_trim_metacog(plan_context, c.plan_context_dim, device=device, batch_size=batch_size)
        scalar_features = pad_or_trim_metacog(scalar_features, 12, device=device, batch_size=batch_size)

        self_state = _match_batch(self_state, batch_size)
        focus_context = _match_batch(focus_context, batch_size)
        affect_latents = _match_batch(affect_latents, batch_size)
        report_latent = _match_batch(report_latent, batch_size)
        plan_context = _match_batch(plan_context, batch_size)
        scalar_features = _match_batch(scalar_features, batch_size)

        h = self.encoder(torch.cat([
            self_state,
            focus_context,
            affect_latents,
            report_latent,
            plan_context,
            scalar_features,
        ], dim=-1))
        raw = self.head(h)
        prob = torch.sigmoid(raw)

        confidence = prob[:, 0:1]
        doubt = prob[:, 1:2]
        verification_need = prob[:, 2:3]
        contradiction_score = prob[:, 3:4]
        action_hold = prob[:, 4:5]
        report_trust = prob[:, 5:6]
        self_consistency = prob[:, 6:7]
        evidence_sufficiency = prob[:, 7:8]

        # Deterministic safety overlays: high panic/no viable chain should raise
        # checking and action-hold even before learning has calibrated the head.
        panic = scalar_features[:, 8:9]
        no_viable = scalar_features[:, 9:10]
        broadcast_priority = scalar_features[:, 10:11]
        inner_speech_conf = scalar_features[:, 2:3]
        self_conf = scalar_features[:, 0:1]

        verification_need = torch.clamp(torch.maximum(verification_need, 0.45 * panic + 0.35 * no_viable), 0.0, 1.0)
        action_hold = torch.clamp(torch.maximum(action_hold, 0.50 * panic + 0.35 * no_viable), 0.0, 1.0)
        confidence = torch.clamp(0.55 * confidence + 0.25 * self_conf + 0.20 * inner_speech_conf, 0.0, 1.0)
        doubt = torch.clamp(torch.maximum(doubt, verification_need * (1.0 - confidence)), 0.0, 1.0)
        evidence_sufficiency = torch.clamp(0.70 * evidence_sufficiency + 0.30 * broadcast_priority, 0.0, 1.0)

        return {
            "metacognitive_confidence": confidence,
            "doubt": doubt,
            "verification_need": verification_need,
            "contradiction_score": contradiction_score,
            "action_hold": action_hold,
            "report_trust": report_trust,
            "self_consistency": self_consistency,
            "evidence_sufficiency": evidence_sufficiency,
            "should_verify": (verification_need > float(c.verification_threshold)).float(),
            "should_hold_action": (action_hold > float(c.action_hold_threshold)).float(),
            "high_doubt": (doubt > float(c.doubt_threshold)).float(),
        }


def render_metacognition_text(m: Dict[str, torch.Tensor]) -> str:
    def f(key: str) -> float:
        try:
            v = m.get(key)
            if torch.is_tensor(v):
                return float(v.detach().cpu().reshape(-1)[0].item())
            return float(v or 0.0)
        except Exception:
            return 0.0

    confidence = f("metacognitive_confidence")
    doubt = f("doubt")
    verify = f("verification_need")
    hold = f("action_hold")
    if hold > 0.70:
        return f"hold action; verify before acting | conf={confidence:.2f} doubt={doubt:.2f}"
    if verify > 0.55 or doubt > 0.50:
        return f"check this before trusting it | conf={confidence:.2f} doubt={doubt:.2f} verify={verify:.2f}"
    return f"report seems usable | conf={confidence:.2f} doubt={doubt:.2f}"


__all__ = [
    "MetacognitionMonitor",
    "MetacognitionMonitorConfig",
    "pad_or_trim_metacog",
    "render_metacognition_text",
]
