from __future__ import annotations

"""
M7 self-bound inner speech decoder.

Stage-6 architecture rule:
    M7 does not read raw M5 symbolic_report as its primary source.
    M7 verbalizes already self-bound thought content:
        M15 active_thought_chain / plan_context
        M9 self_bound_context / subjective_affect_state
        M10-M11 affect_latents

The decoder returns a structured inner_speech packet. It intentionally keeps a
small heuristic text renderer so visualizers have readable output even before a
trained language head exists.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def pad_or_trim_inner_speech(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
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


def _scalar(x, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


@dataclass
class InnerSpeechDecoderConfig:
    enabled: bool = True
    active_thought_dim: int = 128
    plan_context_dim: int = 256
    self_bound_context_dim: int = 396
    subjective_affect_dim: int = 16
    affect_latent_dim: int = 12
    hidden_dim: int = 256
    report_latent_dim: int = 128
    vocab_size: int = 2048
    max_tokens: int = 24


class InnerSpeechDecoder(nn.Module):
    def __init__(self, cfg: Optional[InnerSpeechDecoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or InnerSpeechDecoderConfig()
        c = self.cfg
        input_dim = c.active_thought_dim + c.plan_context_dim + c.self_bound_context_dim + c.subjective_affect_dim + c.affect_latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
        )
        self.report_latent = nn.Sequential(
            nn.Linear(c.hidden_dim, c.report_latent_dim),
            nn.LayerNorm(c.report_latent_dim),
        )
        self.confidence_head = nn.Sequential(nn.Linear(c.hidden_dim, 1), nn.Sigmoid())
        self.token_head = nn.Linear(c.hidden_dim, c.max_tokens * c.vocab_size)

    def forward(
        self,
        *,
        active_thought: torch.Tensor,
        plan_context: Optional[torch.Tensor] = None,
        self_bound_context: Optional[torch.Tensor] = None,
        subjective_affect_state: Optional[torch.Tensor] = None,
        affect_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        c = self.cfg
        active_thought = pad_or_trim_inner_speech(active_thought, c.active_thought_dim, device=device)
        batch_size = active_thought.shape[0]
        plan_context = pad_or_trim_inner_speech(plan_context, c.plan_context_dim, device=device, batch_size=batch_size)
        self_bound_context = pad_or_trim_inner_speech(self_bound_context, c.self_bound_context_dim, device=device, batch_size=batch_size)
        subjective_affect_state = pad_or_trim_inner_speech(subjective_affect_state, c.subjective_affect_dim, device=device, batch_size=batch_size)
        affect_latents = pad_or_trim_inner_speech(affect_latents, c.affect_latent_dim, device=device, batch_size=batch_size)

        plan_context = _match_batch(plan_context, batch_size)
        self_bound_context = _match_batch(self_bound_context, batch_size)
        subjective_affect_state = _match_batch(subjective_affect_state, batch_size)
        affect_latents = _match_batch(affect_latents, batch_size)

        h = self.encoder(torch.cat([active_thought, plan_context, self_bound_context, subjective_affect_state, affect_latents], dim=-1))
        report_latent = self.report_latent(h)
        confidence = self.confidence_head(h)
        logits = self.token_head(h).reshape(batch_size, c.max_tokens, c.vocab_size)
        token_ids = torch.argmax(logits, dim=-1)
        return {
            "report_latent": report_latent,
            "confidence": confidence,
            "text_token_logits": logits,
            "text_token_ids": token_ids,
        }


def render_inner_speech_text(*, self_core: Dict, thought_chain: Dict, affect: Dict, confidence: float = 0.0) -> str:
    """Readable symbolic report for debug UI before the language head is trained."""
    agency = _scalar(self_core.get("agency_score"), 0.0)
    ownership = _scalar(self_core.get("body_ownership_score"), 0.0)
    continuity = _scalar(self_core.get("self_continuity_score"), 0.0)
    focus_binding = _scalar(self_core.get("focus_binding_score"), 0.0)
    affect_binding = _scalar(self_core.get("affect_binding_score"), 0.0)
    panic = _scalar(affect.get("panic_latent"), 0.0)
    comfort = _scalar(affect.get("comfort_latent"), 0.0)
    valence = _scalar(affect.get("valence"), 0.0)

    metrics = thought_chain.get("thought_chain_metrics", {}) if isinstance(thought_chain.get("thought_chain_metrics"), dict) else {}
    readiness = _scalar(metrics.get("planning_readiness"), 0.0)
    urgency = _scalar(metrics.get("urgency"), 0.0)

    parts = []
    if agency > 0.55:
        parts.append("I am causing this action")
    else:
        parts.append("I am observing weak agency")

    if ownership > 0.55 and continuity > 0.55:
        parts.append("my body state is continuous")
    elif ownership > 0.35:
        parts.append("my body signals are partly mine")
    else:
        parts.append("body ownership is uncertain")

    if focus_binding > 0.55:
        parts.append("the current focus is self-bound")
    else:
        parts.append("the current focus is not yet strongly self-bound")

    if affect_binding > 0.55:
        if panic > comfort:
            parts.append("the affect feels tense")
        elif comfort > 0.35 or valence > 0.1:
            parts.append("the affect feels safe enough")
        else:
            parts.append("the affect is present")
    else:
        parts.append("affect is weakly bound")

    if readiness > 0.55:
        parts.append("I can form a plan")
    elif urgency > 0.55:
        parts.append("attention is urgent but planning is uncertain")
    else:
        parts.append("I am still forming the thought chain")

    parts.append(f"confidence {confidence:.2f}")
    return ". ".join(parts) + "."


__all__ = [
    "InnerSpeechDecoder",
    "InnerSpeechDecoderConfig",
    "pad_or_trim_inner_speech",
    "render_inner_speech_text",
]
