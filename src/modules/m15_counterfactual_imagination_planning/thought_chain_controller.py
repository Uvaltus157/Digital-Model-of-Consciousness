from __future__ import annotations

"""
M15 ThoughtChainController.

Stage-5 architecture rule:
    M15 receives self-bound context from M9, focused world-model content from M5,
    and affect latents from M10/M11. It creates a compact active thought chain
    and plan context for future counterfactual planning.

This module intentionally does not generate inner speech text. M7 can later
verbalize the self-bound active_thought_chain.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def pad_or_trim_thought_chain(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
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
class ThoughtChainControllerConfig:
    enabled: bool = True
    self_bound_context_dim: int = 396
    subjective_affect_dim: int = 16
    focus_context_dim: int = 256
    affect_latent_dim: int = 12
    hidden_dim: int = 256
    thought_dim: int = 128
    plan_context_dim: int = 256
    chain_len: int = 4


class ThoughtChainController(nn.Module):
    def __init__(self, cfg: Optional[ThoughtChainControllerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ThoughtChainControllerConfig()
        c = self.cfg
        input_dim = c.self_bound_context_dim + c.subjective_affect_dim + c.focus_context_dim + c.affect_latent_dim

        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
        )
        self.seed_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.thought_dim),
            nn.LayerNorm(c.thought_dim),
        )
        self.recurrent = nn.GRUCell(c.thought_dim + c.affect_latent_dim, c.thought_dim)
        self.gate = nn.Sequential(nn.Linear(c.thought_dim * 2 + c.affect_latent_dim, c.thought_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(c.thought_dim)
        self.plan_head = nn.Sequential(
            nn.Linear(c.thought_dim + c.hidden_dim + c.affect_latent_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.plan_context_dim),
            nn.LayerNorm(c.plan_context_dim),
        )
        self.metrics_head = nn.Sequential(
            nn.Linear(c.thought_dim + c.plan_context_dim + c.affect_latent_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        *,
        self_bound_context: torch.Tensor,
        subjective_affect_state: Optional[torch.Tensor] = None,
        focus_context: Optional[torch.Tensor] = None,
        affect_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        c = self.cfg

        self_bound_context = pad_or_trim_thought_chain(self_bound_context, c.self_bound_context_dim, device=device)
        batch_size = self_bound_context.shape[0]
        subjective_affect_state = pad_or_trim_thought_chain(subjective_affect_state, c.subjective_affect_dim, device=device, batch_size=batch_size)
        focus_context = pad_or_trim_thought_chain(focus_context, c.focus_context_dim, device=device, batch_size=batch_size)
        affect_latents = pad_or_trim_thought_chain(affect_latents, c.affect_latent_dim, device=device, batch_size=batch_size)

        subjective_affect_state = _match_batch(subjective_affect_state, batch_size)
        focus_context = _match_batch(focus_context, batch_size)
        affect_latents = _match_batch(affect_latents, batch_size)

        encoded = self.input_encoder(torch.cat([self_bound_context, subjective_affect_state, focus_context, affect_latents], dim=-1))
        seed = self.seed_head(encoded)
        h = seed
        chain = []
        for _ in range(int(c.chain_len)):
            drive = torch.cat([h, affect_latents], dim=-1)
            candidate = self.recurrent(drive, h)
            gate = self.gate(torch.cat([h, candidate, affect_latents], dim=-1))
            h = self.norm(gate * candidate + (1.0 - gate) * h)
            chain.append(h)

        active_chain = torch.stack(chain, dim=1) if chain else h.unsqueeze(1)
        active_thought = active_chain[:, -1]
        plan_context = self.plan_head(torch.cat([active_thought, encoded, affect_latents], dim=-1))
        metrics = self.metrics_head(torch.cat([active_thought, plan_context, affect_latents], dim=-1))

        return {
            "candidate_thought_chain": active_chain,
            "active_thought_chain": active_chain,
            "active_thought": active_thought,
            "thought_seed": seed,
            "plan_context": plan_context,
            "thought_chain_metrics": {
                "stability": metrics[:, 0:1],
                "urgency": metrics[:, 1:2],
                "self_relevance": metrics[:, 2:3],
                "planning_readiness": metrics[:, 3:4],
            },
        }


__all__ = [
    "ThoughtChainController",
    "ThoughtChainControllerConfig",
    "pad_or_trim_thought_chain",
]
