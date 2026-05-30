from __future__ import annotations

"""
M10 Global Conscious Broadcast.

Architecture rule:
    M10 is not the limbic motivator. M11 produces affect/motivation latents.
    M10 is the selector/gate that decides which candidate material is broadcast
    into the conscious/self-binding branch.

Inputs:
    - M5/M15 focus_context
    - optional raw_focus_context before M15 enhancement
    - M15 active thought / best chain / plan context / predicted affect delta
    - M11 affect_latents

Output:
    out["broadcast"] packet and broadcast_latent suitable for M9 focus input.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def pad_or_trim_broadcast(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
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
class GlobalBroadcastConfig:
    enabled: bool = True
    focus_context_dim: int = 256
    affect_latent_dim: int = 12
    thought_dim: int = 128
    plan_context_dim: int = 256
    hidden_dim: int = 256
    num_sources: int = 4
    broadcast_threshold: float = 0.35


class GlobalConsciousBroadcastGate(nn.Module):
    """Competition/selector/gate for conscious broadcast access."""

    SOURCE_NAMES = ("m15_focus", "m5_raw_focus", "m15_chain", "m11_affect")

    def __init__(self, cfg: Optional[GlobalBroadcastConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or GlobalBroadcastConfig()
        c = self.cfg

        self.raw_focus_proj = nn.Sequential(
            nn.Linear(c.focus_context_dim, c.focus_context_dim),
            nn.LayerNorm(c.focus_context_dim),
        )
        self.chain_proj = nn.Sequential(
            nn.Linear(c.thought_dim + c.plan_context_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.focus_context_dim),
            nn.LayerNorm(c.focus_context_dim),
        )
        self.affect_proj = nn.Sequential(
            nn.Linear(c.affect_latent_dim, c.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.focus_context_dim),
            nn.LayerNorm(c.focus_context_dim),
        )
        self.source_score = nn.Sequential(
            nn.Linear(c.focus_context_dim + c.affect_latent_dim + 5, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, 1),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(c.focus_context_dim + c.affect_latent_dim + 5, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(c.focus_context_dim + c.affect_latent_dim + 5, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _importance_features(
        self,
        *,
        affect_latents: torch.Tensor,
        best_chain_score: torch.Tensor,
        predicted_affect_delta: torch.Tensor,
        no_viable_chain: torch.Tensor,
        panic_trigger: torch.Tensor,
    ) -> torch.Tensor:
        # Affect layout follows M11 EmotionalDrive._build_affect_packet():
        # [valence, arousal, pain, stress, fear, panic, comfort, relief, curiosity, discovery, coherence, expected_delta]
        batch = affect_latents.shape[0]
        if affect_latents.shape[-1] >= 12:
            arousal = affect_latents[..., 1:2]
            pain = affect_latents[..., 2:3]
            stress = affect_latents[..., 3:4]
            fear = affect_latents[..., 4:5]
            panic = affect_latents[..., 5:6]
            curiosity = affect_latents[..., 8:9]
            discovery = affect_latents[..., 9:10]
            expected_delta = affect_latents[..., 11:12]
        else:
            z = torch.zeros(batch, 1, device=affect_latents.device, dtype=affect_latents.dtype)
            arousal = pain = stress = fear = panic = curiosity = discovery = expected_delta = z

        affect_priority = torch.clamp(
            0.24 * arousal
            + 0.22 * pain
            + 0.22 * stress
            + 0.18 * fear
            + 0.32 * panic
            + 0.14 * curiosity
            + 0.10 * discovery,
            0.0,
            1.0,
        )
        chain_priority = torch.clamp(0.55 * best_chain_score + 0.45 * torch.relu(predicted_affect_delta), 0.0, 1.0)
        danger_priority = torch.clamp(torch.maximum(panic_trigger, no_viable_chain) + panic + stress + fear, 0.0, 1.0)
        expected_priority = torch.clamp(torch.relu(expected_delta) + torch.relu(predicted_affect_delta), 0.0, 1.0)
        return torch.cat([affect_priority, chain_priority, danger_priority, expected_priority, arousal], dim=-1)

    def forward(
        self,
        *,
        focus_context: torch.Tensor,
        raw_focus_context: Optional[torch.Tensor] = None,
        active_thought: Optional[torch.Tensor] = None,
        plan_context: Optional[torch.Tensor] = None,
        affect_latents: Optional[torch.Tensor] = None,
        best_chain_score: Optional[torch.Tensor] = None,
        predicted_affect_delta: Optional[torch.Tensor] = None,
        no_viable_chain: Optional[torch.Tensor] = None,
        panic_trigger: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | str]:
        device = next(self.parameters()).device
        c = self.cfg

        focus_context = pad_or_trim_broadcast(focus_context, c.focus_context_dim, device=device)
        batch_size = focus_context.shape[0]
        raw_focus_context = pad_or_trim_broadcast(raw_focus_context, c.focus_context_dim, device=device, batch_size=batch_size)
        active_thought = pad_or_trim_broadcast(active_thought, c.thought_dim, device=device, batch_size=batch_size)
        plan_context = pad_or_trim_broadcast(plan_context, c.plan_context_dim, device=device, batch_size=batch_size)
        affect_latents = pad_or_trim_broadcast(affect_latents, c.affect_latent_dim, device=device, batch_size=batch_size)
        best_chain_score = pad_or_trim_broadcast(best_chain_score, 1, device=device, batch_size=batch_size)
        predicted_affect_delta = pad_or_trim_broadcast(predicted_affect_delta, 1, device=device, batch_size=batch_size)
        no_viable_chain = pad_or_trim_broadcast(no_viable_chain, 1, device=device, batch_size=batch_size)
        panic_trigger = pad_or_trim_broadcast(panic_trigger, 1, device=device, batch_size=batch_size)

        raw_focus_context = _match_batch(raw_focus_context, batch_size)
        active_thought = _match_batch(active_thought, batch_size)
        plan_context = _match_batch(plan_context, batch_size)
        affect_latents = _match_batch(affect_latents, batch_size)
        best_chain_score = _match_batch(best_chain_score, batch_size)
        predicted_affect_delta = _match_batch(predicted_affect_delta, batch_size)
        no_viable_chain = _match_batch(no_viable_chain, batch_size)
        panic_trigger = _match_batch(panic_trigger, batch_size)

        features = self._importance_features(
            affect_latents=affect_latents,
            best_chain_score=best_chain_score,
            predicted_affect_delta=predicted_affect_delta,
            no_viable_chain=no_viable_chain,
            panic_trigger=panic_trigger,
        )

        candidates = torch.stack([
            focus_context,
            self.raw_focus_proj(raw_focus_context),
            self.chain_proj(torch.cat([active_thought, plan_context], dim=-1)),
            self.affect_proj(affect_latents),
        ], dim=1)

        expanded_affect = affect_latents.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        expanded_features = features.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        score_input = torch.cat([candidates, expanded_affect, expanded_features], dim=-1)
        source_logits = self.source_score(score_input).squeeze(-1)
        source_weights = torch.softmax(source_logits, dim=-1)
        selected_idx = torch.argmax(source_weights, dim=-1)

        weighted_context = torch.sum(source_weights.unsqueeze(-1) * candidates, dim=1)
        gate_input = torch.cat([weighted_context, affect_latents, features], dim=-1)
        broadcast_gate = self.context_gate(gate_input)
        priority = self.priority_head(gate_input)
        gated_context = broadcast_gate * weighted_context + (1.0 - broadcast_gate) * focus_context

        selected_sources = [self.SOURCE_NAMES[int(i.detach().cpu().item())] for i in selected_idx]
        # For batch size 1 keep a readable scalar string, for larger batches keep a
        # comma-separated diagnostic string. Numeric selected_idx remains tensor.
        selected_source = selected_sources[0] if len(selected_sources) == 1 else ",".join(selected_sources)

        return {
            "broadcast_latent": gated_context,
            "selected_idx": selected_idx,
            "selected_source": selected_source,
            "source_logits": source_logits,
            "source_weights": source_weights,
            "priority": priority,
            "broadcast_gate": broadcast_gate,
            "urgency": features[..., 2:3],
            "affect_priority": features[..., 0:1],
            "chain_priority": features[..., 1:2],
            "expected_affect_priority": features[..., 3:4],
            "arousal": features[..., 4:5],
            "broadcast_threshold": torch.tensor([float(c.broadcast_threshold)], device=device, dtype=gated_context.dtype),
            "broadcast_allowed": (priority > float(c.broadcast_threshold)).float(),
        }


__all__ = [
    "GlobalBroadcastConfig",
    "GlobalConsciousBroadcastGate",
    "pad_or_trim_broadcast",
]
