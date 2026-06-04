from __future__ import annotations

"""
M5 FocusFeedbackBoundary.

This is the correct M15 -> M5 receptor layer.

It receives a post-self M15 enhanced_focus_context from the previous step and
projects it into the current M5 boundary:

    focus_context_seed
        -> learned receptor
        -> workspace_seed bias
        -> preconscious_seed bias
        -> report refresh

It is intentionally NOT a raw self_state injection.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _pad_or_trim(x: Optional[torch.Tensor], dim: int, *, ref: torch.Tensor) -> torch.Tensor:
    if x is None or not torch.is_tensor(x):
        return torch.zeros(ref.shape[0], dim, device=ref.device, dtype=ref.dtype)

    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    x = x.to(device=ref.device, dtype=ref.dtype).float()
    if x.shape[0] != ref.shape[0]:
        if x.shape[0] == 1:
            x = x.expand(ref.shape[0], -1)
        else:
            x = x[: ref.shape[0]]

    if x.shape[-1] > dim:
        return x[..., :dim]
    if x.shape[-1] < dim:
        pad = torch.zeros(x.shape[0], dim - x.shape[-1], device=ref.device, dtype=ref.dtype)
        return torch.cat([x, pad], dim=-1)
    return x


def _gate_tensor(gate: Any, *, ref: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if torch.is_tensor(gate):
        g = gate.to(device=ref.device, dtype=ref.dtype).float()
        if g.ndim == 0:
            g = g.reshape(1, 1)
        elif g.ndim == 1:
            g = g.reshape(-1, 1)
        elif g.ndim > 2:
            g = g.reshape(g.shape[0], -1)[:, :1]
    else:
        try:
            value = float(gate) if gate is not None else float(default)
        except Exception:
            value = float(default)
        g = torch.full((ref.shape[0], 1), value, device=ref.device, dtype=ref.dtype)

    if g.shape[0] != ref.shape[0]:
        g = g[:1].expand(ref.shape[0], -1)
    return g.clamp(0.0, 0.35)


class FocusFeedbackBoundary(nn.Module):
    """
    Learned boundary receptor inside M5.

    It performs two injections:
    1) pre-fusion injection into workspace_seed;
    2) post-Workspace injection into preconscious_seed.

    The external runtime gate protects against recursive self-runaway, while the
    learned receptor decides how much of the seed is actually relevant to M5.
    """

    def __init__(
        self,
        *,
        focus_context_dim: int = 256,
        workspace_seed_dim: int = 256,
        thought_dim: int = 192,
        hidden_dim: int = 256,
        max_gate: float = 0.35,
    ) -> None:
        super().__init__()
        self.focus_context_dim = int(focus_context_dim)
        self.workspace_seed_dim = int(workspace_seed_dim)
        self.thought_dim = int(thought_dim)
        self.max_gate = float(max_gate)

        self.seed_encoder = nn.Sequential(
            nn.Linear(self.focus_context_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(self.workspace_seed_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.receptor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.learned_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.workspace_delta = nn.Sequential(
            nn.Linear(hidden_dim, self.workspace_seed_dim),
            nn.Tanh(),
        )
        self.preconscious_delta = nn.Sequential(
            nn.Linear(hidden_dim, self.thought_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        *,
        workspace_seed: torch.Tensor,
        focus_context_seed: Optional[torch.Tensor] = None,
        focus_context_seed_gate: Any = None,
    ) -> Dict[str, torch.Tensor]:
        if not torch.is_tensor(focus_context_seed):
            zero_gate = torch.zeros(workspace_seed.shape[0], 1, device=workspace_seed.device, dtype=workspace_seed.dtype)
            return {
                "active": zero_gate,
                "workspace_seed": workspace_seed,
                "external_gate": zero_gate,
                "learned_gate": zero_gate,
                "total_gate": zero_gate,
                "workspace_delta": torch.zeros_like(workspace_seed),
                "preconscious_delta": torch.zeros(workspace_seed.shape[0], self.thought_dim, device=workspace_seed.device, dtype=workspace_seed.dtype),
                "seed_norm": zero_gate,
            }

        seed = _pad_or_trim(focus_context_seed, self.focus_context_dim, ref=workspace_seed)
        current = _pad_or_trim(workspace_seed, self.workspace_seed_dim, ref=workspace_seed)

        h_seed = self.seed_encoder(seed)
        h_current = self.current_encoder(current)
        h = self.receptor(torch.cat([h_current, h_seed], dim=-1))

        ext_gate = _gate_tensor(focus_context_seed_gate, ref=workspace_seed, default=0.0)
        learned = self.learned_gate(h).to(dtype=workspace_seed.dtype)
        total_gate = (ext_gate * learned).clamp(0.0, self.max_gate)

        w_delta = self.workspace_delta(h).to(dtype=workspace_seed.dtype)
        p_delta = self.preconscious_delta(h).to(dtype=workspace_seed.dtype)
        workspace_seed_out = workspace_seed + total_gate * w_delta

        return {
            "active": (total_gate > 0.0).float(),
            "workspace_seed": workspace_seed_out,
            "external_gate": ext_gate,
            "learned_gate": learned,
            "total_gate": total_gate,
            "workspace_delta": w_delta,
            "preconscious_delta": p_delta,
            "seed_norm": seed.norm(dim=-1, keepdim=True),
        }

    def apply_preconscious_seed(self, preconscious_seed: torch.Tensor, packet: Dict[str, torch.Tensor]) -> torch.Tensor:
        delta = packet.get("preconscious_delta")
        gate = packet.get("total_gate")
        if not torch.is_tensor(delta) or not torch.is_tensor(gate):
            return preconscious_seed
        if delta.shape[-1] != preconscious_seed.shape[-1]:
            delta = _pad_or_trim(delta, preconscious_seed.shape[-1], ref=preconscious_seed)
        if gate.shape[0] != preconscious_seed.shape[0]:
            gate = gate[:1].expand(preconscious_seed.shape[0], -1)
        return preconscious_seed + gate.to(preconscious_seed.dtype) * delta.to(preconscious_seed.dtype)


__all__ = ["FocusFeedbackBoundary"]
