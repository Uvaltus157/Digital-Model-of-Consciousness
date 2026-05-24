from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LongDynamicObjectMemoryState:
    h: torch.Tensor
    ema: torch.Tensor
    prev_static: torch.Tensor
    steps: int = 0
    dynamic_steps: int = 0


class LongDynamicObjectMemory(nn.Module):
    """
    Neural long-memory module: z_static stream -> z_dynamic_object proposal.

    Phase 1 keeps output_dim == existing vision proposal dim, so the current
    inner_object_system.fusion() can map the dynamic proposal into latent slot
    space without changing the object decoder interfaces.
    """

    def __init__(self, input_dim: int, context_dim: int = 6, hidden_dim: int = 128,
                 ema_alpha: float = 0.08, residual_scale: float = 0.35) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.ema_alpha = float(ema_alpha)
        self.residual_scale = float(residual_scale)
        self.in_proj = nn.Sequential(
            nn.Linear(self.input_dim + self.context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.conf_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.context_dim, max(8, self.hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, self.hidden_dim // 2), 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def initial_state(self, z_static: torch.Tensor) -> LongDynamicObjectMemoryState:
        if z_static.ndim == 1:
            z_static = z_static.unsqueeze(0)
        batch = int(z_static.shape[0])
        h = torch.zeros(batch, self.hidden_dim, device=z_static.device, dtype=z_static.dtype)
        return LongDynamicObjectMemoryState(
            h=h,
            ema=z_static.detach().clone(),
            prev_static=z_static.detach().clone(),
            steps=0,
            dynamic_steps=0,
        )

    def _coerce_context(self, context: Optional[torch.Tensor], z_static: torch.Tensor) -> torch.Tensor:
        batch = int(z_static.shape[0])
        if context is None:
            return torch.zeros(batch, self.context_dim, device=z_static.device, dtype=z_static.dtype)
        context = context.to(device=z_static.device, dtype=z_static.dtype)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != batch:
            context = context.expand(batch, -1)
        if context.shape[-1] < self.context_dim:
            pad = torch.zeros(batch, self.context_dim - context.shape[-1], device=z_static.device, dtype=z_static.dtype)
            context = torch.cat([context, pad], dim=-1)
        elif context.shape[-1] > self.context_dim:
            context = context[:, : self.context_dim]
        return context

    def forward(self, z_static: torch.Tensor, context: Optional[torch.Tensor],
                state: Optional[LongDynamicObjectMemoryState], dynamic_active: bool):
        if z_static.ndim == 1:
            z_static = z_static.unsqueeze(0)
        z_static = z_static.float()
        if state is None:
            state = self.initial_state(z_static)
        context = self._coerce_context(context, z_static)
        h_prev = state.h.detach().to(device=z_static.device, dtype=z_static.dtype)
        ema_prev = state.ema.detach().to(device=z_static.device, dtype=z_static.dtype)
        prev_static = state.prev_static.detach().to(device=z_static.device, dtype=z_static.dtype)
        x = self.in_proj(torch.cat([z_static, context], dim=-1))
        h = self.gru(x, h_prev)
        ema = (1.0 - self.ema_alpha) * ema_prev + self.ema_alpha * z_static
        z_dynamic = ema + self.residual_scale * self.out_proj(h)
        confidence = self.conf_proj(torch.cat([h, context], dim=-1))
        dz = torch.mean(torch.abs(z_static.detach() - prev_static), dim=-1, keepdim=True)
        new_state = LongDynamicObjectMemoryState(
            h=h,
            ema=ema,
            prev_static=z_static.detach(),
            steps=int(state.steps) + 1,
            dynamic_steps=int(state.dynamic_steps) + (1 if bool(dynamic_active) else 0),
        )
        diag = {
            'long_dynamic_steps': float(new_state.steps),
            'long_dynamic_active_steps': float(new_state.dynamic_steps),
            'long_dynamic_dz': float(dz.mean().detach().cpu().item()),
            'long_dynamic_confidence': float(confidence.mean().detach().cpu().item()),
            'long_dynamic_z_static_norm': float(z_static.detach().float().norm(dim=-1).mean().cpu().item()),
            'long_dynamic_z_dynamic_norm': float(z_dynamic.detach().float().norm(dim=-1).mean().cpu().item()),
        }
        return z_dynamic, confidence, new_state, diag
