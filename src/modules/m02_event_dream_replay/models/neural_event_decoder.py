from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeuralEventDecoderConfig:
    """
    Trainable level-5 decoder.

    Previous deterministic decoder:
        stored z_before/z_after -> interpolate -> replay.

    Neural decoder:
        event_code + slot latent + sentence role features
        -> predicted delta_z / z_after.

    This is the first learnable bridge from code-sentences to latent dynamics.
    """
    enabled: bool = True
    latent_dim: int = 128
    event_code_dim: int = 8
    role_dim: int = 16
    hidden_dim: int = 256
    loss_weight: float = 0.05
    max_delta: float = 0.35


_VERBS = {
    "changes": 0,
    "latent_changes": 1,
    "touch_changes": 2,
    "self_moves_changes": 3,
    "dream_replays": 4,
}
_CONTEXTS = {
    "latent": 0,
    "contact": 1,
    "self_action": 2,
    "dream": 3,
    "unknown": 4,
}


class NeuralEventDecoder(nn.Module):
    """
    Small MLP that learns:
        z_before + event_code + sentence_roles -> delta_z

    It does not replace ObjectImaginationHead2D.
    It predicts the latent trajectory input that ObjectImaginationHead2D/3D
    can decode into inner images.
    """

    def __init__(self, cfg: Optional[NeuralEventDecoderConfig] = None):
        super().__init__()
        self.cfg = cfg or NeuralEventDecoderConfig()
        latent_dim = int(self.cfg.latent_dim)
        event_dim = int(self.cfg.event_code_dim)
        role_dim = int(self.cfg.role_dim)
        hidden = int(self.cfg.hidden_dim)

        self.role_embed = nn.Embedding(32, role_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + event_dim + role_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def _role_ids(self, roles: Dict[str, Any], device) -> torch.Tensor:
        if not isinstance(roles, dict):
            roles = {}
        verb = str(roles.get("verb", "changes") or "changes")
        ctx = str(roles.get("context", "unknown") or "unknown")
        verb_id = _VERBS.get(verb, 0)
        ctx_id = _CONTEXTS.get(ctx, _CONTEXTS["unknown"])
        return torch.tensor([verb_id, 16 + ctx_id], device=device, dtype=torch.long)

    def role_features(self, roles: Dict[str, Any], batch: int, device) -> torch.Tensor:
        ids = self._role_ids(roles, device)
        emb = self.role_embed(ids).reshape(1, -1)
        return emb.repeat(batch, 1)

    def forward(
        self,
        z_before: torch.Tensor,
        event_code: torch.Tensor,
        roles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_before.ndim == 1:
            z_before = z_before.unsqueeze(0)
        z_before = z_before.float()

        if event_code.ndim == 1:
            event_code = event_code.unsqueeze(0)
        event_code = event_code.float().to(device=z_before.device)
        if event_code.shape[0] != z_before.shape[0]:
            event_code = event_code[:1].repeat(z_before.shape[0], 1)

        # Pad/trim event_code.
        event_dim = int(self.cfg.event_code_dim)
        if event_code.shape[-1] < event_dim:
            event_code = F.pad(event_code, (0, event_dim - event_code.shape[-1]))
        elif event_code.shape[-1] > event_dim:
            event_code = event_code[:, :event_dim]

        role = self.role_features(roles or {}, z_before.shape[0], z_before.device)
        x = torch.cat([z_before, event_code, role], dim=-1)
        raw_delta = self.net(x)

        max_delta = float(self.cfg.max_delta)
        delta = torch.tanh(raw_delta) * max_delta
        z_pred = z_before + delta
        return {
            "pred_delta_z": delta,
            "pred_z_after": z_pred,
        }

    def loss_from_event(self, event: Dict[str, Any], device, dtype=torch.float32) -> Dict[str, torch.Tensor]:
        if not isinstance(event, dict):
            return {}

        z0 = event.get("z_before", None)
        z1 = event.get("z_after", None)
        event_code = event.get("event_code", None)
        roles = event.get("sentence_roles", event.get("roles", {}))

        if not torch.is_tensor(z0) or not torch.is_tensor(z1):
            return {}
        if not torch.is_tensor(event_code):
            return {}

        z0 = z0.detach().to(device=device, dtype=dtype)
        z1 = z1.detach().to(device=device, dtype=dtype)
        if z0.ndim == 1:
            z0 = z0.unsqueeze(0)
        if z1.ndim == 1:
            z1 = z1.unsqueeze(0)

        out = self.forward(z0, event_code.to(device=device, dtype=dtype), roles)
        loss = F.mse_loss(out["pred_z_after"], z1)
        return {
            "event_decoder_loss": loss * float(self.cfg.loss_weight),
            "event_decoder_mse": loss.detach(),
            "event_decoder_pred_z": out["pred_z_after"],
            "event_decoder_pred_delta_z": out["pred_delta_z"],
        }

    def decode_event(self, event: Dict[str, Any], device, dtype=torch.float32) -> Dict[str, Any]:
        z0 = event.get("z_before", None)
        event_code = event.get("event_code", None)
        roles = event.get("sentence_roles", event.get("roles", {}))
        if not torch.is_tensor(z0) or not torch.is_tensor(event_code):
            return {}

        z0 = z0.detach().to(device=device, dtype=dtype)
        if z0.ndim == 1:
            z0 = z0.unsqueeze(0)
        pred = self.forward(z0, event_code.to(device=device, dtype=dtype), roles)
        return {
            "neural_event_decoder_active": True,
            "neural_pred_z_after": pred["pred_z_after"],
            "neural_pred_delta_z": pred["pred_delta_z"],
        }
