from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InnerActionDecoderConfig:
    """
    Step 2 after InnerScenarioMind.

    InnerScenarioMind selects a scenario_z candidate:
        "what could happen internally?"

    InnerActionDecoder turns the selected scenario_z into an intention/action hint:
        "what should the body/hand/gaze try?"

    This first version is safe:
        - it does not directly override the main policy;
        - it outputs intent vectors and confidence;
        - runtime may later choose to blend them into actual action.
    """
    enabled: bool = True
    latent_dim: int = 128
    embodied_dim: int = 15
    hand_dim: int = 34
    hidden_dim: int = 256
    max_intent_norm: float = 0.25
    confidence_threshold: float = 0.10
    blend_to_policy: bool = False
    blend_alpha: float = 0.10


_VERB_IDS = {
    "changes": 0,
    "latent_changes": 1,
    "touch_changes": 2,
    "self_moves_changes": 3,
    "dream_replays": 4,
}

_CONTEXT_IDS = {
    "latent": 0,
    "contact": 1,
    "self_action": 2,
    "dream": 3,
    "unknown": 4,
}


def _role_id(roles: Dict[str, Any], key: str, table: Dict[str, int], default: str) -> int:
    try:
        val = str(roles.get(key, default) or default)
        return int(table.get(val, table.get(default, 0)))
    except Exception:
        return int(table.get(default, 0))


class InnerActionDecoder(nn.Module):
    """
    Neural intent decoder:
        inner_mind_z + role embeddings -> action intention.

    It is not the final motor controller. It is the first bridge from
    coded-world thought to action proposal.
    """

    def __init__(self, cfg: Optional[InnerActionDecoderConfig] = None):
        super().__init__()
        self.cfg = cfg or InnerActionDecoderConfig()

        latent_dim = int(self.cfg.latent_dim)
        hidden = int(self.cfg.hidden_dim)
        role_dim = 16

        self.role_embed = nn.Embedding(32, role_dim)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + role_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.body_head = nn.Linear(hidden, int(self.cfg.embodied_dim))
        self.hand_head = nn.Linear(hidden, int(self.cfg.hand_dim))
        self.conf_head = nn.Linear(hidden, 1)

    def _role_features(self, roles: Dict[str, Any], batch: int, device) -> torch.Tensor:
        if not isinstance(roles, dict):
            roles = {}
        verb_id = _role_id(roles, "verb", _VERB_IDS, "changes")
        ctx_id = 16 + _role_id(roles, "context", _CONTEXT_IDS, "unknown")
        ids = torch.tensor([verb_id, ctx_id], device=device, dtype=torch.long)
        emb = self.role_embed(ids).reshape(1, -1)
        return emb.repeat(batch, 1)

    def forward(self, z: torch.Tensor, roles: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.float()
        role = self._role_features(roles or {}, z.shape[0], z.device)

        x = torch.cat([z, role], dim=-1)
        h = self.trunk(x)

        max_norm = float(self.cfg.max_intent_norm)
        body = torch.tanh(self.body_head(h)) * max_norm
        hand = torch.tanh(self.hand_head(h)) * max_norm
        conf = torch.sigmoid(self.conf_head(h))

        return {
            "intent_body": body,
            "intent_hand": hand,
            "intent_confidence": conf,
        }

    def decode_intention(self, thought: Dict[str, Any], *, device, dtype=torch.float32) -> Dict[str, Any]:
        if not isinstance(thought, dict):
            return {}
        z = thought.get("inner_mind_z")
        if not torch.is_tensor(z):
            return {}

        z = z.detach().to(device=device, dtype=dtype)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        # InnerScenarioMind currently exports sentence and slot, but not full roles.
        # If roles are later exported, this will use them.
        roles = thought.get("inner_mind_selected_roles", {})
        if not isinstance(roles, dict):
            roles = {}

        out = self.forward(z, roles)
        conf = out["intent_confidence"]

        return {
            "inner_action_active": True,
            "inner_action_body": out["intent_body"],
            "inner_action_hand": out["intent_hand"],
            "inner_action_confidence": conf,
            "inner_action_source_sentence": str(thought.get("inner_mind_selected_sentence", "")),
            "inner_action_slot_token": str(thought.get("inner_mind_selected_slot_token", "")),
        }
