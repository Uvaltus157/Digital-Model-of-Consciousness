from __future__ import annotations

"""
self_core.py

Rebuilt SelfCore layer.

Purpose:
    self_model answers: "what is my current body/state?"
    SelfCore answers:  "what is mine, caused by me, continuous as me?"

Core idea:
    M9 keeps a latent model of the agent's own body/self-state and binds it to
    the currently focused M5 context. The M5 focus context contains the active
    workspace, object, thought, reflection, attention and planning latents that
    are relevant right now. M9 turns that focused content into self-bound
    meaning: "this is my perception", "this is my action", "this is affecting
    me", "this belongs to my current self-state".

Inputs:
    body_state
    action / embodied_targets
    tactile
    vestibular
    object_latent / inner_object z
    workspace latent
    focus_context / focused M5 latent packet

Outputs:
    self_state
    predicted_self_state
    agency_score
    body_ownership_score
    self_continuity_score
    focus_binding_score
    intent_action_match
    prediction_outcome_match
    self_uncertainty
    self_change
    self_curiosity
    subjective_state
    self_bound_context
    self_prediction_error
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SelfCoreConfig:
    enabled: bool = True
    body_state_dim: int = 73
    action_dim: int = 15
    tactile_dim: int = 42
    vestibular_dim: int = 24
    object_latent_dim: int = 128
    workspace_dim: int = 256
    focus_context_dim: int = 256
    hidden_dim: int = 256
    self_dim: int = 128
    subjective_dim: int = 16
    continuity_decay: float = 0.96
    agency_smoothing: float = 0.88
    ownership_smoothing: float = 0.88


def pad_or_trim_selfcore(x: torch.Tensor, dim: int, *, device=None, dtype=None) -> torch.Tensor:
    if x is None:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(1, dim, device=device, dtype=dtype)
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


def _zeros_like_batch(ref: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.zeros(ref.shape[0], dim, device=ref.device, dtype=ref.dtype)


def _match_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] == 1:
        return x.expand(batch_size, -1)
    return x[:batch_size]


class SelfCore(nn.Module):
    def __init__(self, cfg: Optional[SelfCoreConfig] = None):
        super().__init__()
        self.cfg = cfg or SelfCoreConfig()

        in_dim = (
            self.cfg.body_state_dim
            + self.cfg.action_dim
            + self.cfg.tactile_dim
            + self.cfg.vestibular_dim
            + self.cfg.object_latent_dim
            + self.cfg.workspace_dim
            + self.cfg.focus_context_dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.self_dim),
            nn.LayerNorm(self.cfg.self_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.cfg.self_dim + self.cfg.action_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.self_dim),
            nn.LayerNorm(self.cfg.self_dim),
        )

        self.metrics_head = nn.Sequential(
            nn.Linear(self.cfg.self_dim * 2 + self.cfg.action_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, 6),
        )

        self.focus_binding_head = nn.Sequential(
            nn.Linear(
                self.cfg.self_dim
                + self.cfg.focus_context_dim
                + self.cfg.object_latent_dim
                + self.cfg.workspace_dim,
                self.cfg.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.subjective_head = nn.Sequential(
            nn.Linear(self.cfg.self_dim + 7, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.subjective_dim),
        )

    def initial_state(self, batch_size: int, device=None) -> Dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        return {
            "self_state": torch.zeros(batch_size, self.cfg.self_dim, device=device),
            "predicted_self_state": torch.zeros(batch_size, self.cfg.self_dim, device=device),
            "agency_score": torch.zeros(batch_size, 1, device=device),
            "body_ownership_score": torch.zeros(batch_size, 1, device=device),
            "self_continuity_score": torch.zeros(batch_size, 1, device=device),
        }

    def forward(
        self,
        prev_state: Optional[Dict[str, torch.Tensor]],
        body_state: torch.Tensor,
        action: torch.Tensor,
        tactile: torch.Tensor,
        vestibular: torch.Tensor,
        object_latent: torch.Tensor,
        workspace: torch.Tensor,
        focus_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        cfg = self.cfg

        body_state = pad_or_trim_selfcore(body_state, cfg.body_state_dim, device=device)
        batch_size = body_state.shape[0]
        action = pad_or_trim_selfcore(action, cfg.action_dim, device=device)
        tactile = pad_or_trim_selfcore(tactile, cfg.tactile_dim, device=device)
        vestibular = pad_or_trim_selfcore(vestibular, cfg.vestibular_dim, device=device)
        object_latent = pad_or_trim_selfcore(object_latent, cfg.object_latent_dim, device=device)
        workspace = pad_or_trim_selfcore(workspace, cfg.workspace_dim, device=device)
        if focus_context is None:
            focus_context = _zeros_like_batch(body_state, cfg.focus_context_dim)
        else:
            focus_context = pad_or_trim_selfcore(focus_context, cfg.focus_context_dim, device=device)

        action = _match_batch(action, batch_size)
        tactile = _match_batch(tactile, batch_size)
        vestibular = _match_batch(vestibular, batch_size)
        object_latent = _match_batch(object_latent, batch_size)
        workspace = _match_batch(workspace, batch_size)
        focus_context = _match_batch(focus_context, batch_size)

        prev_state = prev_state or self.initial_state(batch_size, device)

        x = torch.cat([body_state, action, tactile, vestibular, object_latent, workspace, focus_context], dim=-1)
        raw_self = self.encoder(x)

        prev_self = prev_state.get("self_state", torch.zeros_like(raw_self)).to(raw_self.device)
        prev_pred = prev_state.get("predicted_self_state", torch.zeros_like(raw_self)).to(raw_self.device)
        prev_agency = prev_state.get("agency_score", torch.zeros(raw_self.shape[0], 1, device=raw_self.device)).to(raw_self.device)
        prev_ownership = prev_state.get("body_ownership_score", torch.zeros(raw_self.shape[0], 1, device=raw_self.device)).to(raw_self.device)
        prev_continuity = prev_state.get("self_continuity_score", torch.zeros(raw_self.shape[0], 1, device=raw_self.device)).to(raw_self.device)

        prev_self = _match_batch(prev_self, batch_size)
        prev_pred = _match_batch(prev_pred, batch_size)
        prev_agency = _match_batch(prev_agency, batch_size)
        prev_ownership = _match_batch(prev_ownership, batch_size)
        prev_continuity = _match_batch(prev_continuity, batch_size)

        self_state = cfg.continuity_decay * prev_self + (1.0 - cfg.continuity_decay) * raw_self
        predicted_self = self.predictor(torch.cat([self_state, action], dim=-1))

        pred_error = torch.mean((raw_self - prev_pred.detach()) ** 2, dim=-1, keepdim=True)
        continuity_raw = torch.exp(-pred_error).clamp(0.0, 1.0)

        action_mag = torch.tanh(torch.norm(action, dim=-1, keepdim=True))
        tactile_mag = torch.tanh(torch.norm(tactile, dim=-1, keepdim=True))
        vestibular_mag = torch.tanh(torch.norm(vestibular, dim=-1, keepdim=True))
        sensor_response = torch.clamp(0.55 * tactile_mag + 0.45 * vestibular_mag, 0.0, 1.0)
        ownership_heuristic = torch.clamp(0.30 + 0.70 * sensor_response, 0.0, 1.0)

        metrics = torch.sigmoid(self.metrics_head(torch.cat([self_state, prev_self, action], dim=-1)))
        intent_action_match = metrics[:, 0:1]
        prediction_outcome_match = metrics[:, 1:2]
        learned_ownership = metrics[:, 2:3]
        self_uncertainty = metrics[:, 3:4]
        self_change = metrics[:, 4:5]
        self_curiosity = metrics[:, 5:6]

        agency_raw = torch.clamp(
            0.42 * intent_action_match + 0.35 * continuity_raw + 0.23 * action_mag,
            0.0,
            1.0,
        )
        ownership_raw = torch.clamp(
            0.55 * learned_ownership + 0.45 * ownership_heuristic,
            0.0,
            1.0,
        )

        agency = cfg.agency_smoothing * prev_agency + (1.0 - cfg.agency_smoothing) * agency_raw
        ownership = cfg.ownership_smoothing * prev_ownership + (1.0 - cfg.ownership_smoothing) * ownership_raw
        continuity = cfg.continuity_decay * prev_continuity + (1.0 - cfg.continuity_decay) * continuity_raw

        focus_binding = self.focus_binding_head(torch.cat([self_state, focus_context, object_latent, workspace], dim=-1))
        self_bound_context = torch.cat([self_state, focus_context], dim=-1)

        subjective_state = self.subjective_head(torch.cat([
            self_state,
            agency,
            ownership,
            continuity,
            self_uncertainty,
            self_change,
            self_curiosity,
            focus_binding,
        ], dim=-1))

        return {
            "self_state": self_state,
            "raw_self_state": raw_self,
            "predicted_self_state": predicted_self,
            "agency_score": agency,
            "body_ownership_score": ownership,
            "self_continuity_score": continuity,
            "focus_binding_score": focus_binding,
            "intent_action_match": intent_action_match,
            "prediction_outcome_match": prediction_outcome_match,
            "self_uncertainty": self_uncertainty,
            "self_change": self_change,
            "self_curiosity": self_curiosity,
            "subjective_state": subjective_state,
            "self_bound_context": self_bound_context,
            "focus_context": focus_context,
            "self_prediction_error": pred_error,
        }


def build_self_experience_text(out: Dict[str, torch.Tensor]) -> str:
    def val(name: str) -> float:
        x = out.get(name)
        if x is None:
            return 0.0
        try:
            if torch.is_tensor(x):
                return float(x.detach().cpu().reshape(-1)[0].item())
            return float(x)
        except Exception:
            return 0.0

    agency = val("agency_score")
    ownership = val("body_ownership_score")
    continuity = val("self_continuity_score")
    uncertainty = val("self_uncertainty")
    curiosity = val("self_curiosity")
    change = val("self_change")
    focus_binding = val("focus_binding_score")

    parts = []
    if agency > 0.65:
        parts.append("Action feels self-caused.")
    elif agency > 0.35:
        parts.append("Action is partially linked to self.")
    else:
        parts.append("Agency is weak.")

    if ownership > 0.65:
        parts.append("Body signals feel owned.")
    elif ownership > 0.35:
        parts.append("Body ownership is partial.")
    else:
        parts.append("Body ownership is weak.")

    if continuity > 0.65:
        parts.append("Self-state is continuous.")
    elif continuity > 0.35:
        parts.append("Self-continuity is partial.")
    else:
        parts.append("Self-state is fragmented.")

    if focus_binding > 0.65:
        parts.append("Focused content feels self-bound.")
    elif focus_binding > 0.35:
        parts.append("Focused content is partially self-bound.")
    else:
        parts.append("Focused content is weakly bound to self.")

    if uncertainty > 0.60:
        parts.append("Uncertainty is high.")
    if curiosity > 0.55:
        parts.append("Exploration drive is active.")
    if change > 0.60:
        parts.append("Self-state is changing.")

    return " ".join(parts)


__all__ = [
    "SelfCore",
    "SelfCoreConfig",
    "build_self_experience_text",
    "pad_or_trim_selfcore",
]
