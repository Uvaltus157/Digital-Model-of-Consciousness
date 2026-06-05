from __future__ import annotations

"""
M4 Long Dynamic Memory.

Architecture role:
    M4 is the long-term dynamic identity layer.

    M1 produces inner object latents.
    M2 stores event/dream traces.
    M4 binds a stable dynamic identity over time: OBJ token, slot binding,
    stability, replay latent and event history summary.

Runtime contract:
    out["long_dynamic_memory"] = {
        dynamic_identity_context,
        dynamic_memory_gate,
        identity_token,
        identity_stability,
        identity_novelty,
        passport_count,
        passport_slot,
        selected_sentence,
        episode_summary,
        replay_z,
    }

This controller is intentionally lightweight. It uses DynamicObjectPassportManager
as the persistent identity store and exports a compact tensor context for
M13/M2/M15/M9.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def pad_or_trim_dynamic(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
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
    x = x.to(
        device=device if device is not None else x.device,
        dtype=dtype or torch.float32,
    )
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().float().reshape(-1)[0].cpu().item())
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _device_from(out: Dict, obj: Dict) -> torch.device:
    for container in (out, obj):
        if isinstance(container, dict):
            for value in container.values():
                if torch.is_tensor(value):
                    return value.device
                if isinstance(value, dict):
                    for nested in value.values():
                        if torch.is_tensor(nested):
                            return nested.device
    return torch.device("cpu")


@dataclass
class LongDynamicMemoryConfig:
    enabled: bool = True
    context_dim: int = 256
    focus_blend: float = 0.18
    blend_into_focus: bool = False
    stability_threshold: float = 0.12
    novelty_threshold: float = 0.35
    use_passport_manager: bool = True
    use_event_memory: bool = True


class LongDynamicMemory:
    def __init__(self, cfg: Optional[LongDynamicMemoryConfig] = None) -> None:
        self.cfg = cfg or LongDynamicMemoryConfig()
        self.last_packet: Dict[str, Any] = {}

    def _passport_info(self, *, obj: Dict, passport_manager: Any, event_memory: Any, dream_mode: bool, global_step: int) -> Dict[str, Any]:
        if passport_manager is None or not bool(self.cfg.use_passport_manager):
            return {}
        try:
            # Observe is idempotent enough for runtime use: existing slot/token wins
            # and EMA fields are updated slowly.
            return passport_manager.observe(
                obj,
                event_memory=event_memory if bool(self.cfg.use_event_memory) else None,
                dream_mode=bool(dream_mode),
                global_step=int(global_step),
            )
        except Exception:
            return {}

    def _selected_passport(self, passport_manager: Any, token: str = "") -> Dict[str, Any]:
        if passport_manager is None:
            return {}
        try:
            p = passport_manager.select_for_replay(token=token)
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

    def _make_context(self, *, obj: Dict, info: Dict, passport: Dict, out: Dict, device: torch.device) -> torch.Tensor:
        c = self.cfg
        z_obj = pad_or_trim_dynamic(obj.get("z_obj"), int(c.context_dim), device=device)
        replay_z = pad_or_trim_dynamic(passport.get("replay_z"), int(c.context_dim), device=device)
        focus = pad_or_trim_dynamic(out.get("focus_context"), int(c.context_dim), device=device)
        event_replay = out.get("event_dream_replay", {}) if isinstance(out.get("event_dream_replay"), dict) else {}
        event_ctx = pad_or_trim_dynamic(event_replay.get("replay_context"), int(c.context_dim), device=device)

        stability = _scalar(info.get("passport_confidence_ema"), _scalar(passport.get("confidence_ema"), 0.0))
        similarity = _scalar(info.get("passport_similarity"), _scalar(passport.get("last_similarity"), 0.0))
        dyn = _scalar(info.get("passport_dynamic_score"), _scalar(passport.get("dynamic_score_ema"), 0.0))
        gate = max(0.0, min(1.0, 0.45 * stability + 0.35 * max(0.0, similarity) + 0.20 * dyn))

        scalar_tail = torch.tensor(
            [[
                float(gate),
                float(stability),
                float(max(0.0, similarity)),
                float(dyn),
                float(_scalar(info.get("passport_count"), len(getattr(passport, "keys", lambda: [])()) if isinstance(passport, dict) else 0.0)),
                float(_scalar(info.get("passport_slot"), _scalar(passport.get("lives_in_slot"), 0.0))),
                float(1.0 if bool(info.get("passport_created", False)) else 0.0),
                float(_scalar(event_replay.get("event_salience"), 0.0)),
            ]],
            device=device,
            dtype=torch.float32,
        )
        scalar_tail = pad_or_trim_dynamic(scalar_tail, int(c.context_dim), device=device)
        context = 0.40 * z_obj + 0.30 * replay_z + 0.20 * focus + 0.10 * event_ctx + 0.10 * scalar_tail
        return context

    def compute(
        self,
        *,
        out: Dict,
        obj: Dict,
        passport_manager: Any = None,
        event_memory: Any = None,
        dream_mode: bool = False,
        global_step: int = 0,
    ) -> Dict[str, Any]:
        device = _device_from(out, obj)
        c = self.cfg
        if not isinstance(obj, dict):
            obj = {}

        info = self._passport_info(
            obj=obj,
            passport_manager=passport_manager,
            event_memory=event_memory,
            dream_mode=bool(dream_mode),
            global_step=int(global_step),
        )
        token = str(info.get("passport_token", "") or obj.get("passport_token", "") or obj.get("slot_token", "") or "")
        passport = self._selected_passport(passport_manager, token=token)
        if not token and passport:
            token = str(passport.get("token", ""))

        context = self._make_context(obj=obj, info=info, passport=passport, out=out, device=device)
        stability = _scalar(info.get("passport_confidence_ema"), _scalar(passport.get("confidence_ema"), 0.0))
        similarity = _scalar(info.get("passport_similarity"), _scalar(passport.get("last_similarity"), 0.0))
        dyn = _scalar(info.get("passport_dynamic_score"), _scalar(passport.get("dynamic_score_ema"), 0.0))
        novelty = max(0.0, min(1.0, 1.0 - max(0.0, similarity)))
        gate_value = max(0.0, min(1.0, 0.45 * stability + 0.35 * max(0.0, similarity) + 0.20 * dyn))
        gate = torch.tensor([[gate_value]], device=device, dtype=torch.float32)

        replay_z = passport.get("replay_z") if isinstance(passport, dict) else None
        if torch.is_tensor(replay_z):
            replay_z = replay_z.detach().to(device=device, dtype=torch.float32)
            if replay_z.ndim == 1:
                replay_z = replay_z.unsqueeze(0)
        else:
            replay_z = pad_or_trim_dynamic(obj.get("z_obj"), int(c.context_dim), device=device)

        sentence = str(info.get("passport_sentence", "") or passport.get("last_sentence", "") or "")
        episode = str(info.get("passport_episode_summary", "") or passport.get("episode_summary", "") or "")
        source = str(info.get("passport_source", "") or "passport")

        packet: Dict[str, Any] = {
            "dynamic_identity_context": context,
            "dynamic_memory_gate": gate,
            "should_bind_identity": (gate >= float(c.stability_threshold)).float(),
            "identity_token": token,
            "identity_stability": torch.tensor([[float(stability)]], device=device, dtype=torch.float32),
            "identity_similarity": torch.tensor([[float(similarity)]], device=device, dtype=torch.float32),
            "identity_novelty": torch.tensor([[float(novelty)]], device=device, dtype=torch.float32),
            "identity_dynamic_score": torch.tensor([[float(dyn)]], device=device, dtype=torch.float32),
            "passport_count": torch.tensor([[float(info.get("passport_count", len(getattr(passport_manager, "passports", {}) or {})))]], device=device, dtype=torch.float32),
            "passport_slot": torch.tensor([[float(_scalar(info.get("passport_slot"), _scalar(passport.get("lives_in_slot"), 0.0)))]], device=device, dtype=torch.float32),
            "passport_created": torch.tensor([[1.0 if bool(info.get("passport_created", False)) else 0.0]], device=device, dtype=torch.float32),
            "selected_sentence": sentence,
            "episode_summary": episode,
            "identity_source": source,
            "replay_z": replay_z,
        }
        self.last_packet = packet
        return packet


__all__ = [
    "LongDynamicMemory",
    "LongDynamicMemoryConfig",
    "pad_or_trim_dynamic",
]
