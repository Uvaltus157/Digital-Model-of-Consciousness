from __future__ import annotations

"""
M2 Event / Dream Replay.

Architecture role:
    M2 is not the generic training replay buffer. It is the event/dream layer.
    It reads compact event-code memory, M13 autobiographical context and current
    affect/metacognitive signals, then produces a replay packet for the
    pre-self branch.

Runtime contract:
    out["event_dream_replay"] = {
        replay_context,
        replay_gate,
        event_salience,
        dream_pressure,
        should_replay,
        selected_event_sentence,
        selected_event_kind,
        replay_source,
    }

M15 can then search chains with replayed event context available, and M13 can
later store the resulting episode.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def pad_or_trim_replay(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
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


def _device_from_out(out: Dict) -> torch.device:
    for value in out.values():
        if torch.is_tensor(value):
            return value.device
        if isinstance(value, dict):
            for nested in value.values():
                if torch.is_tensor(nested):
                    return nested.device
    return torch.device("cpu")


@dataclass
class EventDreamReplayConfig:
    enabled: bool = True
    replay_context_dim: int = 256
    event_code_dim: int = 8
    replay_threshold: float = 0.35
    focus_blend: float = 0.15
    # Legacy path. Keep available, but default runtime should use the
    # M5 FocusFeedbackBoundary seed input instead of direct focus mutation.
    blend_replay_into_focus: bool = False
    use_m13_context: bool = True
    use_m4_context: bool = True
    m4_context_weight: float = 0.20
    use_event_memory: bool = True
    max_recent_events_scan: int = 16
    seed_to_m5_boundary: bool = True
    seed_gate_gain: float = 1.0
    apply_stage: str = "pre_observe"  # both | pre_observe | main
    seed_only_in_sleep: bool = True


class EventDreamReplay:
    def __init__(self, cfg: Optional[EventDreamReplayConfig] = None) -> None:
        self.cfg = cfg or EventDreamReplayConfig()
        self.last_packet: Dict[str, Any] = {}

    def _latest_event(self, event_memory: Any) -> Dict[str, Any]:
        if event_memory is None:
            return {}
        latest = getattr(event_memory, "last_event", None)
        if isinstance(latest, dict):
            return latest
        try:
            events = list(getattr(event_memory, "events", []) or [])
            if events:
                return events[-1]
        except Exception:
            pass
        return {}

    def _select_replay_event(self, event_memory: Any) -> tuple[Dict[str, Any], int]:
        if event_memory is None:
            return {}, -1
        try:
            events = list(getattr(event_memory, "events", []) or [])
        except Exception:
            events = []
        if not events:
            latest = self._latest_event(event_memory)
            return latest, 0 if latest else -1

        recent = events[-max(1, int(self.cfg.max_recent_events_scan)):]
        best_idx = len(events) - len(recent)
        best_score = -1.0
        best = recent[-1]
        for local_idx, ev in enumerate(recent):
            score = (
                0.45 * _scalar(ev.get("delta_norm"), 0.0)
                + 0.35 * _scalar(ev.get("contact_norm"), 0.0)
                + 0.20 * _scalar(ev.get("action_norm"), 0.0)
                + 0.15 * _scalar(ev.get("touch_strength"), 0.0)
                + 0.10 * (1.0 if bool(ev.get("dream_mode", False)) else 0.0)
            )
            if score >= best_score:
                best_score = score
                best = ev
                best_idx = len(events) - len(recent) + local_idx
        return best if isinstance(best, dict) else {}, int(best_idx)

    def _event_vector(self, event: Dict[str, Any], device: torch.device) -> torch.Tensor:
        code = event.get("event_code") if isinstance(event, dict) else None
        if torch.is_tensor(code):
            return pad_or_trim_replay(code, int(self.cfg.replay_context_dim), device=device)
        values = [
            _scalar(event.get("slot"), 0.0),
            _scalar(event.get("confidence"), 0.0),
            _scalar(event.get("delta_norm"), 0.0),
            _scalar(event.get("action_norm"), 0.0),
            _scalar(event.get("contact_norm"), 0.0),
            _scalar(event.get("vision_strength"), 0.0),
            _scalar(event.get("touch_strength"), 0.0),
            1.0 if bool(event.get("dream_mode", False)) else 0.0,
        ]
        x = torch.tensor([values], dtype=torch.float32, device=device)
        return pad_or_trim_replay(x, int(self.cfg.replay_context_dim), device=device)

    def compute(self, *, out: Dict, event_memory: Any = None, dream_mode: bool = False) -> Dict[str, Any]:
        device = _device_from_out(out)
        c = self.cfg
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        meta = out.get("metacognition", {}) if isinstance(out.get("metacognition"), dict) else {}
        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        memory13 = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}
        memory4 = out.get("long_dynamic_memory", {}) if isinstance(out.get("long_dynamic_memory"), dict) else {}

        event, event_idx = self._select_replay_event(event_memory) if bool(c.use_event_memory) else ({}, -1)
        event_vec = self._event_vector(event, device)
        m13_context = pad_or_trim_replay(memory13.get("retrieved_context"), int(c.replay_context_dim), device=device)
        m4_context = pad_or_trim_replay(memory4.get("dynamic_identity_context"), int(c.replay_context_dim), device=device)
        focus_context = pad_or_trim_replay(out.get("focus_context"), int(c.replay_context_dim), device=device)

        panic = _scalar(affect.get("panic_latent"), 0.0)
        stress = _scalar(affect.get("stress_latent"), 0.0)
        curiosity = _scalar(affect.get("curiosity_latent"), 0.0)
        doubt = _scalar(meta.get("doubt"), 0.0)
        no_viable = _scalar(tc.get("no_viable_chain"), 0.0)
        predicted_delta = _scalar(tc.get("predicted_affect_delta"), 0.0)
        memory_relevance = _scalar(memory13.get("retrieval_relevance"), 0.0)
        identity_stability = _scalar(memory4.get("identity_stability"), 0.0)
        identity_novelty = _scalar(memory4.get("identity_novelty"), 0.0)
        dynamic_memory_gate = _scalar(memory4.get("dynamic_memory_gate"), 0.0)
        event_delta = _scalar(event.get("delta_norm"), 0.0)
        event_contact = _scalar(event.get("contact_norm"), 0.0)
        event_action = _scalar(event.get("action_norm"), 0.0)

        event_salience_value = max(
            0.0,
            min(
                1.0,
                0.25 * panic
                + 0.20 * stress
                + 0.18 * curiosity
                + 0.18 * doubt
                + 0.20 * no_viable
                + 0.18 * max(0.0, -predicted_delta)
                + 0.22 * memory_relevance
                + 0.12 * identity_stability
                + 0.10 * identity_novelty
                + 0.10 * dynamic_memory_gate
                + 0.30 * event_delta
                + 0.30 * event_contact
                + 0.15 * event_action,
            ),
        )
        dream_pressure_value = max(0.0, min(1.0, event_salience_value + (0.25 if dream_mode else 0.0)))
        replay_gate_value = 1.0 if dream_pressure_value >= float(c.replay_threshold) else 0.0

        m13_weight = 0.35 if bool(c.use_m13_context) else 0.0
        event_weight = 0.45 if event else 0.0
        m4_weight = float(c.m4_context_weight) if bool(getattr(c, "use_m4_context", True)) and dynamic_memory_gate > 0.0 else 0.0
        focus_weight = max(0.0, 1.0 - m13_weight - event_weight - m4_weight)
        replay_context = (
            focus_weight * focus_context
            + event_weight * event_vec
            + m13_weight * m13_context
            + m4_weight * m4_context
        )

        replay_gate = torch.tensor([[replay_gate_value]], dtype=torch.float32, device=device)
        event_salience = torch.tensor([[event_salience_value]], dtype=torch.float32, device=device)
        dream_pressure = torch.tensor([[dream_pressure_value]], dtype=torch.float32, device=device)
        should_replay = replay_gate.clone()

        source = "none"
        if event:
            source = "m02_event_memory"
        elif bool(c.use_m13_context) and torch.is_tensor(memory13.get("retrieved_context")):
            source = "m13_autobiographical_memory"
        elif torch.is_tensor(out.get("focus_context")):
            source = "current_focus"

        packet: Dict[str, Any] = {
            "replay_context": replay_context,
            "replay_gate": replay_gate,
            "event_salience": event_salience,
            "dream_pressure": dream_pressure,
            "should_replay": should_replay,
            "dream_mode": torch.tensor([[1.0 if dream_mode else 0.0]], dtype=torch.float32, device=device),
            "selected_event_index": torch.tensor([[float(event_idx)]], dtype=torch.float32, device=device),
            "selected_event_sentence": str(event.get("semantic_sentence", event.get("sentence", ""))) if event else "",
            "selected_event_kind": str(event.get("kind", "")) if event else "",
            "selected_event_slot_token": str(event.get("slot_token", "")) if event else "",
            "selected_episode_summary": str(memory13.get("summary", memory13.get("last_summary", ""))),
            "selected_identity_token": str(memory4.get("identity_token", "")),
            "selected_identity_sentence": str(memory4.get("selected_sentence", "")),
            "identity_stability": torch.tensor([[identity_stability]], dtype=torch.float32, device=device),
            "identity_novelty": torch.tensor([[identity_novelty]], dtype=torch.float32, device=device),
            "dynamic_memory_gate": torch.tensor([[dynamic_memory_gate]], dtype=torch.float32, device=device),
            "replay_source": source,
        }
        self.last_packet = packet
        return packet


__all__ = [
    "EventDreamReplay",
    "EventDreamReplayConfig",
    "pad_or_trim_replay",
]
