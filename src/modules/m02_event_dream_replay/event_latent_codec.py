from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.modules.m14_semantic_grounding.slot_vocabulary import SlotVocabulary, SlotVocabularyConfig
from src.modules.m07_inner_speech_thoughts.event_sentence_composer import EventSentenceComposer, EventSentenceComposerConfig
from src.modules.m07_inner_speech_thoughts.event_sentence_memory import EventSentenceMemory, EventSentenceMemoryConfig
from src.modules.m02_event_dream_replay.event_scenario_decoder import EventScenarioDecoder, EventScenarioDecoderConfig


@dataclass
class EventLatentCodecConfig:
    """
    Runtime event-code memory.

    This is not language yet. It is the first "DNA-like" code layer:
        object slot latent + delta_z + action/contact/context -> code sentence.

    The sentence is symbolic/debuggable, while event_code is numeric and can later
    become trainable input for an EventDecoder / dream replay module.
    """
    enabled: bool = True
    latent_dim: int = 128
    max_events: int = 512
    delta_threshold: float = 0.015
    action_threshold: float = 0.010
    contact_threshold: float = 0.010
    record_in_sleep: bool = True
    keep_z_snapshots: bool = True
    use_slot_vocabulary: bool = True
    slot_token_prefix: str = "OBJ"
    compose_semantic_sentences: bool = True
    sentence_language: str = "code"  # code/en/ru
    use_sentence_memory: bool = True
    max_sentences: int = 512
    max_episodes: int = 64
    episode_gap_steps: int = 25
    new_episode_on_slot_change: bool = False
    use_scenario_decoder: bool = True
    scenario_max_replay_steps: int = 32
    scenario_interpolate_steps: int = 3
    scenario_loop: bool = True


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _tensor_norm(x: Any) -> float:
    try:
        if x is None or not torch.is_tensor(x):
            return 0.0
        return float(x.detach().float().norm(dim=-1).mean().cpu().item())
    except Exception:
        try:
            return float(x.detach().float().abs().mean().cpu().item())
        except Exception:
            return 0.0


def _mean_abs(x: Any) -> float:
    try:
        if x is None or not torch.is_tensor(x):
            return 0.0
        return float(x.detach().float().abs().mean().cpu().item())
    except Exception:
        return 0.0


class EventLatentSentenceMemory:
    """
    Stores compact latent event "sentences".

    One entry describes a transition:
        z_before(slot) -> z_after(slot)
        plus action/contact/vision/touch context.

    This is intentionally runtime-only for step 1:
        - no extra optimizer group;
        - no extra loss;
        - saved/restored through checkpointing.py.
    """

    def __init__(self, cfg: Optional[EventLatentCodecConfig] = None):
        self.cfg = cfg or EventLatentCodecConfig()
        self.events = deque(maxlen=max(1, int(self.cfg.max_events)))
        self.last_event: Optional[Dict[str, Any]] = None
        self.slot_vocabulary = SlotVocabulary(SlotVocabularyConfig(
            enabled=bool(getattr(self.cfg, "use_slot_vocabulary", True)),
            max_slots=32,
            latent_dim=int(getattr(self.cfg, "latent_dim", 128)),
            token_prefix=str(getattr(self.cfg, "slot_token_prefix", "OBJ")),
        ))
        self.sentence_composer = EventSentenceComposer(EventSentenceComposerConfig(
            enabled=bool(getattr(self.cfg, "compose_semantic_sentences", True)),
            language=str(getattr(self.cfg, "sentence_language", "code")),
        ))
        self.sentence_memory = EventSentenceMemory(EventSentenceMemoryConfig(
            enabled=bool(getattr(self.cfg, "use_sentence_memory", True)),
            max_sentences=int(getattr(self.cfg, "max_sentences", getattr(self.cfg, "max_events", 512))),
            max_episodes=int(getattr(self.cfg, "max_episodes", 64)),
            episode_gap_steps=int(getattr(self.cfg, "episode_gap_steps", 25)),
            new_episode_on_slot_change=bool(getattr(self.cfg, "new_episode_on_slot_change", False)),
        ))
        self.scenario_decoder = EventScenarioDecoder(EventScenarioDecoderConfig(
            enabled=bool(getattr(self.cfg, "use_scenario_decoder", True)),
            max_replay_steps=int(getattr(self.cfg, "scenario_max_replay_steps", 32)),
            interpolate_steps=int(getattr(self.cfg, "scenario_interpolate_steps", 3)),
            loop=bool(getattr(self.cfg, "scenario_loop", True)),
        ))

    def __len__(self) -> int:
        return len(self.events)

    def _slot_index(self, obj: Dict[str, Any]) -> int:
        idx = obj.get("active_slot_index", None)
        try:
            if torch.is_tensor(idx):
                return int(idx.detach().reshape(-1)[0].cpu().item())
            if idx is not None:
                return int(idx)
        except Exception:
            pass
        return 0

    def _slot_z_from_state(self, state: Dict[str, Any], slot_idx: int, fallback: torch.Tensor) -> torch.Tensor:
        z_slots = state.get("z_obj_slots") if isinstance(state, dict) else None
        try:
            if torch.is_tensor(z_slots):
                s = int(z_slots.shape[1])
                slot_idx = max(0, min(int(slot_idx), s - 1))
                return z_slots[:, slot_idx, :].detach()
        except Exception:
            pass

        z = state.get("z_obj") if isinstance(state, dict) else None
        if torch.is_tensor(z):
            return z.detach()
        return fallback.detach()

    def _make_code_sentence(
        self,
        *,
        step: int,
        slot_idx: int,
        slot_token: str,
        delta_norm: float,
        confidence: float,
        action_norm: float,
        contact_norm: float,
        vision_strength: float,
        touch_strength: float,
        dream_mode: bool,
        kind: str,
    ) -> str:
        def bin3(v: float, lo: float, hi: float) -> str:
            if v < lo:
                return "LOW"
            if v < hi:
                return "MID"
            return "HIGH"

        parts = [
            "EVT",
            f"t={int(step)}",
            f"SLOT_{int(slot_idx)}",
            str(slot_token),
            f"KIND={kind}",
            f"DZ_{bin3(delta_norm, self.cfg.delta_threshold, self.cfg.delta_threshold * 4.0)}",
            f"ACT_{bin3(action_norm, self.cfg.action_threshold, self.cfg.action_threshold * 5.0)}",
            f"TOUCH_{bin3(contact_norm, self.cfg.contact_threshold, self.cfg.contact_threshold * 5.0)}",
            f"VISION_{bin3(vision_strength, 0.20, 0.55)}",
            f"CONF_{bin3(confidence, 0.10, 0.45)}",
        ]
        if dream_mode:
            parts.append("DREAM")
        if touch_strength > 0.35 or contact_norm > self.cfg.contact_threshold:
            parts.append("CONTACT_BOUND")
        if action_norm > self.cfg.action_threshold:
            parts.append("SELF_CAUSED")
        return " ".join(parts)

    def encode_step(
        self,
        *,
        prev_state: Dict[str, Any],
        obj: Dict[str, Any],
        obs: Dict[str, Any],
        out: Dict[str, Any],
        dream_mode: bool,
        global_step: int,
    ) -> Dict[str, Any]:
        z_after = obj.get("z_obj")
        if not torch.is_tensor(z_after):
            return {}

        slot_idx = self._slot_index(obj)
        z_before = self._slot_z_from_state(prev_state, slot_idx, z_after)
        if z_before.shape[-1] != z_after.shape[-1]:
            z_before = torch.zeros_like(z_after)

        delta = (z_after.detach() - z_before.detach())
        delta_norm = float(delta.norm(dim=-1).mean().detach().cpu().item())

        confidence = _as_float(obj.get("confidence"), 0.0)
        action_norm = _tensor_norm(out.get("embodied_targets")) + 0.25 * _tensor_norm(out.get("hand_ctrl")) + 0.25 * _tensor_norm(out.get("leg_ctrl"))
        contact_norm = _mean_abs(obs.get("tactile"))
        vision_strength = _as_float(obj.get("vision_strength"), 0.0)
        touch_strength = _as_float(obj.get("touch_strength"), 0.0)

        if dream_mode and not bool(self.cfg.record_in_sleep):
            active = False
        else:
            active = (
                delta_norm >= float(self.cfg.delta_threshold)
                or action_norm >= float(self.cfg.action_threshold)
                or contact_norm >= float(self.cfg.contact_threshold)
                or touch_strength > 0.35
                or bool(dream_mode)
            )

        if not active:
            return {
                "event_active": False,
                "event_delta_norm": delta_norm,
                "event_memory_size": len(self.events),
            }

        if dream_mode:
            kind = "dream_replay"
        elif contact_norm >= float(self.cfg.contact_threshold) or touch_strength > 0.35:
            kind = "contact_transition"
        elif action_norm >= float(self.cfg.action_threshold):
            kind = "self_motion_transition"
        else:
            kind = "latent_transition"

        slot_token = f"SLOT_{int(slot_idx)}"
        slot_vocab_entry = {}
        if bool(getattr(self.cfg, "use_slot_vocabulary", True)):
            slot_vocab_entry = self.slot_vocabulary.update_from_observation(
                slot_id=int(slot_idx),
                z_obj=z_after,
                obj=obj,
                confidence=float(confidence),
                vision_strength=float(vision_strength),
                touch_strength=float(touch_strength),
                event_kind=str(kind),
                event_sentence="",
                dream_mode=bool(dream_mode),
            )
            slot_token = str(slot_vocab_entry.get("token", slot_token))

        sentence = self._make_code_sentence(
            step=int(global_step),
            slot_idx=slot_idx,
            slot_token=slot_token,
            delta_norm=delta_norm,
            confidence=confidence,
            action_norm=action_norm,
            contact_norm=contact_norm,
            vision_strength=vision_strength,
            touch_strength=touch_strength,
            dream_mode=bool(dream_mode),
            kind=kind,
        )

        if slot_vocab_entry:
            try:
                slot_vocab_entry["last_event_sentence"] = sentence
                self.slot_vocabulary.entries[int(slot_idx)] = slot_vocab_entry
            except Exception:
                pass

        # Numeric first-stage event code: small, stable vector for visualizers
        # and future neural EventDecoder.
        event_code = torch.tensor(
            [[
                float(slot_idx),
                float(confidence),
                float(delta_norm),
                float(action_norm),
                float(contact_norm),
                float(vision_strength),
                float(touch_strength),
                1.0 if dream_mode else 0.0,
            ]],
            device=z_after.device,
            dtype=z_after.dtype,
        )

        entry: Dict[str, Any] = {
            "step": int(global_step),
            "slot": int(slot_idx),
            "kind": kind,
            "sentence": sentence,
            "slot_token": slot_token,
            "slot_vocabulary_entry": slot_vocab_entry,
            "slot_description": self.slot_vocabulary.describe(slot_idx) if bool(getattr(self.cfg, "use_slot_vocabulary", True)) else "",
            "confidence": float(confidence),
            "delta_norm": float(delta_norm),
            "action_norm": float(action_norm),
            "contact_norm": float(contact_norm),
            "vision_strength": float(vision_strength),
            "touch_strength": float(touch_strength),
            "dream_mode": bool(dream_mode),
            "event_code": event_code.detach().cpu(),
        }

        if bool(self.cfg.keep_z_snapshots):
            entry["z_before"] = z_before.detach().cpu()
            entry["z_after"] = z_after.detach().cpu()
            entry["delta_z"] = delta.detach().cpu()

        # Compose the next grammar layer from tokens.
        # Token sentence: EVT ... SLOT_1 OBJ_001 KIND=...
        # Semantic sentence: SENT SUBJ=OBJ_001 VERB=touch_changes ...
        try:
            composed = self.sentence_composer.compose(entry)
            if composed:
                entry.update(composed)
        except Exception:
            composed = {}

        # Level 3: store composed sentences as ordered narrative/episode memory.
        try:
            if bool(getattr(self.cfg, "use_sentence_memory", True)):
                sent_info = self.sentence_memory.add(entry)
                if sent_info:
                    entry.update(sent_info)
        except Exception:
            sent_info = {}

        self.events.append(entry)
        self.last_event = entry

        return {
            "event_active": True,
            "event_code": event_code,
            "event_delta_norm": torch.tensor([[delta_norm]], device=z_after.device, dtype=z_after.dtype),
            "event_memory_size": torch.tensor([[float(len(self.events))]], device=z_after.device, dtype=z_after.dtype),
            "event_slot_index": torch.tensor([[float(slot_idx)]], device=z_after.device, dtype=z_after.dtype),
            "event_code_sentence": sentence,
            "event_kind": kind,
            "slot_token": slot_token,
            "slot_description": self.slot_vocabulary.describe(slot_idx) if bool(getattr(self.cfg, "use_slot_vocabulary", True)) else "",
            "slot_vocabulary_size": torch.tensor([[float(len(self.slot_vocabulary))]], device=z_after.device, dtype=z_after.dtype),
            "semantic_sentence": entry.get("semantic_sentence", ""),
            "semantic_code_sentence": entry.get("semantic_code_sentence", ""),
            "sentence_roles": entry.get("sentence_roles", {}),
            "sentence_language": entry.get("sentence_language", str(getattr(self.cfg, "sentence_language", "code"))),
            "sentence_memory_size": torch.tensor([[float(entry.get("sentence_memory_size", len(self.sentence_memory)))]], device=z_after.device, dtype=z_after.dtype),
            "episode_id": torch.tensor([[float(entry.get("episode_id", 0))]], device=z_after.device, dtype=z_after.dtype),
            "episode_size": torch.tensor([[float(entry.get("episode_size", 0))]], device=z_after.device, dtype=z_after.dtype),
            "episode_summary": entry.get("episode_summary", ""),
        }

    def latest_sentence(self) -> str:
        if not self.last_event:
            return ""
        return str(self.last_event.get("sentence", ""))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": {
                "enabled": bool(self.cfg.enabled),
                "latent_dim": int(self.cfg.latent_dim),
                "max_events": int(self.cfg.max_events),
                "delta_threshold": float(self.cfg.delta_threshold),
                "action_threshold": float(self.cfg.action_threshold),
                "contact_threshold": float(self.cfg.contact_threshold),
                "record_in_sleep": bool(self.cfg.record_in_sleep),
                "keep_z_snapshots": bool(self.cfg.keep_z_snapshots),
                "use_slot_vocabulary": bool(getattr(self.cfg, "use_slot_vocabulary", True)),
                "slot_token_prefix": str(getattr(self.cfg, "slot_token_prefix", "OBJ")),
                "compose_semantic_sentences": bool(getattr(self.cfg, "compose_semantic_sentences", True)),
                "sentence_language": str(getattr(self.cfg, "sentence_language", "code")),
                "use_sentence_memory": bool(getattr(self.cfg, "use_sentence_memory", True)),
                "max_sentences": int(getattr(self.cfg, "max_sentences", 512)),
                "max_episodes": int(getattr(self.cfg, "max_episodes", 64)),
                "episode_gap_steps": int(getattr(self.cfg, "episode_gap_steps", 25)),
                "new_episode_on_slot_change": bool(getattr(self.cfg, "new_episode_on_slot_change", False)),
                "use_scenario_decoder": bool(getattr(self.cfg, "use_scenario_decoder", True)),
                "scenario_max_replay_steps": int(getattr(self.cfg, "scenario_max_replay_steps", 32)),
                "scenario_interpolate_steps": int(getattr(self.cfg, "scenario_interpolate_steps", 3)),
                "scenario_loop": bool(getattr(self.cfg, "scenario_loop", True)),
            },
            "events": list(self.events),
            "last_event": self.last_event,
            "slot_vocabulary": self.slot_vocabulary.state_dict(),
            "sentence_memory": self.sentence_memory.state_dict(),
            "scenario_decoder": self.scenario_decoder.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        cfg = state.get("cfg", {})
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if hasattr(self.cfg, k):
                    try:
                        setattr(self.cfg, k, v)
                    except Exception:
                        pass
        max_events = max(1, int(getattr(self.cfg, "max_events", 512)))
        self.events = deque(state.get("events", []), maxlen=max_events)
        self.last_event = state.get("last_event", None)
        if "slot_vocabulary" in state:
            try:
                self.slot_vocabulary.load_state_dict(state["slot_vocabulary"])
            except Exception:
                pass
        if "sentence_memory" in state:
            try:
                self.sentence_memory.load_state_dict(state["sentence_memory"])
            except Exception:
                pass
        if "scenario_decoder" in state:
            try:
                self.scenario_decoder.load_state_dict(state["scenario_decoder"])
            except Exception:
                pass
