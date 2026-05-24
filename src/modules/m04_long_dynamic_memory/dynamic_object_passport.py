from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DynamicObjectPassportConfig:
    """
    DynamicObjectPassport = semantic identity over time.

    It is NOT a second set of slots.

    SLOT_N:
        address where latent image is stored in ObjectSlotMemory.

    OBJ_NNN / DynamicObjectPassport:
        semantic identity bound to SLOT_N.
        It stores why this is a stable dynamic object:
            latent signature
            source profile
            event history summary
            confidence over time
            replay z for inner-world reproduction.
    """
    enabled: bool = True
    max_passports: int = 32
    latent_dim: int = 128
    token_prefix: str = "OBJ"

    similarity_threshold: float = 0.72
    confidence_ema_decay: float = 0.94
    signature_ema_decay: float = 0.97

    min_dynamic_score: float = 0.010
    min_confidence_to_create: float = 0.02
    create_scene_passport: bool = True

    replay_enabled: bool = True
    replay_in_sleep: bool = True
    decode_to_second_order: bool = True


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _as_z(x: Any, latent_dim: int) -> Optional[torch.Tensor]:
    try:
        if not torch.is_tensor(x):
            return None
        z = x.detach().float()
        if z.ndim == 0:
            return None
        if z.ndim > 1:
            z = z.reshape(-1, z.shape[-1])[0]
        if z.shape[-1] < latent_dim:
            z = F.pad(z, (0, latent_dim - z.shape[-1]))
        elif z.shape[-1] > latent_dim:
            z = z[:latent_dim]
        return z.cpu()
    except Exception:
        return None


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        a = a.detach().float().reshape(1, -1)
        b = b.detach().float().reshape(1, -1)
        return float(F.cosine_similarity(a, b, dim=-1).cpu().item())
    except Exception:
        return 0.0


class DynamicObjectPassportManager:
    """
    Owns semantic object identities and binds them to ObjectSlotMemory slots.

    Main rules:
        - a slot is a memory address;
        - a passport is identity over time;
        - reproduction starts from passport.replay_z and then decodes to 2nd level.
    """

    def __init__(self, cfg: Optional[DynamicObjectPassportConfig] = None):
        self.cfg = cfg or DynamicObjectPassportConfig()
        self.passports: Dict[str, Dict[str, Any]] = {}
        self.slot_to_token: Dict[int, str] = {}
        self.next_id: int = 0
        self.last_active_token: str = ""
        self.replay_cursor: int = 0

    def __len__(self) -> int:
        return len(self.passports)

    def _new_token(self, preferred_slot: Optional[int] = None) -> str:
        if preferred_slot is not None:
            token = f"{self.cfg.token_prefix}_{int(preferred_slot):03d}"
            if token not in self.passports:
                self.next_id = max(self.next_id, int(preferred_slot) + 1)
                return token

        while True:
            token = f"{self.cfg.token_prefix}_{self.next_id:03d}"
            self.next_id += 1
            if token not in self.passports:
                return token

    def _slot_index(self, obj: Dict[str, Any]) -> int:
        for key in ("active_slot_index", "event_slot_index"):
            try:
                v = obj.get(key)
                if torch.is_tensor(v):
                    return int(v.detach().reshape(-1)[0].cpu().item())
                if v is not None:
                    return int(v)
            except Exception:
                pass
        return 0

    def _token_from_obj(self, obj: Dict[str, Any], slot: int) -> str:
        tok = str(obj.get("slot_token", "") or "")
        if tok:
            return tok
        return self.slot_to_token.get(int(slot), f"{self.cfg.token_prefix}_{int(slot):03d}")

    def _dynamic_score(self, obj: Dict[str, Any]) -> float:
        dz = _f(obj.get("event_delta_norm"), _f(obj.get("delta_norm"), 0.0))
        touch = max(_f(obj.get("touch_strength"), 0.0), _f(obj.get("contact_norm"), 0.0))
        vision = _f(obj.get("vision_strength"), 0.0)
        action = _f(obj.get("action_norm"), 0.0)
        scenario = _f(obj.get("scenario_active"), 0.0)
        return float(dz + 0.25 * touch + 0.10 * vision + 0.10 * action + 0.05 * scenario)

    def _source(self, obj: Dict[str, Any], dream_mode: bool) -> str:
        if dream_mode:
            return "dream"
        touch = max(_f(obj.get("touch_strength"), 0.0), _f(obj.get("contact_norm"), 0.0))
        vision = _f(obj.get("vision_strength"), 0.0)
        if touch > 0.05 and vision > 0.20:
            return "vision+touch"
        if touch > 0.05:
            return "touch"
        if vision > 0.20:
            return "vision"
        return "latent"

    def _latest_sentence(self, obj: Dict[str, Any], event_memory: Any = None) -> str:
        for key in ("semantic_sentence", "semantic_code_sentence", "event_code_sentence", "scenario_sentence"):
            s = str(obj.get(key, "") or "")
            if s:
                return s
        try:
            ev = getattr(event_memory, "last_event", None)
            if isinstance(ev, dict):
                return str(ev.get("semantic_sentence", "") or ev.get("sentence", "") or "")
        except Exception:
            pass
        return ""

    def _event_summary(self, event_memory: Any = None) -> str:
        try:
            sm = getattr(event_memory, "sentence_memory", None)
            if sm is not None and hasattr(sm, "latest_episode_summary"):
                return str(sm.latest_episode_summary() or "")
        except Exception:
            pass
        return ""

    def _make_passport(self, token: str, slot: int, z: torch.Tensor, obj: Dict[str, Any], event_memory: Any, dream_mode: bool) -> Dict[str, Any]:
        source = self._source(obj, dream_mode)
        sentence = self._latest_sentence(obj, event_memory)
        return {
            "token": str(token),
            "lives_in_slot": int(slot),
            "created_step": int(obj.get("global_step", 0) or 0),
            "updates": 0,
            "confidence_ema": 0.0,
            "dynamic_score_ema": 0.0,
            "latent_signature": z.detach().cpu(),
            "replay_z": z.detach().cpu(),
            "source_profile": {source: 1},
            "formed_by": [source],
            "event_count": 0,
            "last_sentence": sentence,
            "episode_summary": self._event_summary(event_memory),
            "last_similarity": 1.0,
            "status": "active",
        }

    def _match_existing(self, z: torch.Tensor, slot: int, token_hint: str) -> Tuple[str, float]:
        # Existing token bound to this slot wins unless it is very far.
        if int(slot) in self.slot_to_token:
            tok = self.slot_to_token[int(slot)]
            p = self.passports.get(tok)
            if p is not None and torch.is_tensor(p.get("latent_signature")):
                sim = _cos_sim(z, p["latent_signature"])
                if sim >= float(self.cfg.similarity_threshold) * 0.85:
                    return tok, sim

        # Explicit OBJ token from SlotVocabulary can also win.
        if token_hint in self.passports:
            p = self.passports[token_hint]
            if torch.is_tensor(p.get("latent_signature")):
                sim = _cos_sim(z, p["latent_signature"])
                if sim >= float(self.cfg.similarity_threshold) * 0.85:
                    return token_hint, sim

        best_tok = ""
        best_sim = -1.0
        for tok, p in self.passports.items():
            sig = p.get("latent_signature")
            if not torch.is_tensor(sig):
                continue
            sim = _cos_sim(z, sig)
            if sim > best_sim:
                best_tok = tok
                best_sim = sim

        if best_tok and best_sim >= float(self.cfg.similarity_threshold):
            return best_tok, best_sim
        return "", best_sim

    def observe(self, obj: Dict[str, Any], *, event_memory: Any = None, dream_mode: bool = False, global_step: int = 0) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        z = _as_z(obj.get("z_obj"), int(self.cfg.latent_dim))
        if z is None:
            return {}

        slot = self._slot_index(obj)
        token_hint = self._token_from_obj(obj, slot)
        dyn = self._dynamic_score(obj)
        conf = _f(obj.get("confidence"), 0.0)
        source = self._source(obj, dream_mode)

        # Permit scene passport at slot 0 even before strong event dynamics.
        can_create = (
            dyn >= float(self.cfg.min_dynamic_score)
            or conf >= float(self.cfg.min_confidence_to_create)
            or (slot == 0 and bool(self.cfg.create_scene_passport))
            or dream_mode
        )

        tok, sim = self._match_existing(z, slot, token_hint)
        created = False
        if not tok:
            if not can_create or len(self.passports) >= int(self.cfg.max_passports):
                return {
                    "passport_active": False,
                    "passport_dynamic_score": dyn,
                    "passport_similarity": sim,
                    "passport_count": len(self.passports),
                }
            tok = token_hint if token_hint and token_hint not in self.passports else self._new_token(preferred_slot=slot)
            self.passports[tok] = self._make_passport(tok, slot, z, obj, event_memory, dream_mode)
            created = True

        p = self.passports[tok]
        p["updates"] = int(p.get("updates", 0)) + 1
        p["lives_in_slot"] = int(slot)
        p["last_step"] = int(global_step)
        p["last_similarity"] = float(sim if sim >= 0.0 else 1.0)

        # EMA updates.
        cd = float(self.cfg.confidence_ema_decay)
        sd = float(self.cfg.signature_ema_decay)
        p["confidence_ema"] = cd * float(p.get("confidence_ema", 0.0)) + (1.0 - cd) * conf
        p["dynamic_score_ema"] = cd * float(p.get("dynamic_score_ema", 0.0)) + (1.0 - cd) * dyn

        old_sig = p.get("latent_signature")
        if torch.is_tensor(old_sig):
            p["latent_signature"] = (sd * old_sig.detach().float().cpu() + (1.0 - sd) * z.detach().float().cpu())
        else:
            p["latent_signature"] = z.detach().cpu()
        p["replay_z"] = z.detach().cpu()

        prof = dict(p.get("source_profile", {}))
        prof[source] = int(prof.get(source, 0)) + 1
        p["source_profile"] = prof
        formed = set(p.get("formed_by", []))
        formed.add(source)
        p["formed_by"] = sorted(formed)

        sent = self._latest_sentence(obj, event_memory)
        if sent:
            p["last_sentence"] = sent
            p["event_count"] = int(p.get("event_count", 0)) + 1
        ep = self._event_summary(event_memory)
        if ep:
            p["episode_summary"] = ep

        self.passports[tok] = p
        self.slot_to_token[int(slot)] = tok
        self.last_active_token = tok

        return {
            "passport_active": True,
            "passport_created": created,
            "passport_token": tok,
            "passport_slot": int(slot),
            "passport_count": len(self.passports),
            "passport_similarity": float(p.get("last_similarity", 1.0)),
            "passport_dynamic_score": float(dyn),
            "passport_confidence_ema": float(p.get("confidence_ema", 0.0)),
            "passport_source": source,
            "passport_sentence": str(p.get("last_sentence", "")),
            "passport_episode_summary": str(p.get("episode_summary", "")),
        }

    def select_for_replay(self, token: str = "") -> Optional[Dict[str, Any]]:
        if token and token in self.passports:
            return self.passports[token]
        if self.last_active_token and self.last_active_token in self.passports:
            return self.passports[self.last_active_token]
        if self.passports:
            return list(self.passports.values())[-1]
        return None

    def reproduce_inner_world(self, token: str = "", *, device=None, dtype=torch.float32) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "replay_enabled", True)):
            return {}

        p = self.select_for_replay(token)
        if p is None:
            return {}

        z = p.get("replay_z")
        if not torch.is_tensor(z):
            z = p.get("latent_signature")
        if not torch.is_tensor(z):
            return {}

        z = z.detach().to(device=device, dtype=dtype)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        return {
            "passport_replay_active": True,
            "passport_replay_token": str(p.get("token", "")),
            "passport_replay_slot": int(p.get("lives_in_slot", 0)),
            "passport_inner_world_z": z,
            "passport_replay_sentence": str(p.get("last_sentence", "")),
            "passport_replay_episode_summary": str(p.get("episode_summary", "")),
            "passport_replay_confidence": float(p.get("confidence_ema", 0.0)),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "passports": self.passports,
            "slot_to_token": self.slot_to_token,
            "next_id": int(self.next_id),
            "last_active_token": str(self.last_active_token),
            "replay_cursor": int(self.replay_cursor),
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
        self.passports = state.get("passports", {}) if isinstance(state.get("passports", {}), dict) else {}
        self.slot_to_token = {int(k): str(v) for k, v in dict(state.get("slot_to_token", {})).items()}
        self.next_id = int(state.get("next_id", len(self.passports)) or 0)
        self.last_active_token = str(state.get("last_active_token", "") or "")
        self.replay_cursor = int(state.get("replay_cursor", 0) or 0)
