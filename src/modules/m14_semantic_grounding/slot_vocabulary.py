from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


SHAPE_NAMES = ["unknown", "sphere", "box", "cylinder"]


@dataclass
class SlotVocabularyConfig:
    """
    Runtime vocabulary for ObjectSlotMemory slots.

    Slot index is only an address:
        SLOT_1

    Slot vocabulary gives that address a stable internal word:
        OBJ_001

    The real meaning is still the latent vector z_obj_slots[slot].
    The vocabulary stores a compact passport around it:
        latent signature, confidence, age, decoded attributes, source profile,
        event counters, and last event sentence.
    """
    enabled: bool = True
    max_slots: int = 10
    latent_dim: int = 128
    token_prefix: str = "OBJ"
    top_k_signature_dims: int = 8
    confidence_ema_decay: float = 0.92
    signature_ema_decay: float = 0.96
    min_confidence_to_name: float = 0.05


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _tensor_1d(x: Any, default_dim: int = 0) -> Optional[torch.Tensor]:
    try:
        if x is None or not torch.is_tensor(x):
            return None
        z = x.detach().float()
        if z.ndim == 0:
            z = z.view(1)
        if z.ndim > 1:
            z = z.reshape(-1, z.shape[-1])[0]
        if default_dim > 0:
            if z.shape[-1] > default_dim:
                z = z[:default_dim]
            elif z.shape[-1] < default_dim:
                z = F.pad(z, (0, default_dim - z.shape[-1]))
        return z.cpu()
    except Exception:
        return None


class SlotVocabulary:
    """
    Small runtime symbol table for slots.

    It does NOT decide object identity and does NOT hand-label cube/sphere/etc.
    It only gives each slot a stable internal token and a latent fingerprint.
    """

    def __init__(self, cfg: Optional[SlotVocabularyConfig] = None):
        self.cfg = cfg or SlotVocabularyConfig()
        self.entries: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def token_for(self, slot_id: int) -> str:
        slot_id = int(slot_id)
        return f"{self.cfg.token_prefix}_{slot_id:03d}"

    def _empty_entry(self, slot_id: int) -> Dict[str, Any]:
        return {
            "slot_id": int(slot_id),
            "token": self.token_for(slot_id),
            "updates": 0,
            "confidence_ema": 0.0,
            "z_norm_ema": 0.0,
            "dominant_source": "unknown",
            "top_dims": [],
            "top_values": [],
            "shape_name": "unknown",
            "color_rgb": [0.0, 0.0, 0.0],
            "size": 0.0,
            "hardness": 0.0,
            "stability": 0.0,
            "novelty": 0.0,
            "event_counts": {},
            "last_event_kind": "",
            "last_event_sentence": "",
            "formed_by": [],
        }

    def _infer_source(self, vision_strength: float, touch_strength: float, dream_mode: bool) -> str:
        if dream_mode:
            return "dream"
        if touch_strength > 0.35 and vision_strength > 0.20:
            return "vision+touch"
        if touch_strength > 0.35:
            return "touch"
        if vision_strength > 0.20:
            return "vision"
        return "latent"

    def _shape_name(self, obj: Dict[str, Any]) -> str:
        try:
            logits = obj.get("shape_logits")
            if torch.is_tensor(logits):
                idx = int(torch.argmax(logits.detach().float().reshape(-1, logits.shape[-1])[0]).cpu().item())
                return SHAPE_NAMES[idx] if 0 <= idx < len(SHAPE_NAMES) else "unknown"
        except Exception:
            pass
        return "unknown"

    def update_from_observation(
        self,
        *,
        slot_id: int,
        z_obj: torch.Tensor,
        obj: Dict[str, Any],
        confidence: float,
        vision_strength: float,
        touch_strength: float,
        event_kind: str = "",
        event_sentence: str = "",
        dream_mode: bool = False,
    ) -> Dict[str, Any]:
        slot_id = int(slot_id)
        entry = self.entries.get(slot_id)
        if entry is None:
            entry = self._empty_entry(slot_id)

        z = _tensor_1d(z_obj, int(self.cfg.latent_dim))
        decay = float(self.cfg.confidence_ema_decay)
        sig_decay = float(self.cfg.signature_ema_decay)

        entry["updates"] = int(entry.get("updates", 0)) + 1
        entry["confidence_ema"] = decay * float(entry.get("confidence_ema", 0.0)) + (1.0 - decay) * float(confidence)

        if z is not None:
            z_norm = float(torch.linalg.vector_norm(z).item())
            entry["z_norm_ema"] = sig_decay * float(entry.get("z_norm_ema", 0.0)) + (1.0 - sig_decay) * z_norm

            k = max(1, int(self.cfg.top_k_signature_dims))
            vals, idx = torch.topk(torch.abs(z), k=min(k, z.numel()))
            top_dims = [int(i) for i in idx.tolist()]
            top_values = [float(z[i].item()) for i in top_dims]
            entry["top_dims"] = top_dims
            entry["top_values"] = top_values

        source = self._infer_source(float(vision_strength), float(touch_strength), bool(dream_mode))
        entry["dominant_source"] = source
        formed_by = set(entry.get("formed_by", []))
        formed_by.add(source)
        entry["formed_by"] = sorted(formed_by)

        entry["shape_name"] = self._shape_name(obj)
        try:
            c = obj.get("color_rgb")
            if torch.is_tensor(c):
                cv = c.detach().float().reshape(-1)[:3].cpu().tolist()
                entry["color_rgb"] = [float(v) for v in cv]
        except Exception:
            pass

        for key in ("size", "hardness", "stability", "novelty"):
            entry[key] = _f(obj.get(key), float(entry.get(key, 0.0)))

        if event_kind:
            counts = dict(entry.get("event_counts", {}))
            counts[event_kind] = int(counts.get(event_kind, 0)) + 1
            entry["event_counts"] = counts
            entry["last_event_kind"] = str(event_kind)
        if event_sentence:
            entry["last_event_sentence"] = str(event_sentence)

        self.entries[slot_id] = entry
        return entry

    def get(self, slot_id: int) -> Dict[str, Any]:
        slot_id = int(slot_id)
        return self.entries.get(slot_id, self._empty_entry(slot_id))

    def describe(self, slot_id: int) -> str:
        e = self.get(slot_id)
        dims = ",".join(str(i) for i in e.get("top_dims", [])[:5])
        return (
            f"{e['token']} slot={slot_id} conf={e.get('confidence_ema', 0.0):.2f} "
            f"src={e.get('dominant_source', 'unknown')} shape={e.get('shape_name', 'unknown')} "
            f"z={e.get('z_norm_ema', 0.0):.2f} top=[{dims}]"
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "entries": self.entries,
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
        entries = state.get("entries", {})
        if isinstance(entries, dict):
            self.entries = {int(k): v for k, v in entries.items()}
