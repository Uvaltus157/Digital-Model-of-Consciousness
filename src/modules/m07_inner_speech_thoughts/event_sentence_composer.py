from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EventSentenceComposerConfig:
    """
    Grammar layer above event tokens.

    Token layer:
        EVT t=120 SLOT_1 OBJ_001 KIND=contact_transition DZ_MID ...

    Sentence layer:
        OBJ_001 was touched and changed moderately; confidence is high.

    This is still deterministic/debug grammar, not LLM text generation.
    Later it can be replaced by a trainable EventLanguageHead.
    """
    enabled: bool = True
    language: str = "code"  # "code", "en", "ru"
    include_debug_tokens: bool = True


class EventSentenceComposer:
    def __init__(self, cfg: EventSentenceComposerConfig | None = None):
        self.cfg = cfg or EventSentenceComposerConfig()

    def _level(self, value: float, lo: float, hi: float) -> str:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if v < lo:
            return "low"
        if v < hi:
            return "mid"
        return "high"

    def _verb_from_kind(self, kind: str, contact: float, action: float, dream: bool) -> str:
        kind = str(kind or "")
        if dream:
            return "dream_replays"
        if "contact" in kind or contact > 0.010:
            return "touch_changes"
        if "self_motion" in kind or action > 0.010:
            return "self_moves_changes"
        if "latent" in kind:
            return "latent_changes"
        return "changes"

    def _code_sentence(self, event: Dict[str, Any]) -> str:
        token = str(event.get("slot_token") or f"SLOT_{event.get('slot', 0)}")
        slot = int(event.get("slot", 0) or 0)
        kind = str(event.get("kind", "latent_transition") or "latent_transition")
        dz = float(event.get("delta_norm", 0.0) or 0.0)
        action = float(event.get("action_norm", 0.0) or 0.0)
        contact = float(event.get("contact_norm", 0.0) or 0.0)
        conf = float(event.get("confidence", 0.0) or 0.0)
        vision = float(event.get("vision_strength", 0.0) or 0.0)
        touch = float(event.get("touch_strength", 0.0) or 0.0)
        dream = bool(event.get("dream_mode", False))

        verb = self._verb_from_kind(kind, contact, action, dream)
        dz_l = self._level(dz, 0.015, 0.060).upper()
        act_l = self._level(action, 0.010, 0.050).upper()
        touch_l = self._level(max(contact, touch), 0.010, 0.050).upper()
        conf_l = self._level(conf, 0.10, 0.45).upper()
        vision_l = self._level(vision, 0.20, 0.55).upper()

        modifiers = []
        if contact > 0.010 or touch > 0.35:
            modifiers.append("CONTACT_BOUND")
        if action > 0.010:
            modifiers.append("SELF_CAUSED")
        if dream:
            modifiers.append("DREAM")
        if conf > 0.45:
            modifiers.append("STABLE")
        if dz > 0.060:
            modifiers.append("BIG_CHANGE")

        return " ".join([
            "SENT",
            f"SUBJ={token}",
            f"ADDR=SLOT_{slot}",
            f"VERB={verb}",
            f"KIND={kind}",
            f"DZ={dz_l}",
            f"ACT={act_l}",
            f"TOUCH={touch_l}",
            f"VISION={vision_l}",
            f"CONF={conf_l}",
            "MODS=" + ("+".join(modifiers) if modifiers else "none"),
        ])

    def _en_sentence(self, event: Dict[str, Any]) -> str:
        token = str(event.get("slot_token") or f"SLOT_{event.get('slot', 0)}")
        kind = str(event.get("kind", "latent_transition") or "latent_transition")
        dz = float(event.get("delta_norm", 0.0) or 0.0)
        action = float(event.get("action_norm", 0.0) or 0.0)
        contact = float(event.get("contact_norm", 0.0) or 0.0)
        conf = float(event.get("confidence", 0.0) or 0.0)
        dream = bool(event.get("dream_mode", False))

        dz_word = {"low": "slightly", "mid": "noticeably", "high": "strongly"}[self._level(dz, 0.015, 0.060)]
        conf_word = {"low": "weak", "mid": "forming", "high": "stable"}[self._level(conf, 0.10, 0.45)]

        if dream:
            return f"{token} is replayed internally in dream mode and changes {dz_word}; memory is {conf_word}."
        if "contact" in kind or contact > 0.010:
            return f"{token} is touched; its latent image changes {dz_word}; memory is {conf_word}."
        if "self_motion" in kind or action > 0.010:
            return f"{token} changes during self-caused motion; latent shift is {dz_word}; memory is {conf_word}."
        return f"{token} changes {dz_word} as an internal latent transition; memory is {conf_word}."

    def _ru_sentence(self, event: Dict[str, Any]) -> str:
        token = str(event.get("slot_token") or f"SLOT_{event.get('slot', 0)}")
        kind = str(event.get("kind", "latent_transition") or "latent_transition")
        dz = float(event.get("delta_norm", 0.0) or 0.0)
        action = float(event.get("action_norm", 0.0) or 0.0)
        contact = float(event.get("contact_norm", 0.0) or 0.0)
        conf = float(event.get("confidence", 0.0) or 0.0)
        dream = bool(event.get("dream_mode", False))

        dz_word = {"low": "слабо", "mid": "заметно", "high": "сильно"}[self._level(dz, 0.015, 0.060)]
        conf_word = {"low": "слабая", "mid": "формируется", "high": "устойчивая"}[self._level(conf, 0.10, 0.45)]

        if dream:
            return f"{token} проигрывается внутри во сне и меняется {dz_word}; память {conf_word}."
        if "contact" in kind or contact > 0.010:
            return f"{token} получил контакт; его латентный образ меняется {dz_word}; память {conf_word}."
        if "self_motion" in kind or action > 0.010:
            return f"{token} изменился из-за собственного движения; сдвиг латента {dz_word}; память {conf_word}."
        return f"{token} меняется {dz_word} как внутренний латентный переход; память {conf_word}."

    def compose(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        lang = str(getattr(self.cfg, "language", "code") or "code").lower().strip()
        code = self._code_sentence(event)
        if lang == "ru":
            natural = self._ru_sentence(event)
        elif lang == "en":
            natural = self._en_sentence(event)
        else:
            natural = code

        token = str(event.get("slot_token") or f"SLOT_{event.get('slot', 0)}")
        kind = str(event.get("kind", "latent_transition") or "latent_transition")
        contact = float(event.get("contact_norm", 0.0) or 0.0)
        action = float(event.get("action_norm", 0.0) or 0.0)
        dream = bool(event.get("dream_mode", False))

        roles = {
            "subject": token,
            "address": f"SLOT_{int(event.get('slot', 0) or 0)}",
            "verb": self._verb_from_kind(kind, contact, action, dream),
            "object": "latent_image",
            "context": "dream" if dream else ("contact" if contact > 0.010 else ("self_action" if action > 0.010 else "latent")),
        }

        return {
            "semantic_sentence": natural,
            "semantic_code_sentence": code,
            "sentence_roles": roles,
            "sentence_language": lang,
        }
