from __future__ import annotations

"""
english_inner_speech_teacher.py

Small English inner-speech teacher for ConsciousDreamer symbolic reports.

It mirrors the Russian teacher interface:
    vocab.encode/decode
    teacher.build_report(...)
    teacher.target_ids(...)
    teacher.report_loss(...)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AnglishInnerSpeechVocab:
    tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<bos>", "<eos>", "<unk>",
        "i", "see", "think", "feel", "touch", "remember",
        "want", "plan", "explore", "approach", "take", "push",
        "object", "sphere", "cube", "cylinder", "shape", "color", "material",
        "green", "red", "blue", "yellow", "dark", "light",
        "round", "edged", "long", "solid", "smooth", "unknown",
        "with", "hand", "fingers", "left", "right", "in", "front", "of", "me",
        "focus", "memory", "similar", "episode", "new", "familiar",
        "weak", "strong", "contact", "no", "yes",
        "confident", "uncertain", "inside", "model", "goal", "action",
        "and", "but", "now", "next", "movement",
    ])

    def __post_init__(self):
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def normalize(self, text: str) -> str:
        text = text.lower()
        for ch in ",.!?:;()[]{}\"'":
            text = text.replace(ch, " ")
        return " ".join(text.split())

    def encode(self, text: str, max_len: int = 48, add_bos_eos: bool = True) -> List[int]:
        text = self.normalize(text)
        ids = []
        if add_bos_eos:
            ids.append(self.bos_id)
        for w in text.split():
            ids.append(self.stoi.get(w, self.unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        ids = ids[:max_len]
        while len(ids) < max_len:
            ids.append(self.pad_id)
        return ids

    def decode(self, ids: List[int] | torch.Tensor, skip_special: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().reshape(-1).tolist()
        words = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if skip_special and tok in ("<pad>", "<bos>", "<eos>"):
                continue
            words.append(tok)
        return " ".join(words)


@dataclass
class InnerSpeechTeacherConfig:
    max_text_tokens: int = 48
    touch_threshold: float = 0.15
    memory_threshold: float = 1.0
    object_conf_threshold: float = 0.45


class AnglishInnerSpeechTeacher:
    shape_names = {
        0: "sphere",
        1: "cube",
        2: "cylinder",
    }

    color_names = {
        0: "unknown",
        1: "green",
        2: "red",
        3: "blue",
        4: "yellow",
    }

    def __init__(self, vocab: Optional[AnglishInnerSpeechVocab] = None, cfg: Optional[InnerSpeechTeacherConfig] = None) -> None:
        self.vocab = vocab or AnglishInnerSpeechVocab()
        self.cfg = cfg or InnerSpeechTeacherConfig()

    def _safe_float(self, x, default: float = 0.0) -> float:
        if x is None:
            return float(default)
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().cpu().reshape(-1)[0].item())
        arr = np.asarray(x).reshape(-1)
        return float(arr[0]) if arr.size else float(default)

    def _safe_int(self, x, default: int = 0) -> int:
        return int(round(self._safe_float(x, float(default))))

    def infer_object_words(self, out: Dict) -> Tuple[str, str]:
        imagery = out.get("object_imagery", {})
        shape_id = self._safe_int(imagery.get("shape_id"), default=-1)
        color_id = self._safe_int(imagery.get("color_id"), default=-1)
        conf = self._safe_float(imagery.get("object_confidence"), default=0.0)

        shape = self.shape_names.get(shape_id, "object")
        color = self.color_names.get(color_id, "unknown")
        if conf < self.cfg.object_conf_threshold:
            shape = "object"
            color = "unknown"
        return color, shape

    def build_report(self, obs: Dict, out: Dict) -> str:
        tactile = obs.get("tactile")
        touch_sum = self._safe_float(tactile.sum() if torch.is_tensor(tactile) else np.asarray(tactile).sum(), 0.0)

        curiosity = self._safe_float(out.get("values", {}).get("curiosity"), 0.0)
        coherence = self._safe_float(out.get("values", {}).get("coherence"), 0.0)
        memory_used = self._safe_float(out.get("memory", {}).get("memory_usage", torch.zeros(1)).sum(), 0.0)

        color, shape = self.infer_object_words(out)
        parts = []

        if color != "unknown" and shape != "object":
            parts += ["i", "see", color, shape]
        elif shape != "object":
            parts += ["i", "see", shape]
        else:
            parts += ["i", "see", "object"]

        parts += ["object", "in", "focus"]

        if touch_sum > self.cfg.touch_threshold:
            parts += ["i", "touch", "object", "with", "fingers", "yes", "contact"]
        else:
            parts += ["no", "contact"]

        if memory_used > self.cfg.memory_threshold:
            parts += ["remember", "similar", "episode"]
        else:
            parts += ["object", "new"]

        if curiosity > 0.55:
            parts += ["want", "explore", "object", "with", "hand"]
        elif coherence > 0.55:
            parts += ["plan", "next", "movement"]
        else:
            parts += ["think", "object", "uncertain"]

        return " ".join(parts)

    def target_ids(self, obs: Dict, out: Dict, device=None) -> torch.Tensor:
        text = self.build_report(obs, out)
        ids = self.vocab.encode(text, max_len=self.cfg.max_text_tokens)
        return torch.tensor([ids], dtype=torch.long, device=device)

    def report_loss(self, symbolic_report: Dict, target_ids: torch.Tensor) -> torch.Tensor:
        logits = symbolic_report["text_logits"]
        t = target_ids
        if logits.shape[1] != t.shape[1]:
            n = min(logits.shape[1], t.shape[1])
            logits = logits[:, :n]
            t = t[:, :n]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            t.reshape(-1),
            ignore_index=self.vocab.pad_id,
        )


EnglishInnerSpeechVocab = AnglishInnerSpeechVocab
EnglishInnerSpeechTeacher = AnglishInnerSpeechTeacher
InnerSpeechVocab = AnglishInnerSpeechVocab
InnerSpeechTeacher = AnglishInnerSpeechTeacher


if __name__ == "__main__":
    vocab = AnglishInnerSpeechVocab()
    teacher = AnglishInnerSpeechTeacher(vocab)
    print("vocab size:", vocab.size)
    print(vocab.encode("i see green cube", max_len=12))
    print(vocab.decode(vocab.encode("i see green cube", max_len=12)))
