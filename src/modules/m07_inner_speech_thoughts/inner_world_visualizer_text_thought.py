from __future__ import annotations

"""
inner_world_visualizer_text_thought.py

Extends object-image visualizer with readable inner thought text:

- Predicted thought text: model decoded text_token_ids
- Target thought text: teacher-generated target report
- Match score: rough token overlap
- Still shows symbol_ids / phoneme_ids / text_token_ids

Expected:
    out["symbolic_report"]["text_token_ids"]
    optional out["decoded_report"]
    optional out["target_report"]

If decoded/target strings are not in out, pass them as arguments to render/show/save.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Dict, Optional

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, get_cv2_last_key

from src.modules.m01_object_imagery.inner_world_visualizer_object_image import DreamerInnerWorldVisualizerV2
from src.modules.m07_inner_speech_thoughts.inner_world_visualizer import draw_block, draw_text, draw_code_tokens
from src.modules.m07_inner_speech_thoughts.english_inner_speech_teacher import InnerSpeechVocab


def wrap_text(text: str, max_chars: int = 64) -> list[str]:
    words = str(text).split()
    lines = []
    cur = []
    n = 0
    for w in words:
        if n + len(w) + 1 > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def simple_text_match(pred: str, target: str) -> float:
    p = [w for w in str(pred).split() if w]
    t = [w for w in str(target).split() if w]
    if not p or not t:
        return 0.0
    ps = set(p)
    ts = set(t)
    return len(ps & ts) / max(1, len(ts))


class DreamerInnerWorldVisualizerV3(DreamerInnerWorldVisualizerV2):
    def __init__(self, cfg=None, vocab: Optional[InnerSpeechVocab] = None) -> None:
        super().__init__(cfg)
        self.speech_vocab = vocab or InnerSpeechVocab()

    def decode_predicted_text(self, symbolic_report: Optional[Dict]) -> str:
        if not symbolic_report:
            return ""
        ids = symbolic_report.get("text_token_ids")
        if ids is None:
            return ""
        if torch.is_tensor(ids):
            if ids.ndim == 2:
                ids = ids[0]
        return self.speech_vocab.decode(ids, skip_special=True)

    def _draw_symbolic_report(
        self,
        canvas: np.ndarray,
        symbolic_report: Optional[Dict],
        x: int,
        y: int,
        w: int,
        h: int,
        predicted_text: Optional[str] = None,
        target_text: Optional[str] = None,
    ):
        cfg = self.cfg
        draw_block(canvas, x, y, w, h, "Readable inner speech / мысль модели", cfg)

        if not symbolic_report:
            draw_text(canvas, "symbolic report not provided", (x + 14, y + 36), cfg.text_color, 0.52, 1)
            return

        if predicted_text is None:
            predicted_text = self.decode_predicted_text(symbolic_report)
        if target_text is None:
            target_text = ""

        conf = 0.0
        c = symbolic_report.get("confidence")
        if c is not None:
            if torch.is_tensor(c):
                conf = float(c.detach().cpu().reshape(-1)[0].item())
            else:
                conf = float(np.asarray(c).reshape(-1)[0])

        match = simple_text_match(predicted_text, target_text)

        draw_text(canvas, f"inner speech confidence={conf:.3f}", (x + 14, y + 34), cfg.accent5, 0.52, 1)
        draw_text(canvas, f"text match={match:.3f}", (x + 310, y + 34), cfg.accent2, 0.52, 1)

        # Predicted text
        draw_text(canvas, "МЫСЛЬ МОДЕЛИ:", (x + 14, y + 70), cfg.accent1, 0.58, 1)
        yy = y + 98
        for line in wrap_text(predicted_text, max_chars=72)[:4]:
            draw_text(canvas, line, (x + 24, yy), cfg.text_color, 0.58, 1)
            yy += 26

        # Target text
        draw_text(canvas, "ЦЕЛЕВАЯ МЫСЛЬ / УЧИТЕЛЬ:", (x + 14, y + 210), cfg.accent3, 0.58, 1)
        yy = y + 238
        for line in wrap_text(target_text, max_chars=72)[:4]:
            draw_text(canvas, line, (x + 24, yy), cfg.accent3, 0.54, 1)
            yy += 24

        # Codes at bottom
        draw_text(canvas, "internal codes:", (x + 14, y + h - 126), cfg.text_color, 0.5, 1)
        draw_code_tokens(canvas, symbolic_report.get("symbol_ids"), (x + 10, y + h - 110, w - 20, 36), cfg, "S", cfg.accent5, max_rows=1)
        draw_code_tokens(canvas, symbolic_report.get("phoneme_ids"), (x + 10, y + h - 74, w - 20, 34), cfg, "P", cfg.accent2, max_rows=1)
        draw_code_tokens(canvas, symbolic_report.get("text_token_ids"), (x + 10, y + h - 40, w - 20, 34), cfg, "T", cfg.accent1, max_rows=1)

    def render(
        self,
        out: Dict,
        symbolic_report: Optional[Dict] = None,
        predicted_text: Optional[str] = None,
        target_text: Optional[str] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        canvas = np.full((cfg.height, cfg.width, 3), cfg.bg_color, dtype=np.uint8)

        draw_text(canvas, cfg.title + " + Readable Thought", (24, 34), cfg.text_color, 0.9, 2)

        if symbolic_report is None:
            symbolic_report = out.get("symbolic_report")

        if predicted_text is None:
            predicted_text = out.get("decoded_report")
        if target_text is None:
            target_text = out.get("target_report")

        self._draw_core_graph(canvas, out, 20, 56, 720, 520)
        self._draw_attention(canvas, out, 760, 56, 300, 230)
        self._draw_memory(canvas, out, 1080, 56, 700, 230)
        self._draw_imagination(canvas, out, 760, 306, 500, 270)
        self._draw_stats(canvas, out, 1280, 306, 500, 270)
        self._draw_object_imagery(canvas, out, 20, 596, 900, 470)
        self._draw_symbolic_report(canvas, symbolic_report, 940, 596, 840, 470, predicted_text, target_text)

        return canvas

    def show(
        self,
        out: Dict,
        symbolic_report: Optional[Dict] = None,
        predicted_text: Optional[str] = None,
        target_text: Optional[str] = None,
        window_name: str = "dreamer inner world v3",
        delay_ms: int = 1,
    ) -> int:
        frame = self.render(out, symbolic_report, predicted_text, target_text)
        submit_cv2_frame(window_name, frame, frame.shape[1], frame.shape[0])
        return int(get_cv2_last_key())

    def save(
        self,
        path: str,
        out: Dict,
        symbolic_report: Optional[Dict] = None,
        predicted_text: Optional[str] = None,
        target_text: Optional[str] = None,
    ) -> None:
        frame = self.render(out, symbolic_report, predicted_text, target_text)
        cv2.imwrite(path, frame)
