from __future__ import annotations


if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

"""
inner_world_visualizer_text_thought_unicode.py

Autonomous Unicode-safe inner-world visualizer.
Does not rely on cv2.putText for text; all text is drawn with PIL, so Russian
Cyrillic is displayed correctly.

Shows:
- readable model thought
- teacher target thought
- text match
- internal S/P/T codes
- object imagery panel if out["object_imagery"] exists
- attention/memory/imagination summary
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, get_cv2_last_key

from src.modules.m07_inner_speech_thoughts.unicode_text_draw import draw_text
from src.modules.m07_inner_speech_thoughts.english_inner_speech_teacher import InnerSpeechVocab
from src.modules.m01_object_imagery.models.object_imagery_decoder import blended_object_image


@dataclass
class InnerWorldVizConfig:
    width: int = 1800
    height: int = 1100
    bg_color: Tuple[int, int, int] = (10, 14, 24)
    panel_color: Tuple[int, int, int] = (18, 24, 40)
    text_color: Tuple[int, int, int] = (235, 238, 245)
    accent1: Tuple[int, int, int] = (80, 180, 255)
    accent2: Tuple[int, int, int] = (120, 255, 180)
    accent3: Tuple[int, int, int] = (255, 210, 120)
    accent4: Tuple[int, int, int] = (255, 140, 140)
    accent5: Tuple[int, int, int] = (210, 170, 255)
    title: str = "ConsciousDreamer Inner World"
    debug_vocab: List[str] = field(default_factory=list)


def to_np(x):
    if x is None:
        return np.array([], dtype=np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_scalar(x, default=0.0) -> float:
    a = to_np(x).reshape(-1)
    if a.size == 0:
        return float(default)
    return float(a[0])


def safe_int(x, default=0) -> int:
    return int(round(safe_scalar(x, default)))


def norm01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def draw_block(img, x, y, w, h, title, cfg: InnerWorldVizConfig):
    cv2.rectangle(img, (x, y), (x + w, y + h), cfg.panel_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 70, 96), 1, cv2.LINE_AA)
    draw_text(img, title, (x + 12, y + 8), cfg.text_color, 0.58, 1)


def wrap_text(text: str, max_chars: int = 64) -> list[str]:
    words = str(text).split()
    lines, cur, n = [], [], 0
    for w in words:
        if n + len(w) + 1 > max_chars and cur:
            lines.append(" ".join(cur))
            cur, n = [w], len(w)
        else:
            cur.append(w)
            n += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def simple_text_match(pred: str, target: str) -> float:
    p = set(str(pred).split())
    t = set(str(target).split())
    if not p or not t:
        return 0.0
    return len(p & t) / max(1, len(t))


def draw_code_line(img, ids, x, y, prefix, color, max_items=16):
    arr = to_np(ids).astype(np.int64)
    if arr.size == 0:
        draw_text(img, f"{prefix}: <нет данных>", (x, y), color, 0.44, 1)
        return
    if arr.ndim == 2:
        arr = arr[0]
    line = " ".join([f"{prefix}{int(v)}" for v in arr[:max_items].tolist()])
    draw_text(img, line, (x, y), color, 0.42, 1)


def chw_to_bgr_u8(x):
    if torch.is_tensor(x):
        if x.ndim == 4:
            x = x[0]
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
        if arr.ndim == 4:
            arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def gray_to_bgr_u8(x):
    if torch.is_tensor(x):
        if x.ndim == 4:
            x = x[0]
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
        if arr.ndim == 4:
            arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(arr, cv2.COLORMAP_TURBO)


class DreamerInnerWorldVisualizerV3:
    def __init__(self, cfg: Optional[InnerWorldVizConfig] = None, vocab: Optional[InnerSpeechVocab] = None) -> None:
        self.cfg = cfg or InnerWorldVizConfig()
        self.speech_vocab = vocab or InnerSpeechVocab()

    def decode_predicted_text(self, symbolic_report: Optional[Dict]) -> str:
        if not symbolic_report:
            return ""
        ids = symbolic_report.get("text_token_ids")
        if ids is None:
            return ""
        if torch.is_tensor(ids) and ids.ndim == 2:
            ids = ids[0]
        return self.speech_vocab.decode(ids, skip_special=True)

    def draw_core(self, img, out, x, y, w, h):
        cfg = self.cfg
        draw_block(img, x, y, w, h, "Ядро внутреннего мира", cfg)
        cx, cy = x + w // 2, y + h // 2

        nodes = [
            ("workspace", out.get("workspace_out"), cfg.accent3, -180, 0),
            ("thought", out.get("thoughts", {}).get("thought"), cfg.accent1, 120, -90),
            ("reflection", out.get("reflection_out", {}).get("reflection"), cfg.accent5, 150, 95),
            ("object_repr", out.get("object_repr"), cfg.accent4, -160, -120),
            ("memory", out.get("memory", {}).get("memory_context"), cfg.accent2, -150, 120),
            ("body_self", out.get("selves", {}).get("body_self"), (170, 210, 255), 0, -145),
        ]

        pts = {}
        for name, vec, color, dx, dy in nodes:
            a = to_np(vec).reshape(-1)
            jitter = a[:2] if a.size >= 2 else np.zeros(2)
            px = int(cx + dx + float(jitter[0]) * 30)
            py = int(cy + dy - float(jitter[1]) * 30)
            pts[name] = (px, py)
            cv2.circle(img, (px, py), 18, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 21, tuple(min(255, c + 35) for c in color), 1, cv2.LINE_AA)
            draw_text(img, name, (px + 24, py - 8), color, 0.46, 1)

        for a, b in [("workspace", "thought"), ("workspace", "reflection"), ("workspace", "object_repr"), ("workspace", "memory"), ("memory", "thought"), ("object_repr", "thought"), ("body_self", "reflection")]:
            cv2.arrowedLine(img, pts[a], pts[b], (75, 92, 130), 2, cv2.LINE_AA, tipLength=0.08)

        curiosity = safe_scalar(out.get("values", {}).get("curiosity"), 0.0)
        coherence = safe_scalar(out.get("values", {}).get("coherence"), 0.0)
        self_conf = safe_scalar(out.get("reflection_out", {}).get("self_confidence"), 0.0)
        draw_text(img, f"любопытство={curiosity:.3f}", (x + 20, y + h - 70), cfg.accent2, 0.48, 1)
        draw_text(img, f"связность={coherence:.3f}", (x + 20, y + h - 45), cfg.accent3, 0.48, 1)
        draw_text(img, f"уверенность Я={self_conf:.3f}", (x + 20, y + h - 20), cfg.accent5, 0.48, 1)

    def draw_attention(self, img, out, x, y, w, h):
        cfg = self.cfg
        draw_block(img, x, y, w, h, "Внимание по модальностям", cfg)
        weights = to_np(out.get("attention", {}).get("modality_weights")).reshape(-1)
        labels = ["зрение", "поза", "тело", "осязание", "мотор", "объект", "действие"]
        if weights.size == 0:
            draw_text(img, "нет данных внимания", (x + 12, y + 42), cfg.text_color, 0.48, 1)
            return
        weights = weights[:len(labels)]
        maxv = max(1e-6, float(weights.max()))
        yy = y + 44
        for lab, val in zip(labels, weights):
            bw = int((w - 135) * float(val) / maxv)
            draw_text(img, lab, (x + 12, yy), cfg.text_color, 0.42, 1)
            cv2.rectangle(img, (x + 100, yy - 12), (x + 100 + bw, yy + 4), cfg.accent1, -1, cv2.LINE_AA)
            draw_text(img, f"{float(val):.2f}", (x + w - 55, yy), cfg.accent1, 0.38, 1)
            yy += 24

    def draw_memory(self, img, out, x, y, w, h):
        cfg = self.cfg
        draw_block(img, x, y, w, h, "Автобиографическая память", cfg)
        usage = to_np(out.get("memory", {}).get("memory_usage")).reshape(-1)
        if usage.size:
            n = min(48, usage.size)
            vals = usage[:n]
            bw = max(3, (w - 30) // n)
            for i in range(n):
                bh = int((h - 65) * float(np.clip(vals[i], 0, 1)))
                cv2.rectangle(img, (x + 15 + i * bw, y + h - 18 - bh), (x + 15 + i * bw + bw - 2, y + h - 18), cfg.accent2, -1)
            draw_text(img, f"активно слотов={float(usage.sum()):.2f}", (x + 15, y + 44), cfg.accent2, 0.46, 1)
        else:
            draw_text(img, "память пока пуста", (x + 15, y + 44), cfg.text_color, 0.46, 1)

    def draw_object_image(self, img, out, x, y, w, h):
        cfg = self.cfg
        draw_block(img, x, y, w, h, "Внутренний образ объекта", cfg)
        imagery = out.get("object_imagery")
        if not imagery:
            draw_text(img, "object_imagery отсутствует", (x + 14, y + 42), cfg.text_color, 0.5, 1)
            return

        rgb = chw_to_bgr_u8(blended_object_image(imagery["rgb"], imagery["alpha"]))
        alpha = gray_to_bgr_u8(imagery["alpha"])
        depth = gray_to_bgr_u8(imagery["depth"])

        ih = h - 86
        iw = (w - 40) // 3
        y0, x0 = y + 42, x + 10
        for i, im in enumerate([rgb, alpha, depth]):
            im = cv2.resize(im, (iw, ih))
            xx = x0 + i * (iw + 10)
            img[y0:y0 + ih, xx:xx + iw] = im
        draw_text(img, "RGB", (x0, y + h - 34), cfg.text_color, 0.45, 1)
        draw_text(img, "маска", (x0 + iw + 10, y + h - 34), cfg.text_color, 0.45, 1)
        draw_text(img, "глубина", (x0 + 2 * (iw + 10), y + h - 34), cfg.text_color, 0.45, 1)

        draw_text(img, f"форма={safe_int(imagery.get('shape_id'), -1)}", (x + 12, y + 25), cfg.accent3, 0.42, 1)
        draw_text(img, f"цвет={safe_int(imagery.get('color_id'), -1)}", (x + 120, y + 25), cfg.accent2, 0.42, 1)
        draw_text(img, f"уверенность={safe_scalar(imagery.get('object_confidence'), 0):.2f}", (x + 225, y + 25), cfg.accent1, 0.42, 1)

    def draw_imagination(self, img, out, x, y, w, h):
        cfg = self.cfg
        draw_block(img, x, y, w, h, "Воображаемые будущие", cfg)
        iv = to_np(out.get("imagined", {}).get("imagined_value")).reshape(-1)
        it = to_np(out.get("imagined", {}).get("imagined_touch")).reshape(-1)
        if iv.size == 0:
            draw_text(img, "нет воображения", (x + 12, y + 42), cfg.text_color, 0.5, 1)
            return
        vals = norm01(iv)
        touches = norm01(it) if it.size else np.zeros_like(vals)
        cx, cy = x + 110, y + h // 2 + 10
        for i, val in enumerate(vals):
            ang = -1.0 + 2.0 * i / max(1, len(vals) - 1)
            r = int(55 + 80 * float(val))
            px, py = int(cx + np.cos(ang) * r), int(cy + np.sin(ang) * r)
            col = (70, int(120 + 120 * float(val)), int(100 + 130 * float(touches[i] if i < touches.size else 0)))
            cv2.arrowedLine(img, (cx, cy), (px, py), col, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.circle(img, (px, py), 5, col, -1, cv2.LINE_AA)
            draw_text(img, f"{float(iv[i]):.2f}", (px + 7, py), col, 0.36, 1)

    def draw_text_thought(self, img, out, symbolic_report, x, y, w, h, predicted_text=None, target_text=None):
        cfg = self.cfg
        draw_block(img, x, y, w, h, "Читаемая внутренняя речь", cfg)

        if symbolic_report is None:
            symbolic_report = out.get("symbolic_report")
        if predicted_text is None:
            predicted_text = out.get("decoded_report") or self.decode_predicted_text(symbolic_report)
        if target_text is None:
            target_text = out.get("target_report") or ""

        conf = safe_scalar(symbolic_report.get("confidence") if symbolic_report else None, 0.0)
        match = simple_text_match(predicted_text, target_text)

        draw_text(img, f"уверенность речи={conf:.3f}", (x + 14, y + 32), cfg.accent5, 0.48, 1)
        draw_text(img, f"совпадение={match:.3f}", (x + 250, y + 32), cfg.accent2, 0.48, 1)

        draw_text(img, "МЫСЛЬ МОДЕЛИ:", (x + 14, y + 70), cfg.accent1, 0.58, 1)
        yy = y + 100
        for line in wrap_text(predicted_text, 64)[:4]:
            draw_text(img, line, (x + 24, yy), cfg.text_color, 0.56, 1)
            yy += 28

        draw_text(img, "ЦЕЛЕВАЯ МЫСЛЬ:", (x + 14, y + 220), cfg.accent3, 0.58, 1)
        yy = y + 250
        for line in wrap_text(target_text, 64)[:4]:
            draw_text(img, line, (x + 24, yy), cfg.accent3, 0.52, 1)
            yy += 25

        if symbolic_report:
            draw_code_line(img, symbolic_report.get("symbol_ids"), x + 18, y + h - 92, "S", cfg.accent5)
            draw_code_line(img, symbolic_report.get("phoneme_ids"), x + 18, y + h - 62, "P", cfg.accent2)
            draw_code_line(img, symbolic_report.get("text_token_ids"), x + 18, y + h - 32, "T", cfg.accent1)

    def decode_predicted_text(self, symbolic_report: Optional[Dict]) -> str:
        if not symbolic_report:
            return ""
        ids = symbolic_report.get("text_token_ids")
        if ids is None:
            return ""
        if torch.is_tensor(ids) and ids.ndim == 2:
            ids = ids[0]
        return self.speech_vocab.decode(ids, skip_special=True)

    def render(self, out: Dict, symbolic_report: Optional[Dict] = None, predicted_text: Optional[str] = None, target_text: Optional[str] = None) -> np.ndarray:
        cfg = self.cfg
        img = np.full((cfg.height, cfg.width, 3), cfg.bg_color, dtype=np.uint8)
        draw_text(img, cfg.title + " — визуализация внутреннего мира", (24, 18), cfg.text_color, 0.82, 2)

        self.draw_core(img, out, 20, 56, 720, 520)
        self.draw_attention(img, out, 760, 56, 300, 230)
        self.draw_memory(img, out, 1080, 56, 700, 230)
        self.draw_imagination(img, out, 760, 306, 500, 270)
        self.draw_object_image(img, out, 20, 596, 900, 470)
        self.draw_text_thought(img, out, symbolic_report, 940, 596, 840, 470, predicted_text, target_text)

        # compact stats panel
        draw_block(img, 1280, 306, 500, 270, "Сводка состояния", cfg)
        stats = [
            ("действие", safe_int(out.get("action_ids"), -1)),
            ("фокус", safe_int(out.get("focus", {}).get("focus_idx"), -1)),
            ("любопытство", safe_scalar(out.get("values", {}).get("curiosity"), 0)),
            ("связность", safe_scalar(out.get("values", {}).get("coherence"), 0)),
            ("уверенность Я", safe_scalar(out.get("reflection_out", {}).get("self_confidence"), 0)),
        ]
        yy = 350
        for k, v in stats:
            if isinstance(v, int):
                s = f"{k}: {v}"
            else:
                s = f"{k}: {v:.3f}"
            draw_text(img, s, (1300, yy), cfg.text_color, 0.48, 1)
            yy += 28
        return img

    def show(self, out: Dict, symbolic_report: Optional[Dict] = None, predicted_text: Optional[str] = None, target_text: Optional[str] = None, window_name: str = "dreamer inner world unicode", delay_ms: int = 1) -> int:
        frame = self.render(out, symbolic_report, predicted_text, target_text)
        submit_cv2_frame(window_name, frame, frame.shape[1], frame.shape[0])
        return int(get_cv2_last_key())

    def save(self, path: str, out: Dict, symbolic_report: Optional[Dict] = None, predicted_text: Optional[str] = None, target_text: Optional[str] = None) -> None:
        cv2.imwrite(path, self.render(out, symbolic_report, predicted_text, target_text))
