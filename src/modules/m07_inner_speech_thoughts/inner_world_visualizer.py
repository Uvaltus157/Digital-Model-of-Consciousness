from __future__ import annotations

"""
inner_world_visualizer.py

Visualizer for the current DMoC inner-world pipeline.

Architecture naming rule:
    M5 shows preconscious workspace / candidate / model reflection / body
    context. It must not present those tensors as self-aware thought or
    autobiographical memory.

    True self-binding is produced by M9 SelfCore. True inner speech should be
    supplied by M7 after M9, as out["inner_speech"] or out["conscious_report"].
    Legacy out["symbolic_report"] is still accepted as a fallback for old runs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.modules.m07_inner_speech_thoughts.models.symbolic_report_language import decode_debug_tokens, DEFAULT_DEBUG_TOKENS_RU
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, get_cv2_last_key, close_cv2_window


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
    title: str = "DMoC M5 Preconscious Workspace"
    trail_len: int = 64
    debug_vocab: List[str] = field(default_factory=lambda: list(DEFAULT_DEBUG_TOKENS_RU))


def _to_np(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_dict(x) -> Dict:
    return x if isinstance(x, dict) else {}


def _first_tensor(*values):
    for value in values:
        if torch.is_tensor(value):
            return value
        if value is not None:
            arr = _to_np(value)
            if arr.size:
                return value
    return None


def _preconscious_thought(out: Dict):
    p = _as_dict(out.get("preconscious_thoughts"))
    legacy = _as_dict(out.get("thoughts"))
    return _first_tensor(p.get("thought_candidate"), p.get("candidate"), legacy.get("thought"))


def _preconscious_sequence(out: Dict):
    p = _as_dict(out.get("preconscious_thoughts"))
    legacy = _as_dict(out.get("thoughts"))
    return _first_tensor(p.get("candidate_sequence"), legacy.get("thought_sequence"))


def _preconscious_memory(out: Dict) -> Dict:
    return _as_dict(out.get("preconscious_memory") or out.get("memory"))


def _preconscious_reflection(out: Dict) -> Dict:
    return _as_dict(out.get("preconscious_reflection_out") or out.get("reflection_out"))


def _body_context(out: Dict):
    bc = _as_dict(out.get("body_context"))
    legacy = _as_dict(out.get("selves"))
    return _first_tensor(bc.get("body_context"), legacy.get("body_self"))


def _model_confidence(out: Dict):
    r = _preconscious_reflection(out)
    return _first_tensor(r.get("model_confidence"), r.get("self_confidence"))


def _inner_speech_report(out: Dict) -> Optional[Dict]:
    for key in ("inner_speech", "conscious_report", "m7_inner_speech", "symbolic_report"):
        value = out.get(key)
        if isinstance(value, dict):
            return value
    return None


def _vec2(x, scale: float = 1.0) -> np.ndarray:
    a = _to_np(x).astype(np.float32).reshape(-1)
    if a.size < 2:
        a = np.pad(a, (0, max(0, 2 - a.size)))
    return a[:2] * scale


def _proj(v: np.ndarray, cx: int, cy: int, scale: float) -> Tuple[int, int]:
    return int(cx + float(v[0]) * scale), int(cy - float(v[1]) * scale)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _safe_scalar(x, default=0.0) -> float:
    a = _to_np(x).reshape(-1)
    if a.size == 0:
        return float(default)
    return float(a[0])


def _safe_int(x, default=0) -> int:
    a = _to_np(x).reshape(-1)
    if a.size == 0:
        return int(default)
    return int(a[0])


def _norm_scalar(x) -> float:
    a = _to_np(x).astype(np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a))


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], color=(235, 238, 245), scale=0.5, thickness=1):
    cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_block(img: np.ndarray, x: int, y: int, w: int, h: int, title: str, cfg: InnerWorldVizConfig):
    cv2.rectangle(img, (x, y), (x + w, y + h), cfg.panel_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 70, 96), 1, cv2.LINE_AA)
    draw_text(img, title, (x + 12, y + 24), cfg.text_color, 0.6, 1)


def draw_node(img: np.ndarray, p: Tuple[int, int], r: int, color: Tuple[int, int, int], label: Optional[str] = None):
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p, r + 2, tuple(min(255, c + 40) for c in color), 1, cv2.LINE_AA)
    if label:
        draw_text(img, label, (p[0] + r + 6, p[1] - 4), color, 0.50, 1)


def draw_arrow(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2):
    cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.08)


def draw_trail(img: np.ndarray, pts: List[Tuple[int, int]], color: Tuple[int, int, int]):
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts) - 1)
        c = tuple(int(color[j] * alpha) for j in range(3))
        cv2.line(img, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


def draw_bar_chart(img: np.ndarray, values: np.ndarray, labels: List[str], rect: Tuple[int, int, int, int], cfg: InnerWorldVizConfig, color: Tuple[int, int, int], value_fmt: str = "{:.2f}"):
    x, y, w, h = rect
    values = _to_np(values).reshape(-1).astype(np.float32)
    if values.size == 0:
        draw_text(img, "no data", (x + 12, y + 30), cfg.text_color, 0.5, 1)
        return
    n = min(len(labels), values.size)
    vals = values[:n]
    vals01 = _norm01(vals if vals.size > 1 else np.array([0.0, vals[0] if vals.size else 0.0], dtype=np.float32))
    if vals.size == 1:
        vals01 = np.array([vals01[-1]], dtype=np.float32)
    bar_h = max(18, (h - 30) // max(1, n))
    for i in range(n):
        yy = y + 26 + i * bar_h
        xx0 = x + 110
        bw = int((w - 130) * float(vals01[i]))
        cv2.rectangle(img, (xx0, yy), (xx0 + bw, yy + bar_h - 8), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (xx0, yy), (x + w - 10, yy + bar_h - 8), (70, 80, 100), 1, cv2.LINE_AA)
        draw_text(img, labels[i], (x + 10, yy + bar_h - 12), cfg.text_color, 0.48, 1)
        draw_text(img, value_fmt.format(float(vals[i])), (x + w - 80, yy + bar_h - 12), color, 0.46, 1)


def draw_code_tokens(img: np.ndarray, ids: np.ndarray, rect: Tuple[int, int, int, int], cfg: InnerWorldVizConfig, token_prefix: str, color: Tuple[int, int, int], max_rows: int = 3):
    x, y, w, h = rect
    ids = _to_np(ids).astype(np.int64)
    if ids.size == 0:
        draw_text(img, "no codes", (x + 12, y + 28), cfg.text_color, 0.5, 1)
        return
    if ids.ndim == 1:
        ids = ids[None, :]
    ids = ids[:max_rows]
    yy = y + 28
    for r in range(ids.shape[0]):
        parts = [f"{token_prefix}{int(v)}" for v in ids[r].tolist()]
        draw_text(img, " ".join(parts[:12]), (x + 12, yy), color, 0.46, 1)
        yy += 22
        if yy > y + h - 10:
            break


class DreamerInnerWorldVisualizer:
    def __init__(self, cfg: Optional[InnerWorldVizConfig] = None) -> None:
        self.cfg = cfg or InnerWorldVizConfig()
        self.workspace_trail: List[Tuple[int, int]] = []
        self.memory_trail: List[Tuple[int, int]] = []
        self.thought_trail: List[Tuple[int, int]] = []

    def _update_trail(self, trail: List[Tuple[int, int]], p: Tuple[int, int]) -> None:
        trail.append(p)
        if len(trail) > self.cfg.trail_len:
            del trail[:-self.cfg.trail_len]

    def _draw_core_graph(self, canvas: np.ndarray, out: Dict, x: int, y: int, w: int, h: int):
        cfg = self.cfg
        draw_block(canvas, x, y, w, h, "M5 preconscious latent graph", cfg)

        workspace = out.get("workspace_out")
        candidate = _preconscious_thought(out)
        reflection = _preconscious_reflection(out).get("reflection")
        object_repr = out.get("object_repr")
        memory_context = _preconscious_memory(out).get("memory_context")
        body_context = _body_context(out)

        cx, cy = x + w // 2, y + h // 2 + 10
        p_workspace = _proj(_vec2(workspace), cx - 60, cy, 120)
        p_candidate = _proj(_vec2(candidate), cx + 180, cy - 120, 90)
        p_reflect = _proj(_vec2(reflection), cx + 120, cy + 140, 85)
        p_object = _proj(_vec2(object_repr), cx - 190, cy - 120, 85)
        p_memory = _proj(_vec2(memory_context), cx - 170, cy + 150, 85)
        p_body = _proj(_vec2(body_context), cx + 10, cy - 175, 75)

        self._update_trail(self.workspace_trail, p_workspace)
        self._update_trail(self.memory_trail, p_memory)
        self._update_trail(self.thought_trail, p_candidate)

        for a, b in [
            (p_workspace, p_candidate), (p_workspace, p_reflect), (p_workspace, p_object),
            (p_workspace, p_memory), (p_workspace, p_body), (p_memory, p_candidate),
            (p_object, p_candidate), (p_body, p_reflect),
        ]:
            draw_arrow(canvas, a, b, (75, 92, 130), 2)

        draw_trail(canvas, self.workspace_trail, cfg.accent3)
        draw_trail(canvas, self.memory_trail, cfg.accent2)
        draw_trail(canvas, self.thought_trail, cfg.accent1)

        draw_node(canvas, p_workspace, 26, cfg.accent3, "workspace")
        draw_node(canvas, p_candidate, 18, cfg.accent1, "preconscious_candidate")
        draw_node(canvas, p_reflect, 18, cfg.accent5, "model_reflection")
        draw_node(canvas, p_object, 18, cfg.accent4, "object_repr")
        draw_node(canvas, p_memory, 18, cfg.accent2, "preconscious_memory")
        draw_node(canvas, p_body, 16, (170, 210, 255), "body_context")

        coherence = _safe_scalar(_as_dict(out.get("values")).get("coherence"), 0.0)
        curiosity = _safe_scalar(_as_dict(out.get("values")).get("curiosity"), 0.0)
        model_conf = _safe_scalar(_model_confidence(out), 0.0)
        cv2.circle(canvas, p_workspace, int(50 + 40 * coherence), (58, 78, 148), 1, cv2.LINE_AA)
        cv2.circle(canvas, p_workspace, int(76 + 35 * curiosity), (48, 68, 118), 1, cv2.LINE_AA)
        draw_text(canvas, f"coherence={coherence:.3f}", (x + 20, y + h - 50), cfg.accent3, 0.5, 1)
        draw_text(canvas, f"curiosity={curiosity:.3f}", (x + 190, y + h - 50), cfg.accent2, 0.5, 1)
        draw_text(canvas, f"model_conf={model_conf:.3f}", (x + 350, y + h - 50), cfg.accent5, 0.5, 1)

        seq = _preconscious_sequence(out)
        if seq is not None:
            ts = _to_np(seq)
            if ts.ndim == 3:
                ts = ts[0]
            if ts.ndim >= 2:
                for i in range(min(ts.shape[0], 8)):
                    p = _proj(ts[i][:2], p_workspace[0], p_workspace[1], 42 + i * 18)
                    cv2.circle(canvas, p, 4, cfg.accent1, -1, cv2.LINE_AA)
                    cv2.line(canvas, p_workspace, p, (45, 90, 120), 1, cv2.LINE_AA)

    def _draw_attention(self, canvas: np.ndarray, out: Dict, x: int, y: int, w: int, h: int):
        draw_block(canvas, x, y, w, h, "M5 modality attention", self.cfg)
        weights = _as_dict(out.get("attention")).get("modality_weights")
        labels = ["vision", "pose", "body", "tactile", "motor", "object", "action"]
        if weights is not None:
            ww = _to_np(weights)
            if ww.ndim >= 2:
                ww = ww[0]
            draw_bar_chart(canvas, ww, labels, (x + 10, y + 18, w - 20, h - 28), self.cfg, self.cfg.accent1)
        else:
            draw_text(canvas, "attention not found", (x + 12, y + 34), self.cfg.text_color, 0.5, 1)

    def _draw_memory(self, canvas: np.ndarray, out: Dict, x: int, y: int, w: int, h: int):
        draw_block(canvas, x, y, w, h, "M5 preconscious episode memory", self.cfg)
        memory = _preconscious_memory(out)
        usage = memory.get("memory_usage")
        weights = memory.get("memory_weights")
        if usage is not None:
            u = _to_np(usage).reshape(-1)
            n_show = min(32, u.size)
            if n_show > 0:
                vals = u[:n_show]
                x0, y0 = x + 14, y + 42
                bw = max(3, (w - 30) // n_show)
                for i in range(n_show):
                    bar_h = int((h - 110) * float(np.clip(vals[i], 0.0, 1.0)))
                    cv2.rectangle(canvas, (x0 + i * bw, y + h - 24 - bar_h), (x0 + i * bw + bw - 3, y + h - 24), self.cfg.accent2, -1, cv2.LINE_AA)
                draw_text(canvas, "usage", (x + 14, y + h - 8), self.cfg.text_color, 0.45, 1)
        if weights is not None:
            wts = _to_np(weights)
            if wts.ndim == 2:
                wts = wts[0]
            top_idx = np.argsort(-wts)[:6]
            yy = y + 30
            draw_text(canvas, "top preconscious recalls:", (x + w - 215, yy), self.cfg.text_color, 0.48, 1)
            yy += 22
            for idx in top_idx.tolist():
                draw_text(canvas, f"slot {idx}: {float(wts[idx]):.3f}", (x + w - 170, yy), self.cfg.accent2, 0.45, 1)
                yy += 20

    def _draw_imagination(self, canvas: np.ndarray, out: Dict, x: int, y: int, w: int, h: int):
        draw_block(canvas, x, y, w, h, "M5 imagined action futures", self.cfg)
        iv = _as_dict(out.get("imagined")).get("imagined_value")
        it = _as_dict(out.get("imagined")).get("imagined_touch")
        if iv is None or it is None:
            draw_text(canvas, "imagined futures not found", (x + 12, y + 34), self.cfg.text_color, 0.5, 1)
            return
        iv = _to_np(iv); it = _to_np(it)
        if iv.ndim == 2: iv = iv[0]
        if it.ndim == 2: it = it[0]
        center = (x + 140, y + h // 2 + 20)
        vals01 = _norm01(iv); touch01 = _norm01(it)
        for i in range(len(iv)):
            ang = -1.1 + 2.2 * i / max(1, len(iv) - 1)
            radius = int(55 + 85 * float(vals01[i]))
            p = (int(center[0] + np.cos(ang) * radius), int(center[1] + np.sin(ang) * radius))
            col = (70, int(130 + 110 * float(vals01[i])), int(90 + 150 * float(touch01[i])))
            draw_arrow(canvas, center, p, col, 2)
            draw_node(canvas, p, 6, col)
            draw_text(canvas, f"v={float(iv[i]):.2f}", (p[0] + 6, p[1] - 4), col, 0.42, 1)
            draw_text(canvas, f"t={float(it[i]):.2f}", (p[0] + 6, p[1] + 12), col, 0.42, 1)
        draw_text(canvas, f"chosen action={_safe_int(out.get('action_ids'), -1)}", (x + 300, y + 50), self.cfg.accent1, 0.56, 1)
        draw_text(canvas, f"focus idx={_safe_int(_as_dict(out.get('focus')).get('focus_idx'), -1)}", (x + 300, y + 76), self.cfg.accent3, 0.56, 1)

    def _draw_symbolic_report(self, canvas: np.ndarray, symbolic_report: Optional[Dict], x: int, y: int, w: int, h: int):
        draw_block(canvas, x, y, w, h, "M7 inner speech / conscious report", self.cfg)
        if not symbolic_report:
            draw_text(canvas, "M7 report not provided yet", (x + 14, y + 36), self.cfg.text_color, 0.52, 1)
            draw_text(canvas, "M5 now outputs only preconscious_report; true inner speech comes after M9.", (x + 14, y + 62), self.cfg.accent1, 0.5, 1)
            return
        symbol_ids = symbolic_report.get("symbol_ids")
        phoneme_ids = symbolic_report.get("phoneme_ids")
        text_ids = symbolic_report.get("text_token_ids")
        conf = _safe_scalar(symbolic_report.get("confidence"), 0.0)
        draw_text(canvas, f"inner speech confidence={conf:.3f}", (x + 14, y + 34), self.cfg.accent5, 0.52, 1)
        draw_code_tokens(canvas, symbol_ids, (x + 10, y + 48, w - 20, 84), self.cfg, "S", self.cfg.accent3)
        draw_code_tokens(canvas, phoneme_ids, (x + 10, y + 136, w - 20, 84), self.cfg, "P", self.cfg.accent2)
        draw_code_tokens(canvas, text_ids, (x + 10, y + 224, w - 20, 84), self.cfg, "T", self.cfg.accent1)
        draw_text(canvas, "decoded debug text:", (x + 14, y + 328), self.cfg.text_color, 0.5, 1)
        try:
            lines = decode_debug_tokens(text_ids, self.cfg.debug_vocab, max_tokens=16) if text_ids is not None else []
        except Exception:
            lines = []
        if not lines:
            lines = ["<no decoded text>"]
        yy = y + 352
        for line in lines[:4]:
            draw_text(canvas, line, (x + 14, yy), self.cfg.accent4, 0.54, 1)
            yy += 24

    def _draw_stats(self, canvas: np.ndarray, out: Dict, x: int, y: int, w: int, h: int):
        draw_block(canvas, x, y, w, h, "State summary", self.cfg)
        stats = [
            ("workspace_norm", _norm_scalar(out.get("workspace_out"))),
            ("candidate_norm", _norm_scalar(_preconscious_thought(out))),
            ("model_reflection_norm", _norm_scalar(_preconscious_reflection(out).get("reflection"))),
            ("object_repr_norm", _norm_scalar(out.get("object_repr"))),
            ("preconscious_memory_norm", _norm_scalar(_preconscious_memory(out).get("memory_context"))),
            ("value_latent_norm", _norm_scalar(_as_dict(out.get("values")).get("value_latent"))),
            ("curiosity", _safe_scalar(_as_dict(out.get("values")).get("curiosity"), 0.0)),
            ("coherence", _safe_scalar(_as_dict(out.get("values")).get("coherence"), 0.0)),
            ("model_confidence", _safe_scalar(_model_confidence(out), 0.0)),
            ("action_id", float(_safe_int(out.get("action_ids"), -1))),
            ("focus_idx", float(_safe_int(_as_dict(out.get("focus")).get("focus_idx"), -1))),
        ]
        yy = y + 34
        for k, v in stats:
            draw_text(canvas, f"{k}: {v:.3f}" if not k.endswith("_id") and k != "focus_idx" else f"{k}: {int(v)}", (x + 12, yy), self.cfg.text_color, 0.5, 1)
            yy += 24

    def render(self, out: Dict, symbolic_report: Optional[Dict] = None) -> np.ndarray:
        cfg = self.cfg
        canvas = np.full((cfg.height, cfg.width, 3), cfg.bg_color, dtype=np.uint8)
        draw_text(canvas, cfg.title, (24, 34), cfg.text_color, 0.9, 2)
        if symbolic_report is None:
            symbolic_report = _inner_speech_report(out)
        self._draw_core_graph(canvas, out, 20, 56, 760, 560)
        self._draw_attention(canvas, out, 800, 56, 500, 250)
        self._draw_memory(canvas, out, 1320, 56, 460, 250)
        self._draw_imagination(canvas, out, 800, 326, 500, 290)
        self._draw_stats(canvas, out, 1320, 326, 460, 290)
        self._draw_symbolic_report(canvas, symbolic_report, 20, 636, 1760, 430)
        return canvas

    def show(self, out: Dict, symbolic_report: Optional[Dict] = None, window_name: str = "dreamer inner world", delay_ms: int = 1) -> int:
        frame = self.render(out, symbolic_report)
        submit_cv2_frame(window_name, frame, frame.shape[1], frame.shape[0])
        return int(get_cv2_last_key())

    def save(self, path: str, out: Dict, symbolic_report: Optional[Dict] = None) -> None:
        frame = self.render(out, symbolic_report)
        cv2.imwrite(path, frame)


def render_inner_world(out: Dict, symbolic_report: Optional[Dict] = None, cfg: Optional[InnerWorldVizConfig] = None) -> np.ndarray:
    viz = DreamerInnerWorldVisualizer(cfg)
    return viz.render(out, symbolic_report)


if __name__ == "__main__":
    cfg = InnerWorldVizConfig()
    viz = DreamerInnerWorldVisualizer(cfg)
    out = {
        "workspace_out": torch.randn(1, 256),
        "preconscious_thoughts": {
            "thought_candidate": torch.randn(1, 192),
            "candidate_sequence": torch.randn(1, 4, 192),
        },
        "preconscious_reflection_out": {
            "reflection": torch.randn(1, 192),
            "model_confidence": torch.tensor([[0.72]]),
        },
        "object_repr": torch.randn(1, 128),
        "preconscious_memory": {
            "memory_context": torch.randn(1, 256),
            "memory_weights": torch.softmax(torch.randn(1, 256), dim=-1),
            "memory_usage": torch.rand(256),
        },
        "attention": {"modality_weights": torch.softmax(torch.randn(1, 7), dim=-1)},
        "imagined": {"imagined_value": torch.randn(1, 5), "imagined_touch": torch.sigmoid(torch.randn(1, 5))},
        "values": {"value_latent": torch.randn(1, 128), "curiosity": torch.tensor([[0.58]]), "coherence": torch.tensor([[0.63]])},
        "body_context": {"body_context": torch.randn(1, 192)},
        "action_ids": torch.tensor([5]),
        "focus": {"focus_idx": torch.tensor([2])},
        "embodied_targets": torch.randn(1, 11),
        "hand_ctrl": torch.sigmoid(torch.randn(1, 34)),
    }
    frame = viz.render(out)
    submit_cv2_frame("dreamer inner world", frame, frame.shape[1], frame.shape[0])
    while True:
        key = int(get_cv2_last_key())
        if key in (27, ord("q")):
            break
    close_cv2_window("dreamer inner world")
