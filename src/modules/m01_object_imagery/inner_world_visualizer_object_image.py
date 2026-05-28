from __future__ import annotations


if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Dict, Optional

import cv2
import numpy as np
import torch

from src.modules.m07_inner_speech_thoughts.inner_world_visualizer import (
    DreamerInnerWorldVisualizer,
    draw_block,
    draw_text,
    _safe_int,
    _safe_scalar,
)
from src.modules.m01_object_imagery.models.object_imagery_decoder import blended_object_image


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


class DreamerInnerWorldVisualizerV2(DreamerInnerWorldVisualizer):
    def _draw_object_imagery(self, canvas: np.ndarray, out: Dict, x: int, y: int, w: int, h: int):
        cfg = self.cfg
        draw_block(canvas, x, y, w, h, "Internal object image", cfg)
        imagery = out.get("object_imagery")
        if not imagery:
            draw_text(canvas, "object_imagery not found", (x + 14, y + 36), cfg.text_color, 0.55, 1)
            draw_text(canvas, "Use ConsciousDreamer object imagery", (x + 14, y + 62), cfg.accent1, 0.5, 1)
            return

        comp = blended_object_image(imagery["rgb"], imagery["alpha"])
        rgb_bgr = chw_to_bgr_u8(comp)
        alpha_bgr = gray_to_bgr_u8(imagery["alpha"])
        depth_bgr = gray_to_bgr_u8(imagery["depth"])

        img_h = h - 90
        img_w = (w - 40) // 3
        rgb_bgr = cv2.resize(rgb_bgr, (img_w, img_h))
        alpha_bgr = cv2.resize(alpha_bgr, (img_w, img_h))
        depth_bgr = cv2.resize(depth_bgr, (img_w, img_h))

        x0 = x + 10
        y0 = y + 34
        canvas[y0:y0 + img_h, x0:x0 + img_w] = rgb_bgr
        canvas[y0:y0 + img_h, x0 + img_w + 10:x0 + 2 * img_w + 10] = alpha_bgr
        canvas[y0:y0 + img_h, x0 + 2 * img_w + 20:x0 + 3 * img_w + 20] = depth_bgr

        draw_text(canvas, "RGB", (x0, y + h - 40), cfg.text_color, 0.5, 1)
        draw_text(canvas, "ALPHA", (x0 + img_w + 10, y + h - 40), cfg.text_color, 0.5, 1)
        draw_text(canvas, "DEPTH", (x0 + 2 * img_w + 20, y + h - 40), cfg.text_color, 0.5, 1)

        draw_text(canvas, f"shape_id={_safe_int(imagery.get('shape_id'), -1)}", (x + 12, y + 24), cfg.accent3, 0.48, 1)
        draw_text(canvas, f"color_id={_safe_int(imagery.get('color_id'), -1)}", (x + 140, y + 24), cfg.accent2, 0.48, 1)
        draw_text(canvas, f"material_id={_safe_int(imagery.get('material_id'), -1)}", (x + 270, y + 24), cfg.accent5, 0.48, 1)
        draw_text(canvas, f"obj_conf={_safe_scalar(imagery.get('object_confidence'), 0.0):.3f}", (x + 440, y + 24), cfg.accent1, 0.48, 1)

    def render(self, out: Dict, symbolic_report: Optional[Dict] = None) -> np.ndarray:
        cfg = self.cfg
        canvas = np.full((cfg.height, cfg.width, 3), cfg.bg_color, dtype=np.uint8)

        draw_text(canvas, cfg.title + " + Object Imagery", (24, 34), cfg.text_color, 0.9, 2)
        self._draw_core_graph(canvas, out, 20, 56, 720, 520)
        self._draw_attention(canvas, out, 760, 56, 300, 230)
        self._draw_memory(canvas, out, 1080, 56, 700, 230)
        self._draw_imagination(canvas, out, 760, 306, 500, 270)
        self._draw_stats(canvas, out, 1280, 306, 500, 270)
        self._draw_object_imagery(canvas, out, 20, 596, 900, 470)
        self._draw_symbolic_report(canvas, symbolic_report, 940, 596, 840, 470)

        return canvas
