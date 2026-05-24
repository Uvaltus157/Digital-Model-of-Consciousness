from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_unicode_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


@lru_cache(maxsize=64)
def get_font(size: int = 18) -> ImageFont.ImageFont:
    path = _find_unicode_font()
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_text_unicode(
    img_bgr: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color_bgr: Tuple[int, int, int] = (235, 238, 245),
    size: int = 18,
    thickness: int = 1,
) -> None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    font = get_font(size)

    x, y = int(pos[0]), int(pos[1])
    offsets = [(0, 0)]
    if thickness > 1:
        offsets += [(1, 0), (0, 1)]
    if thickness > 2:
        offsets += [(-1, 0), (0, -1)]

    for dx, dy in offsets:
        draw.text((x + dx, y + dy), str(text), font=font, fill=color_rgb)

    img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def cv_scale_to_px(scale: float) -> int:
    return max(10, int(30 * float(scale)))


def draw_text(img_bgr: np.ndarray, text: str, pos: Tuple[int, int], color=(235, 238, 245), scale=0.5, thickness=1):
    draw_text_unicode(img_bgr, text, pos, color_bgr=color, size=cv_scale_to_px(scale), thickness=thickness)
