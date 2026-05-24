from __future__ import annotations

"""
inner_world_visualizer_text_thought_hybrid.py

Hybrid fix:
- keeps the original graph/layout implementation from inner_world_visualizer_text_thought.py
- replaces ONLY text drawing with a Unicode-safe PIL drawer
- avoids rewriting graph geometry, bar charts, arrows, nodes, panels

Use this if the fully rewritten Unicode visualizer changed/spoiled the charts.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Tuple
import numpy as np

from src.modules.m07_inner_speech_thoughts.unicode_text_draw import draw_text as _unicode_draw_text
import src.modules.m07_inner_speech_thoughts.inner_world_visualizer as _base_viz
import src.modules.m01_object_imagery.inner_world_visualizer_object_image as _v2
import src.modules.m07_inner_speech_thoughts.inner_world_visualizer_text_thought as _v3


def draw_text_unicode_safe(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color=(235, 238, 245),
    scale=0.5,
    thickness=1,
):
    _unicode_draw_text(img, str(text), pos, color=color, scale=scale, thickness=thickness)


# Monkey-patch draw_text functions while preserving existing graph rendering code.
_base_viz.draw_text = draw_text_unicode_safe
_v2.draw_text = draw_text_unicode_safe
_v3.draw_text = draw_text_unicode_safe

DreamerInnerWorldVisualizerV3 = _v3.DreamerInnerWorldVisualizerV3
wrap_text = _v3.wrap_text
simple_text_match = _v3.simple_text_match
