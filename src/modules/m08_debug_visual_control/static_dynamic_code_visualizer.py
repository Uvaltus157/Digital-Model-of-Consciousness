from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, get_cv2_last_key


@dataclass
class StaticDynamicCodeVisualizerConfig:
    enabled: bool = True
    window_name: str = "static/dynamic code heatmaps"
    width: int = 1500
    height: int = 900
    delay_ms: int = 1
    show_every_steps: int = 1
    heatmap_height: int = 120


def _as_vec(x: Any) -> Optional[np.ndarray]:
    try:
        if not torch.is_tensor(x):
            return None
        t = x.detach().float().cpu()
        if t.numel() == 0:
            return None
        if t.ndim > 1:
            t = t.reshape(-1, t.shape[-1])[0]
        return t.reshape(-1).numpy()
    except Exception:
        return None


def _scalar(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _txt(x: Any, n: int = 120) -> str:
    try:
        if x is None:
            return ""
        return str(x).replace("\n", " ").replace("\r", " ")[:n]
    except Exception:
        return ""


def _first_present(obj: Dict[str, Any], *keys: str) -> Any:
    """
    Return first non-None value without using Python `or`.

    Important:
        torch.Tensor cannot be used in boolean context.
        This bug made the cv2 window open but stay completely blank because
        draw() crashed before imshow().
    """
    for key in keys:
        try:
            if key in obj and obj[key] is not None:
                return obj[key]
        except Exception:
            pass
    return None


class StaticDynamicCodeVisualizer:
    """
    Shows the code ladder:

        z_static
            primary sensory proposal / static scene code

        z_dynamic
            object latent code in ObjectSlotMemory / DynamicObjectPassport

        scenario_z / inner_mind_z
            temporal rollout / imagined future

    The key view is z_static as a heatmap, so we can see what enters the
    dynamic object pipeline before it becomes a passport.
    """

    def __init__(self, cfg: Optional[StaticDynamicCodeVisualizerConfig] = None):
        self.cfg = cfg or StaticDynamicCodeVisualizerConfig()
        self.window_name = self.cfg.window_name
        self.created = False
        self._display_size = (int(self.cfg.width), int(self.cfg.height))

    def close(self) -> None:
        close_cv2_window(self.window_name)
        self.created = False

    def _ensure(self) -> None:
        # Window creation/resize happens in the dedicated OpenCV GUI thread.
        self.created = True

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        # Do not query OpenCV window geometry from the model/runtime thread.
        w, h = self._display_size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame

    def _heatmap(self, vec: Optional[np.ndarray], w: int, h: int, title: str, subtitle: str = "") -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (8, 12, 20)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (65, 85, 110), 1)
        cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 255), 1, cv2.LINE_AA)
        if subtitle:
            cv2.putText(img, subtitle[:150], (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (175, 200, 235), 1, cv2.LINE_AA)

        if vec is None or len(vec) == 0:
            cv2.putText(img, "no code", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 150, 180), 1, cv2.LINE_AA)
            return img

        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        v_clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v_norm = v_clean - np.min(v_clean)
        denom = np.max(v_norm) + 1e-6
        v_norm = v_norm / denom
        heat = (v_norm * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
        heat = cv2.resize(heat, (w - 24, max(24, h - 78)), interpolation=cv2.INTER_NEAREST)
        img[64:64 + heat.shape[0], 12:12 + heat.shape[1]] = heat

        stats = f"dim={len(vec)} min={float(np.min(v_clean)):.3f} max={float(np.max(v_clean)):.3f} mean={float(np.mean(v_clean)):.3f} norm={float(np.linalg.norm(v_clean)):.3f}"
        cv2.putText(img, stats, (12, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 245), 1, cv2.LINE_AA)
        return img

    def _info_panel(self, obj: Dict[str, Any], w: int, h: int) -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (8, 12, 20)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (65, 85, 110), 1)

        lines = [
            "Code ladder / current identity",
            "",
            "z_static:",
            "  primary sensory proposal before object memory",
            "  source: summarize_vision_tensors / proposal stream",
            "",
            "z_dynamic:",
            "  object latent after ObjectSlotMemory + passport binding",
            "  source: z_obj or passport_inner_world_z",
            "",
            "scenario_z:",
            "  z_dynamic unfolded through time",
            "  source: EventScenarioDecoder / NeuralEventDecoder / InnerScenarioMind",
            "",
            f"passport_token: {_txt(obj.get('passport_token'), 80)}",
            f"passport_slot: {_scalar(obj.get('passport_slot')):.0f}",
            f"passport_count: {_scalar(obj.get('passport_count')):.0f}",
            f"passport_source: {_txt(obj.get('passport_source'), 80)}",
            f"passport_similarity: {_scalar(obj.get('passport_similarity')):.4f}",
            f"passport_dynamic_score: {_scalar(obj.get('passport_dynamic_score')):.4f}",
            f"passport_debug_z_distance: {_scalar(obj.get('passport_debug_z_distance')):.4f}",
            f"passport_debug_z_cosine: {_scalar(obj.get('passport_debug_z_cosine')):.4f}",
            "",
            f"sentence: {_txt(obj.get('passport_sentence') or obj.get('semantic_sentence'), 150)}",
        ]

        y = 28
        for i, line in enumerate(lines):
            color = (245, 245, 255) if i == 0 else (205, 220, 240)
            if line.startswith("z_static") or line.startswith("z_dynamic") or line.startswith("scenario_z"):
                color = (255, 220, 140)
            if line.startswith("passport_token"):
                color = (200, 255, 190)
            cv2.putText(img, line[:165], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44 if i else 0.62, color, 1, cv2.LINE_AA)
            y += 23 if line else 10
            if y > h - 10:
                break
        return img

    def draw(self, obj: Dict[str, Any], global_step: int = 0, pump_events: bool = True) -> None:
        if not bool(self.cfg.enabled):
            return
        self._ensure()

        W, H = int(self.cfg.width), int(self.cfg.height)
        header_h = 58
        body_h = H - header_h
        col_w = W // 2
        row_h = max(80, body_h // 4)
        last_row_h = max(80, body_h - 3 * row_h)

        header = np.zeros((header_h, W, 3), dtype=np.uint8)
        header[:] = (5, 9, 15)
        cv2.putText(header, "Static/Dynamic Code Debug — z_static -> z_dynamic -> scenario_z", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 255), 1, cv2.LINE_AA)
        cv2.putText(header, f"step={int(global_step)} | z_static is sensor/proposal code; z_dynamic is object/passport code; scenario_z is imagined temporal code",
                    (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (175, 200, 235), 1, cv2.LINE_AA)

        z_static = _as_vec(_first_present(obj, "z_static", "static_sensor_code"))
        z_dynamic = _as_vec(_first_present(obj, "z_dynamic", "dynamic_object_code", "passport_inner_world_z", "z_obj"))
        z_obj = _as_vec(_first_present(obj, "z_obj"))
        z_passport = _as_vec(_first_present(obj, "passport_inner_world_z"))
        z_scenario = _as_vec(_first_present(obj, "scenario_z", "inner_mind_z", "scenario_code"))

        left = np.concatenate([
            self._heatmap(z_static, col_w, row_h, "z_static heatmap", "static sensory/proposal code before ObjectSlotMemory"),
            self._heatmap(z_dynamic, col_w, row_h, "z_dynamic heatmap", "dynamic object/passport code in first-order inner world"),
            self._heatmap(z_scenario, col_w, row_h, "scenario_z / inner_mind_z heatmap", "imagined temporal code / selected thought"),
            self._heatmap(z_passport, col_w, last_row_h, "passport_inner_world_z heatmap", "replay_z from DynamicObjectPassport"),
        ], axis=0)

        right = np.concatenate([
            self._heatmap(z_obj, col_w, row_h, "current z_obj heatmap", "ObjectSlotMemory active object latent"),
            self._info_panel(obj, col_w, body_h - row_h),
        ], axis=0)

        # Final guard against one-column height mismatch caused by resize rounding.
        if left.shape[0] != right.shape[0]:
            target_h = max(left.shape[0], right.shape[0])
            left = cv2.resize(left, (left.shape[1], target_h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (right.shape[1], target_h), interpolation=cv2.INTER_AREA)

        frame = np.concatenate([header, np.concatenate([left, right], axis=1)], axis=0)
        frame = self._fit(frame)
        submit_cv2_frame(self.window_name, frame, int(self.cfg.width), int(self.cfg.height))
        return
