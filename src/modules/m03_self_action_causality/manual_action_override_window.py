
from __future__ import annotations

"""
manual_action_override_window.py

OpenCV slider window for manually overriding the neural body/rig action outputs:
    vx, vy, vz, yaw_rate, pitch_rate, roll_rate, head_yaw, head_pitch, head_roll

When enabled, it replaces embodied_targets[0:6] before DynamicAgentRigController
uses them. Arms/hands/legs can still be controlled by the neural model.
"""

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class ManualActionOverrideConfig:
    enabled: bool = True
    window_name: str = "manual body action override"
    max_linear: float = 1.0
    max_angular: float = 1.0
    slider_abs: int = 1000
    width: int = 760
    height: int = 480
    show_every_steps: int = 1


class ManualActionOverrideWindow:
    def __init__(self, cfg: ManualActionOverrideConfig | None = None):
        self.cfg = cfg or ManualActionOverrideConfig()
        self.created = False
        self.active = False
        self.values = {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0,
            "pitch_rate": 0.0,
            "roll_rate": 0.0,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
        }
        self.names = list(self.values.keys())

    def _trackbar_name(self, name: str) -> str:
        return name

    def _slider_to_value(self, slider: int, angular: bool = False) -> float:
        centered = (float(slider) - float(self.cfg.slider_abs)) / float(self.cfg.slider_abs)
        scale = self.cfg.max_angular if angular else self.cfg.max_linear
        return float(np.clip(centered * scale, -scale, scale))

    def _value_to_slider(self, value: float, angular: bool = False) -> int:
        scale = self.cfg.max_angular if angular else self.cfg.max_linear
        norm = float(np.clip(value / max(scale, 1e-6), -1.0, 1.0))
        return int(round((norm + 1.0) * self.cfg.slider_abs))

    def create(self):
        if self.created:
            return
        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.window_name, int(self.cfg.width), int(self.cfg.height))

        for name in self.names:
            angular = name in ("yaw_rate", "pitch_rate", "roll_rate", "head_yaw", "head_pitch", "head_roll")
            cv2.createTrackbar(
                self._trackbar_name(name),
                self.cfg.window_name,
                self._value_to_slider(0.0, angular=angular),
                self.cfg.slider_abs * 2,
                lambda _v: None,
            )
        self.created = True
        self.active = True

    def close(self):
        try:
            cv2.destroyWindow(self.cfg.window_name)
        except Exception:
            pass
        self.created = False
        self.active = False

    def read_values(self) -> Dict[str, float]:
        if not self.created:
            return self.values
        for name in self.names:
            try:
                slider = cv2.getTrackbarPos(self._trackbar_name(name), self.cfg.window_name)
            except Exception:
                slider = self.cfg.slider_abs
            angular = name in ("yaw_rate", "pitch_rate", "roll_rate", "head_yaw", "head_pitch", "head_roll")
            self.values[name] = self._slider_to_value(slider, angular=angular)
        return dict(self.values)

    def as_vector(self) -> np.ndarray:
        vals = self.read_values()
        return np.array([
            vals["vx"],
            vals["vy"],
            vals["vz"],
            vals["yaw_rate"],
            vals["pitch_rate"],
            vals["roll_rate"],
            vals["head_yaw"],
            vals["head_pitch"],
            vals["head_roll"],
        ], dtype=np.float32)

    def reset_sliders(self):
        self.create()
        for name in self.names:
            angular = name in ("yaw_rate", "pitch_rate", "roll_rate", "head_yaw", "head_pitch", "head_roll")
            cv2.setTrackbarPos(self._trackbar_name(name), self.cfg.window_name, self._value_to_slider(0.0, angular=angular))
        self.read_values()

    def draw(self, neural_vector=None, override_vector=None):
        self.create()
        vals = self.read_values()
        frame = np.zeros((int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
        frame[:] = (12, 16, 24)

        cv2.putText(frame, "manual action override: neural body output is intercepted", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "ESC/Q: close override | R: reset sliders", (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1, cv2.LINE_AA)

        y = 88
        for name in self.names:
            v = vals[name]
            cv2.putText(frame, f"{name:10s}: {v:+.3f}", (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 235, 245), 1, cv2.LINE_AA)
            x0, x1 = 220, int(self.cfg.width) - 40
            mid = (x0 + x1) // 2
            cv2.line(frame, (x0, y - 8), (x1, y - 8), (70, 80, 100), 1)
            cv2.line(frame, (mid, y - 18), (mid, y + 2), (120, 120, 140), 1)
            maxv = self.cfg.max_angular if name in ("yaw_rate", "pitch_rate", "roll_rate", "head_yaw", "head_pitch", "head_roll") else self.cfg.max_linear
            pos = int(mid + (x1 - x0) * 0.5 * (v / max(maxv, 1e-6)))
            color = (80, 230, 120) if v >= 0 else (100, 160, 255)
            cv2.rectangle(frame, (min(mid, pos), y - 15), (max(mid, pos), y - 2), color, -1)
            y += 42

        if neural_vector is not None:
            neural_vector = np.asarray(neural_vector).reshape(-1)
            text = "neural raw: " + np.array2string(neural_vector[:9], precision=2, suppress_small=True)
            cv2.putText(frame, text[:120], (14, int(self.cfg.height) - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1, cv2.LINE_AA)

        cv2.imshow(self.cfg.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()
        elif key == ord("r"):
            self.reset_sliders()
