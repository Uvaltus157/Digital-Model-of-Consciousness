from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, get_cv2_last_key


@dataclass
class PassportDebugVisualizerConfig:
    enabled: bool = True
    window_name: str = "passport debug: live vs replay"
    width: int = 1560
    height: int = 980
    delay_ms: int = 1
    show_every_steps: int = 1


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
        s = str(x or "")
    except Exception:
        s = ""
    return s.replace("\n", " ").replace("\r", " ")[:n]


def _to_np_img(x: Any, size: Tuple[int, int], mode: str = "rgb") -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if x is None or not torch.is_tensor(x):
        cv2.putText(img, "none", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 140, 170), 1, cv2.LINE_AA)
        return img

    t = x.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    elif t.ndim == 2:
        t = t.unsqueeze(-1)

    arr = t.numpy()
    if arr.ndim == 2:
        arr = arr[..., None]

    if mode == "rgb":
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr[..., :3]
        # Accept either 0..1 or arbitrary normalized values.
        if arr.max() <= 1.5 and arr.min() >= -0.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = arr - arr.min()
            arr = arr / (arr.max() + 1e-6) * 255.0
        arr = arr.astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        a = arr[..., 0]
        a = a - np.nanmin(a)
        a = a / (np.nanmax(a) + 1e-6)
        arr = (a * 255.0).astype(np.uint8)
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_VIRIDIS)

    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)


class PassportDebugVisualizer:
    """
    Debugger for the key question:

        Does DynamicObjectPassport reproduce the same internal object?

    It shows:
        live z_obj decode
        passport replay z decode
        difference metrics
        passport state/passport sentence
    """

    def __init__(self, cfg: Optional[PassportDebugVisualizerConfig] = None):
        self.cfg = cfg or PassportDebugVisualizerConfig()
        self.window_name = self.cfg.window_name
        self.created = False
        self._display_size = (int(self.cfg.width), int(self.cfg.height))

    def close(self) -> None:
        close_cv2_window(self.window_name)
        self.created = False

    def _ensure(self) -> None:
        # Window creation/resize is owned by src/platform/gui/opencv_gui_thread.py.
        self.created = True

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        # Do not query OpenCV window geometry from the model/runtime thread.
        w, h = self._display_size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame

    def _metric_panel(self, obj: Dict[str, Any], live: Dict[str, Any], replay: Dict[str, Any], w: int, h: int) -> np.ndarray:
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (8, 12, 20)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (65, 85, 110), 1)

        z_live = live.get("z_obj") or obj.get("z_obj")
        z_replay = obj.get("passport_inner_world_z")
        z_dist = 0.0
        z_cos = 0.0
        try:
            if torch.is_tensor(z_live) and torch.is_tensor(z_replay):
                a = z_live.detach().float().reshape(1, -1)
                b = z_replay.detach().float().to(a.device).reshape(1, -1)
                d = min(a.shape[-1], b.shape[-1])
                a = a[:, :d]
                b = b[:, :d]
                z_dist = float((a - b).norm(dim=-1).mean().cpu().item())
                z_cos = float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().cpu().item())
        except Exception:
            pass

        # Approx image diff if both decoded.
        rgb_diff = 0.0
        try:
            lr = live.get("rgb")
            rr = replay.get("rgb")
            if torch.is_tensor(lr) and torch.is_tensor(rr):
                rgb_diff = float((lr.detach().float().cpu() - rr.detach().float().cpu()).abs().mean().item())
        except Exception:
            pass

        lines = [
            "DynamicObjectPassport debug",
            f"passport_active: {_scalar(obj.get('passport_active')) > 0.5}",
            f"passport_token: {_txt(obj.get('passport_token'), 60)}",
            f"passport_slot: {_scalar(obj.get('passport_slot')):.0f}",
            f"passport_count: {_scalar(obj.get('passport_count')):.0f}",
            f"created_this_step: {_scalar(obj.get('passport_created')) > 0.5}",
            f"similarity: {_scalar(obj.get('passport_similarity')):.4f}",
            f"dynamic_score: {_scalar(obj.get('passport_dynamic_score')):.4f}",
            f"confidence_ema: {_scalar(obj.get('passport_confidence_ema')):.4f}",
            f"source: {_txt(obj.get('passport_source'), 60)}",
            "",
            "Live vs replay:",
            f"z_distance: {z_dist:.5f}",
            f"z_cosine: {z_cos:.5f}",
            f"rgb_abs_diff: {rgb_diff:.5f}",
            f"replay_active: {_scalar(obj.get('passport_replay_active')) > 0.5}",
            f"second_order_decoded: {_scalar(obj.get('passport_second_order_decoded')) > 0.5}",
            "",
            "sentence:",
            _txt(obj.get("passport_sentence") or obj.get("passport_replay_sentence"), 160),
            "",
            "episode:",
            _txt(obj.get("passport_episode_summary") or obj.get("passport_replay_episode_summary"), 160),
        ]

        y = 30
        for i, line in enumerate(lines):
            color = (245, 245, 255) if i == 0 else (205, 220, 240)
            if "z_distance" in line or "z_cosine" in line or "rgb_abs_diff" in line:
                color = (255, 220, 140)
            if "passport_token" in line:
                color = (210, 255, 190)
            cv2.putText(panel, line[:150], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44 if i else 0.62, color, 1, cv2.LINE_AA)
            y += 24 if line else 12
            if y > h - 12:
                break
        return panel

    def draw(self, obj: Dict[str, Any], live_decoded: Dict[str, Any], replay_decoded: Dict[str, Any], global_step: int = 0) -> None:
        if not bool(self.cfg.enabled):
            return
        self._ensure()

        W, H = int(self.cfg.width), int(self.cfg.height)
        header_h = 58
        panel_h = H - header_h
        col_w = W // 3
        img_h = panel_h // 3

        header = np.zeros((header_h, W, 3), dtype=np.uint8)
        header[:] = (5, 9, 15)
        cv2.putText(header, "Passport Debug — live object vs DynamicObjectPassport replay", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 255), 1, cv2.LINE_AA)
        cv2.putText(header, f"step={int(global_step)} | SLOT_N = storage | OBJ_NNN/passport = semantic identity | replay_z = first-order inner world",
                    (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (175, 200, 235), 1, cv2.LINE_AA)

        live_rgb = _to_np_img(live_decoded.get("rgb"), (col_w, img_h), "rgb")
        live_depth = _to_np_img(live_decoded.get("depth"), (col_w, img_h), "depth")
        live_mask = _to_np_img(live_decoded.get("mask"), (col_w, img_h), "depth")

        rep_rgb = _to_np_img(replay_decoded.get("rgb"), (col_w, img_h), "rgb")
        rep_depth = _to_np_img(replay_decoded.get("depth"), (col_w, img_h), "depth")
        rep_mask = _to_np_img(replay_decoded.get("mask"), (col_w, img_h), "depth")

        # Difference views
        diff_rgb = np.zeros_like(live_rgb)
        try:
            diff_rgb = cv2.absdiff(live_rgb, rep_rgb)
        except Exception:
            pass
        diff_depth = np.zeros_like(live_depth)
        try:
            diff_depth = cv2.absdiff(live_depth, rep_depth)
        except Exception:
            pass
        metrics = self._metric_panel(obj, live_decoded, replay_decoded, col_w, panel_h)

        def label(img: np.ndarray, text: str) -> np.ndarray:
            out = img.copy()
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 25), (0, 0, 0), -1)
            cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 255), 1, cv2.LINE_AA)
            return out

        left = np.concatenate([
            label(live_rgb, "LIVE decode: current z_obj RGB"),
            label(live_depth, "LIVE decode: depth"),
            label(live_mask, "LIVE decode: mask"),
        ], axis=0)
        mid = np.concatenate([
            label(rep_rgb, "PASSPORT replay decode: replay_z RGB"),
            label(rep_depth, "PASSPORT replay decode: depth"),
            label(rep_mask, "PASSPORT replay decode: mask"),
        ], axis=0)
        right = np.concatenate([
            label(diff_rgb, "ABS DIFF: RGB"),
            label(diff_depth, "ABS DIFF: depth"),
            metrics[:panel_h - 2 * img_h],
        ], axis=0)

        # Normalize column heights.
        def fit_h(x: np.ndarray, h: int) -> np.ndarray:
            if x.shape[0] == h:
                return x
            return cv2.resize(x, (x.shape[1], h), interpolation=cv2.INTER_AREA)

        left = fit_h(left, panel_h)
        mid = fit_h(mid, panel_h)
        right = fit_h(right, panel_h)

        frame = np.concatenate([header, np.concatenate([left, mid, right], axis=1)], axis=0)
        frame = self._fit(frame)
        submit_cv2_frame(self.window_name, frame, int(self.cfg.width), int(self.cfg.height))
        key = int(get_cv2_last_key())
        if key in (27, ord("q")):
            self.close()
