
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window


SHAPE_NAMES = ["unknown", "sphere", "box", "cylinder"]


def _tensor_rgb_to_bgr_u8(x: torch.Tensor, size: int = 192) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def _tensor_gray_colormap(x: torch.Tensor, size: int = 192) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / max(float(arr.max()), 1e-6)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = cv2.applyColorMap(arr, cv2.COLORMAP_TURBO)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def _bar_panel(title: str, values, width=420, height=160, color=(80, 220, 120)) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (12, 16, 24)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 90, 110), 1)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return panel
    vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
    vals = np.clip(vals, 0.0, None)
    vmax = max(float(vals.max()), 1e-6)

    left, right, top, bottom = 10, width - 10, 32, height - 14
    slot = max(4, int((right - left) / max(1, len(vals))))
    bw = max(3, int(slot * 0.68))
    for i, v in enumerate(vals):
        x0 = left + i * slot
        x1 = min(x0 + bw, right - 1)
        h = int((bottom - top) * float(v / vmax))
        cv2.rectangle(panel, (x0, bottom - h), (x1, bottom), color, -1)
    return panel


def _draw_axis(panel, title):
    h, w = panel.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (70, 90, 110), 1)
    cv2.line(panel, (cx, 12), (cx, h - 12), (60, 80, 110), 1, cv2.LINE_AA)
    cv2.line(panel, (12, cy), (w - 12, cy), (60, 80, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)


def _render_point_cloud_views(points: np.ndarray, conf: Optional[np.ndarray], size=220):
    if conf is None:
        conf = np.ones((points.shape[0], 1), dtype=np.float32)
    conf = np.clip(conf.reshape(-1), 0.0, 1.0)
    views = []
    specs = [
        ("point cloud XY", 0, 1, (80, 220, 120)),
        ("point cloud XZ", 0, 2, (70, 170, 255)),
        ("point cloud YZ", 1, 2, (255, 180, 80)),
    ]
    for title, ax1, ax2, base_color in specs:
        panel = np.zeros((size, size, 3), dtype=np.uint8)
        panel[:] = (10, 14, 22)
        _draw_axis(panel, title)
        h, w = panel.shape[:2]
        cx, cy = w // 2, h // 2
        s = min(w, h) * 0.38
        for p, c in zip(points, conf):
            x = int(cx + float(p[ax1]) * s)
            y = int(cy - float(p[ax2]) * s)
            col = tuple(min(255, int(v * (0.45 + 0.55 * float(c)))) for v in base_color)
            cv2.circle(panel, (x, y), max(1, int(1 + 2 * float(c))), col, -1, cv2.LINE_AA)
        views.append(panel)
    return views


def _render_voxel_projections(vox: np.ndarray, size=220):
    # vox shape [R,R,R]
    projs = [
        ("voxel max XY", vox.max(axis=2)),
        ("voxel max XZ", vox.max(axis=1)),
        ("voxel max YZ", vox.max(axis=0)),
    ]
    out = []
    for title, img in projs:
        img = img.astype(np.float32)
        img = img - img.min()
        img = img / max(float(img.max()), 1e-6)
        img = (img * 255.0).astype(np.uint8)
        panel = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        panel = cv2.resize(panel, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, (0, 0), (size - 1, size - 1), (70, 90, 110), 1)
        cv2.rectangle(panel, (4, 4), (150, 24), (20, 20, 20), -1)
        cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        out.append(panel)
    return out


class InnerObject3DVisualizer:
    def __init__(self, window_name: str = "inner object imagery 3D", width: int = 1420, height: int = 980):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.created = False

    def close(self):
        close_cv2_window(self.window_name)
        self.created = False

    def _ensure(self):
        self.created = True

    def draw(self, obj: Dict[str, torch.Tensor], obs: Optional[Dict] = None, tactile_values=None) -> None:
        self._ensure()

        rgb = _tensor_rgb_to_bgr_u8(obj["rgb"], size=200)
        depth = _tensor_gray_colormap(obj["depth"], size=200)
        mask = _tensor_gray_colormap(obj["mask"], size=200)

        top_left = np.concatenate([rgb, depth, mask], axis=1)
        labels = ["decoded internal RGB", "decoded depth", "decoded mask"]
        x = 6
        for lab in labels:
            cv2.rectangle(top_left, (x, 6), (x + 180, 28), (20, 20, 20), -1)
            cv2.putText(top_left, lab, (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
            x += 200

        shape_probs = torch.softmax(obj["shape_logits"][0], dim=-1).detach().cpu().numpy()
        shape_idx = int(np.argmax(shape_probs))
        color = obj["color_rgb"][0].detach().cpu().numpy()
        confidence = float(obj["confidence"][0, 0].detach().cpu().item())
        vision_strength = float(obj["vision_strength"][0, 0].detach().cpu().item())
        touch_strength = float(obj["touch_strength"][0, 0].detach().cpu().item())
        update_gate = float(obj["update_gate_mean"][0, 0].detach().cpu().item())
        size_v = float(obj["size"][0, 0].detach().cpu().item())
        hardness = float(obj["hardness"][0, 0].detach().cpu().item())
        stability = float(obj["stability"][0, 0].detach().cpu().item())
        novelty = float(obj["novelty"][0, 0].detach().cpu().item())

        pts = obj["point_cloud"][0].detach().cpu().numpy()
        pc_conf = obj["point_conf"][0].detach().cpu().numpy() if "point_conf" in obj else None
        vox = obj["voxel_occ"][0, 0].detach().cpu().numpy()

        info = np.zeros((200, 580, 3), dtype=np.uint8)
        info[:] = (10, 14, 22)
        lines = [
            "forming internal object image + internal 3D model",
            f"shape: {SHAPE_NAMES[shape_idx]}  probs={np.round(shape_probs, 2)}",
            f"confidence: {confidence:.3f}",
            f"vision contribution: {vision_strength:.3f}",
            f"touch contribution: {touch_strength:.3f}",
            f"slot update gate: {update_gate:.3f}",
            f"size: {size_v:.3f}  hardness: {hardness:.3f}",
            f"stability: {stability:.3f}  novelty: {novelty:.3f}",
            f"point cloud: {pts.shape[0]} pts  voxel res: {vox.shape[0]}^3",
            f"internal color rgb: {np.round(color, 2)}",
        ]
        y = 24
        for line in lines:
            cv2.putText(info, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 245), 1, cv2.LINE_AA)
            y += 18

        color_box = np.zeros((48, 96, 3), dtype=np.uint8)
        color_box[:] = tuple(int(c * 255) for c in color[::-1])
        info[145:193, 465:561] = color_box

        top_row = np.concatenate([top_left, info], axis=1)

        pc_views = np.concatenate(_render_point_cloud_views(pts, pc_conf, size=220), axis=1)
        vox_views = np.concatenate(_render_voxel_projections(vox, size=220), axis=1)
        middle_row = np.concatenate([pc_views, vox_views], axis=1)

        if obs is not None and "left" in obs and "right" in obs:
            left = _tensor_rgb_to_bgr_u8(obs["left"], size=180)
            right = _tensor_rgb_to_bgr_u8(obs["right"], size=180)
            d = _tensor_gray_colormap(obs["depth"], size=180) if "depth" in obs else np.zeros_like(left)
            obs_row = np.concatenate([left, right, d], axis=1)
            focus_applied = False
            try:
                focus_applied = bool(obs.get("depth_focus_applied", False))
            except Exception:
                focus_applied = False
            olabs = ["left", "right", "depth_input:focus" if focus_applied else "depth_input:raw"]
            ox = 6
            for lab in olabs:
                label_w = max(80, min(172, 10 + len(str(lab)) * 8))
                cv2.rectangle(obs_row, (ox, 6), (ox + label_w, 24), (20, 20, 20), -1)
                cv2.putText(obs_row, lab, (ox + 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                ox += 180
        else:
            obs_row = np.zeros((180, 540, 3), dtype=np.uint8)
            obs_row[:] = (12, 16, 24)

        if tactile_values is None:
            tactile_values = []
        bars = _bar_panel("tactile contacts feeding object slot", tactile_values, width=780, height=180)
        bottom_row = np.concatenate([obs_row, bars], axis=1)

        total_w = max(top_row.shape[1], middle_row.shape[1], bottom_row.shape[1])
        def pad_to(img):
            if img.shape[1] < total_w:
                return cv2.copyMakeBorder(img, 0, 0, 0, total_w - img.shape[1], cv2.BORDER_CONSTANT, value=(8, 8, 8))
            return img

        top_row = pad_to(top_row)
        middle_row = pad_to(middle_row)
        bottom_row = pad_to(bottom_row)

        header = np.zeros((40, total_w, 3), dtype=np.uint8)
        header[:] = (6, 10, 16)
        cv2.putText(header, "inner object visualizer: vision + touch -> object slot -> 2D imagery + canonical internal 3D model", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (240, 240, 240), 1, cv2.LINE_AA)

        frame = np.concatenate([header, top_row, middle_row, bottom_row], axis=0)
        submit_cv2_frame(self.window_name, frame, int(self.width), int(self.height))
