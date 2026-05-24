from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, consume_cv2_key


SHAPE_NAMES = ["unknown", "sphere", "box", "cylinder"]


@dataclass
class InnerObjectVisualizerV2Config:
    window_name: str = "inner object imagery v2"
    width: int = 1520
    height: int = 1260
    history_len: int = 240
    panel_size: int = 220
    delay_ms: int = 1
    max_slots: int = 10


def _to_numpy(x, default=None) -> np.ndarray:
    if x is None:
        return np.asarray(default if default is not None else [], dtype=np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _scalar(obj: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        arr = _to_numpy(obj.get(key)).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(np.nan_to_num(arr[0], nan=default, posinf=default, neginf=default))
    except Exception:
        return float(default)


def _vector(obj: Dict[str, Any], key: str, default_len: int = 0) -> np.ndarray:
    try:
        x = obj.get(key)
        if x is None:
            return np.zeros(default_len, dtype=np.float32)
        return _to_numpy(x).reshape(-1).astype(np.float32)
    except Exception:
        return np.zeros(default_len, dtype=np.float32)


def _tensor_rgb_to_bgr_u8(x: torch.Tensor | np.ndarray, size: int = 192) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.size == 0:
        arr = np.zeros((size, size, 3), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3:
        arr = np.zeros((size, size, 3), dtype=np.float32)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def _tensor_gray_colormap(
    x: torch.Tensor | np.ndarray,
    size: int = 192,
    normalize: bool = True,
    focus_depth: float | None = None,
    focus_half_range: float | None = None,
) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0)).mean(axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.mean(axis=-1)
    if arr.size == 0:
        arr = np.zeros((size, size), dtype=np.float32)
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        valid = np.isfinite(arr) & (arr > 1e-6)
        if focus_depth is not None and np.isfinite(float(focus_depth)) and bool(valid.any()):
            half = float(focus_half_range) if focus_half_range is not None else 0.85
            half = max(0.10, half)
            lo = max(0.0, float(focus_depth) - half)
            hi = float(focus_depth) + half
        else:
            values = arr[valid] if bool(valid.any()) else arr.reshape(-1)
            lo = float(np.percentile(values, 1.0)) if values.size else 0.0
            hi = float(np.percentile(values, 99.0)) if values.size else 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if abs(hi - lo) < 1e-6:
            lo = float(arr.min()) if arr.size else 0.0
            hi = float(arr.max()) if arr.size else 1.0
        arr = np.zeros_like(arr, dtype=np.float32) if abs(hi - lo) < 1e-6 else (arr - lo) / (hi - lo)
        if focus_depth is not None:
            arr[~valid] = 1.0
    arr = np.clip(arr, 0.0, 1.0)
    if focus_depth is not None:
        arr = 1.0 - arr
    arr = (arr * 255.0).astype(np.uint8)
    arr = cv2.applyColorMap(arr, cv2.COLORMAP_TURBO)
    if focus_depth is not None:
        try:
            edges = cv2.Canny(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), 35, 110)
            arr[edges > 0] = (255, 255, 255)
        except Exception:
            pass
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def _depth_focus_from_obs(obs: Optional[Dict]) -> Tuple[float | None, float | None, str]:
    if not isinstance(obs, dict):
        return None, None, ""
    try:
        focus_depth = float(obs.get("depth_focus_depth", 0.0))
        if not np.isfinite(focus_depth) or focus_depth <= 0.0:
            return None, None, ""
        half_range = float(obs.get("depth_focus_half_range", 0.85))
        return focus_depth, half_range, str(obs.get("depth_focus_label", ""))
    except Exception:
        return None, None, ""


def _put_label(img: np.ndarray, text: str, x: int = 8, y: int = 24):
    cv2.rectangle(img, (x - 4, y - 18), (x + max(170, len(text) * 8), y + 6), (8, 12, 18), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 245, 255), 1, cv2.LINE_AA)


def _safe_mask_stats(mask_tensor) -> Tuple[float, float, float]:
    try:
        m = _to_numpy(mask_tensor)
        if m.ndim == 4:
            m = m[0]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        m = np.clip(m.astype(np.float32), 0.0, 1.0)
        mean = float(m.mean())
        area = float((m > 0.35).mean())
        entropy = float(-(m * np.log(m + 1e-6) + (1 - m) * np.log(1 - m + 1e-6)).mean())
        return mean, area, entropy
    except Exception:
        return 0.0, 0.0, 0.0


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1.0 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,         1.0 - (xx+zz), yz - wx],
        [xz - wy,         yz + wx,       1.0 - (xx+yy)],
    ], dtype=np.float64)


def _estimate_object_visibility_from_obs(obs: Optional[Dict]) -> Tuple[bool, float, float, str]:
    try:
        if obs is None or "pose" not in obs or "object_state" not in obs:
            return False, -1.0, 0.0, "visibility: no pose/object_state -> not seeing"
        pose = _to_numpy(obs["pose"]).reshape(-1)
        obj = _to_numpy(obs["object_state"]).reshape(-1)
        if pose.size < 7 or obj.size < 3:
            return False, -1.0, 0.0, "visibility: malformed obs -> not seeing"
        cam_pos = pose[:3].astype(np.float64)
        quat = pose[3:7].astype(np.float64)
        rot = _quat_to_rotmat(quat)
        forward = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        pts = obj.reshape(-1, 3).astype(np.float64)
        best_cos, best_dist = -1.0, 1e9
        for pt in pts:
            vec = pt - cam_pos
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                continue
            cosang = float(np.dot(forward, vec / dist))
            if cosang > best_cos:
                best_cos, best_dist = cosang, dist
        visible = bool(best_cos > 0.58 and best_dist < 12.0)
        return visible, best_cos, best_dist, f"visibility_est={visible} cos={best_cos:.3f} dist={best_dist:.2f}"
    except Exception as e:
        return False, -1.0, 0.0, f"visibility: error -> not seeing {e}"


def _decoded_signal_stats(obj: Dict[str, Any]) -> Tuple[float, float, float, bool]:
    try:
        rgb = _to_numpy(obj.get("rgb"))
        depth = _to_numpy(obj.get("depth"))
        rgb_c = 0.0 if rgb.size == 0 else float(np.nanstd(rgb.astype(np.float32)))
        depth_c = 0.0 if depth.size == 0 else float(np.nanstd(depth.astype(np.float32)))
        _, mask_area, _ = _safe_mask_stats(obj.get("mask"))
        decoder_visible = bool((rgb_c > 0.015 or depth_c > 0.015) and mask_area > 0.01)
        return rgb_c, depth_c, mask_area, decoder_visible
    except Exception:
        return 0.0, 0.0, 0.0, False


def _slot_bar_panel(z: np.ndarray, width: int, height: int, title: str = "Active slot / z_obj") -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 14, 22)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 245, 255), 1, cv2.LINE_AA)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if z.size == 0:
        cv2.putText(panel, "z_obj missing", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 150, 190), 1, cv2.LINE_AA)
        return panel
    shown = z[:min(96, z.size)]
    max_abs = max(float(np.max(np.abs(shown))), 1e-6)
    left, right = 12, width - 12
    mid = height // 2 + 18
    top, bottom = 42, height - 18
    cv2.line(panel, (left, mid), (right, mid), (70, 85, 105), 1)
    slot = max(3, int((right - left) / max(1, shown.size)))
    bw = max(2, int(slot * 0.68))
    for i, v in enumerate(shown):
        x0 = left + i * slot
        h = int((bottom - top) * 0.45 * abs(float(v)) / max_abs)
        if v >= 0:
            y0, y1, color = mid - h, mid, (90, 220, 120)
        else:
            y0, y1, color = mid, mid + h, (90, 150, 255)
        cv2.rectangle(panel, (x0, y0), (min(x0 + bw, right), y1), color, -1)
    norm = float(np.linalg.norm(z))
    active = int(np.sum(np.abs(z) > 1e-3))
    cv2.putText(panel, f"norm={norm:.3f} active={active}/{z.size} showing={shown.size}", (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 220, 235), 1, cv2.LINE_AA)
    return panel



def _get_slot_latent_for_display(obj: Dict[str, Any], requested_slot: Optional[int], active_slot_index: int) -> Tuple[np.ndarray, int, str]:
    """
    Return the latent vector that should be visualized.

    Priority:
        requested slot from keys 0..9 -> obj["z_obj_slots"][0, requested]
        active slot                    -> obj["z_obj"]
        fallback                       -> empty vector

    This makes the latent panel show the selected memory slot, not only whatever
    slot the runtime currently marked as active.
    """
    z_slots = _to_numpy(obj.get("z_obj_slots"))
    if z_slots.size > 0:
        if z_slots.ndim == 3:
            z_slots = z_slots[0]
        elif z_slots.ndim == 1:
            z_slots = z_slots.reshape(1, -1)

        if z_slots.ndim == 2 and z_slots.shape[0] > 0:
            if requested_slot is not None:
                idx = max(0, min(int(requested_slot), z_slots.shape[0] - 1))
                return np.asarray(z_slots[idx], dtype=np.float32).reshape(-1), idx, f"KEY slot {idx} / z_obj_slots[{idx}]"
            idx = max(0, min(int(active_slot_index), z_slots.shape[0] - 1))
            return np.asarray(z_slots[idx], dtype=np.float32).reshape(-1), idx, f"ACTIVE slot {idx} / z_obj_slots[{idx}]"

    z = _vector(obj, "z_obj", 0)
    return z, int(active_slot_index), f"ACTIVE slot {active_slot_index} / z_obj"


def _latent_heatmap_panel(z: np.ndarray, width: int, height: int, title: str = "latent code") -> np.ndarray:
    """
    Render a 128-dim latent code as an 8x16 heatmap plus diagnostics.

    Green-ish/yellow values = positive, blue/purple values = negative after
    symmetric normalization around zero.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 14, 22)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)
    cv2.putText(panel, title[:72], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 255), 1, cv2.LINE_AA)

    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if z.size == 0:
        cv2.putText(panel, "latent missing", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 150, 190), 1, cv2.LINE_AA)
        return panel

    shown = z[:128]
    if shown.size < 128:
        shown = np.pad(shown, (0, 128 - shown.size), mode="constant")

    max_abs = max(float(np.max(np.abs(shown))), 1e-6)
    normed = (shown / max_abs + 1.0) * 0.5
    heat = np.clip(normed.reshape(8, 16) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)

    heat_top = 42
    heat_h = max(80, height - 92)
    heat_w = width - 24
    heat_img = cv2.resize(heat, (heat_w, heat_h), interpolation=cv2.INTER_NEAREST)
    panel[heat_top:heat_top + heat_h, 12:12 + heat_w] = heat_img

    # Grid lines for 8x16 cells.
    cell_w = heat_w / 16.0
    cell_h = heat_h / 8.0
    for i in range(17):
        x = int(12 + i * cell_w)
        cv2.line(panel, (x, heat_top), (x, heat_top + heat_h), (12, 18, 28), 1)
    for j in range(9):
        y = int(heat_top + j * cell_h)
        cv2.line(panel, (12, y), (12 + heat_w, y), (12, 18, 28), 1)

    norm = float(np.linalg.norm(z))
    active = int(np.sum(np.abs(z) > 1e-3))
    top_idx = np.argsort(-np.abs(z))[:5]
    top_text = " ".join([f"{int(i)}:{z[int(i)]:+.2f}" for i in top_idx])
    cv2.putText(panel, f"norm={norm:.3f} active={active}/{z.size} max_abs={max_abs:.3f}", (12, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 220, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, f"top |z|: {top_text[:70]}", (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 235), 1, cv2.LINE_AA)
    return panel

def _multi_slot_panel(obj: Dict[str, Any], width: int, height: int, max_slots: int = 10) -> np.ndarray:
    """
    Draw all object slots in a compact two-column layout, highlighting:
        ACTIVE = runtime active slot
        KEY    = slot requested by keyboard 0..9
    """
    max_slots = int(max(1, max_slots))
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 14, 22)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)
    cv2.putText(panel, f"multi-slot object memory: showing {max_slots} slots", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 255), 1, cv2.LINE_AA)

    z_slots = _to_numpy(obj.get("z_obj_slots"))
    conf = _to_numpy(obj.get("confidence_slots"))
    mem = _to_numpy(obj.get("memory_stability_slots"))
    dream = _to_numpy(obj.get("dream_activation_slots"))
    age = _to_numpy(obj.get("slot_age"))
    bind = _to_numpy(obj.get("slot_binding"))
    sim = _to_numpy(obj.get("slot_similarity"))
    upd = _to_numpy(obj.get("slot_update_strength"))

    active_idx = int(_scalar(obj, "active_slot_index", 0.0))
    requested_idx = None
    try:
        requested_idx = obj.get("_requested_dream_slot_index", None)
        if requested_idx is not None:
            requested_idx = int(requested_idx)
    except Exception:
        requested_idx = None

    if z_slots.size == 0:
        cv2.putText(panel, "z_obj_slots missing - single-slot mode", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 165, 190), 1, cv2.LINE_AA)
        return panel

    if z_slots.ndim == 3:
        z_slots = z_slots[0]
    elif z_slots.ndim == 1:
        z_slots = z_slots.reshape(1, -1)

    n = min(int(max_slots), int(z_slots.shape[0]))
    if n <= 0:
        return panel

    def val(arr, i, default=0.0):
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return float(default)
            if a.ndim == 3:
                return float(a[0, i, 0])
            if a.ndim == 2:
                return float(a[i, 0] if a.shape[0] > i else default)
            if a.ndim == 1:
                return float(a[i] if a.shape[0] > i else default)
        except Exception:
            pass
        return float(default)

    cols = 2 if n > 5 else 1
    rows = int(np.ceil(n / cols))
    top = 42
    bottom = height - 12
    gap = 8
    col_w = int((width - 24 - gap * (cols - 1)) / cols)
    row_h = max(28, int((bottom - top) / max(1, rows)))

    for i in range(n):
        col = 0 if i < rows else 1
        row = i if i < rows else i - rows
        x0 = 12 + col * (col_w + gap)
        y0 = top + row * row_h
        x1 = x0 + col_w - 2
        y1 = min(y0 + row_h - 5, bottom - 1)

        c = float(np.clip(val(conf, i), 0.0, 1.0))
        m = float(np.clip(val(mem, i, c), 0.0, 1.0))
        d = float(np.clip(val(dream, i, 0.0), 0.0, 1.0))
        a = float(max(0.0, val(age, i)))
        b = float(np.clip(val(bind, i), 0.0, 1.0))
        u = float(np.clip(val(upd, i), 0.0, 1.0))
        sr = float(np.clip(val(sim, i), -1.0, 1.0))
        s01 = (sr + 1.0) * 0.5
        z = np.asarray(z_slots[i], dtype=np.float32).reshape(-1)
        z_norm = float(np.linalg.norm(z)) if z.size else 0.0

        is_active = (i == active_idx)
        is_requested = (requested_idx is not None and i == requested_idx)
        border = (65, 85, 110)
        if is_active:
            border = (80, 240, 120)
        if is_requested:
            border = (255, 120, 230)

        bg = (20, 34, 32) if is_active else (12, 18, 28)
        if is_requested:
            bg = (38, 18, 40)
        cv2.rectangle(panel, (x0, y0), (x1, y1), bg, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), border, 2 if is_active or is_requested else 1)

        title = f"slot {i}"
        if is_active:
            title += " ACTIVE"
        if is_requested:
            title += " KEY"
        cv2.putText(panel, title, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, border, 1, cv2.LINE_AA)

        bar_x = x0 + 78
        bar_y = y0 + 7
        bar_w = max(55, x1 - bar_x - 8)
        bar_h = 5
        bars = [
            ("m", m, (80, 240, 120)),     # memory stability
            ("c", c, (80, 170, 255)),     # current/sensory confidence
            ("d", d, (255, 120, 230)),    # dream activation
            ("u", u, (230, 120, 255)),    # update strength
        ]
        for bi, (name, vv, color) in enumerate(bars):
            yy = bar_y + bi * 9
            cv2.putText(panel, name, (bar_x - 14, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 195, 215), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (bar_x, yy), (bar_x + bar_w, yy + bar_h), (35, 45, 60), -1)
            cv2.rectangle(panel, (bar_x, yy), (bar_x + int(bar_w * float(np.clip(vv, 0, 1))), yy + bar_h), color, -1)

        cv2.putText(panel, f"age={a:.0f} z={z_norm:.2f} sim={sr:+.2f}", (x0 + 8, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 220, 235), 1, cv2.LINE_AA)

    cv2.putText(panel, "M=memory  C=current/sensory confidence  D=dream activation  U=update | press 0..9 to select slot latent", (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (155, 170, 195), 1, cv2.LINE_AA)
    return panel



def _long_dynamic_cv2_panel(obj: Dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 14, 22)

    ready = _scalar(obj, "long_dynamic_ready", 0.0)
    write = _scalar(obj, "long_dynamic_slot_update_allowed", 0.0)
    persisted = _scalar(obj, "long_dynamic_slot_persisted", 0.0)
    props = _scalar(obj, "semantic_proposal_count", 0.0)
    slot = _scalar(obj, "semantic_updated_slot", -1.0)
    conf = _scalar(obj, "long_dynamic_confidence", 0.0)
    dyn_raw = _scalar(obj, "dynamic_object_confidence_raw", conf)
    dyn_eff = _scalar(obj, "dynamic_object_confidence", 0.0)
    formed_conf = _scalar(obj, "object_formed_confidence", 0.0)
    streak = _scalar(obj, "long_dynamic_ready_streak", 0.0)
    steps = _scalar(obj, "long_dynamic_steps", 0.0)
    active_steps = _scalar(obj, "long_dynamic_active_steps", 0.0)
    dyn = _scalar(obj, "dynamic_score", 0.0)
    nov = _scalar(obj, "scene_novelty", 0.0)
    inter = _scalar(obj, "interaction", 0.0)
    dz = _scalar(obj, "long_dynamic_dz", 0.0)
    z_static = _scalar(obj, "long_dynamic_z_static_norm", 0.0)
    z_dynamic = _scalar(obj, "long_dynamic_z_dynamic_norm", 0.0)
    loss = _scalar(obj, "long_dynamic_loss", 0.0)
    recon = _scalar(obj, "long_dynamic_recon", 0.0)
    updates = _scalar(obj, "long_dynamic_train_updates", 0.0)

    color = (80, 240, 120) if ready > 0.5 else (90, 150, 255)
    if write > 0.5:
        color = (90, 255, 120)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), color, 2)
    cv2.putText(panel, "LONG DYNAMIC MEMORY", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

    def lamp(x: int, y: int, flag: float, label: str):
        c = (60, 240, 90) if flag > 0.5 else (60, 60, 220)
        cv2.circle(panel, (x, y), 10, c, -1)
        cv2.circle(panel, (x, y), 10, (230, 235, 245), 1)
        cv2.putText(panel, label, (x + 18, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 235, 245), 1, cv2.LINE_AA)

    lamp(22, 56, ready, "READY")
    lamp(118, 56, write, "WRITE")

    lines = [
        f"props={props:.0f} slot={slot:.0f} persisted={persisted:.0f}",
        f"dyn_raw={dyn_raw:.3f} dyn_eff={dyn_eff:.3f}",
        f"formed_conf={formed_conf:.3f} streak={streak:.0f}",
        f"steps={steps:.0f} active={active_steps:.0f}",
        f"dyn={dyn:.4f} novelty={nov:.4f}",
        f"interaction={inter:.4f} dz={dz:.5f}",
        f"z_static={z_static:.2f}",
        f"z_dynamic={z_dynamic:.2f}",
        f"loss={loss:.2e} recon={recon:.2e}",
        f"train_updates={updates:.0f}",
    ]

    y = 86
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 235, 245), 1, cv2.LINE_AA)
        y += 19
        if y > height - 12:
            break

    if ready <= 0.5 and write <= 0.5:
        hint = "observing / no slot write"
    elif ready > 0.5 and write > 0.5:
        hint = "dynamic object -> slot"
    else:
        hint = "ready, waiting write"
    cv2.putText(panel, hint[:34], (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 190, 220), 1, cv2.LINE_AA)
    return panel



def _four_d_preview_image_panel(obj: Dict[str, Any], width: int, height: int) -> np.ndarray:
    """Embedded Step 3C 4D/deformed render preview for Inner Object."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 14, 22)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)

    render_valid = _scalar(obj, "slot_4d_playback_render_valid", 0.0)
    deformation_used = _scalar(obj, "slot_4d_playback_deformation_used", 0.0)
    phase = _scalar(obj, "slot_4d_playback_phase", 0.0)
    frames = _scalar(obj, "slot_4d_playback_frames", 0.0)
    pred_delta = _scalar(obj, "slot_4d_playback_pred_delta_norm", 0.0)
    backend_cuda = _scalar(obj, "slot_4d_playback_backend_is_cuda", 0.0)
    backend = "cuda_3dgs" if backend_cuda > 0.5 else "torch_lowres"

    title_color = (90, 255, 130) if render_valid > 0.5 else (80, 170, 255)
    cv2.putText(panel, "4D / 3DGS PREVIEW IMAGE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, title_color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"r={int(render_valid > 0.5)} d={int(deformation_used > 0.5)} f={frames:.0f} phase={phase:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 230, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, f"delta={pred_delta:.3f} {backend}", (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (165, 185, 220), 1, cv2.LINE_AA)

    top = 76
    gap = 8
    img_size = max(64, min((width - 22 - 2 * gap) // 3, height - top - 12))

    if render_valid <= 0.5 or obj.get("slot_4d_playback_rgb") is None:
        cv2.putText(panel, "waiting render_valid=1", (14, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 170, 205), 1, cv2.LINE_AA)
        cv2.putText(panel, "slot_4d_playback_rgb/depth/alpha", (14, top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 145, 180), 1, cv2.LINE_AA)
        return panel

    rgb = cv2.flip(_tensor_rgb_to_bgr_u8(obj.get("slot_4d_playback_rgb"), size=img_size), 0)
    depth = cv2.flip(_tensor_gray_colormap(obj.get("slot_4d_playback_depth"), size=img_size), 0)
    alpha = cv2.flip(_tensor_gray_colormap(obj.get("slot_4d_playback_alpha"), size=img_size, normalize=False), 0)

    for i, (label, img) in enumerate([("RGB", rgb), ("depth", depth), ("alpha", alpha)]):
        x = 8 + i * (img_size + gap)
        y = top
        panel[y:y + img_size, x:x + img_size] = img
        cv2.rectangle(panel, (x, y), (x + img_size - 1, y + img_size - 1), (95, 120, 150), 1)
        _put_label(panel[y:y + img_size, x:x + img_size], label, 6, 18)

    return panel


def _four_d_playback_panel(obj: Dict[str, Any], width: int, height: int) -> np.ndarray:
    """
    Step 3D UI: compact 4D playback status panel.
    No training or rendering happens here. This is only an Inner Object UI panel.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 14, 22)

    frames = _scalar(obj, "slot_4d_playback_frames", 0.0)
    phase = _scalar(obj, "slot_4d_playback_phase", 0.0)
    pred_delta = _scalar(obj, "slot_4d_playback_pred_delta_norm", 0.0)
    render_valid = _scalar(obj, "slot_4d_playback_render_valid", 0.0)
    deformation_used = _scalar(obj, "slot_4d_playback_deformation_used", 0.0)
    preview_fps = _scalar(obj, "slot_4d_playback_preview_fps", 0.0)
    backend_cuda = _scalar(obj, "slot_4d_playback_backend_is_cuda", 0.0)

    timeline_frames = _scalar(obj, "slot_4d_timeline_frames", 0.0)
    temporal_span = _scalar(obj, "slot_4d_temporal_span", 0.0)
    timeline_motion = _scalar(obj, "slot_4d_motion_norm", 0.0)
    timeline_gauss = _scalar(obj, "slot_4d_gaussian_count", 0.0)

    deform_updates = _scalar(obj, "slot_4d_deformation_updates", 0.0)
    deform_loss = _scalar(obj, "slot_4d_deformation_loss", 0.0)
    deform_delta = _scalar(obj, "slot_4d_deformation_pred_delta_norm", 0.0)
    deform_samples = _scalar(obj, "slot_4d_deformation_sample_count", 0.0)

    active_slot = _scalar(obj, "active_slot_index", 0.0)
    formed_conf = _scalar(obj, "object_formed_confidence", 0.0)

    ok = bool(render_valid > 0.5 and deformation_used > 0.5 and frames > 0.0)
    border = (90, 255, 130) if ok else (80, 170, 255)
    if render_valid <= 0.5:
        border = (90, 90, 220)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border, 2)
    cv2.putText(panel, "4D PLAYBACK / DEFORMED OBJECT PREVIEW", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, border, 1, cv2.LINE_AA)
    cv2.putText(panel, f"active_slot={active_slot:.0f} formed_conf={formed_conf:.3f} backend={'cuda_3dgs' if backend_cuda > 0.5 else 'torch_lowres'}",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1, cv2.LINE_AA)

    def lamp(x: int, y: int, flag: float, label: str):
        c = (60, 240, 90) if flag > 0.5 else (55, 65, 95)
        if label == "RENDER" and flag <= 0.5:
            c = (70, 70, 210)
        cv2.circle(panel, (x, y), 9, c, -1)
        cv2.circle(panel, (x, y), 9, (230, 235, 245), 1)
        cv2.putText(panel, label, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 235, 245), 1, cv2.LINE_AA)

    lamp(22, 74, render_valid, "RENDER")
    lamp(112, 74, deformation_used, "DEFORM")
    lamp(210, 74, 1.0 if frames > 0.0 else 0.0, "PLAY")

    bar_x, bar_y = 330, 66
    bar_w = max(120, width - bar_x - 24)
    cv2.putText(panel, f"phase={phase:.3f}", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 230, 245), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (35, 45, 60), -1)
    px = bar_x + int(bar_w * float(np.clip(phase, 0.0, 1.0)))
    cv2.rectangle(panel, (bar_x, bar_y), (px, bar_y + 12), (80, 210, 255), -1)
    cv2.circle(panel, (px, bar_y + 6), 7, (255, 235, 120), -1)

    col_w = max(180, (width - 36) // 3)
    x0 = 12
    y0 = 104
    columns = [
        ("timeline", [f"frames={timeline_frames:.0f}", f"span={temporal_span:.0f}", f"motion={timeline_motion:.4f}", f"gaussians={timeline_gauss:.0f}"], (160, 210, 255)),
        ("deformation", [f"updates={deform_updates:.0f}", f"loss={deform_loss:.3e}", f"delta={deform_delta:.4f}", f"samples={deform_samples:.0f}"], (230, 160, 255)),
        ("playback", [f"frames={frames:.0f}", f"pred_delta={pred_delta:.4f}", f"fps={preview_fps:.1f}", f"render_valid={int(render_valid > 0.5)}"], (130, 255, 170)),
    ]

    for ci, (title, lines, color) in enumerate(columns):
        xx = x0 + ci * col_w
        cv2.rectangle(panel, (xx, y0 - 20), (min(xx + col_w - 8, width - 12), height - 12), (12, 18, 28), 1)
        cv2.putText(panel, title, (xx + 8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        yy = y0 + 22
        for line in lines:
            cv2.putText(panel, line[:32], (xx + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 230, 245), 1, cv2.LINE_AA)
            yy += 18

    hint = "Step 3D: visual status only | Step 3C computes playback; this panel displays it"
    cv2.putText(panel, hint[:120], (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (155, 175, 205), 1, cv2.LINE_AA)
    return panel


class InnerObjectVisualizerV2:
    def __init__(self, cfg: InnerObjectVisualizerV2Config | None = None, **kwargs):
        if cfg is None:
            cfg = InnerObjectVisualizerV2Config(**kwargs)
        self.cfg = cfg
        self.window_name = cfg.window_name
        self.width = int(cfg.width)
        self.height = int(cfg.height)
        self.created = False
        self.hist: Dict[str, List[float]] = {
            "confidence": [], "vision_gate_raw": [], "vision_objectness": [], "vision_activity": [],
            "touch": [], "gate": [], "z_norm": [], "mask_area": [], "mask_entropy": [], "memory_score": [],
            "ldm_streak": [], "dyn_eff": [], "formed_conf": [], "ldm_loss_x1e6": [], "ldm_recon_x1e5": [], "z_dynamic_norm": [],
        }
        self.frames_since_seen = 0
        self.last_status = "INIT"
        self._display_size: Tuple[int, int] = (max(self.width, 1520), max(self.height, 1260))
        self._freeratio_applied: bool = False
        self.event_sentence_history: List[str] = []
        self.last_event_sentence: str = ""

    def close(self):
        close_cv2_window(self.window_name)
        self.created = False

    def _ensure(self):
        # Window creation/resize happens in the dedicated OpenCV GUI thread.
        self.created = True

    def _push(self, key: str, value: float):
        arr = self.hist.setdefault(key, [])
        arr.append(float(np.nan_to_num(value)))
        if len(arr) > self.cfg.history_len:
            del arr[:len(arr) - self.cfg.history_len]

    def _plot_history(self, width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (10, 14, 22)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)
        cv2.putText(panel, "temporal diagnostics", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 255), 1, cv2.LINE_AA)
        series = [
            ("confidence", (80, 240, 120), 1.0), ("vision_gate_raw", (80, 170, 255), 1.0),
            ("vision_objectness", (120, 255, 180), 1.0), ("vision_activity", (60, 210, 255), 1.0),
            ("touch", (255, 180, 80), 1.0), ("memory_score", (230, 120, 255), 1.0),
            ("mask_area", (180, 220, 80), 1.0),
        ]
        left, right, top, bottom = 46, width - 16, 42, height - 34
        cv2.line(panel, (left, bottom), (right, bottom), (70, 85, 105), 1)
        cv2.line(panel, (left, top), (left, bottom), (70, 85, 105), 1)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            y = int(bottom - frac * (bottom - top))
            cv2.line(panel, (left, y), (right, y), (32, 42, 58), 1)
        for name, color, vmax in series:
            vals = np.asarray(self.hist.get(name, []), dtype=np.float32)
            if vals.size < 2:
                continue
            vals = np.clip(np.nan_to_num(vals[-self.cfg.history_len:]) / max(vmax, 1e-6), 0, 1)
            xs = np.linspace(left, right, len(vals)).astype(np.int32)
            ys = (bottom - vals * (bottom - top)).astype(np.int32)
            cv2.polylines(panel, [np.stack([xs, ys], axis=1).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        lx, ly = left + 8, height - 10
        for name, color, _ in series:
            cv2.circle(panel, (lx, ly - 4), 4, color, -1)
            cv2.putText(panel, name, (lx + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 220, 235), 1, cv2.LINE_AA)
            lx += 120
        return panel


    def _plot_dynamic_slot_history(self, width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (10, 14, 22)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)
        cv2.putText(panel, "dynamic slot formation / observed slot", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 245, 255), 1, cv2.LINE_AA)

        series = [
            ("streak", "ldm_streak", (255, 210, 90), 1.0),
            ("dyn_eff", "dyn_eff", (80, 210, 255), 1.0),
            ("formed_conf", "formed_conf", (80, 255, 130), 1.0),
            ("loss_x1e6", "ldm_loss_x1e6", (90, 120, 255), 1.0),
            ("recon_x1e5", "ldm_recon_x1e5", (220, 120, 255), 1.0),
            ("z_dyn", "z_dynamic_norm", (255, 170, 80), 3.0),
        ]

        left, right, top, bottom = 48, width - 18, 44, height - 38
        cv2.line(panel, (left, bottom), (right, bottom), (70, 85, 105), 1)
        cv2.line(panel, (left, top), (left, bottom), (70, 85, 105), 1)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            y = int(bottom - frac * (bottom - top))
            cv2.line(panel, (left, y), (right, y), (32, 42, 58), 1)
            cv2.putText(panel, f"{frac:.2f}", (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 150, 175), 1, cv2.LINE_AA)

        last_values = []
        for label, key, color, vmax in series:
            vals = np.asarray(self.hist.get(key, []), dtype=np.float32)
            last = float(vals[-1]) if vals.size else 0.0
            last_values.append((label, last, color))
            if vals.size < 2:
                continue
            vals = np.clip(np.nan_to_num(vals[-self.cfg.history_len:]) / max(vmax, 1e-6), 0, 1)
            xs = np.linspace(left, right, len(vals)).astype(np.int32)
            ys = (bottom - vals * (bottom - top)).astype(np.int32)
            cv2.polylines(panel, [np.stack([xs, ys], axis=1).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)

        lx, ly = left + 8, height - 12
        for label, last, color in last_values:
            cv2.circle(panel, (lx, ly - 4), 4, color, -1)
            cv2.putText(panel, f"{label}={last:.3f}", (lx + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 235), 1, cv2.LINE_AA)
            lx += 122

        cv2.putText(panel, "goal: static -> all near 0 | rotating object -> streak rises, dyn_eff/formed_conf > 0",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 170, 200), 1, cv2.LINE_AA)
        return panel

    def _diagnose(
        self,
        confidence: float,
        vision: float,
        touch: float,
        z_norm: float,
        mask_area: float,
        camera_visible: bool = True,
        decoder_visible: bool = False,
        dream_mode: bool = False,
        dream_empty: bool = False,
    ) -> Tuple[str, float, str]:
        seeing = bool(camera_visible)
        touch_active = touch > 0.08

        if dream_mode:
            # In full sleep, external seeing is impossible even if old geometry
            # metadata exists in obs. Dream status has priority over SEEING.
            self.frames_since_seen += 1
        else:
            self.frames_since_seen = 0 if seeing else self.frames_since_seen + 1

        memory_score = float(np.clip(
            0.35 * confidence
            + 0.30 * np.tanh(z_norm / 4.0)
            + 0.20 * np.clip(mask_area * 4.0, 0, 1)
            + 0.10 * float(decoder_visible)
            + 0.05 * touch_active,
            0.0,
            1.0,
        ))

        if dream_mode and dream_empty:
            status = "DREAM : EMPTY"
            explanation = "all sensors are off; no stable internal object slot is available for dream decoding"
        elif dream_mode:
            status = "DREAM : DECODING"
            explanation = "all sensors are off; decoder is rendering from internal slot memory"
        elif seeing:
            status = "SEEING : DECODING" if decoder_visible else "SEEING OBJECT / NO DECODE"
            explanation = "object is in camera view" + (" and decoded internal image/mask is visible" if decoder_visible else ", but decoder output is still empty or flat")
        elif memory_score > 0.35 and self.frames_since_seen > 2:
            status = "MEMORY TRACE" if decoder_visible else "LATENT MEMORY / NO DECODE"
            explanation = "camera no longer sees object, but multi-slot memory still holds an object trace"
        elif z_norm > 1.0 and not decoder_visible:
            status = "LATENT ONLY"
            explanation = "active object slot latent is present, but decoded RGB/depth/mask are empty"
        elif confidence < 0.12 and memory_score < 0.20:
            status = "LOST / EMPTY SLOT"
            explanation = "object slot collapsed or has not formed yet"
        else:
            status = "WEAK / FORMING"
            explanation = "representation exists but is unstable or visibility is uncertain"
        self.last_status = status
        return status, memory_score, explanation

    def _info_panel(self, width: int, height: int, lines: List[str], status: str) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (10, 14, 22)
        color = (80, 240, 120)
        if "DREAM : EMPTY" in status:
            color = (120, 140, 180)
        elif "DREAM" in status:
            color = (255, 120, 230)
        elif "MEMORY" in status:
            color = (230, 120, 255)
        elif "LOST" in status:
            color = (90, 90, 220)
        elif "WEAK" in status:
            color = (80, 190, 255)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), color, 2)
        cv2.putText(panel, status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
        y = 64
        for line in lines:
            cv2.putText(panel, line[:88], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 235, 245), 1, cv2.LINE_AA)
            y += 21
            if y > height - 12:
                break
        return panel

    def _tactile_panel(self, tactile_values, width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (10, 14, 22)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)
        cv2.putText(panel, "tactile contacts feeding slots", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 245, 255), 1, cv2.LINE_AA)
        vals = np.asarray(tactile_values if tactile_values is not None else [], dtype=np.float32).reshape(-1)
        vals = np.clip(np.nan_to_num(vals, nan=0.0), 0.0, 1.0)
        if vals.size:
            left, right, top, bottom = 12, width - 12, 42, height - 18
            slot = max(3, int((right - left) / max(1, vals.size)))
            bw = max(2, int(slot * 0.68))
            for i, v in enumerate(vals):
                x0 = left + i * slot
                h = int((bottom - top) * float(v))
                color = (80, 240, 120) if v > 0.05 else (60, 80, 100)
                cv2.rectangle(panel, (x0, bottom - h), (min(x0 + bw, right), bottom), color, -1)
        return panel

    def _push_event_sentence(self, sentence: str) -> None:
        sentence = str(sentence or "").strip()
        if not sentence:
            return
        if sentence == self.last_event_sentence:
            return
        self.last_event_sentence = sentence
        self.event_sentence_history.append(sentence)
        if len(self.event_sentence_history) > 10:
            del self.event_sentence_history[:len(self.event_sentence_history) - 10]

    def _event_sentence_panel(self, obj: Dict[str, Any], width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (10, 14, 22)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (65, 85, 110), 1)

        active = bool(_scalar(obj, "event_active", 0.0) > 0.5)
        delta = _scalar(obj, "event_delta_norm", 0.0)
        mem_size = _scalar(obj, "event_memory_size", 0.0)
        slot = _scalar(obj, "event_slot_index", -1.0)
        slot_token = str(obj.get("slot_token", "") or "")
        slot_desc = str(obj.get("slot_description", "") or "")
        vocab_size = _scalar(obj, "slot_vocabulary_size", 0.0)

        sentence = ""
        try:
            sentence = str(obj.get("event_code_sentence", "") or obj.get("event_code_text", "") or "")
        except Exception:
            sentence = ""

        if sentence:
            self._push_event_sentence(sentence)

        title_color = (120, 240, 180) if active else (150, 165, 190)
        cv2.putText(panel, "event latent code sentences", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, title_color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"active={active} slot={slot:.0f} token={slot_token or 'n/a'} delta_z={delta:.4f} events={mem_size:.0f} vocab={vocab_size:.0f}",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 220, 235), 1, cv2.LINE_AA)
        if slot_desc:
            cv2.putText(panel, slot_desc[:130], (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 205, 245), 1, cv2.LINE_AA)

        # Current event sentence, then recent history.
        y = 92
        if sentence:
            cv2.putText(panel, "current:", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 130), 1, cv2.LINE_AA)
            y += 22
            for chunk_i in range(0, len(sentence), 105):
                cv2.putText(panel, sentence[chunk_i:chunk_i + 105], (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 255), 1, cv2.LINE_AA)
                y += 18
                if y > height - 24:
                    break
        else:
            cv2.putText(panel, "current: no active event sentence yet", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (145, 160, 185), 1, cv2.LINE_AA)
            y += 26

        cv2.putText(panel, "recent DNA-like event stream:", (12, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 205, 245), 1, cv2.LINE_AA)
        y += 34
        recent = self.event_sentence_history[-5:]
        if not recent:
            cv2.putText(panel, "empty", (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 140, 170), 1, cv2.LINE_AA)
        else:
            for s in recent:
                cv2.putText(panel, "• " + s[:120], (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 220, 235), 1, cv2.LINE_AA)
                y += 18
                if y > height - 14:
                    break
        return panel


    def draw(self, obj: Dict[str, torch.Tensor], obs: Optional[Dict] = None, tactile_values=None) -> None:
        self._ensure()
        if not hasattr(self, "requested_dream_slot_index"):
            self.requested_dream_slot_index = None
        while True:
            key = int(consume_cv2_key())
            if key == 255:
                break
            self.last_key = int(key)
            if ord("0") <= key <= ord("9"):
                self.requested_dream_slot_index = int(chr(key))
            elif key in (ord("a"), ord("A")):
                self.requested_dream_slot_index = None

        ps = int(self.cfg.panel_size)

        rgb = cv2.flip(_tensor_rgb_to_bgr_u8(obj.get("rgb"), size=ps), 0)
        depth = cv2.flip(_tensor_gray_colormap(obj.get("depth"), size=ps), 0)
        mask = cv2.flip(_tensor_gray_colormap(obj.get("mask"), size=ps, normalize=False), 0)
        _put_label(rgb, "internal RGB")
        _put_label(depth, "internal depth")
        _put_label(mask, "object mask")

        confidence = _scalar(obj, "confidence", 0.0)
        slot_confidence_raw = _scalar(obj, "slot_confidence_raw", confidence)
        dynamic_object_confidence_raw = _scalar(obj, "dynamic_object_confidence_raw", 0.0)
        dynamic_object_confidence = _scalar(obj, "dynamic_object_confidence", 0.0)
        object_formed_confidence = _scalar(obj, "object_formed_confidence", 0.0)
        object_formed_ready = _scalar(obj, "object_formed_ready", 0.0) > 0.5
        ldm_streak = _scalar(obj, "long_dynamic_ready_streak", 0.0)
        ldm_loss = _scalar(obj, "long_dynamic_loss", 0.0)
        ldm_recon = _scalar(obj, "long_dynamic_recon", 0.0)
        ldm_loss_x1e6 = _scalar(obj, "long_dynamic_loss_x1e6", ldm_loss * 1.0e6)
        ldm_recon_x1e5 = _scalar(obj, "long_dynamic_recon_x1e5", ldm_recon * 1.0e5)
        z_dynamic_norm = _scalar(obj, "long_dynamic_z_dynamic_norm", 0.0)
        dyn_eff = dynamic_object_confidence
        formed_conf = object_formed_confidence
        vision = _scalar(obj, "vision_strength", 0.0)
        vision_activity = _scalar(obj, "vision_activity", 0.0)
        vision_objectness = float(np.clip(1.0 - vision, 0.0, 1.0))
        touch = _scalar(obj, "touch_strength", 0.0)
        update_gate = _scalar(obj, "update_gate_mean", 0.0)
        active_slot_index = int(_scalar(obj, "active_slot_index", 0.0))
        active_slot_age = _scalar(obj, "active_slot_age", 0.0)
        active_slot_binding = _scalar(obj, "active_slot_binding", 0.0)
        active_slot_similarity = _scalar(obj, "active_slot_similarity", 0.0)
        dream_mode = _scalar(obj, "sleep_dream_mode", 0.0) > 0.5
        dream_empty = _scalar(obj, "dream_empty_mode", 0.0) > 0.5
        dream_tick = _scalar(obj, "dream_tick", 0.0)
        dream_delta = _scalar(obj, "dream_latent_delta", 0.0)
        stability = _scalar(obj, "stability", 0.0)
        novelty = _scalar(obj, "novelty", 0.0)
        size = _scalar(obj, "size", 0.0)
        hardness = _scalar(obj, "hardness", 0.0)
        z = _vector(obj, "z_obj", 0)
        z_norm = float(np.linalg.norm(z)) if z.size else 0.0
        mask_mean, mask_area, mask_entropy = _safe_mask_stats(obj.get("mask"))

        shape_probs = None
        shape_name = "unknown"
        try:
            logits = obj.get("shape_logits")
            if logits is not None:
                shape_probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
                shape_name = SHAPE_NAMES[int(np.argmax(shape_probs))]
        except Exception:
            pass

        color = _vector(obj, "color_rgb", 3)[:3]
        if color.size < 3:
            color = np.zeros(3, dtype=np.float32)

        rgb_contrast, depth_contrast, _, decoder_visible = _decoded_signal_stats(obj)
        camera_visible, _, _, visibility_debug = _estimate_object_visibility_from_obs(obs)
        status, memory_score, explanation = self._diagnose(
            object_formed_confidence,
            vision,
            touch,
            z_norm,
            mask_area,
            camera_visible=camera_visible,
            decoder_visible=decoder_visible,
            dream_mode=dream_mode,
            dream_empty=dream_empty,
        )

        for k, v in [("confidence", confidence), ("vision_gate_raw", vision), ("vision_objectness", vision_objectness), ("vision_activity", vision_activity), ("touch", touch), ("gate", update_gate), ("z_norm", np.tanh(z_norm / 4.0)), ("mask_area", mask_area), ("mask_entropy", mask_entropy / 0.7), ("memory_score", memory_score), ("formed_confidence", object_formed_confidence), ("ldm_streak", min(ldm_streak / max(1.0, 8.0), 1.0)), ("dyn_eff", dyn_eff), ("formed_conf", formed_conf), ("ldm_loss_x1e6", ldm_loss_x1e6), ("ldm_recon_x1e5", ldm_recon_x1e5), ("z_dynamic_norm", z_dynamic_norm)]:
            self._push(k, v)

        if obs is not None and "left" in obs:
            cam = cv2.flip(_tensor_rgb_to_bgr_u8(obs.get("left"), size=ps), 0)
            focus_applied = bool(obs.get("depth_focus_applied", False)) if isinstance(obs, dict) else False
            focus_label = str(obs.get("depth_focus_label", "")) if isinstance(obs, dict) else ""
            cam_depth = cv2.flip(
                _tensor_gray_colormap(obs.get("depth"), size=ps) if "depth" in obs else np.zeros_like(cam),
                0,
            )
            _put_label(cam, "current camera left")
            label = "depth_input:raw"
            if focus_applied:
                label = "depth_input:focus"
            _put_label(cam_depth, label)
        else:
            cam = np.zeros((ps, ps, 3), dtype=np.uint8); cam[:] = (10, 14, 22); _put_label(cam, "camera missing")
            cam_depth = np.zeros_like(cam); _put_label(cam_depth, "depth missing")

        lines = [
            explanation,
            f"active_slot={active_slot_index} age={active_slot_age:.0f} formed_conf={object_formed_confidence:.3f} memory={memory_score:.3f}",
            f"slot_conf_raw={slot_confidence_raw:.3f} dyn_raw={dynamic_object_confidence_raw:.3f}",
            f"dyn_eff={dynamic_object_confidence:.3f} formed_ready={object_formed_ready}",
            f"binding={active_slot_binding:.3f} similarity={active_slot_similarity:.3f}",
            f"dream_mode={dream_mode} dream_empty={dream_empty} tick={dream_tick:.0f} delta={dream_delta:.4f}",
            f"vision_gate_raw={vision:.3f} objectness={vision_objectness:.3f} activity={vision_activity:.3f}",
            f"touch={touch:.3f} update_gate={update_gate:.3f}",
            visibility_debug,
            f"decoder_visible={decoder_visible} rgb_std={rgb_contrast:.4f} depth_std={depth_contrast:.4f}",
            f"z_obj_norm={z_norm:.3f} mask_area={mask_area:.3f} mask_mean={mask_mean:.3f}",
            f"shape={shape_name} probs={np.round(shape_probs, 2) if shape_probs is not None else 'n/a'}",
            f"size={size:.3f} hardness={hardness:.3f} stability={stability:.3f} novelty={novelty:.3f}",
            f"internal_color_rgb={np.round(color, 2)}",
            "turn camera away: stable slot confidence/age => object memory",
        ]
        info = self._info_panel(width=560, height=ps, lines=lines, status=status)
        long_dynamic_panel = _long_dynamic_cv2_panel(obj, width=300, height=ps)
        preview_image_panel = _four_d_preview_image_panel(obj, width=520, height=ps)
        try:
            color_box = np.zeros((60, 90, 3), dtype=np.uint8)
            color_box[:] = tuple(int(np.clip(c, 0, 1) * 255) for c in color[::-1])
            info[ps - 72:ps - 12, 455:545] = color_box
        except Exception:
            pass

        top_row = np.concatenate([rgb, depth, mask, long_dynamic_panel, preview_image_panel], axis=1)
        obs_row = np.concatenate([cam, cam_depth], axis=1)
        # The user can press 0..9 to inspect a concrete slot latent.
        # This panel shows z_obj_slots[selected] when a key was pressed,
        # otherwise it shows the active slot latent.
        try:
            current_requested_slot = getattr(self, "requested_dream_slot_index", None)
            if current_requested_slot is not None:
                obj["_requested_dream_slot_index"] = int(current_requested_slot)
            elif "_requested_dream_slot_index" in obj:
                obj.pop("_requested_dream_slot_index", None)
        except Exception:
            pass

        selected_z, selected_slot_idx, selected_title = _get_slot_latent_for_display(
            obj,
            getattr(self, "requested_dream_slot_index", None),
            active_slot_index,
        )
        selected_z_norm = float(np.linalg.norm(selected_z)) if selected_z.size else 0.0

        tactile_panel = self._tactile_panel(tactile_values, width=420, height=ps)
        slot_panel = _slot_bar_panel(selected_z, width=520, height=ps, title=selected_title)
        latent_heat = _latent_heatmap_panel(selected_z, width=360, height=ps, title=f"latent heatmap / slot {selected_slot_idx}")
        history = self._plot_history(width=660, height=ps)
        dynamic_history = self._plot_dynamic_slot_history(width=660, height=ps)

        # Layout:
        #   row 1: current camera/depth/tactile inputs + dream/decoding status.
        #   row 2: decoded internal object + long-dynamic state.
        #   row 3: slot-detail panels, starting with active/selected slot.
        #   bottom: all object slots stay low for memory overview.
        camera_row = np.concatenate([obs_row, tactile_panel, info], axis=1)
        slot_detail_row = np.concatenate([slot_panel, latent_heat, dynamic_history], axis=1)

        multi_slot = _multi_slot_panel(obj, width=slot_detail_row.shape[1], height=max(240, int(ps * 1.10)), max_slots=int(self.cfg.max_slots))
        event_panel = self._event_sentence_panel(obj, width=slot_detail_row.shape[1], height=max(150, int(ps * 0.72)))
        playback_panel = _four_d_playback_panel(obj, width=slot_detail_row.shape[1], height=max(210, int(ps * 0.95)))

        total_w = max(top_row.shape[1], camera_row.shape[1], slot_detail_row.shape[1], multi_slot.shape[1], event_panel.shape[1], playback_panel.shape[1], preview_image_panel.shape[1])

        def pad_w(img):
            if img.shape[1] >= total_w:
                return img
            return cv2.copyMakeBorder(img, 0, 0, 0, total_w - img.shape[1], cv2.BORDER_CONSTANT, value=(6, 10, 16))

        top_row = pad_w(top_row)
        camera_row = pad_w(camera_row)
        slot_detail_row = pad_w(slot_detail_row)
        multi_slot = pad_w(multi_slot)
        event_panel = pad_w(event_panel)
        playback_panel = pad_w(playback_panel)

        header = np.zeros((56, total_w, 3), dtype=np.uint8)
        header[:] = (6, 10, 16)
        cv2.putText(header, "Inner Object Imagery V2: current camera/depth/tactile | decoded object + 4D preview | slot diagnostics | 4D playback | memory", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (240, 245, 255), 1, cv2.LINE_AA)
        current_requested_slot = getattr(self, "requested_dream_slot_index", None)
        key_slot_text = f"selected_slot={current_requested_slot}" if current_requested_slot is not None else "selected_slot=ACTIVE"
        cv2.putText(header, f"status={status} | active_slot={active_slot_index} | {key_slot_text} | keys 0..9 select slot, A=active", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 190, 220), 1, cv2.LINE_AA)

        frame = np.concatenate([header, camera_row, top_row, slot_detail_row, playback_panel, event_panel, multi_slot], axis=0)

        # Do not query OpenCV window geometry from the model/runtime thread.
        target_w, target_h = self._display_size

        # Exact image resize to the configured display size.
        # imshow uses both target_w and target_h instead of preserving aspect.
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            interp = cv2.INTER_AREA if (frame.shape[1] > target_w or frame.shape[0] > target_h) else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)

        submit_cv2_frame(self.window_name, frame, max(int(getattr(self, "width", 1520)), 1520), max(int(getattr(self, "height", 1260)), 1260))
