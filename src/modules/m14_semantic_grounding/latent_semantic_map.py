
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, get_cv2_last_key


@dataclass
class LatentSemanticMapConfig:
    enabled: bool = True
    window_name: str = "latent semantic map"
    width: int = 1600
    height: int = 980
    max_history: int = 320
    show_every_steps: int = 1
    delay_ms: int = 1
    thumbnail_size: int = 82
    max_thumbnails: int = 6
    point_radius: int = 4
    draw_grid: bool = True
    follow_inner_world_toggle: bool = True


def _to_numpy_1d(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        arr = x.detach().float().cpu().numpy()
    else:
        try:
            arr = np.asarray(x, dtype=np.float32)
        except Exception:
            return None
    if arr.size == 0:
        return None
    arr = arr.reshape(-1).astype(np.float32)
    return arr


def _to_scalar(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().cpu().reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0])
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_text(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\n", " ").strip()


def _rgb_tensor_to_bgr_u8(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        arr = x.detach().float().cpu().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class _LatentStreamHistory:
    def __init__(self, max_history: int) -> None:
        self.vectors: Deque[np.ndarray] = deque(maxlen=max_history)
        self.meta: Deque[dict] = deque(maxlen=max_history)

    def append(self, vec: Optional[np.ndarray], meta: dict) -> None:
        if vec is None:
            return
        self.vectors.append(vec.astype(np.float32))
        self.meta.append(dict(meta))

    def matrix(self) -> Optional[np.ndarray]:
        if len(self.vectors) == 0:
            return None
        dim = min(v.shape[0] for v in self.vectors)
        if dim <= 0:
            return None
        return np.stack([v[:dim] for v in self.vectors], axis=0)

    def __len__(self) -> int:
        return len(self.vectors)


class LatentSemanticMapVisualizer:
    def __init__(self, cfg: LatentSemanticMapConfig) -> None:
        self.cfg = cfg
        self.streams: Dict[str, _LatentStreamHistory] = {
            "object": _LatentStreamHistory(cfg.max_history),
            "workspace": _LatentStreamHistory(cfg.max_history),
            "memory": _LatentStreamHistory(cfg.max_history),
            "thought": _LatentStreamHistory(cfg.max_history),
        }
        self.thumbnail_history: Deque[np.ndarray] = deque(maxlen=cfg.max_thumbnails)
        self.last_canvas: Optional[np.ndarray] = None

    def _extract_streams(self, out: dict) -> Dict[str, Optional[np.ndarray]]:
        memory = out.get("memory") if isinstance(out.get("memory"), dict) else {}
        thoughts = out.get("thoughts") if isinstance(out.get("thoughts"), dict) else {}

        thought_vec = None
        if "thought" in thoughts:
            thought_vec = _to_numpy_1d(thoughts.get("thought"))
        elif "thought_trace" in out:
            tr = out.get("thought_trace")
            tr_np = _to_numpy_1d(tr)
            thought_vec = tr_np

        memory_vec = None
        for key in ("memory_context", "memory_readout", "memory_summary", "summary"):
            if isinstance(memory, dict) and key in memory:
                memory_vec = _to_numpy_1d(memory.get(key))
                if memory_vec is not None:
                    break
            if key in out:
                memory_vec = _to_numpy_1d(out.get(key))
                if memory_vec is not None:
                    break

        workspace_vec = None
        for key in ("workspace_out", "workspace", "workspace_state"):
            if key in out:
                workspace_vec = _to_numpy_1d(out.get(key))
                if workspace_vec is not None:
                    break

        object_vec = None
        for key in ("object_repr", "object_latent", "object_slots"):
            if key in out:
                object_vec = _to_numpy_1d(out.get(key))
                if object_vec is not None:
                    break

        return {
            "object": object_vec,
            "workspace": workspace_vec,
            "memory": memory_vec,
            "thought": thought_vec,
        }

    def _extract_meta(self, out: dict) -> dict:
        focus_idx = 0
        focus = out.get("focus")
        if isinstance(focus, dict) and "focus_idx" in focus:
            focus_idx = int(_to_scalar(focus["focus_idx"], 0))
        elif "focus_idx" in out:
            focus_idx = int(_to_scalar(out["focus_idx"], 0))

        values = out.get("values") if isinstance(out.get("values"), dict) else {}
        curiosity = _to_scalar(values.get("curiosity"), 0.0)
        coherence = _to_scalar(values.get("coherence"), 0.0)

        report = out.get("decoded_report")
        if report is None:
            report = out.get("target_report")
        report = _safe_text(report)

        object_imagery = out.get("object_imagery") if isinstance(out.get("object_imagery"), dict) else {}
        confidence = _to_scalar(object_imagery.get("object_confidence"), 0.0)

        return {
            "focus_idx": focus_idx,
            "curiosity": curiosity,
            "coherence": coherence,
            "report": report,
            "object_confidence": confidence,
        }

    def _append_thumbnail(self, out: dict) -> None:
        imagery = out.get("object_imagery")
        if not isinstance(imagery, dict):
            return
        img = _rgb_tensor_to_bgr_u8(imagery.get("rgb"))
        if img is None:
            return
        thumb = cv2.resize(img, (self.cfg.thumbnail_size, self.cfg.thumbnail_size), interpolation=cv2.INTER_AREA)
        self.thumbnail_history.append(thumb)

    def update(self, out: dict) -> None:
        streams = self._extract_streams(out)
        meta = self._extract_meta(out)
        for name, vec in streams.items():
            self.streams[name].append(vec, meta)
        self._append_thumbnail(out)

    def _project_2d(self, mat: np.ndarray) -> np.ndarray:
        if mat is None or mat.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if mat.shape[0] == 1:
            return np.zeros((1, 2), dtype=np.float32)

        x = mat.astype(np.float32)
        x = x - x.mean(axis=0, keepdims=True)

        try:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            if vt.shape[0] >= 2:
                comp = vt[:2]
            elif vt.shape[0] == 1:
                comp = np.concatenate([vt[:1], np.zeros_like(vt[:1])], axis=0)
            else:
                comp = np.zeros((2, x.shape[1]), dtype=np.float32)
            proj = x @ comp.T
        except np.linalg.LinAlgError:
            proj = np.zeros((x.shape[0], 2), dtype=np.float32)

        proj = proj.astype(np.float32)
        scale = np.max(np.abs(proj))
        if scale > 1e-6:
            proj /= scale
        return proj

    def _draw_axes(self, img: np.ndarray, rect: Tuple[int, int, int, int], title: str) -> None:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (48, 58, 82), 1, cv2.LINE_AA)
        if self.cfg.draw_grid:
            for frac in (0.25, 0.5, 0.75):
                gx = x + int(frac * w)
                gy = y + int(frac * h)
                cv2.line(img, (gx, y), (gx, y + h), (25, 30, 44), 1, cv2.LINE_AA)
                cv2.line(img, (x, gy), (x + w, gy), (25, 30, 44), 1, cv2.LINE_AA)
            cv2.line(img, (x + w // 2, y), (x + w // 2, y + h), (60, 70, 96), 1, cv2.LINE_AA)
            cv2.line(img, (x, y + h // 2), (x + w, y + h // 2), (60, 70, 96), 1, cv2.LINE_AA)
        cv2.putText(img, title, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 230, 250), 1, cv2.LINE_AA)

    def _draw_projection(self, img: np.ndarray, rect: Tuple[int, int, int, int], title: str, history: _LatentStreamHistory) -> None:
        self._draw_axes(img, rect, title)
        x, y, w, h = rect
        mat = history.matrix()
        if mat is None or mat.shape[0] == 0:
            cv2.putText(img, "no history yet", (x + 12, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (128, 140, 170), 1, cv2.LINE_AA)
            return

        proj = self._project_2d(mat)
        if proj.shape[0] == 0:
            return

        def map_pt(px: float, py: float) -> Tuple[int, int]:
            sx = x + int((px * 0.44 + 0.5) * w)
            sy = y + int((0.5 - py * 0.44) * h)
            return sx, sy

        pts = [map_pt(float(p[0]), float(p[1])) for p in proj]

        for i in range(1, len(pts)):
            alpha = i / max(1, len(pts) - 1)
            c = (
                int(40 + 140 * alpha),
                int(60 + 100 * alpha),
                int(90 + 135 * alpha),
            )
            cv2.line(img, pts[i - 1], pts[i], c, 1, cv2.LINE_AA)

        for i, p in enumerate(pts):
            alpha = i / max(1, len(pts) - 1)
            focus = int(history.meta[i].get("focus_idx", 0))
            color = (
                int(45 + 18 * (focus % 7) + 130 * alpha),
                int(85 + 20 * ((focus + 2) % 7) + 80 * alpha),
                int(110 + 18 * ((focus + 4) % 7) + 70 * alpha),
            )
            r = self.cfg.point_radius if i < len(pts) - 1 else self.cfg.point_radius + 3
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)

        current_meta = history.meta[-1]
        cv2.putText(
            img,
            f"n={len(history)} focus={current_meta.get('focus_idx', 0)}",
            (x + 10, y + h - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 220, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"cur={current_meta.get('curiosity', 0.0):.2f} coh={current_meta.get('coherence', 0.0):.2f}",
            (x + 10, y + h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (140, 185, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_sidebar(self, img: np.ndarray, rect: Tuple[int, int, int, int], out: dict) -> None:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (48, 58, 82), 1, cv2.LINE_AA)
        cv2.putText(img, "meaning synthesis", (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (235, 238, 250), 1, cv2.LINE_AA)

        meta = self._extract_meta(out)
        lines = [
            f"focus_idx: {meta['focus_idx']}",
            f"curiosity: {meta['curiosity']:.3f}",
            f"coherence: {meta['coherence']:.3f}",
            f"object_conf: {meta['object_confidence']:.3f}",
            "",
            "inner speech:",
        ]
        yy = y + 54
        for line in lines:
            cv2.putText(img, line, (x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 220, 236), 1, cv2.LINE_AA)
            yy += 22

        report = meta["report"] or "(empty)"
        wrapped = self._wrap(report, max_chars=30, max_lines=5)
        for line in wrapped:
            cv2.putText(img, line, (x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 180), 1, cv2.LINE_AA)
            yy += 22

        yy += 12
        cv2.putText(img, "emergent pattern cue", (x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 238, 250), 1, cv2.LINE_AA)
        yy += 24

        cues = self._describe_cues()
        for cue in cues:
            cv2.putText(img, cue, (x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 210, 120), 1, cv2.LINE_AA)
            yy += 22

        yy += 8
        cv2.putText(img, "object imagery history", (x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 238, 250), 1, cv2.LINE_AA)
        yy += 14

        thumb_x = x + 10
        thumb_y = yy + 10
        for i, thumb in enumerate(self.thumbnail_history):
            xx = thumb_x + i * (self.cfg.thumbnail_size + 8)
            if xx + self.cfg.thumbnail_size > x + w - 8:
                break
            img[thumb_y:thumb_y + self.cfg.thumbnail_size, xx:xx + self.cfg.thumbnail_size] = thumb
            cv2.rectangle(img, (xx, thumb_y), (xx + self.cfg.thumbnail_size, thumb_y + self.cfg.thumbnail_size), (62, 72, 92), 1, cv2.LINE_AA)

    def _describe_cues(self) -> List[str]:
        cues = []
        if len(self.streams["object"]) >= 4:
            obj = self.streams["object"].matrix()
            if obj is not None:
                spread = float(np.std(obj[-min(24, obj.shape[0]):], axis=0).mean())
                if spread < 0.10:
                    cues.append("object attractor stabilizing")
                elif spread < 0.25:
                    cues.append("object cluster becoming denser")
                else:
                    cues.append("object concept still diffuse")
        if len(self.streams["workspace"]) >= 4:
            w = self.streams["workspace"].matrix()
            if w is not None and w.shape[0] >= 2:
                drift = float(np.linalg.norm(w[-1] - w[-2]))
                if drift < 0.05:
                    cues.append("workspace state recurrent")
                else:
                    cues.append("workspace still exploring")
        if len(cues) == 0:
            cues.append("collecting latent traces")
        return cues[:5]

    def _wrap(self, text: str, max_chars: int = 28, max_lines: int = 5) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines = []
        cur = ""
        for w in words:
            candidate = f"{cur} {w}".strip()
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                lines.append(cur)
                cur = w
                if len(lines) >= max_lines - 1:
                    break
        if cur and len(lines) < max_lines:
            lines.append(cur)
        return lines

    def render(self, out: dict) -> np.ndarray:
        canvas = np.full((self.cfg.height, self.cfg.width, 3), (12, 16, 24), dtype=np.uint8)

        left_w = int(self.cfg.width * 0.72)
        gap = 14
        rects = {
            "object": (18, 18, left_w // 2 - 24, self.cfg.height // 2 - 28),
            "workspace": (left_w // 2 + 6, 18, left_w // 2 - 24, self.cfg.height // 2 - 28),
            "memory": (18, self.cfg.height // 2 + 8, left_w // 2 - 24, self.cfg.height // 2 - 32),
            "thought": (left_w // 2 + 6, self.cfg.height // 2 + 8, left_w // 2 - 24, self.cfg.height // 2 - 32),
        }
        side_rect = (left_w + gap, 18, self.cfg.width - left_w - 2 * gap - 8, self.cfg.height - 36)

        self._draw_projection(canvas, rects["object"], "object_repr latent map", self.streams["object"])
        self._draw_projection(canvas, rects["workspace"], "workspace latent map", self.streams["workspace"])
        self._draw_projection(canvas, rects["memory"], "memory latent map", self.streams["memory"])
        self._draw_projection(canvas, rects["thought"], "thought latent map", self.streams["thought"])
        self._draw_sidebar(canvas, side_rect, out)

        title = "emerging semantic images in latent space"
        cv2.putText(canvas, title, (20, self.cfg.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 200, 255), 1, cv2.LINE_AA)

        self.last_canvas = canvas
        return canvas

    def show(self, out: dict, delay_ms: Optional[int] = None) -> int:
        self.update(out)
        canvas = self.render(out)
        submit_cv2_frame(self.cfg.window_name, canvas, canvas.shape[1], canvas.shape[0])
        return int(get_cv2_last_key())

    def save(self, path: str) -> None:
        if self.last_canvas is not None:
            cv2.imwrite(path, self.last_canvas)

    def close(self) -> None:
        close_cv2_window(self.cfg.window_name)
