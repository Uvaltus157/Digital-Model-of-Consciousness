from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _tensor_np(x: Any) -> np.ndarray | None:
    if x is None or not torch.is_tensor(x):
        return None
    return x.detach().float().cpu().numpy().astype(np.float32)


def _descriptor(xyz: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] < 3:
        arr = np.zeros((0, 3), dtype=np.float32)
    arr = arr[:, :3]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        return {"point_count": 0, "centroid": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0], "radius": 0.0}
    c = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    r = float(np.mean(np.linalg.norm(arr - c.reshape(1, 3), axis=-1)))
    return {"point_count": int(arr.shape[0]), "centroid": [float(v) for v in c], "std": [float(v) for v in s], "radius": r}


def _dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    ac = np.asarray(a.get("centroid", [0, 0, 0]), dtype=np.float32)
    bc = np.asarray(b.get("centroid", [0, 0, 0]), dtype=np.float32)
    ast = np.asarray(a.get("std", [0, 0, 0]), dtype=np.float32)
    bst = np.asarray(b.get("std", [0, 0, 0]), dtype=np.float32)
    return float(np.linalg.norm(ac - bc) + np.linalg.norm(ast - bst) + abs(float(a.get("radius", 0.0)) - float(b.get("radius", 0.0))))


class SlotObjectMemoryManager:
    def __init__(self, root: str | Path = "./checkpoint/slot_memory", match_threshold: float = 0.30) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.match_threshold = float(match_threshold)
        self.latest: dict[int, dict[str, Any]] = {}

    def _dir(self, slot_id: int, target_name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(target_name or f"slot_{slot_id}"))
        d = self.root / f"slot_{int(slot_id)}_{safe}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _arrays(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz = _tensor_np(getattr(state, "xyz", None))
        if xyz is None or xyz.ndim != 2 or xyz.shape[-1] < 3:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
        xyz = xyz[:, :3]
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]

        color = None
        c = getattr(state, "color_logit", None)
        if torch.is_tensor(c):
            color = torch.sigmoid(c.detach().float()).cpu().numpy().astype(np.float32)
            color = color[:, :3] if color.ndim == 2 and color.shape[-1] >= 3 else None
        if color is None:
            color = np.ones((mask.shape[0], 3), dtype=np.float32) * 0.7
        color = color[mask] if color.shape[0] == mask.shape[0] else np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.7
        color = np.nan_to_num(color, nan=0.7, posinf=1.0, neginf=0.0)

        alpha = None
        o = getattr(state, "opacity_logit", None)
        if torch.is_tensor(o):
            alpha = torch.sigmoid(o.detach().float()).cpu().numpy().astype(np.float32).reshape(-1)
        if alpha is None:
            alpha = np.ones((mask.shape[0],), dtype=np.float32)
        alpha = alpha[mask] if alpha.shape[0] == mask.shape[0] else np.ones((xyz.shape[0],), np.float32)
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=0.0)
        return xyz.astype(np.float32), np.clip(color, 0, 1), np.clip(alpha, 0, 1)

    def _deform(self, slot_id: int, raw_xyz: np.ndarray, deformation_trainer: Any, phase: float) -> tuple[np.ndarray, bool]:
        model = (getattr(deformation_trainer, "models", {}) or {}).get(int(slot_id)) if deformation_trainer is not None else None
        if model is None or raw_xyz.shape[0] == 0:
            return raw_xyz.copy(), False
        try:
            with torch.no_grad():
                try:
                    device = next(model.parameters()).device
                    dtype = next(model.parameters()).dtype
                except Exception:
                    device, dtype = "cpu", torch.float32
                x = torch.as_tensor(raw_xyz, device=device, dtype=dtype)
                d = model(x[:, :3], torch.tensor(float(phase), device=x.device, dtype=x.dtype))
                out = (x[:, :3] + d).detach().float().cpu().numpy().astype(np.float32)
                if out.shape != raw_xyz.shape or not np.all(np.isfinite(out)):
                    return raw_xyz.copy(), False
                return out, True
        except Exception:
            return raw_xyz.copy(), False

    def list_memories(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("slot_*_*/metadata.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return out

    def find_best_match(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        best = {"matched": False, "distance": 999999.0, "path": "", "target_name": "", "slot_id": -1}
        for item in self.list_memories():
            d = _dist(descriptor, item.get("descriptor", {}))
            if d < float(best["distance"]):
                best = {"matched": bool(d <= self.match_threshold), "distance": float(d), "path": str(item.get("path", "")), "target_name": str(item.get("target_name", "")), "slot_id": int(item.get("slot_id", -1))}
        return best

    def save_from_runtime(self, *, slot_id: int, target_name: str, gaussian_reconstructor: Any, deformation_trainer: Any = None, playback_renderer: Any = None) -> dict[str, Any]:
        states = getattr(gaussian_reconstructor, "states", {}) or {}
        state = states.get(int(slot_id))
        if state is None:
            return {"written": False, "reason": "no_state", "slot_id": int(slot_id), "target_name": str(target_name)}
        raw, color, alpha = self._arrays(state)
        if raw.shape[0] == 0:
            return {"written": False, "reason": "no_points", "slot_id": int(slot_id), "target_name": str(target_name)}

        pm = (getattr(playback_renderer, "last_metrics", {}) or {}).get(int(slot_id)) if playback_renderer is not None else None
        phase = float(getattr(pm, "playback_phase", 0.0) or 0.0) if pm is not None else 0.0
        deformed, used = self._deform(int(slot_id), raw, deformation_trainer, phase)
        desc = _descriptor(raw)
        best = self.find_best_match(desc)

        d = self._dir(int(slot_id), str(target_name))
        npz = d / "object_points_latest.npz"
        meta_path = d / "metadata.json"
        np.savez_compressed(npz, raw_xyz=raw, deformed_xyz=deformed, color=color, alpha=alpha)
        meta = {
            "slot_id": int(slot_id),
            "target_name": str(target_name),
            "path": str(d),
            "npz_path": str(npz),
            "updated_unix": time.time(),
            "raw_points": int(raw.shape[0]),
            "deformed_points": int(deformed.shape[0]),
            "deformation_used": bool(used),
            "playback_phase": float(phase),
            "descriptor": desc,
            "best_match": best,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.latest[int(slot_id)] = meta
        return {"written": True, "reason": "ok", **meta}
