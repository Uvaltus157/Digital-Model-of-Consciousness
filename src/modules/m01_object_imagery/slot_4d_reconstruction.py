
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class Slot4DFrame:
    slot_id: int
    target_name: str
    live_step: int
    gaussian_count: int
    updates: int
    recon_loss: float
    formed_conf: float
    z_dynamic_norm: float
    xyz_mean: list[float]
    xyz_std: list[float]
    xyz_sample: np.ndarray | None
    backend: str


class Slot4DTimelineBuffer:
    """Compact per-slot temporal history of Gaussian states."""

    def __init__(self, max_frames_per_slot: int = 256, sample_points: int = 128) -> None:
        self.max_frames_per_slot = max(2, int(max_frames_per_slot))
        self.sample_points = max(8, int(sample_points))
        self.frames: dict[int, deque[Slot4DFrame]] = defaultdict(
            lambda: deque(maxlen=self.max_frames_per_slot)
        )

    def add(self, frame: Slot4DFrame) -> dict[str, Any]:
        sid = int(frame.slot_id)
        prev = self.frames[sid][-1] if self.frames[sid] else None
        self.frames[sid].append(frame)

        motion_norm = 0.0
        mean_delta = [0.0, 0.0, 0.0]
        if prev is not None:
            a = np.asarray(frame.xyz_mean, dtype=np.float32)
            b = np.asarray(prev.xyz_mean, dtype=np.float32)
            d = a - b
            motion_norm = float(np.linalg.norm(d))
            mean_delta = d.astype(float).tolist()

        q = self.frames[sid]
        first_step = int(q[0].live_step) if q else int(frame.live_step)
        temporal_span = int(frame.live_step) - first_step

        return {
            "slot_id": sid,
            "target_name": frame.target_name,
            "frame_count": int(len(q)),
            "live_step": int(frame.live_step),
            "temporal_span": int(temporal_span),
            "gaussian_count": int(frame.gaussian_count),
            "updates": int(frame.updates),
            "recon_loss": float(frame.recon_loss),
            "formed_conf": float(frame.formed_conf),
            "z_dynamic_norm": float(frame.z_dynamic_norm),
            "motion_norm": float(motion_norm),
            "mean_delta_x": float(mean_delta[0]),
            "mean_delta_y": float(mean_delta[1]),
            "mean_delta_z": float(mean_delta[2]),
            "backend": str(frame.backend),
            "valid": bool(frame.gaussian_count > 0),
        }

    def count(self, slot_id: int) -> int:
        return len(self.frames.get(int(slot_id), ()))

    def latest(self, slot_id: int) -> Slot4DFrame | None:
        q = self.frames.get(int(slot_id))
        if not q:
            return None
        return q[-1]

    def slot_summary(self, slot_id: int) -> dict[str, Any]:
        sid = int(slot_id)
        q = self.frames.get(sid)
        if not q:
            return {
                "slot_id": sid,
                "frame_count": 0,
                "target_name": "unknown",
                "temporal_span": 0,
                "gaussian_count": 0,
                "updates": 0,
                "motion_norm": 0.0,
            }
        latest = q[-1]
        first = q[0]
        return {
            "slot_id": sid,
            "frame_count": int(len(q)),
            "target_name": latest.target_name,
            "temporal_span": int(latest.live_step) - int(first.live_step),
            "gaussian_count": int(latest.gaussian_count),
            "updates": int(latest.updates),
            "recon_loss": float(latest.recon_loss),
            "formed_conf": float(latest.formed_conf),
            "z_dynamic_norm": float(latest.z_dynamic_norm),
            "backend": str(latest.backend),
        }


class Slot4DReconstructor:
    """
    Step 3A: timeline over per-slot Gaussian states.

    This is not a deformation model yet. It captures state(t) so Step 3B can
    learn base_gaussians + time_code -> deformation.
    """

    def __init__(self, max_frames_per_slot: int = 256, sample_points: int = 128) -> None:
        self.timeline = Slot4DTimelineBuffer(
            max_frames_per_slot=max_frames_per_slot,
            sample_points=sample_points,
        )
        self.latest_metrics: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _state_xyz_sample(state: Any, sample_points: int) -> tuple[int, list[float], list[float], np.ndarray | None]:
        xyz = getattr(state, "xyz", None)
        if xyz is None or not torch.is_tensor(xyz):
            return 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], None
        with torch.no_grad():
            x = xyz.detach().float().cpu()
            if x.ndim != 2 or x.shape[-1] < 3:
                return 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], None
            x = x[:, :3]
            n = int(x.shape[0])
            mean = x.mean(dim=0).numpy().astype(float).tolist()
            std = x.std(dim=0, unbiased=False).numpy().astype(float).tolist()
            if n > int(sample_points):
                idx = torch.linspace(0, n - 1, int(sample_points)).long()
                sample = x.index_select(0, idx).numpy().astype(np.float32)
            else:
                sample = x.numpy().astype(np.float32)
            return n, mean, std, sample

    def add_from_gaussian_reconstructor(
        self,
        *,
        slot_id: int,
        target_name: str,
        live_step: int,
        gaussian_reconstructor: Any,
        formed_conf: float,
        z_dynamic_norm: float,
    ) -> dict[str, Any]:
        sid = int(slot_id)
        states = getattr(gaussian_reconstructor, "states", {}) or {}
        state = states.get(sid)
        if state is None:
            return {
                "slot_id": sid,
                "target_name": str(target_name),
                "frame_count": self.timeline.count(sid),
                "live_step": int(live_step),
                "temporal_span": 0,
                "gaussian_count": 0,
                "updates": 0,
                "recon_loss": 0.0,
                "formed_conf": float(formed_conf),
                "z_dynamic_norm": float(z_dynamic_norm),
                "motion_norm": 0.0,
                "backend": "none",
                "valid": False,
                "reason": "no_gaussian_state",
            }

        gaussian_count, xyz_mean, xyz_std, xyz_sample = self._state_xyz_sample(
            state, self.timeline.sample_points
        )

        metrics_by_slot = getattr(gaussian_reconstructor, "last_metrics", {}) or {}
        gm = metrics_by_slot.get(sid)
        updates = int(getattr(state, "updates", getattr(gm, "updates", 0)) or 0)
        recon_loss = float(getattr(state, "last_total_loss", getattr(gm, "total_loss", 0.0)) or 0.0)
        backend = str(getattr(gm, "backend", "torch_lowres")) if gm is not None else "torch_lowres"

        frame = Slot4DFrame(
            slot_id=sid,
            target_name=str(target_name),
            live_step=int(live_step),
            gaussian_count=int(gaussian_count),
            updates=int(updates),
            recon_loss=float(recon_loss),
            formed_conf=float(formed_conf),
            z_dynamic_norm=float(z_dynamic_norm),
            xyz_mean=xyz_mean,
            xyz_std=xyz_std,
            xyz_sample=xyz_sample,
            backend=backend,
        )
        metrics = self.timeline.add(frame)
        self.latest_metrics[sid] = metrics
        return metrics

    def summary(self) -> dict[str, Any]:
        return {
            "slot_0": self.timeline.slot_summary(0),
            "slot_1": self.timeline.slot_summary(1),
        }
