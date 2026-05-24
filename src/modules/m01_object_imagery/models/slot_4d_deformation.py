
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Slot4DDeformationMetrics:
    slot_id: int
    target_name: str
    enabled: bool
    trainable: bool
    valid: bool
    updates: int
    loss: float
    motion_norm: float
    pred_delta_norm: float
    sample_count: int
    temporal_dt: float
    model_type: str = "Slot4DDeformationModel"


class Slot4DDeformationModel(nn.Module):
    """
    Step 3B: compact neural deformation field.

    It predicts a delta for each Gaussian sample from:
      [xyz, normalized_time] -> delta_xyz

    This is intentionally small and safe. It does not replace the Gaussian
    renderer. It trains from Step 3A timeline frame pairs.
    """

    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        hidden_dim = max(16, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, xyz: torch.Tensor, time_code: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(time_code):
            time_code = torch.tensor(float(time_code), device=xyz.device, dtype=xyz.dtype)
        if time_code.ndim == 0:
            t = time_code.expand(xyz.shape[0], 1)
        elif time_code.ndim == 1:
            t = time_code.reshape(-1, 1).expand(xyz.shape[0], 1)
        else:
            t = time_code.reshape(xyz.shape[0], 1)
        inp = torch.cat([xyz[:, :3], t.to(device=xyz.device, dtype=xyz.dtype)], dim=-1)
        return self.net(inp)


class Slot4DDeformationTrainer:
    """
    Trains one deformation model per object slot from Slot4DTimelineBuffer.

    Input:
      last two Slot4DFrame snapshots for a slot.

    Loss:
      pred_xyz(t1) = xyz(t0) + model(xyz(t0), dt)
      MSE(pred_xyz(t1), xyz(t1)) + small delta regularization
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 96,
        lr: float = 2.0e-3,
        train_steps_per_update: int = 1,
        min_frames: int = 2,
        delta_reg_weight: float = 1.0e-4,
        device: str | torch.device | None = None,
    ) -> None:
        self.hidden_dim = max(16, int(hidden_dim))
        self.lr = float(lr)
        self.train_steps_per_update = max(1, int(train_steps_per_update))
        self.min_frames = max(2, int(min_frames))
        self.delta_reg_weight = float(delta_reg_weight)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.models: dict[int, Slot4DDeformationModel] = {}
        self.optimizers: dict[int, torch.optim.Optimizer] = {}
        self.last_metrics: dict[int, Slot4DDeformationMetrics] = {}

    def _ensure_model(self, slot_id: int) -> Slot4DDeformationModel:
        sid = int(slot_id)
        if sid not in self.models:
            model = Slot4DDeformationModel(hidden_dim=self.hidden_dim).to(self.device)
            self.models[sid] = model
            self.optimizers[sid] = torch.optim.Adam(model.parameters(), lr=self.lr)
        return self.models[sid]

    @staticmethod
    def _frame_sample(frame: Any) -> np.ndarray | None:
        sample = getattr(frame, "xyz_sample", None)
        if sample is None:
            return None
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] < 3 or arr.shape[0] <= 0:
            return None
        return arr[:, :3]

    def train_from_timeline(self, *, slot_id: int, target_name: str, timeline: Any) -> dict[str, Any]:
        sid = int(slot_id)
        q = getattr(timeline, "frames", {}).get(sid)
        if not q or len(q) < self.min_frames:
            metrics = Slot4DDeformationMetrics(
                slot_id=sid,
                target_name=str(target_name),
                enabled=True,
                trainable=False,
                valid=False,
                updates=0,
                loss=0.0,
                motion_norm=0.0,
                pred_delta_norm=0.0,
                sample_count=0,
                temporal_dt=0.0,
            )
            self.last_metrics[sid] = metrics
            return dict(metrics.__dict__)

        prev = q[-2]
        curr = q[-1]
        x0_np = self._frame_sample(prev)
        x1_np = self._frame_sample(curr)
        if x0_np is None or x1_np is None:
            metrics = Slot4DDeformationMetrics(
                slot_id=sid,
                target_name=str(target_name),
                enabled=True,
                trainable=False,
                valid=False,
                updates=int(getattr(self.last_metrics.get(sid), "updates", 0) or 0),
                loss=0.0,
                motion_norm=0.0,
                pred_delta_norm=0.0,
                sample_count=0,
                temporal_dt=0.0,
            )
            self.last_metrics[sid] = metrics
            return dict(metrics.__dict__)

        n = min(int(x0_np.shape[0]), int(x1_np.shape[0]))
        if n <= 0:
            metrics = Slot4DDeformationMetrics(
                slot_id=sid,
                target_name=str(target_name),
                enabled=True,
                trainable=False,
                valid=False,
                updates=0,
                loss=0.0,
                motion_norm=0.0,
                pred_delta_norm=0.0,
                sample_count=0,
                temporal_dt=0.0,
            )
            self.last_metrics[sid] = metrics
            return dict(metrics.__dict__)

        x0 = torch.as_tensor(x0_np[:n], device=self.device, dtype=torch.float32).detach().clone()
        x1 = torch.as_tensor(x1_np[:n], device=self.device, dtype=torch.float32).detach().clone()
        dt_raw = max(float(int(getattr(curr, "live_step", 0)) - int(getattr(prev, "live_step", 0))), 1.0)
        dt = torch.tensor(min(dt_raw / 60.0, 1.0), device=self.device, dtype=torch.float32)

        model = self._ensure_model(sid)
        opt = self.optimizers[sid]
        loss = torch.tensor(0.0, device=self.device)
        pred_delta = torch.zeros_like(x0)

        for _ in range(self.train_steps_per_update):
            pred_delta = model(x0, dt)
            pred = x0 + pred_delta
            recon_loss = F.mse_loss(pred, x1)
            reg = torch.mean(pred_delta ** 2)
            loss = recon_loss + self.delta_reg_weight * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        prev_updates = int(getattr(self.last_metrics.get(sid), "updates", 0) or 0)
        motion_norm = float(torch.mean(torch.linalg.norm(x1 - x0, dim=-1)).detach().cpu().item())
        pred_delta_norm = float(torch.mean(torch.linalg.norm(pred_delta, dim=-1)).detach().cpu().item())
        metrics = Slot4DDeformationMetrics(
            slot_id=sid,
            target_name=str(target_name),
            enabled=True,
            trainable=True,
            valid=True,
            updates=prev_updates + self.train_steps_per_update,
            loss=float(loss.detach().cpu().item()),
            motion_norm=motion_norm,
            pred_delta_norm=pred_delta_norm,
            sample_count=int(n),
            temporal_dt=float(dt_raw),
        )
        self.last_metrics[sid] = metrics
        return dict(metrics.__dict__)

    def summary(self) -> dict[str, Any]:
        return {
            "slot_0": dict(getattr(self.last_metrics.get(0), "__dict__", {}) or {}),
            "slot_1": dict(getattr(self.last_metrics.get(1), "__dict__", {}) or {}),
        }

    def trainable_params(self) -> int:
        return int(sum(p.numel() for m in self.models.values() for p in m.parameters() if p.requires_grad))
