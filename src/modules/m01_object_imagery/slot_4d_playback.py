
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Slot4DPlaybackMetrics:
    slot_id: int
    target_name: str
    enabled: bool
    valid: bool
    render_valid: bool
    deformation_used: bool
    gaussian_count: int
    playback_phase: float
    pred_delta_norm: float
    backend: str
    preview_fps: float
    frame_count: int
    model_type: str = "Slot4DPlaybackRenderer"


class _DeformedGaussianState:
    """
    Read-only Gaussian state wrapper for preview rendering.

    It keeps scale/opacity/color from the current 3DGS state and replaces xyz
    with deformation-aware xyz(t).
    """

    def __init__(self, base_state: Any, xyz: torch.Tensor) -> None:
        self.xyz = xyz
        self.log_scale = getattr(base_state, "log_scale", None)
        self.opacity_logit = getattr(base_state, "opacity_logit", None)
        self.color_logit = getattr(base_state, "color_logit", None)
        self.slot_id = int(getattr(base_state, "slot_id", -1))
        self.target_name = str(getattr(base_state, "target_name", "unknown"))

    @property
    def gaussian_count(self) -> int:
        try:
            return int(self.xyz.shape[0])
        except Exception:
            return 0


class Slot4DPlaybackRenderer:
    """
    Step 3C: deformation-aware 4D playback preview.

    This does not train. It renders an object at a time phase by applying the
    Step 3B deformation model to the current per-slot Gaussian state:

        xyz_t = xyz_base + deformation_model(xyz_base, playback_phase)

    Rendering uses the existing Gaussian renderer/backend adapter, so Step 2B
    backend switching remains the source of truth.
    """

    def __init__(self, *, period_steps: int = 120, strength: float = 1.0) -> None:
        self.period_steps = max(2, int(period_steps))
        self.strength = float(strength)
        self.frame_counts: dict[int, int] = {}
        self.last_metrics: dict[int, Slot4DPlaybackMetrics] = {}
        self.last_preview: dict[int, dict[str, Any]] = {}

    def phase_from_step(self, live_step: int) -> float:
        return float(int(live_step) % self.period_steps) / float(self.period_steps - 1)

    def render_slot(
        self,
        *,
        slot_id: int,
        target_name: str,
        live_step: int,
        gaussian_reconstructor: Any,
        deformation_trainer: Any,
        playback_phase: float | None = None,
    ) -> dict[str, Any]:
        sid = int(slot_id)
        states = getattr(gaussian_reconstructor, "states", {}) or {}
        base_state = states.get(sid)
        if base_state is None:
            metrics = Slot4DPlaybackMetrics(
                slot_id=sid,
                target_name=str(target_name),
                enabled=True,
                valid=False,
                render_valid=False,
                deformation_used=False,
                gaussian_count=0,
                playback_phase=0.0 if playback_phase is None else float(playback_phase),
                pred_delta_norm=0.0,
                backend="none",
                preview_fps=0.0,
                frame_count=int(self.frame_counts.get(sid, 0)),
            )
            self.last_metrics[sid] = metrics
            return dict(metrics.__dict__)

        phase = self.phase_from_step(live_step) if playback_phase is None else float(playback_phase)
        phase = max(0.0, min(1.0, phase))

        xyz = getattr(base_state, "xyz", None)
        if xyz is None or not torch.is_tensor(xyz):
            metrics = Slot4DPlaybackMetrics(
                slot_id=sid,
                target_name=str(target_name),
                enabled=True,
                valid=False,
                render_valid=False,
                deformation_used=False,
                gaussian_count=0,
                playback_phase=phase,
                pred_delta_norm=0.0,
                backend="none",
                preview_fps=0.0,
                frame_count=int(self.frame_counts.get(sid, 0)),
            )
            self.last_metrics[sid] = metrics
            return dict(metrics.__dict__)

        deformation_used = False
        pred_delta_norm = 0.0
        render_state = base_state

        models = getattr(deformation_trainer, "models", {}) or {}
        model = models.get(sid)
        if model is not None:
            try:
                with torch.no_grad():
                    x = xyz.detach()
                    delta = model(x[:, :3], torch.tensor(float(phase), device=x.device, dtype=x.dtype))
                    deformed_xyz = x.clone()
                    deformed_xyz[:, :3] = x[:, :3] + self.strength * delta
                    pred_delta_norm = float(torch.mean(torch.linalg.norm(delta, dim=-1)).detach().cpu().item())
                    render_state = _DeformedGaussianState(base_state, deformed_xyz)
                    deformation_used = True
            except Exception:
                render_state = base_state
                deformation_used = False
                pred_delta_norm = 0.0

        backend = "torch_lowres"
        preview_fps = 0.0
        render_valid = False
        preview_payload: dict[str, Any] = {}
        try:
            adapter = getattr(gaussian_reconstructor, "cuda_adapter", None)
            fallback_renderer = getattr(gaussian_reconstructor, "renderer", None)
            if adapter is not None and fallback_renderer is not None:
                preview = adapter.render_preview(render_state, fallback_renderer)
                preview_payload = dict(preview)
                backend = str(preview.get("backend", "torch_lowres"))
                preview_fps = float(preview.get("fps", 0.0) or 0.0)
                render_valid = bool(preview.get("rgb", None) is not None)
            elif fallback_renderer is not None:
                rendered = fallback_renderer.render(render_state)
                preview_payload = {"rgb": rendered.get("rgb"), "depth": rendered.get("depth"), "alpha": rendered.get("alpha")}
                backend = "torch_lowres"
                render_valid = bool(preview_payload.get("rgb", None) is not None)
        except Exception:
            render_valid = False

        self.frame_counts[sid] = int(self.frame_counts.get(sid, 0)) + (1 if render_valid else 0)
        if render_valid:
            self.last_preview[sid] = {
                "rgb": preview_payload.get("rgb"),
                "depth": preview_payload.get("depth"),
                "alpha": preview_payload.get("alpha"),
                "backend": backend,
                "preview_fps": preview_fps,
            }
        metrics = Slot4DPlaybackMetrics(
            slot_id=sid,
            target_name=str(target_name),
            enabled=True,
            valid=bool(render_valid and deformation_used),
            render_valid=bool(render_valid),
            deformation_used=bool(deformation_used),
            gaussian_count=int(getattr(base_state, "gaussian_count", 0) or 0),
            playback_phase=float(phase),
            pred_delta_norm=float(pred_delta_norm),
            backend=str(backend),
            preview_fps=float(preview_fps),
            frame_count=int(self.frame_counts.get(sid, 0)),
        )
        self.last_metrics[sid] = metrics
        return dict(metrics.__dict__)

    def summary(self) -> dict[str, Any]:
        return {
            "slot_0": dict(getattr(self.last_metrics.get(0), "__dict__", {}) or {}),
            "slot_1": dict(getattr(self.last_metrics.get(1), "__dict__", {}) or {}),
        }
