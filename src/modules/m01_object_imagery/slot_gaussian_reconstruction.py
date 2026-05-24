
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from src.modules.m01_object_imagery.slot_gaussian_cuda_adapter import SlotGaussianCUDAAdapter

@dataclass
class SlotGaussianMetrics:
    slot_id: int
    target_name: str
    initialized: bool
    gaussian_count: int
    updates: int
    rgb_loss: float
    depth_loss: float
    total_loss: float
    render_valid: bool
    backend: str = "torch_lowres"
    requested_backend: str = "auto"
    cuda_available: bool = False
    rasterizer_available: bool = False
    fallback_used: bool = False
    preview_fps: float = 0.0
    import_error: str = ""

class SlotGaussianState:
    def __init__(self, *, slot_id: int, target_name: str, xyz: torch.Tensor, color: torch.Tensor, lr: float) -> None:
        self.slot_id = int(slot_id)
        self.target_name = str(target_name)
        device, dtype = xyz.device, xyz.dtype
        self.xyz = torch.nn.Parameter(xyz.to(device=device, dtype=dtype).contiguous())
        self.log_scale = torch.nn.Parameter(torch.full((xyz.shape[0], 1), -3.0, device=device, dtype=dtype))
        self.opacity_logit = torch.nn.Parameter(torch.zeros((xyz.shape[0], 1), device=device, dtype=dtype))
        color = torch.clamp(color.to(device=device, dtype=dtype), 1e-4, 1.0 - 1e-4)
        self.color_logit = torch.nn.Parameter(torch.logit(color))
        self.optimizer = torch.optim.Adam([self.xyz, self.log_scale, self.opacity_logit, self.color_logit], lr=float(lr))
        self.updates = 0
        self.last_rgb_loss = 0.0
        self.last_depth_loss = 0.0
        self.last_total_loss = 0.0
    @property
    def gaussian_count(self) -> int:
        return int(self.xyz.shape[0])

class SimpleGaussianRenderer:
    def __init__(self, image_size: int = 64, max_render_gaussians: int = 256) -> None:
        self.image_size = max(16, int(image_size))
        self.max_render_gaussians = max(16, int(max_render_gaussians))
    def render(self, state: SlotGaussianState) -> dict[str, torch.Tensor]:
        xyz = state.xyz
        n = int(xyz.shape[0])
        if n <= 0:
            raise RuntimeError("no gaussians")
        if n > self.max_render_gaussians:
            idx = torch.linspace(0, n - 1, self.max_render_gaussians, device=xyz.device).long()
            xyz = xyz.index_select(0, idx)
            log_scale = state.log_scale.index_select(0, idx)
            opacity_logit = state.opacity_logit.index_select(0, idx)
            color_logit = state.color_logit.index_select(0, idx)
        else:
            log_scale = state.log_scale
            opacity_logit = state.opacity_logit
            color_logit = state.color_logit
        h = w = self.image_size
        device, dtype = xyz.device, xyz.dtype
        z = torch.clamp(xyz[:, 2:3], min=1e-3)
        fx = torch.tensor(float(w), device=device, dtype=dtype)
        fy = torch.tensor(float(h), device=device, dtype=dtype)
        cx = torch.tensor(float(w - 1) * 0.5, device=device, dtype=dtype)
        cy = torch.tensor(float(h - 1) * 0.5, device=device, dtype=dtype)
        px = fx * xyz[:, 0:1] / z + cx
        py = -fy * xyz[:, 1:2] / z + cy
        yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij")
        grid_x = xx.reshape(1, h*w)
        grid_y = yy.reshape(1, h*w)
        sigma = torch.clamp(torch.exp(log_scale) * float(w), min=0.75, max=float(w) * 0.25)
        opacity = torch.sigmoid(opacity_logit)
        color = torch.sigmoid(color_logit)
        dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
        weights = torch.exp(-0.5 * dist2 / (sigma ** 2 + 1e-6)) * opacity
        denom = torch.clamp(weights.sum(dim=0, keepdim=True), min=1e-6)
        rgb = (weights.t() @ color) / denom.t()
        depth = (weights.t() @ z) / denom.t()
        alpha = torch.clamp(weights.sum(dim=0).reshape(h, w, 1), 0.0, 1.0)
        return {"rgb": rgb.reshape(h, w, 3), "depth": depth.reshape(h, w, 1), "alpha": alpha}

class SlotGaussianReconstructor:
    def __init__(self, *, image_size: int = 64, max_gaussians: int = 768, max_render_gaussians: int = 256,
                 lr: float = 3e-3, train_steps_per_update: int = 1, depth_weight: float = 0.35,
                 scale_reg_weight: float = 1e-4, opacity_reg_weight: float = 1e-4,
                 renderer_backend: str = "auto", allow_fallback: bool = True, preview_every_steps: int = 1,
                 device: str | torch.device | None = None) -> None:
        self.image_size = max(16, int(image_size))
        self.max_gaussians = max(16, int(max_gaussians))
        self.lr = float(lr)
        self.train_steps_per_update = max(1, int(train_steps_per_update))
        self.depth_weight = float(depth_weight)
        self.scale_reg_weight = float(scale_reg_weight)
        self.opacity_reg_weight = float(opacity_reg_weight)
        self.preview_every_steps = max(1, int(preview_every_steps))
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.renderer = SimpleGaussianRenderer(image_size=self.image_size, max_render_gaussians=max_render_gaussians)
        self.cuda_adapter = SlotGaussianCUDAAdapter(requested_backend=renderer_backend, allow_fallback=allow_fallback)
        self.states: dict[int, SlotGaussianState] = {}
        self.last_metrics: dict[int, SlotGaussianMetrics] = {}
        self.last_preview: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _to_tensor(x: Any, device: torch.device) -> torch.Tensor | None:
        if x is None: return None
        try:
            if torch.is_tensor(x):
                t = x.detach().float().to(device)
            else:
                t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
            if t.ndim == 4: t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3, 4): t = t.permute(1, 2, 0)
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return None

    def _resize_image(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        if x.ndim == 2: x = x.unsqueeze(-1)
        if x.shape[-1] > channels: x = x[..., :channels]
        if x.shape[-1] == 1 and channels == 3: x = x.repeat(1, 1, 3)
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x[0].permute(1, 2, 0).contiguous()

    def _target_from_observation(self, observation: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rgb = self._to_tensor(getattr(observation, "rgb", None), self.device)
        depth = self._to_tensor(getattr(observation, "depth", None), self.device)
        if rgb is not None: rgb = torch.clamp(self._resize_image(rgb, 3), 0.0, 1.0)
        if depth is not None: depth = torch.clamp(self._resize_image(depth, 1), min=0.0)
        return rgb, depth

    def _init_from_points(self, *, slot_id: int, target_name: str, points: Any, colors: Any | None) -> SlotGaussianState:
        pts = self._to_tensor(points, self.device)
        if pts is None or pts.ndim != 2 or pts.shape[-1] < 3:
            raise RuntimeError("invalid point cloud for gaussian init")
        pts = pts[:, :3]
        valid = torch.isfinite(pts).all(dim=-1) & (pts[:, 2] > 1e-6)
        pts = pts[valid]
        if pts.shape[0] <= 0: raise RuntimeError("empty valid point cloud")
        if pts.shape[0] > self.max_gaussians:
            idx = torch.linspace(0, pts.shape[0] - 1, self.max_gaussians, device=pts.device).long()
            pts = pts.index_select(0, idx)
        else:
            idx = None
        col = self._to_tensor(colors, self.device)
        if col is None or col.ndim != 2:
            col = torch.ones((pts.shape[0], 3), device=self.device) * 0.5
        else:
            col = col[:, :3][valid]
            if idx is not None: col = col.index_select(0, idx)
            if col.shape[0] != pts.shape[0]: col = torch.ones((pts.shape[0], 3), device=self.device) * 0.5
            col = torch.clamp(col, 0.0, 1.0)
        state = SlotGaussianState(slot_id=slot_id, target_name=target_name, xyz=pts.detach(), color=col.detach(), lr=self.lr)
        self.states[int(slot_id)] = state
        return state

    def backend_status(self) -> dict[str, Any]:
        return dict(self.cuda_adapter.status().__dict__)

    def train_step(self, *, slot_id: int, target_name: str, observation: Any, points: Any | None, colors: Any | None = None) -> dict[str, Any]:
        slot_id = int(slot_id); target_name = str(target_name or "dynamic_object")
        initialized_now = False
        if slot_id not in self.states:
            if points is None:
                status = self.backend_status()
                return {"slot_id": slot_id, "target_name": target_name, "initialized": False, "gaussian_count": 0, "updates": 0, "rgb_loss": 0.0, "depth_loss": 0.0, "total_loss": 0.0, "render_valid": False, "backend": str(status.get("active_backend", "torch_lowres")), "preview_fps": float(status.get("fps", 0.0) or 0.0), **status}
            state = self._init_from_points(slot_id=slot_id, target_name=target_name, points=points, colors=colors)
            initialized_now = True
        else:
            state = self.states[slot_id]; state.target_name = target_name
        target_rgb, target_depth = self._target_from_observation(observation)
        if target_rgb is None and target_depth is None:
            status = self.backend_status()
            return {"slot_id": slot_id, "target_name": target_name, "initialized": initialized_now, "gaussian_count": state.gaussian_count, "updates": state.updates, "rgb_loss": state.last_rgb_loss, "depth_loss": state.last_depth_loss, "total_loss": state.last_total_loss, "render_valid": False, "backend": str(status.get("active_backend", "torch_lowres")), "preview_fps": float(status.get("fps", 0.0) or 0.0), **status}
        total_loss = None; rgb_loss = torch.tensor(0.0, device=self.device); depth_loss = torch.tensor(0.0, device=self.device); preview = None
        for _ in range(self.train_steps_per_update):
            preview = self.cuda_adapter.render_preview(state, self.renderer)
            rendered = {"rgb": preview["rgb"], "depth": preview["depth"], "alpha": preview.get("alpha")}
            loss = torch.tensor(0.0, device=self.device)
            if target_rgb is not None:
                rgb_loss = F.l1_loss(rendered["rgb"], target_rgb); loss = loss + rgb_loss
            if target_depth is not None:
                mask = target_depth > 1e-6
                if torch.any(mask):
                    depth_loss = F.l1_loss(rendered["depth"][mask], target_depth[mask]); loss = loss + self.depth_weight * depth_loss
            loss = loss + self.scale_reg_weight * torch.mean(torch.exp(state.log_scale) ** 2)
            loss = loss + self.opacity_reg_weight * torch.mean(torch.sigmoid(state.opacity_logit) ** 2)
            state.optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([state.xyz, state.log_scale, state.opacity_logit, state.color_logit], 1.0)
            state.optimizer.step(); state.updates += 1; total_loss = loss.detach()
        state.last_rgb_loss = float(rgb_loss.detach().cpu().item())
        state.last_depth_loss = float(depth_loss.detach().cpu().item())
        state.last_total_loss = float(total_loss.detach().cpu().item()) if total_loss is not None else 0.0
        status = self.backend_status()
        if preview is not None:
            status.update({
                "active_backend": str(preview.get("backend", status.get("active_backend", "torch_lowres"))),
                "requested_backend": str(preview.get("requested_backend", status.get("requested_backend", "auto"))),
                "cuda_available": bool(preview.get("cuda_available", status.get("cuda_available", False))),
                "rasterizer_available": bool(preview.get("rasterizer_available", status.get("rasterizer_available", False))),
                "fallback_used": bool(preview.get("fallback_used", status.get("fallback_used", False))),
                "fps": float(preview.get("fps", status.get("fps", 0.0)) or 0.0),
                "import_error": str(preview.get("import_error", status.get("import_error", ""))),
            })
            self.last_preview[slot_id] = preview
        metrics = SlotGaussianMetrics(
            slot_id=slot_id, target_name=target_name, initialized=bool(initialized_now),
            gaussian_count=state.gaussian_count, updates=state.updates,
            rgb_loss=state.last_rgb_loss, depth_loss=state.last_depth_loss, total_loss=state.last_total_loss,
            render_valid=True, backend=str(status.get("active_backend", "torch_lowres")),
            requested_backend=str(status.get("requested_backend", "auto")),
            cuda_available=bool(status.get("cuda_available", False)),
            rasterizer_available=bool(status.get("rasterizer_available", False)),
            fallback_used=bool(status.get("fallback_used", False)),
            preview_fps=float(status.get("fps", 0.0) or 0.0),
            import_error=str(status.get("import_error", "")))
        self.last_metrics[slot_id] = metrics
        return dict(metrics.__dict__)
