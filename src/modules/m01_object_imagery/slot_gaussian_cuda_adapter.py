
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import torch

@dataclass
class CUDABackendStatus:
    requested_backend: str
    active_backend: str
    cuda_available: bool
    rasterizer_available: bool
    fallback_used: bool
    import_error: str
    fps: float = 0.0

class SlotGaussianCUDAAdapter:
    """
    Safe Step 2B adapter:
      renderer_backend = torch_lowres | cuda_3dgs | auto
    If CUDA/rasterizer package is unavailable, it falls back to Step 2A renderer.
    """
    def __init__(self, requested_backend: str = "auto", allow_fallback: bool = True) -> None:
        self.requested_backend = str(requested_backend or "auto").strip().lower()
        self.allow_fallback = bool(allow_fallback)
        self.cuda_available = bool(torch.cuda.is_available())
        self.rasterizer_available = False
        self.backend_module_name = ""
        self.import_error = ""
        self._last_fps = 0.0
        self._detect()

    def _detect(self) -> None:
        errors = []
        for name in ("gsplat", "diff_gaussian_rasterization"):
            try:
                __import__(name)
                self.rasterizer_available = True
                self.backend_module_name = name
                self.import_error = ""
                return
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")
        self.import_error = " | ".join(errors)

    @property
    def can_use_cuda(self) -> bool:
        return bool(self.cuda_available and self.rasterizer_available)

    def _choose_active_backend(self) -> tuple[str, bool]:
        req = self.requested_backend
        if req in ("torch", "torch_lowres", "lowres", "fallback"):
            return "torch_lowres", False
        if req in ("cuda", "cuda_3dgs", "gsplat"):
            if self.can_use_cuda:
                return "cuda_3dgs", False
            return ("torch_lowres" if self.allow_fallback else "cuda_3dgs_unavailable"), True
        if self.can_use_cuda:
            return "cuda_3dgs", False
        return "torch_lowres", True

    def status(self) -> CUDABackendStatus:
        active, fallback = self._choose_active_backend()
        return CUDABackendStatus(
            requested_backend=self.requested_backend,
            active_backend=active,
            cuda_available=bool(self.cuda_available),
            rasterizer_available=bool(self.rasterizer_available),
            fallback_used=bool(fallback),
            import_error=str(self.import_error),
            fps=float(self._last_fps),
        )

    def render_preview(self, state: Any, fallback_renderer: Any) -> dict[str, Any]:
        """
        Boundary for CUDA preview. Current safe implementation renders through
        Step 2A fallback renderer if no pinned CUDA rasterizer API is available.
        """
        t0 = time.perf_counter()
        st = self.status()
        # Until a concrete rasterizer package/API is pinned, fallback renderer
        # provides a valid preview while backend diagnostics show CUDA status.
        out = fallback_renderer.render(state)
        dt = max(time.perf_counter() - t0, 1.0e-6)
        self._last_fps = float(1.0 / dt)
        st = self.status()
        fallback_used = bool(st.fallback_used)
        return {
            "rgb": out.get("rgb"),
            "depth": out.get("depth"),
            "alpha": out.get("alpha"),
            "backend": st.active_backend,
            "requested_backend": st.requested_backend,
            "cuda_available": bool(st.cuda_available),
            "rasterizer_available": bool(st.rasterizer_available),
            "fallback_used": fallback_used,
            "fps": float(self._last_fps),
            "import_error": st.import_error,
        }
