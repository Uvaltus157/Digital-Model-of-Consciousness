from __future__ import annotations

"""M5 simulated learned latent prototypes.

IMIT MODULE.

This file intentionally lives under:

    src/modules/m05_world_model_attention_workspace/imit/

It does NOT train M5 and does NOT overwrite real neural-network weights.
It provides deterministic object latent prototypes for diagnostic imitation:

    cube
    tetrahedron
    cube↔tetrahedron morph

The prototype enters M5 through the same path as other feedback/replay seeds:

    focus_context_seed
    focus_context_seed_gate
    ↓
    FocusFeedbackBoundary

Interpretation:
    - If downstream monitors react, the circuit is wired.
    - It does not prove that M5 has truly learned cube/tetrahedron semantics.
"""

from typing import Any, Dict
import math

import torch


_SHAPE_DESCRIPTORS: Dict[str, Dict[str, float]] = {
    "cube": {
        "faces": 6.0,
        "edges": 12.0,
        "vertices": 8.0,
        "triangular_faces": 0.0,
        "square_faces": 6.0,
        "symmetry_order": 24.0,
        "sharp_edges": 12.0,
        "compactness": 0.82,
        "axis_aligned": 1.0,
        "simplex": 0.0,
        "corner_density": 0.80,
        "flatness": 1.0,
    },
    "tetrahedron": {
        "faces": 4.0,
        "edges": 6.0,
        "vertices": 4.0,
        "triangular_faces": 4.0,
        "square_faces": 0.0,
        "symmetry_order": 12.0,
        "sharp_edges": 6.0,
        "compactness": 0.66,
        "axis_aligned": 0.0,
        "simplex": 1.0,
        "corner_density": 1.0,
        "flatness": 1.0,
    },
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _shape_descriptor(name: str) -> Dict[str, float]:
    key = str(name or "cube").lower().strip()
    if key in ("tetra", "tetrahedron", "pyramid"):
        key = "tetrahedron"
    elif key in ("box", "hexahedron", "cuboid"):
        key = "cube"
    return dict(_SHAPE_DESCRIPTORS.get(key, _SHAPE_DESCRIPTORS["cube"]))


def _descriptor_to_vector(desc: Dict[str, float], dim: int = 256, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    dim = max(16, int(dim))
    ordered = [
        "faces",
        "edges",
        "vertices",
        "triangular_faces",
        "square_faces",
        "symmetry_order",
        "sharp_edges",
        "compactness",
        "axis_aligned",
        "simplex",
        "corner_density",
        "flatness",
    ]
    raw = torch.zeros(dim, dtype=torch.float32, device=device)
    normalizers = {
        "faces": 12.0,
        "edges": 24.0,
        "vertices": 16.0,
        "triangular_faces": 12.0,
        "square_faces": 12.0,
        "symmetry_order": 24.0,
        "sharp_edges": 24.0,
        "compactness": 1.0,
        "axis_aligned": 1.0,
        "simplex": 1.0,
        "corner_density": 1.0,
        "flatness": 1.0,
    }

    for i, key in enumerate(ordered):
        raw[i] = _safe_float(desc.get(key), 0.0) / float(normalizers.get(key, 1.0))

    base = sum((i + 1) * _safe_float(desc.get(k), 0.0) for i, k in enumerate(ordered))
    for i in range(len(ordered), dim):
        a = math.sin(0.173 * (i + 1) * (base + 1.0))
        b = math.cos(0.097 * (i + 3) * (_safe_float(desc.get("edges"), 1.0) + 1.0))
        c = math.sin(0.041 * (i + 7) * (_safe_float(desc.get("symmetry_order"), 1.0) + 1.0))
        raw[i] = float(0.45 * a + 0.35 * b + 0.20 * c)

    raw = raw - raw.mean()
    raw = raw / raw.norm().clamp_min(1e-6)
    return raw.reshape(1, dim)


def _cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    try:
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return 0.0
        aa = a.detach().float().reshape(1, -1)
        bb = b.detach().float().reshape(1, -1).to(aa.device)
        n = min(int(aa.shape[-1]), int(bb.shape[-1]))
        if n <= 0:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(aa[:, :n], bb[:, :n], dim=-1).reshape(-1)[0].cpu().item())
    except Exception:
        return 0.0


def _device_from_latest(latest: Any) -> torch.device:
    if isinstance(latest, dict):
        for value in latest.values():
            if torch.is_tensor(value):
                return value.device
            if isinstance(value, dict):
                for vv in value.values():
                    if torch.is_tensor(vv):
                        return vv.device
    return torch.device("cpu")


class M5LatentPrototypeRuntimeMixin:
    def _m5_latent_prototype_dim(self) -> int:
        self_core_cfg = getattr(getattr(self, "cfg", None), "self_core", None)
        return int(getattr(self_core_cfg, "focus_context_dim", 256) or 256)

    def make_m5_latent_prototype(
        self,
        kind: str = "cube",
        *,
        device: torch.device | None = None,
        dim: int | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        dim = int(dim or self._m5_latent_prototype_dim())
        desc = _shape_descriptor(kind)
        latent = _descriptor_to_vector(desc, dim=dim, device=device)
        return latent, desc

    def request_m5_latent_prototype(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = dict(payload or {})
        kind = str(payload.get("kind", payload.get("object_type", "cube"))).lower().strip()

        if kind in ("clear", "stop", "off", "none"):
            self._m5_latent_prototype_state = {
                "active": False,
                "kind": "clear",
                "remaining": 0,
                "duration": 0,
                "gate": 0.0,
                "source": str(payload.get("source", "ipc")),
                "layout": "imit",
            }
            print("[m5_latent_prototype][imit] cleared")
            if hasattr(self, "write_module_debug_status"):
                self.write_module_debug_status()
            return dict(self._m5_latent_prototype_state)

        if kind in ("tetra", "pyramid"):
            kind = "tetrahedron"
        elif kind in ("box", "hexahedron", "cuboid"):
            kind = "cube"
        elif kind not in ("cube", "tetrahedron", "morph"):
            kind = "cube"

        duration = max(1, min(1000, int(_safe_float(payload.get("duration", 120), 120))))
        gate = max(0.0, min(2.0, _safe_float(payload.get("gate", payload.get("intensity", 0.85)), 0.85)))
        alpha = max(0.0, min(1.0, _safe_float(payload.get("alpha", 0.5), 0.5)))

        device = _device_from_latest(getattr(self, "latest_out", {}) or {})
        cube, cube_desc = self.make_m5_latent_prototype("cube", device=device)
        tetra, tetra_desc = self.make_m5_latent_prototype("tetrahedron", device=device)

        if kind == "tetrahedron":
            latent, desc = tetra, tetra_desc
        elif kind == "morph":
            latent = (1.0 - alpha) * cube + alpha * tetra
            latent = latent / latent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            desc = {
                k: (1.0 - alpha) * cube_desc.get(k, 0.0) + alpha * tetra_desc.get(k, 0.0)
                for k in set(cube_desc) | set(tetra_desc)
            }
            desc["morph_alpha"] = alpha
        else:
            latent, desc = cube, cube_desc

        gate_tensor = torch.tensor([[float(gate)]], dtype=torch.float32, device=latent.device)
        self._m5_latent_prototype_seed = latent.detach()
        self._m5_latent_prototype_gate = gate_tensor.detach()
        self._m5_latent_prototype_state = {
            "active": True,
            "kind": kind,
            "remaining": int(duration),
            "duration": int(duration),
            "gate": float(gate),
            "alpha": float(alpha),
            "seed_norm": float(latent.detach().float().norm().cpu().item()),
            "cube_similarity": _cosine(latent, cube),
            "tetra_similarity": _cosine(latent, tetra),
            "descriptor": dict(desc),
            "source": str(payload.get("source", "ipc")),
            "started_step": int(getattr(self, "global_step", 0)),
            "target_m5_boundary": "FocusFeedbackBoundary(workspace_seed + simulated_learned_object_latent)",
            "layout": "imit",
        }

        print(
            "[m5_latent_prototype][imit] "
            f"kind={kind} gate={gate:.3f} duration={duration} "
            f"cube_sim={self._m5_latent_prototype_state['cube_similarity']:.3f} "
            f"tetra_sim={self._m5_latent_prototype_state['tetra_similarity']:.3f}"
        )
        if hasattr(self, "write_module_debug_status"):
            self.write_module_debug_status()
        return dict(self._m5_latent_prototype_state)

    def get_m5_latent_prototype_focus_seed(self, stage: str = "model_step"):
        del stage
        state = getattr(self, "_m5_latent_prototype_state", None)
        if not isinstance(state, dict) or not bool(state.get("active", False)):
            return None, None

        remaining = int(state.get("remaining", 0) or 0)
        if remaining <= 0:
            state["active"] = False
            self._m5_latent_prototype_state = state
            return None, None

        seed = getattr(self, "_m5_latent_prototype_seed", None)
        gate = getattr(self, "_m5_latent_prototype_gate", None)
        if not torch.is_tensor(seed):
            return None, None

        state["remaining"] = max(0, remaining - 1)
        state["active"] = bool(state["remaining"] > 0)
        self._m5_latent_prototype_state = state

        latest = getattr(self, "latest_out", None)
        if isinstance(latest, dict):
            packet = latest.setdefault("m5_latent_prototype", {})
            if isinstance(packet, dict):
                packet.update(dict(state))
                packet["next_focus_context_seed"] = seed.detach()
                packet["next_focus_context_seed_gate"] = gate.detach() if torch.is_tensor(gate) else gate
                packet["seed_source"] = "m5_latent_prototype_simulator"
                packet["layout"] = "imit"

        return seed.detach(), gate.detach() if torch.is_tensor(gate) else gate

    def m5_latent_prototype_status(self) -> Dict[str, Any]:
        state = getattr(self, "_m5_latent_prototype_state", None)
        if not isinstance(state, dict):
            return {"active": False, "kind": "", "remaining": 0, "layout": "imit"}

        out = dict(state)
        seed = getattr(self, "_m5_latent_prototype_seed", None)
        gate = getattr(self, "_m5_latent_prototype_gate", None)
        out["seed_norm"] = float(seed.detach().float().norm().cpu().item()) if torch.is_tensor(seed) else 0.0
        out["gate"] = float(gate.detach().reshape(-1)[0].cpu().item()) if torch.is_tensor(gate) else float(out.get("gate", 0.0) or 0.0)
        out["layout"] = "imit"
        return out


__all__ = ["M5LatentPrototypeRuntimeMixin"]
