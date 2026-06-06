
from __future__ import annotations

"""M1 inner-object slot latent imitator.

Imitator layout:
    src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py

It simulates M1 object latents for inner-object slots. It does not train or
overwrite model weights. It feeds deterministic z_obj-like cube/tetrahedron
latents into the existing inner-object proposal path so ObjectSlotMemory writes
specific slots and inner_object_viz can display the selected slot.
"""

from typing import Any, Dict
import math
import torch

_DESCRIPTORS: Dict[str, Dict[str, float]] = {
    "cube": {
        "faces": 6.0, "edges": 12.0, "vertices": 8.0,
        "square_faces": 6.0, "triangular_faces": 0.0,
        "symmetry": 24.0, "corner_density": 0.80,
        "axis_aligned": 1.0, "simplex": 0.0,
    },
    "tetrahedron": {
        "faces": 4.0, "edges": 6.0, "vertices": 4.0,
        "square_faces": 0.0, "triangular_faces": 4.0,
        "symmetry": 12.0, "corner_density": 1.0,
        "axis_aligned": 0.0, "simplex": 1.0,
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

def _normalize_kind(kind: str) -> str:
    k = str(kind or "cube").lower().strip()
    if k in ("tetra", "pyramid"):
        return "tetrahedron"
    if k in ("box", "hexahedron", "cuboid"):
        return "cube"
    if k in ("morph", "cube↔tetra"):
        return "morph"
    if k in ("cube_tetra", "cube_tetra_fill", "cube+tetra", "cube_and_tetra"):
        return "cube_tetra"
    if k in ("clear", "none", "off", "stop"):
        return "clear"
    return k if k in ("cube", "tetrahedron") else "cube_tetra"

def _descriptor(kind: str, alpha: float = 0.5) -> Dict[str, float]:
    kind = _normalize_kind(kind)
    if kind == "morph":
        a = max(0.0, min(1.0, float(alpha)))
        c, t = _DESCRIPTORS["cube"], _DESCRIPTORS["tetrahedron"]
        d = {k: (1.0 - a) * c.get(k, 0.0) + a * t.get(k, 0.0) for k in set(c) | set(t)}
        d["morph_alpha"] = a
        return d
    return dict(_DESCRIPTORS["tetrahedron" if kind == "tetrahedron" else "cube"])

def _descriptor_to_latent(desc: Dict[str, float], dim: int, device: torch.device) -> torch.Tensor:
    dim = max(16, int(dim))
    keys = ["faces","edges","vertices","square_faces","triangular_faces","symmetry","corner_density","axis_aligned","simplex","morph_alpha"]
    norm = {"faces":12.0,"edges":24.0,"vertices":16.0,"square_faces":12.0,"triangular_faces":12.0,"symmetry":24.0,
            "corner_density":1.0,"axis_aligned":1.0,"simplex":1.0,"morph_alpha":1.0}
    z = torch.zeros(dim, dtype=torch.float32, device=device)
    for i, k in enumerate(keys):
        z[i] = _safe_float(desc.get(k), 0.0) / float(norm.get(k, 1.0))
    base = sum((i + 1) * _safe_float(desc.get(k), 0.0) for i, k in enumerate(keys))
    for i in range(len(keys), dim):
        z[i] = float(
            0.40 * math.sin(0.113 * (i + 1) * (base + 1.0))
            + 0.35 * math.cos(0.071 * (i + 3) * (_safe_float(desc.get("edges"), 1.0) + 1.0))
            + 0.25 * math.sin(0.037 * (i + 5) * (_safe_float(desc.get("symmetry"), 1.0) + 1.0))
        )
    z = z - z.mean()
    z = z / z.norm().clamp_min(1e-6)
    return z.reshape(1, dim)

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        aa = a.detach().float().reshape(1, -1)
        bb = b.detach().float().reshape(1, -1).to(aa.device)
        n = min(aa.shape[-1], bb.shape[-1])
        return float(torch.nn.functional.cosine_similarity(aa[:, :n], bb[:, :n], dim=-1).item())
    except Exception:
        return 0.0

class M1ObjectSlotLatentImitRuntimeMixin:
    def _m1_imit_latent_dim(self) -> int:
        cfg_obj = getattr(getattr(self, "cfg", None), "object_image", None)
        return int(getattr(cfg_obj, "latent_dim", 128) or 128)

    def make_m1_object_slot_latent(self, kind: str, *, alpha: float = 0.5, device: torch.device | None = None, dim: int | None = None):
        dim = int(dim or self._m1_imit_latent_dim())
        desc = _descriptor(kind, alpha=alpha)
        z = _descriptor_to_latent(desc, dim=dim, device=device or torch.device("cpu"))
        return z, desc

    def request_m1_object_slot_latents(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = dict(payload or {})
        kind = _normalize_kind(str(payload.get("kind", "cube_tetra")))
        if kind == "clear":
            self._m1_object_slot_imit_state = {"active": False, "kind": "clear", "remaining": 0, "layout": "imit"}
            print("[m1_object_slot_imit] cleared")
            if hasattr(self, "write_module_debug_status"):
                self.write_module_debug_status()
            return dict(self._m1_object_slot_imit_state)

        duration = max(1, min(1000, int(_safe_float(payload.get("duration", 180), 180))))
        gate = max(0.0, min(2.0, _safe_float(payload.get("gate", 1.0), 1.0)))
        alpha = max(0.0, min(1.0, _safe_float(payload.get("alpha", 0.5), 0.5)))
        auto_select = bool(payload.get("auto_select_slot", True))
        cfg_obj = getattr(getattr(self, "cfg", None), "object_image", None)
        n_slots = max(1, int(getattr(cfg_obj, "num_slots", 10) or 10))

        items = []
        if kind in ("cube", "tetrahedron", "morph"):
            default_slot = 1 if kind == "cube" else 2 if kind == "tetrahedron" else 3
            slot = int(_safe_float(payload.get("slot", payload.get("selected_slot", default_slot)), default_slot))
            items.append({"kind": kind, "slot": max(0, min(n_slots - 1, slot)), "alpha": alpha})
        else:
            cube_slot = int(_safe_float(payload.get("cube_slot", 1), 1))
            tetra_slot = int(_safe_float(payload.get("tetra_slot", 2), 2))
            items.append({"kind": "cube", "slot": max(0, min(n_slots - 1, cube_slot)), "alpha": 0.0})
            items.append({"kind": "tetrahedron", "slot": max(0, min(n_slots - 1, tetra_slot)), "alpha": 1.0})
            if bool(payload.get("include_morph", False)):
                morph_slot = int(_safe_float(payload.get("morph_slot", 3), 3))
                items.append({"kind": "morph", "slot": max(0, min(n_slots - 1, morph_slot)), "alpha": alpha})

        selected_slot = int(payload.get("selected_slot", items[-1]["slot"] if items else 0))
        selected_slot = max(0, min(n_slots - 1, selected_slot))
        state = {
            "active": True, "kind": kind, "remaining": int(duration), "duration": int(duration),
            "gate": float(gate), "alpha": float(alpha), "items": list(items),
            "selected_slot": int(selected_slot), "auto_select_slot": bool(auto_select),
            "source": str(payload.get("source", "ipc")), "layout": "imit",
            "target": "M1 build_inner_object_vision_proposals -> ObjectSlotMemory.force_slot_index",
            "started_step": int(getattr(self, "global_step", 0)),
        }
        self._m1_object_slot_imit_state = state
        if auto_select:
            self._m1_select_inner_object_slot(selected_slot)
        print(
            "[m1_object_slot_imit][request] "
            f"kind={kind} items={[(i['kind'], i['slot']) for i in items]} "
            f"selected_slot={selected_slot} duration={duration}"
        )
        if hasattr(self, "write_module_debug_status"):
            self.write_module_debug_status()
        return dict(state)

    def _m1_select_inner_object_slot(self, slot: int) -> None:
        slot = int(slot)
        try:
            if getattr(self, "inner_object_viz", None) is not None:
                self.inner_object_viz.requested_dream_slot_index = slot
        except Exception:
            pass
        self._ipc_inner_object_dream_slot_index = slot
        self._m1_object_slot_imit_selected_slot = slot

    def get_m1_imit_inner_object_proposals(self, scene: torch.Tensor | None = None):
        state = getattr(self, "_m1_object_slot_imit_state", None)
        if not isinstance(state, dict) or not bool(state.get("active", False)):
            return None
        remaining = int(state.get("remaining", 0) or 0)
        if remaining <= 0:
            state["active"] = False
            self._m1_object_slot_imit_state = state
            return None

        device = scene.device if torch.is_tensor(scene) else torch.device("cpu")
        dim = self._m1_imit_latent_dim()
        gate = float(state.get("gate", 1.0) or 1.0)
        cube_ref, _ = self.make_m1_object_slot_latent("cube", device=device, dim=dim)
        tetra_ref, _ = self.make_m1_object_slot_latent("tetrahedron", device=device, dim=dim)
        latents, slots, kinds, names, details = [], [], [], [], []

        for item in list(state.get("items", []) or []):
            k = _normalize_kind(str(item.get("kind", "cube")))
            slot = int(item.get("slot", 0))
            alpha = float(item.get("alpha", state.get("alpha", 0.5)))
            z, desc = self.make_m1_object_slot_latent(k, alpha=alpha, device=device, dim=dim)
            z = z * gate
            latents.append(z)
            slots.append(slot)
            kinds.append("m1_imit_dynamic_object")
            names.append(k)
            details.append({
                "kind": k, "slot": int(slot), "alpha": float(alpha),
                "norm": float(z.detach().float().norm().cpu().item()),
                "cube_similarity": _cos(z, cube_ref), "tetra_similarity": _cos(z, tetra_ref),
                "descriptor": desc,
            })
        if not latents:
            return None

        state["remaining"] = max(0, remaining - 1)
        state["active"] = bool(state["remaining"] > 0)
        state["last_details"] = details
        state["last_slots"] = list(slots)
        state["last_kinds"] = list(kinds)
        state["last_names"] = list(names)
        self._m1_object_slot_imit_state = state
        proposals = torch.stack(latents, dim=1)
        print(
            "[m1_object_slot_imit][proposal] "
            f"slots={slots} names={names} shape={tuple(proposals.shape)} "
            f"norms={[round(float(d['norm']), 4) for d in details]}"
        )

        return {
            "proposals": proposals,
            "target_slots": slots,
            "proposal_kinds": kinds,
            "target_names": names,
            "details": details,
        }

    def m1_object_slot_imit_status(self) -> Dict[str, Any]:
        state = getattr(self, "_m1_object_slot_imit_state", None)
        if not isinstance(state, dict):
            return {"active": False, "kind": "", "remaining": 0, "layout": "imit"}
        out = dict(state)
        out.setdefault("layout", "imit")
        return out

__all__ = ["M1ObjectSlotLatentImitRuntimeMixin"]
