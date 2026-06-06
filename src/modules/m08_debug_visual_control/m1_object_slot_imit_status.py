
from __future__ import annotations
from typing import Any, Dict

import torch


def _slot_metric(system: Any, slot_id: int) -> tuple[float, float]:
    try:
        state = getattr(system, "inner_object_state", {}) or {}
        z_slots = state.get("z_obj_slots")
        c_slots = state.get("confidence_slots")
        z_norm = 0.0
        conf = 0.0
        if torch.is_tensor(z_slots) and z_slots.ndim == 3 and 0 <= int(slot_id) < int(z_slots.shape[1]):
            z_norm = float(z_slots[:, int(slot_id), :].detach().float().norm().cpu().item())
        if torch.is_tensor(c_slots) and c_slots.ndim == 3 and 0 <= int(slot_id) < int(c_slots.shape[1]):
            conf = float(c_slots[:, int(slot_id), :].detach().float().mean().cpu().item())
        return z_norm, conf
    except Exception:
        return 0.0, 0.0


def build_m1_object_slot_imit_status(system: Any) -> Dict[str, Any]:
    if hasattr(system, "m1_object_slot_imit_status"):
        payload = system.m1_object_slot_imit_status()
    else:
        payload = dict(getattr(system, "_m1_object_slot_imit_state", {}) or {})
    if not isinstance(payload, dict):
        payload = {}
    out = dict(payload)
    out.setdefault("active", False)
    out.setdefault("kind", "")
    out.setdefault("remaining", 0)
    out.setdefault("duration", 0)
    out.setdefault("selected_slot", int(getattr(system, "_m1_object_slot_imit_selected_slot", 0) or 0))
    out.setdefault("layout", "imit")
    selected_slot = int(out.get("selected_slot", 0) or 0)
    selected_z_norm, selected_confidence = _slot_metric(system, selected_slot)
    out["selected_slot_z_norm"] = selected_z_norm
    out["selected_slot_confidence"] = selected_confidence
    slot_metrics = {}
    for slot_id in out.get("last_slots", []) or []:
        try:
            sid = int(slot_id)
        except Exception:
            continue
        z_norm, conf = _slot_metric(system, sid)
        slot_metrics[str(sid)] = {"z_norm": z_norm, "confidence": conf}
    out["slot_metrics"] = slot_metrics
    out["global_step"] = int(getattr(system, "global_step", 0) or 0)
    out["is_m1_imit"] = True
    out["note"] = "M1 imit object-slot latent injection: fills inner-object slots and selects slot for inner object 3D display."
    return out

__all__ = ["build_m1_object_slot_imit_status"]
