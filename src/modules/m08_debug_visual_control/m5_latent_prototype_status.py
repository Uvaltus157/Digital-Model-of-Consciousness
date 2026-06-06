from __future__ import annotations

"""Status payload for simulated M5 latent prototypes."""

from typing import Any, Dict


def build_m5_latent_prototype_status(system: Any) -> Dict[str, Any]:
    if hasattr(system, "m5_latent_prototype_status"):
        payload = system.m5_latent_prototype_status()
    else:
        payload = dict(getattr(system, "_m5_latent_prototype_state", {}) or {})

    if not isinstance(payload, dict):
        payload = {}

    out = dict(payload)
    out.setdefault("active", False)
    out.setdefault("kind", "")
    out.setdefault("remaining", 0)
    out.setdefault("duration", 0)
    out.setdefault("gate", 0.0)
    out.setdefault("seed_norm", 0.0)
    out.setdefault("cube_similarity", 0.0)
    out.setdefault("tetra_similarity", 0.0)
    out.setdefault("layout", "imit")
    out.setdefault("target_m5_boundary", "FocusFeedbackBoundary")
    out["global_step"] = int(getattr(system, "global_step", 0) or 0)
    out["is_simulated_learned_latent"] = True
    out["note"] = (
        "Simulated learned object latent prototype from m05/imit. "
        "This does not prove M5 is trained; it tests downstream M5 seed/boundary wiring "
        "as if cube/tetrahedron latents existed."
    )
    return out


__all__ = ["build_m5_latent_prototype_status"]
