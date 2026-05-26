from __future__ import annotations

"""Shared runner/model dimension validation helpers.

The same dimension contract is needed by the structured runner config and by the
canonical ConsciousDreamer config factory. Keeping it here avoids coupling the
canonical M5 config helper back to `src.shared.config`.
"""

from typing import Any


MODEL_DIM_KEYS = (
    "action_dim",
    "embodied_dim",
    "hand_motor_dim",
    "tactile_dim",
    "body_state_dim",
)

EXPECTED_RUNNER_MODEL_DIMS = {
    "action_dim": 24,
    "embodied_dim": 15,
    "hand_motor_dim": 44,
    "tactile_dim": 42,
    "body_state_dim": 83,
}


def cfg_get(cfg: Any, key: str) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg[key]
    raise KeyError(key)


def model_dimensions_from_runner_cfg(cfg: Any) -> dict[str, int]:
    return {key: int(cfg_get(cfg, key)) for key in MODEL_DIM_KEYS}


def validate_runner_model_dimensions(cfg: Any) -> dict[str, int]:
    dims = model_dimensions_from_runner_cfg(cfg)
    bad = {
        key: (dims[key], EXPECTED_RUNNER_MODEL_DIMS[key])
        for key in EXPECTED_RUNNER_MODEL_DIMS
        if dims[key] != EXPECTED_RUNNER_MODEL_DIMS[key]
    }
    if bad:
        raise ValueError(
            "Invalid model dimensions in runner config: "
            + ", ".join(f"{key}={got} expected {expected}" for key, (got, expected) in bad.items())
        )
    return dims


__all__ = [
    "MODEL_DIM_KEYS",
    "EXPECTED_RUNNER_MODEL_DIMS",
    "cfg_get",
    "model_dimensions_from_runner_cfg",
    "validate_runner_model_dimensions",
]
