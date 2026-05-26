from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_model_dimensions_helper_accepts_expected_runner_dims():
    from src.shared.model_dimensions import model_dimensions_from_runner_cfg, validate_runner_model_dimensions

    cfg = SimpleNamespace(
        action_dim=24,
        embodied_dim=15,
        hand_motor_dim=44,
        tactile_dim=42,
        body_state_dim=83,
    )

    dims = model_dimensions_from_runner_cfg(cfg)
    assert dims == {
        "action_dim": 24,
        "embodied_dim": 15,
        "hand_motor_dim": 44,
        "tactile_dim": 42,
        "body_state_dim": 83,
    }
    assert validate_runner_model_dimensions(cfg) == dims


def test_model_dimensions_helper_rejects_mismatched_dims():
    from src.shared.model_dimensions import validate_runner_model_dimensions

    cfg = SimpleNamespace(
        action_dim=24,
        embodied_dim=11,
        hand_motor_dim=44,
        tactile_dim=42,
        body_state_dim=83,
    )

    with pytest.raises(ValueError, match="embodied_dim=11 expected 15"):
        validate_runner_model_dimensions(cfg)
