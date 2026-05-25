from __future__ import annotations

from omegaconf import OmegaConf

from src.apps.runner_config import (
    ALLOWED_TOP_KEYS,
    DEFAULT_MODEL_DIMS,
    DEFAULT_OBJECT_IMAGE_TACTILE_DIM,
    apply_required_runner_defaults,
    build_runner_config,
    filter_runner_config_dict,
    render_resolved_runner_config,
)
from src.shared.config import UnifiedV510Config


def test_filter_runner_config_drops_unknown_top_level_and_nested_keys() -> None:
    raw = OmegaConf.create(
        {
            "mode": "run",
            "unknown_top": 123,
            "runtime": {
                "device": "cpu",
                "seed": 7,
                "unknown_nested": "drop-me",
            },
            "viewer": {
                "allow_mujoco_window": False,
                "unknown_nested": "drop-me-too",
            },
        }
    )

    clean = filter_runner_config_dict(raw)
    assert "unknown_top" not in clean
    assert "unknown_nested" not in clean["runtime"]
    assert "unknown_nested" not in clean["viewer"]
    assert clean["runtime"]["device"] == "cpu"
    assert clean["runtime"]["seed"] == 7


def test_build_runner_config_returns_unified_v510_config() -> None:
    cfg = build_runner_config(
        OmegaConf.create(
            {
                "mode": "train",
                "runtime": {"device": "cpu", "seed": 11},
                "train": {"enabled": False},
            }
        )
    )
    assert isinstance(cfg, UnifiedV510Config)
    assert cfg.mode == "train"
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.seed == 11
    assert cfg.train.enabled is False
    assert cfg.action_dim == DEFAULT_MODEL_DIMS["action_dim"]
    assert cfg.embodied_dim == DEFAULT_MODEL_DIMS["embodied_dim"]
    assert cfg.hand_motor_dim == DEFAULT_MODEL_DIMS["hand_motor_dim"]
    assert cfg.tactile_dim == DEFAULT_MODEL_DIMS["tactile_dim"]
    assert cfg.body_state_dim == DEFAULT_MODEL_DIMS["body_state_dim"]
    assert cfg.object_image.tactile_dim == DEFAULT_OBJECT_IMAGE_TACTILE_DIM


def test_apply_required_runner_defaults_preserves_explicit_dimensions() -> None:
    clean = apply_required_runner_defaults(
        {
            "action_dim": 12,
            "tactile_dim": 34,
            "object_image": {"tactile_dim": 56},
        }
    )

    assert clean["action_dim"] == 12
    assert clean["tactile_dim"] == 34
    assert clean["object_image"]["tactile_dim"] == 56
    assert clean["body_state_dim"] == DEFAULT_MODEL_DIMS["body_state_dim"]


def test_render_resolved_runner_config_contains_known_sections() -> None:
    text = render_resolved_runner_config(OmegaConf.create({"mode": "run"}))
    assert "mode: run" in text
    assert "runtime:" in text
    assert "viewer:" in text


def test_allowed_top_keys_include_current_runner_sections() -> None:
    for key in [
        "mode",
        "runtime",
        "train",
        "viewer",
        "object_image",
        "object_image_open3d",
        "module_debug",
        "module_status_ipc",
        "adaptive_scenario_controller",
    ]:
        assert key in ALLOWED_TOP_KEYS
