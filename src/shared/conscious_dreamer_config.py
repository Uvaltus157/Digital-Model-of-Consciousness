from __future__ import annotations

"""Canonical config helpers for the M5 ConsciousDreamer model.

This file is the runner/app-facing replacement for versioned helpers such as
`make_v23_config_from_unified()`. It keeps app-level code away from historical
implementation names and depends only on the shared model-dimension contract.
"""

from typing import Any

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamerConfig
from src.shared.model_dimensions import validate_runner_model_dimensions


def make_conscious_dreamer_config_from_unified(cfg: Any) -> ConsciousDreamerConfig:
    """Build the canonical M5 config from `UnifiedV510Config`.

    The runner config remains the single source of truth for model dimensions.
    """
    dims = validate_runner_model_dimensions(cfg)

    model_cfg = ConsciousDreamerConfig()
    model_cfg.data.image_height = int(cfg.mujoco_world.height)
    model_cfg.data.image_width = int(cfg.mujoco_world.width)

    model_cfg.data.body_state_dim = dims["body_state_dim"]
    model_cfg.data.tactile_dim = dims["tactile_dim"]
    model_cfg.data.hand_motor_dim = dims["hand_motor_dim"]
    model_cfg.data.embodied_dim = dims["embodied_dim"]
    model_cfg.data.action_dim = dims["action_dim"]

    model_cfg.object_imagery.image_size = 96
    return model_cfg


__all__ = ["make_conscious_dreamer_config_from_unified"]
