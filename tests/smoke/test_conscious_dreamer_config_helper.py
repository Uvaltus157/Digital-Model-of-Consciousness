from __future__ import annotations

from types import SimpleNamespace


def test_canonical_config_helper_builds_latest_config_from_unified_shape():
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamerConfig
    from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified

    cfg = SimpleNamespace(
        mujoco_world=SimpleNamespace(height=72, width=128),
        action_dim=24,
        embodied_dim=15,
        hand_motor_dim=44,
        tactile_dim=42,
        body_state_dim=83,
    )

    model_cfg = make_conscious_dreamer_config_from_unified(cfg)

    assert isinstance(model_cfg, ConsciousDreamerConfig)
    assert model_cfg.data.image_height == 72
    assert model_cfg.data.image_width == 128
    assert model_cfg.data.action_dim == 24
    assert model_cfg.data.embodied_dim == 15
    assert model_cfg.data.hand_motor_dim == 44
    assert model_cfg.data.tactile_dim == 42
    assert model_cfg.data.body_state_dim == 83
    assert model_cfg.object_imagery.image_size == 96
