from __future__ import annotations

from types import SimpleNamespace


def _runner_cfg():
    return SimpleNamespace(
        mujoco_world=SimpleNamespace(height=72, width=128),
        action_dim=24,
        embodied_dim=15,
        hand_motor_dim=44,
        tactile_dim=42,
        body_state_dim=83,
    )


def test_legacy_v23_config_helper_matches_canonical_helper_dimensions():
    from src.shared.config import make_v23_config_from_unified
    from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified

    cfg = _runner_cfg()
    canonical = make_conscious_dreamer_config_from_unified(cfg)
    legacy = make_v23_config_from_unified(cfg)

    assert type(legacy) is type(canonical)
    assert legacy.data.image_height == canonical.data.image_height == 72
    assert legacy.data.image_width == canonical.data.image_width == 128
    assert legacy.data.action_dim == canonical.data.action_dim == 24
    assert legacy.data.embodied_dim == canonical.data.embodied_dim == 15
    assert legacy.data.hand_motor_dim == canonical.data.hand_motor_dim == 44
    assert legacy.data.tactile_dim == canonical.data.tactile_dim == 42
    assert legacy.data.body_state_dim == canonical.data.body_state_dim == 83
    assert legacy.object_imagery.image_size == canonical.object_imagery.image_size == 96


def test_runner_model_factory_legacy_config_alias_matches_canonical_helper():
    from src.apps.runner_model_factory import create_conscious_dreamer_config, create_v23_config

    cfg = _runner_cfg()
    canonical = create_conscious_dreamer_config(cfg, speech_vocab_size=321)
    legacy = create_v23_config(cfg, speech_vocab_size=321)

    assert type(legacy) is type(canonical)
    assert legacy.symbolic_report.text_vocab_size == canonical.symbolic_report.text_vocab_size == 321
    assert legacy.data.action_dim == canonical.data.action_dim == 24
    assert legacy.data.embodied_dim == canonical.data.embodied_dim == 15
    assert legacy.data.hand_motor_dim == canonical.data.hand_motor_dim == 44
    assert legacy.data.tactile_dim == canonical.data.tactile_dim == 42
    assert legacy.data.body_state_dim == canonical.data.body_state_dim == 83
