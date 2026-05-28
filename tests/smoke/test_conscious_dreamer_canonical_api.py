from __future__ import annotations


def test_conscious_dreamer_canonical_api_exports_latest_model():
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
        CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER,
        CONSCIOUS_DREAMER_MODEL_VERSION,
        ConsciousDreamer,
        ConsciousDreamerConfig,
        ConsciousDreamerLatest,
        ConsciousDreamerLatestConfig,
        make_conscious_dreamer_config_from_world,
    )
    from src.modules.m05_world_model_attention_workspace.models import (
        ConsciousDreamer as PackageConsciousDreamer,
        ConsciousDreamerConfig as PackageConsciousDreamerConfig,
        make_conscious_dreamer_config_from_world as package_make_config,
    )
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
        ConsciousDreamerObjectImagery,
        ConsciousDreamerObjectImageryConfig,
    )

    assert ConsciousDreamer is ConsciousDreamerObjectImagery
    assert ConsciousDreamerConfig is ConsciousDreamerObjectImageryConfig
    assert ConsciousDreamerLatest is ConsciousDreamer
    assert ConsciousDreamerLatestConfig is ConsciousDreamerConfig
    assert PackageConsciousDreamer is ConsciousDreamer
    assert PackageConsciousDreamerConfig is ConsciousDreamerConfig
    assert package_make_config is make_conscious_dreamer_config_from_world
    assert CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER == "V23"
    assert CONSCIOUS_DREAMER_MODEL_VERSION.startswith("M5_CONSCIOUS_DREAMER")


def test_conscious_dreamer_canonical_world_config_factory_sets_dimensions():
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
        ConsciousDreamerConfig,
        make_conscious_dreamer_config_from_world,
    )

    cfg = make_conscious_dreamer_config_from_world(
        image_height=64,
        image_width=96,
        body_state_dim=83,
        tactile_dim=42,
        hand_motor_dim=44,
        embodied_dim=15,
        action_dim=24,
        text_vocab_size=123,
    )

    assert isinstance(cfg, ConsciousDreamerConfig)
    assert cfg.data.image_height == 64
    assert cfg.data.image_width == 96
    assert cfg.data.body_state_dim == 83
    assert cfg.data.tactile_dim == 42
    assert cfg.data.hand_motor_dim == 44
    assert cfg.data.embodied_dim == 15
    assert cfg.data.action_dim == 24
    assert cfg.symbolic_report.text_vocab_size == 123
    assert cfg.object_imagery.image_size == 64


def test_runner_model_factory_uses_canonical_api_names():
    import src.apps.runner_model_factory as factory

    assert hasattr(factory, "create_conscious_dreamer")
    assert hasattr(factory, "create_conscious_dreamer_config")
