from __future__ import annotations


def test_legacy_unified_viewer_uses_canonical_conscious_dreamer_imports():
    import src.apps.unified_conscious_viewer as viewer
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
        ConsciousDreamer,
        make_conscious_dreamer_config_from_world,
    )

    assert viewer.ConsciousDreamer is ConsciousDreamer
    assert viewer.make_conscious_dreamer_config_from_world is make_conscious_dreamer_config_from_world
    assert not hasattr(viewer, "ConsciousDreamerV22")
    assert not hasattr(viewer, "make_v22_config_from_world")


def test_legacy_unified_v57_config_defaults_are_importable():
    from src.apps.unified_conscious_viewer import UnifiedV57Config

    cfg = UnifiedV57Config()
    assert cfg.action_dim == 24
    assert cfg.tactile_dim == 42
    assert cfg.inner_world.enabled is True
