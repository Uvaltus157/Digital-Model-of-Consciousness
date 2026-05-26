"""Compatibility wrapper.

Neural model implementation moved into this module's models package.
New location: src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery
"""

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)

__all__ = [
    "ConsciousDreamer",
    "ConsciousDreamerConfig",
    "make_conscious_dreamer_config_from_world",
]
