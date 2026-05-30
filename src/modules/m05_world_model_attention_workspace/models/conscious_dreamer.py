from __future__ import annotations

"""Canonical public API for the M5 ConsciousDreamer world model.

Runtime code should import the current M5 model names from this file:

    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
        ConsciousDreamer,
        ConsciousDreamerConfig,
        make_conscious_dreamer_config_from_world,
    )

M5 is the shared preconscious world-model / attention / focus field. It does
not own true inner speech. M7 should generate inner speech only after M9 binds
M5 focus_context and M10 affect_latents to self.
"""

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought import (
    ConsciousDreamerMemoryThought,
    ConsciousDreamerMemoryThoughtConfig,
)


# Canonical public names used by runner/app code.
ConsciousDreamer = ConsciousDreamerObjectImagery
ConsciousDreamerConfig = ConsciousDreamerObjectImageryConfig

# Explicit aliases for diagnostics/documentation without versioned naming.
ConsciousDreamerLatest = ConsciousDreamer
ConsciousDreamerLatestConfig = ConsciousDreamerConfig
CONSCIOUS_DREAMER_MODEL_FLAVOR = "M5_PRECONSCIOUS_WORLD_MODEL_CANONICAL"
CONSCIOUS_DREAMER_MODEL_ID = "M5_PRECONSCIOUS_WORLD_MODEL"


def make_conscious_dreamer_config_from_world(
    *,
    image_height: int = 128,
    image_width: int = 192,
    body_state_dim: int | None = None,
    tactile_dim: int | None = None,
    hand_motor_dim: int | None = None,
    embodied_dim: int | None = None,
    action_dim: int | None = None,
    symbol_vocab_size: int = 512,
    phoneme_vocab_size: int = 96,
    text_vocab_size: int = 2048,
) -> ConsciousDreamerConfig:
    """Build the canonical M5 config from explicit world/model dimensions.

    Symbol/text vocab args are accepted for API compatibility only. They are no
    longer applied to canonical M5 because inner speech belongs to M7 after M9.
    """
    required_dims = {
        "body_state_dim": body_state_dim,
        "tactile_dim": tactile_dim,
        "hand_motor_dim": hand_motor_dim,
        "embodied_dim": embodied_dim,
        "action_dim": action_dim,
    }
    missing = [name for name, value in required_dims.items() if value is None]
    if missing:
        raise ValueError(
            "make_conscious_dreamer_config_from_world() requires explicit model dimensions. "
            f"Missing: {missing}"
        )

    cfg = ConsciousDreamerConfig()
    cfg.data.image_height = int(image_height)
    cfg.data.image_width = int(image_width)
    cfg.data.body_state_dim = int(body_state_dim)
    cfg.data.tactile_dim = int(tactile_dim)
    cfg.data.hand_motor_dim = int(hand_motor_dim)
    cfg.data.embodied_dim = int(embodied_dim)
    cfg.data.action_dim = int(action_dim)

    # The object-imagery layer fixes input dimensions in __init__, but setting
    # image size here keeps config previews deterministic.
    cfg.object_imagery.image_size = min(int(image_height), int(image_width), 96)
    return cfg


__all__ = [
    "ConsciousDreamer",
    "ConsciousDreamerConfig",
    "ConsciousDreamerLatest",
    "ConsciousDreamerLatestConfig",
    "ConsciousDreamerObjectImagery",
    "ConsciousDreamerObjectImageryConfig",
    "ConsciousDreamerCore",
    "ConsciousDreamerCoreConfig",
    "ConsciousDreamerMemoryThought",
    "ConsciousDreamerMemoryThoughtConfig",
    "make_conscious_dreamer_config_from_world",
    "CONSCIOUS_DREAMER_MODEL_FLAVOR",
    "CONSCIOUS_DREAMER_MODEL_ID",
]
