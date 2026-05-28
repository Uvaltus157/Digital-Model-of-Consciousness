from __future__ import annotations

"""Canonical public API for the M5 ConsciousDreamer world model.

Runtime code should import the current M5 model names from this file:

    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
        ConsciousDreamer,
        ConsciousDreamerConfig,
        make_conscious_dreamer_config_from_world,
    )

This keeps app and runner boundaries on one stable ConsciousDreamer API.
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
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_inner_speech import (
    ConsciousDreamerInnerSpeech,
    ConsciousDreamerInnerSpeechConfig,
)


# Canonical public names used by runner/app code.
ConsciousDreamer = ConsciousDreamerObjectImagery
ConsciousDreamerConfig = ConsciousDreamerObjectImageryConfig

# Explicit latest aliases for diagnostics/documentation.
ConsciousDreamerLatest = ConsciousDreamer
ConsciousDreamerLatestConfig = ConsciousDreamerConfig
CONSCIOUS_DREAMER_MODEL_VERSION = "M5_CONSCIOUS_DREAMER_CANONICAL_V1"
CONSCIOUS_DREAMER_MODEL_ID = "M5_CONSCIOUS_DREAMER"


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

    It intentionally requires all runtime-owned model dimensions to be passed
    explicitly, so config files stay the single source of truth.
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

    cfg.symbolic_report.symbol_vocab_size = int(symbol_vocab_size)
    cfg.symbolic_report.phoneme_vocab_size = int(phoneme_vocab_size)
    cfg.symbolic_report.text_vocab_size = int(text_vocab_size)

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
    "ConsciousDreamerInnerSpeech",
    "ConsciousDreamerInnerSpeechConfig",
    "make_conscious_dreamer_config_from_world",
    "CONSCIOUS_DREAMER_MODEL_VERSION",
    "CONSCIOUS_DREAMER_MODEL_ID",
]
