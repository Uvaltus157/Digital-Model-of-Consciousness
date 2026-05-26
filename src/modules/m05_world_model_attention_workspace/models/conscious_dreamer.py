from __future__ import annotations

"""Canonical public API for the M5 ConsciousDreamer world model.

Historically M5 grew through implementation layers named V2, V21, V22 and V23:

- V2  : base multimodal Dreamer-style stack;
- V21 : memory + thought loop;
- V22 : inner-speech / symbolic report;
- V23 : object-imagery decoder integration.

The active runtime should not import versioned class names directly. New code
should import the canonical names from this file:

    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
        ConsciousDreamer,
        ConsciousDreamerConfig,
    )

The versioned files remain as internal implementation layers and compatibility
imports only. This keeps checkpoints and older tests safer while giving the app a
single stable M5 model name.
"""

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
)


# Canonical public names used by runner/app code.
ConsciousDreamer = ConsciousDreamerV23
ConsciousDreamerConfig = ConsciousDreamerV23Config

# Explicit latest aliases for diagnostics/documentation.
ConsciousDreamerLatest = ConsciousDreamer
ConsciousDreamerLatestConfig = ConsciousDreamerConfig
CONSCIOUS_DREAMER_MODEL_VERSION = "M5_CONSCIOUS_DREAMER_CANONICAL_V1"
CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER = "V23"

# Backward-compatible aliases. Prefer canonical names in new code.
ConsciousDreamerV2_3 = ConsciousDreamerV23


__all__ = [
    "ConsciousDreamer",
    "ConsciousDreamerConfig",
    "ConsciousDreamerLatest",
    "ConsciousDreamerLatestConfig",
    "ConsciousDreamerV23",
    "ConsciousDreamerV23Config",
    "ConsciousDreamerV2_3",
    "CONSCIOUS_DREAMER_MODEL_VERSION",
    "CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER",
]
