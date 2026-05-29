from __future__ import annotations

"""Semantic core facade for the base M5 ConsciousDreamer implementation.

The implementation still lives in `conscious_dreamer_full.py` for backward
compatibility. New internal M5 code should import the core layer from this file.

Naming rule:
    M5 owns preconscious world-model / attention / body-context logic only.
    M9 owns true self-binding. M7 owns true inner speech after M9.
"""

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full import (
    AttentionController,
    ConsciousPlanner,
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
    ConsciousConfig,
    DecoderHeads,
    DreamerDataConfig,
    DreamerLatentConfig,
    ImaginationCore,
    MLPEncoder,
    ObjectRepresentation,
    RSSMCore,
    ReflectiveLoop,
    PreconsciousReflectionLoop,
    BodyContextModel,
    VisionEncoder,
    Workspace,
    make_core_config_from_world,
)

# Compatibility alias for old imports. New code should use BodyContextModel.
SelfModel = BodyContextModel

__all__ = [
    "DreamerDataConfig",
    "DreamerLatentConfig",
    "ConsciousConfig",
    "ConsciousDreamerCore",
    "ConsciousDreamerCoreConfig",
    "VisionEncoder",
    "MLPEncoder",
    "AttentionController",
    "RSSMCore",
    "Workspace",
    "BodyContextModel",
    "SelfModel",
    "ObjectRepresentation",
    "PreconsciousReflectionLoop",
    "ReflectiveLoop",
    "ImaginationCore",
    "ConsciousPlanner",
    "DecoderHeads",
    "make_core_config_from_world",
]
