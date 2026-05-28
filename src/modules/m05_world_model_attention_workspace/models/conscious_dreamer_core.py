from __future__ import annotations

"""Semantic core facade for the base M5 ConsciousDreamer implementation.

The implementation still lives in `conscious_dreamer_full.py` for backward
compatibility. New internal M5 code should import the core layer from this file.
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
    SelfModel,
    VisionEncoder,
    Workspace,
    make_core_config_from_world,
)

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
    "SelfModel",
    "ObjectRepresentation",
    "ReflectiveLoop",
    "ImaginationCore",
    "ConsciousPlanner",
    "DecoderHeads",
    "make_core_config_from_world",
]
