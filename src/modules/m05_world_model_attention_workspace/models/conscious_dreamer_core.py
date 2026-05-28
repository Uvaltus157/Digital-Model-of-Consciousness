from __future__ import annotations

"""Semantic core facade for the base M5 ConsciousDreamer implementation.

The implementation still lives in `conscious_dreamer_full.py` for backward
compatibility. New internal M5 code should import the core layer from this file.
"""

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full import (
    AttentionControllerV2,
    ConsciousPlannerV2,
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
    ConsciousV2Config,
    DecoderHeadsV2,
    DreamerV2DataConfig,
    DreamerV2LatentConfig,
    ImaginationCoreV2,
    MLPEncoder,
    ObjectRepresentationV2,
    RSSMCoreV2,
    ReflectiveLoopV2,
    SelfModelV2,
    VisionEncoder,
    WorkspaceV2,
    make_v2_full_config_from_world,
)

__all__ = [
    "DreamerV2DataConfig",
    "DreamerV2LatentConfig",
    "ConsciousV2Config",
    "ConsciousDreamerCore",
    "ConsciousDreamerCoreConfig",
    "VisionEncoder",
    "MLPEncoder",
    "AttentionControllerV2",
    "RSSMCoreV2",
    "WorkspaceV2",
    "SelfModelV2",
    "ObjectRepresentationV2",
    "ReflectiveLoopV2",
    "ImaginationCoreV2",
    "ConsciousPlannerV2",
    "DecoderHeadsV2",
    "make_v2_full_config_from_world",
]
