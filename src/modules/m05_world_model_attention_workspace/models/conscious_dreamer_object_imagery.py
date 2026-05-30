from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought import (
    ConsciousDreamerMemoryThought,
    ConsciousDreamerMemoryThoughtConfig,
)
from src.modules.m01_object_imagery.models.object_imagery_decoder import ObjectImageryConfig, ObjectImageryDecoder


@dataclass
class ConsciousDreamerObjectImageryConfig(ConsciousDreamerMemoryThoughtConfig):
    object_imagery: ObjectImageryConfig = field(default_factory=ObjectImageryConfig)


class ConsciousDreamerObjectImagery(ConsciousDreamerMemoryThought):
    """M5 object-imagery layer over preconscious M5 latents.

    This layer intentionally does not inherit from the M7/inner-speech path.
    M5 may build object imagery from workspace/object/candidate/reflection, but
    true inner speech is produced later by M7 after M9 self-binding.
    """

    def __init__(self, cfg: ConsciousDreamerObjectImageryConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.cfg.object_imagery.object_dim = cfg.conscious.object_repr_dim
        self.cfg.object_imagery.workspace_dim = cfg.conscious.workspace_dim
        self.cfg.object_imagery.thought_dim = cfg.conscious.thought_dim
        self.cfg.object_imagery.reflection_dim = cfg.conscious.model_reflection_dim
        self.object_imagery_decoder = ObjectImageryDecoder(self.cfg.object_imagery)

    def build_object_imagery(self, out: Dict):
        preconscious = out.get("preconscious_thoughts", {})
        thought_candidate = preconscious.get("thought_candidate")
        if thought_candidate is None:
            thought_candidate = torch.zeros(
                out["workspace_out"].shape[0],
                self.cfg.conscious.thought_dim,
                device=out["workspace_out"].device,
                dtype=out["workspace_out"].dtype,
            )

        preconscious_reflection = out.get("preconscious_reflection_out", {})
        reflection = preconscious_reflection.get("reflection")
        if reflection is None:
            reflection = torch.zeros(
                out["workspace_out"].shape[0],
                self.cfg.conscious.model_reflection_dim,
                device=out["workspace_out"].device,
                dtype=out["workspace_out"].dtype,
            )

        return self.object_imagery_decoder(
            object_repr=out["object_repr"],
            workspace=out["workspace_out"],
            thought=thought_candidate,
            reflection=reflection,
        )

    def step(self, *args, **kwargs):
        out = super().step(*args, **kwargs)
        out["object_imagery"] = self.build_object_imagery(out)
        return out


__all__ = [
    "ConsciousDreamerObjectImagery",
    "ConsciousDreamerObjectImageryConfig",
]
