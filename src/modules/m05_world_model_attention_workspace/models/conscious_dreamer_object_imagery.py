from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_inner_speech import ConsciousDreamerV22, ConsciousDreamerV22Config
from src.modules.m01_object_imagery.models.object_imagery_decoder import ObjectImageryConfig, ObjectImageryDecoder


@dataclass
class ConsciousDreamerV23Config(ConsciousDreamerV22Config):
    object_imagery: ObjectImageryConfig = field(default_factory=ObjectImageryConfig)


class ConsciousDreamerV23(ConsciousDreamerV22):
    def __init__(self, cfg: ConsciousDreamerV23Config) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.cfg.object_imagery.object_dim = cfg.conscious.object_repr_dim
        self.cfg.object_imagery.workspace_dim = cfg.conscious.workspace_dim
        self.cfg.object_imagery.thought_dim = cfg.conscious.thought_dim
        self.cfg.object_imagery.reflection_dim = cfg.conscious.reflective_self_dim
        self.object_imagery_decoder = ObjectImageryDecoder(self.cfg.object_imagery)

    def build_object_imagery(self, out: Dict):
        return self.object_imagery_decoder(
            object_repr=out["object_repr"],
            workspace=out["workspace_out"],
            thought=out["thoughts"]["thought"],
            reflection=out["reflection_out"]["reflection"],
        )

    def step(self, *args, **kwargs):
        out = super().step(*args, **kwargs)
        out["object_imagery"] = self.build_object_imagery(out)
        return out


ConsciousDreamerV2_3 = ConsciousDreamerV23
