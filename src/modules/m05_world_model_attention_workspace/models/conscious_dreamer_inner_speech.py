from __future__ import annotations

"""
conscious_dreamer_inner_speech.py

ConsciousDreamer inner-speech layer:
- based on the memory / preconscious-candidate layer
- adds built-in InnerSpeechLoop / symbolic report layer
- each step returns:
    out["symbolic_report"]
        report_latent
        inner_speech_sequence
        symbol_ids
        phoneme_ids
        text_token_ids
        confidence

Architecture note:
    The M5 input here is a preconscious thought candidate, not yet a
    self-aware conscious thought. Self-binding is performed later by M9.
    M7/M11 can then turn self-bound content into conscious inner speech.

This model keeps the same current M5 step() signature.
"""

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought import (
    ConsciousDreamerMemoryThought,
    ConsciousDreamerMemoryThoughtConfig,
)

from src.modules.m07_inner_speech_thoughts.models.symbolic_report_language import InnerSpeechLoop, SymbolicReportConfig


@dataclass
class ConsciousDreamerInnerSpeechConfig(ConsciousDreamerMemoryThoughtConfig):
    symbolic_report: SymbolicReportConfig = field(default_factory=SymbolicReportConfig)


class ConsciousDreamerInnerSpeech(ConsciousDreamerMemoryThought):
    """Current M5 symbolic report pathway over preconscious candidates."""

    def __init__(self, cfg: ConsciousDreamerInnerSpeechConfig) -> None:
        super().__init__(cfg)
        self.cfg: ConsciousDreamerInnerSpeechConfig = cfg

        c = cfg.conscious
        tm = cfg.thought_memory
        d = cfg.data

        report_input_dim = (
            c.workspace_dim
            + c.thought_dim
            + c.reflective_self_dim
            + c.object_repr_dim
            + tm.memory_dim
            + c.value_dim
            + d.embodied_dim
            + d.hand_motor_dim
        )

        # Force correct dimensions even if cfg.symbolic_report was default-created
        self.symbolic_report_cfg = cfg.symbolic_report
        self.symbolic_report_cfg.input_dim = report_input_dim

        self.inner_speech = InnerSpeechLoop(self.symbolic_report_cfg)

    def build_symbolic_report(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        memory_context = out.get("memory", {}).get("memory_context")
        if memory_context is None:
            memory_context = torch.zeros(
                out["workspace_out"].shape[0],
                self.cfg.thought_memory.memory_dim,
                device=out["workspace_out"].device,
                dtype=out["workspace_out"].dtype,
            )

        preconscious = out.get("preconscious_thoughts", {})
        thought_candidate = preconscious.get("thought_candidate")
        if thought_candidate is None:
            thought_candidate = preconscious.get("candidate")
        if thought_candidate is None:
            thought_candidate = torch.zeros(
                out["workspace_out"].shape[0],
                self.cfg.conscious.thought_dim,
                device=out["workspace_out"].device,
                dtype=out["workspace_out"].dtype,
            )

        latent = torch.cat(
            [
                out["workspace_out"],
                thought_candidate,
                out["reflection_out"]["reflection"],
                out["object_repr"],
                memory_context,
                out["values"]["value_latent"],
                out["embodied_targets"],
                out["hand_ctrl"],
            ],
            dim=-1,
        )
        return self.inner_speech(latent)

    def step(self, *args, **kwargs) -> Dict:
        out = super().step(*args, **kwargs)
        symbolic_report = self.build_symbolic_report(out)
        out["symbolic_report"] = symbolic_report
        return out


def make_inner_speech_config_from_world(
    image_height=128,
    image_width=192,
    body_state_dim=None,
    tactile_dim=None,
    hand_motor_dim=None,
    embodied_dim=None,
    action_dim=None,
    symbol_vocab_size=512,
    phoneme_vocab_size=96,
    text_vocab_size=2048,
):
    cfg = ConsciousDreamerInnerSpeechConfig()

    # runner.yaml is the source of truth for these dimensions.
    required_dims = {
        "body_state_dim": body_state_dim,
        "tactile_dim": tactile_dim,
        "hand_motor_dim": hand_motor_dim,
        "embodied_dim": embodied_dim,
        "action_dim": action_dim,
    }
    missing = [k for k, v in required_dims.items() if v is None]
    if missing:
        raise ValueError(
            "make_inner_speech_config_from_world() does not own model dimensions. "
            "Read them from runner.yaml / UnifiedV510Config and pass them explicitly. "
            f"Missing: {missing}"
        )

    cfg.data.image_height = image_height
    cfg.data.image_width = image_width
    cfg.data.body_state_dim = body_state_dim
    cfg.data.tactile_dim = tactile_dim
    cfg.data.hand_motor_dim = hand_motor_dim
    cfg.data.embodied_dim = embodied_dim
    cfg.data.action_dim = action_dim

    cfg.symbolic_report.symbol_vocab_size = symbol_vocab_size
    cfg.symbolic_report.phoneme_vocab_size = phoneme_vocab_size
    cfg.symbolic_report.text_vocab_size = text_vocab_size

    # input_dim is corrected in __init__
    return cfg

__all__ = [
    "ConsciousDreamerInnerSpeech",
    "ConsciousDreamerInnerSpeechConfig",
    "make_inner_speech_config_from_world",
]


if __name__ == "__main__":
    cfg = make_inner_speech_config_from_world()
    model = ConsciousDreamerInnerSpeech(cfg)
    state = model.initial_state(1, "cpu")

    left = torch.zeros(1, 3, 128, 192)
    right = torch.zeros(1, 3, 128, 192)
    pose = torch.zeros(1, 7)
    body = torch.zeros(1, cfg.data.body_state_dim)
    tactile = torch.zeros(1, cfg.data.tactile_dim)
    hand = torch.zeros(1, cfg.data.hand_motor_dim)
    embodied = torch.zeros(1, cfg.data.embodied_dim)

    out = model.step(
        left,
        right,
        pose,
        body,
        state,
        tactile=tactile,
        hand_motor=hand,
        embodied_action=embodied,
    )

    print("workspace:", out["workspace_out"].shape)
    print("candidate_sequence:", out["preconscious_thoughts"]["candidate_sequence"].shape)
    print("symbol_ids:", out["symbolic_report"]["symbol_ids"].shape)
    print("phoneme_ids:", out["symbolic_report"]["phoneme_ids"].shape)
    print("text_token_ids:", out["symbolic_report"]["text_token_ids"].shape)
