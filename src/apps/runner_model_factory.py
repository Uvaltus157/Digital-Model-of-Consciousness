from __future__ import annotations

"""Model factory helpers for the V5.10 runner.

This module extracts the config-to-model boundary from the heavy runner. Heavy
Torch/model imports stay inside factory functions where possible, while pure
metadata helpers are easy to smoke-test.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModelFactorySnapshot:
    device: str
    seed: int
    text_vocab_size: int | None
    optimizer_lr: float
    optimizer_weight_decay: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def resolve_device_name(cfg: Any) -> str:
    return str(getattr(cfg.runtime, "device", "cpu"))


def resolve_runtime_seed(cfg: Any) -> int:
    return int(getattr(cfg.runtime, "seed", 0))


def create_torch_device(cfg: Any) -> Any:
    import torch

    return torch.device(resolve_device_name(cfg))


def seed_torch(cfg: Any) -> int:
    import torch

    seed = resolve_runtime_seed(cfg)
    torch.manual_seed(seed)
    return seed


def create_v23_config(cfg: Any, speech_vocab_size: int | None = None) -> Any:
    from src.shared.config import make_v23_config_from_unified

    v23_cfg = make_v23_config_from_unified(cfg)
    if speech_vocab_size is not None:
        v23_cfg.symbolic_report.text_vocab_size = int(speech_vocab_size)
    return v23_cfg


def create_conscious_dreamer_v23(cfg: Any, device: Any, speech_vocab_size: int | None = None) -> Any:
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import ConsciousDreamerV23

    return ConsciousDreamerV23(create_v23_config(cfg, speech_vocab_size=speech_vocab_size)).to(device)


def optimizer_kwargs(cfg: Any) -> Dict[str, float]:
    return {
        "lr": float(getattr(cfg.train, "lr", 0.0)),
        "weight_decay": float(getattr(cfg.train, "weight_decay", 0.0)),
    }


def create_base_optimizer(model: Any, cfg: Any) -> Any:
    import torch.optim as optim

    return optim.AdamW(model.parameters(), **optimizer_kwargs(cfg))


def model_factory_snapshot(cfg: Any, speech_vocab_size: int | None = None) -> ModelFactorySnapshot:
    opt = optimizer_kwargs(cfg)
    return ModelFactorySnapshot(
        device=resolve_device_name(cfg),
        seed=resolve_runtime_seed(cfg),
        text_vocab_size=speech_vocab_size,
        optimizer_lr=opt["lr"],
        optimizer_weight_decay=opt["weight_decay"],
    )
