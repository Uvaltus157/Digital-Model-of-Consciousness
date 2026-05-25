from __future__ import annotations

"""Memory/replay/quality/novelty factory helpers for V5.10 runner cleanup."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryFactorySnapshot:
    replay_capacity: int
    replay_min_ready: int
    quality_ema_decay: float
    novelty_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def create_replay_buffer(cfg: Any) -> Any:
    from src.apps.unified_conscious_viewer import ReplayBuffer

    return ReplayBuffer(int(cfg.replay.capacity))


def create_quality_meter(ema_decay: float = 0.98) -> Any:
    from src.apps.unified_conscious_viewer import QualityMeter

    return QualityMeter(ema_decay=float(ema_decay))


def create_novelty_detector(cfg: Any) -> Any:
    from src.apps.unified_conscious_viewer import NoveltyDetector

    return NoveltyDetector(cfg.novelty)


def seed_python_random(cfg: Any) -> int:
    import random

    seed = int(getattr(cfg.runtime, "seed", 0))
    random.seed(seed)
    return seed


def memory_factory_snapshot(cfg: Any) -> MemoryFactorySnapshot:
    return MemoryFactorySnapshot(
        replay_capacity=int(getattr(cfg.replay, "capacity", 0)),
        replay_min_ready=int(getattr(cfg.replay, "min_ready", 0)),
        quality_ema_decay=0.98,
        novelty_enabled=bool(getattr(cfg.novelty, "enabled", True)),
    )
