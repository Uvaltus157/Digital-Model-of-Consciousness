from __future__ import annotations

"""Visualizer factory helpers for V5.10 runner cleanup."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VisualizerFactorySnapshot:
    inner_world_kwargs: Dict[str, Any]
    latent_semantic_kwargs: Dict[str, Any]
    inner_object_kwargs: Dict[str, Any]
    inner_object_open3d_kwargs: Dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_world_kwargs": dict(self.inner_world_kwargs),
            "latent_semantic_kwargs": dict(self.latent_semantic_kwargs),
            "inner_object_kwargs": dict(self.inner_object_kwargs),
            "inner_object_open3d_kwargs": dict(self.inner_object_open3d_kwargs),
        }


def inner_world_viz_kwargs(cfg: Any) -> Dict[str, Any]:
    return {
        "width": cfg.inner_world.width,
        "height": cfg.inner_world.height,
    }


def create_inner_world_visualizer(cfg: Any, speech_vocab: Any | None = None) -> Any:
    from src.modules.m07_inner_speech_thoughts.inner_world_visualizer import InnerWorldVizConfig
    from src.modules.m07_inner_speech_thoughts.inner_world_visualizer_text_thought_hybrid import DreamerInnerWorldVisualizerV3

    viz = DreamerInnerWorldVisualizerV3(InnerWorldVizConfig(**inner_world_viz_kwargs(cfg)))
    if speech_vocab is not None:
        try:
            viz.speech_vocab = speech_vocab
        except Exception:
            pass
    return viz


def visualizer_factory_snapshot(cfg: Any) -> VisualizerFactorySnapshot:
    from src.apps.runner_components import (
        inner_object_open3d_viewer_kwargs,
        inner_object_visualizer_kwargs,
        latent_semantic_map_kwargs,
    )

    return VisualizerFactorySnapshot(
        inner_world_kwargs=inner_world_viz_kwargs(cfg),
        latent_semantic_kwargs=latent_semantic_map_kwargs(cfg),
        inner_object_kwargs=inner_object_visualizer_kwargs(cfg),
        inner_object_open3d_kwargs=inner_object_open3d_viewer_kwargs(cfg),
    )
