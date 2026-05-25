from __future__ import annotations

"""Component factory helpers for the V5.10 runner.

This module is the next extraction boundary after optimizer/services/startup
state. It collects pure config-to-constructor-kwargs helpers first, then small
factory functions. Heavy imports are kept inside factory functions so smoke tests
can import this module without constructing MuJoCo/Open3D/PyQt windows.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ComponentFactorySnapshot:
    component: str
    kwargs: Dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"component": self.component, "kwargs": dict(self.kwargs)}


def inner_object_system_kwargs(cfg: Any) -> Dict[str, Any]:
    """Return kwargs for ObjectInnerImagery3DConfig."""
    object_cfg = cfg.object_image
    return {
        "enabled": object_cfg.enabled,
        "latent_dim": object_cfg.latent_dim,
        "hidden_dim": object_cfg.hidden_dim,
        "image_size": object_cfg.image_size,
        "tactile_dim": object_cfg.tactile_dim,
        "body_dim": cfg.embodied_dim,
        "hand_dim": cfg.hand_motor_dim,
        "leg_dim": cfg.leg_control.leg_motor_dim,
        "point_count": object_cfg.point_count,
        "voxel_res": object_cfg.voxel_res,
        "num_slots": getattr(object_cfg, "num_slots", 4),
        "max_object_proposals": getattr(object_cfg, "max_object_proposals", 4),
        "proposal_slot_lock": getattr(object_cfg, "proposal_slot_lock", True),
        "sleep_freeze_memory_update": getattr(object_cfg, "sleep_freeze_object_slots", False),
        "dream_latent_dynamics": getattr(object_cfg, "dream_latent_dynamics", True),
        "dream_strength": getattr(object_cfg, "dream_strength", 0.025),
        "dream_cycle_slots": getattr(object_cfg, "dream_cycle_slots", False),
        "dream_slot_cycle_steps": getattr(object_cfg, "dream_slot_cycle_steps", 90),
        "dream_empty_confidence_threshold": getattr(object_cfg, "dream_empty_confidence_threshold", 0.05),
    }


def create_inner_object_system(cfg: Any, device: Any) -> Any:
    """Create the M1 inner object representation system."""
    from src.modules.m01_object_imagery.models.object_inner_imagery_3d import (
        InnerObjectRepresentation3DSystem,
        ObjectInnerImagery3DConfig,
    )

    return InnerObjectRepresentation3DSystem(ObjectInnerImagery3DConfig(**inner_object_system_kwargs(cfg))).to(device)


def self_core_kwargs(cfg: Any) -> Dict[str, Any]:
    """Return kwargs for SelfCoreConfig."""
    return {
        "enabled": cfg.self_core.enabled,
        "body_state_dim": cfg.body_state_dim,
        "action_dim": cfg.embodied_dim,
        "tactile_dim": cfg.tactile_dim,
        "vestibular_dim": 24,
        "object_latent_dim": cfg.self_core.object_latent_dim,
        "workspace_dim": cfg.self_core.workspace_dim,
        "hidden_dim": cfg.self_core.hidden_dim,
        "self_dim": cfg.self_core.self_dim,
    }


def create_self_core(cfg: Any, device: Any) -> Any:
    """Create M9 SelfCore."""
    from src.modules.m09_self_core.models.self_core import SelfCore, SelfCoreConfig

    return SelfCore(SelfCoreConfig(**self_core_kwargs(cfg))).to(device)


def inner_object_visualizer_kwargs(cfg: Any) -> Dict[str, Any]:
    object_cfg = cfg.object_image
    return {
        "window_name": object_cfg.window_name,
        "width": max(int(object_cfg.width), 1520),
        "height": max(int(object_cfg.height), 1260),
        "max_slots": getattr(object_cfg, "num_slots", 10),
    }


def create_inner_object_visualizer(cfg: Any) -> Any:
    from src.modules.m01_object_imagery.visualizer_inner_object import InnerObjectVisualizerV2, InnerObjectVisualizerV2Config

    return InnerObjectVisualizerV2(InnerObjectVisualizerV2Config(**inner_object_visualizer_kwargs(cfg)))


def inner_object_open3d_viewer_kwargs(cfg: Any) -> Dict[str, Any]:
    open3d_cfg = cfg.object_image_open3d
    return {
        "enabled": open3d_cfg.enabled,
        "window_name": open3d_cfg.window_name,
        "width": open3d_cfg.width,
        "height": open3d_cfg.height,
        "update_every_steps": open3d_cfg.update_every_steps,
        "point_size": open3d_cfg.point_size,
        "voxel_threshold": open3d_cfg.voxel_threshold,
        "max_voxel_points": open3d_cfg.max_voxel_points,
        "show_voxels": open3d_cfg.show_voxels,
        "use_internal_color": open3d_cfg.use_internal_color,
        "max_slots": open3d_cfg.max_slots,
        "slot_spacing": open3d_cfg.slot_spacing,
        "export_dir": open3d_cfg.export_dir,
    }


def create_inner_object_open3d_viewer(cfg: Any) -> Any:
    from src.modules.m01_object_imagery.inner_object_open3d_viewer import InnerObjectOpen3DViewerV2, InnerObjectOpen3DViewerV2Config

    return InnerObjectOpen3DViewerV2(InnerObjectOpen3DViewerV2Config(**inner_object_open3d_viewer_kwargs(cfg)))


def latent_semantic_map_kwargs(cfg: Any) -> Dict[str, Any]:
    map_cfg = cfg.latent_semantic_map
    return {
        "enabled": map_cfg.enabled,
        "window_name": map_cfg.window_name,
        "width": map_cfg.width,
        "height": map_cfg.height,
        "max_history": map_cfg.max_history,
        "show_every_steps": map_cfg.show_every_steps,
        "delay_ms": map_cfg.delay_ms,
        "thumbnail_size": map_cfg.thumbnail_size,
        "max_thumbnails": map_cfg.max_thumbnails,
        "point_radius": map_cfg.point_radius,
        "draw_grid": map_cfg.draw_grid,
        "follow_inner_world_toggle": map_cfg.follow_inner_world_toggle,
    }


def create_latent_semantic_map_visualizer(cfg: Any) -> Any:
    from src.modules.m14_semantic_grounding.latent_semantic_map import LatentSemanticMapConfig, LatentSemanticMapVisualizer

    return LatentSemanticMapVisualizer(LatentSemanticMapConfig(**latent_semantic_map_kwargs(cfg)))


def emotional_drive_kwargs(cfg: Any) -> Dict[str, Any]:
    drive = cfg.emotional_drive
    return {
        "enabled": drive.enabled,
        "ema_decay": drive.ema_decay,
        "reward_scale": drive.reward_scale,
        "w_gap_fill": drive.w_gap_fill,
        "w_coherence_gain": drive.w_coherence_gain,
        "w_object_conf_gain": drive.w_object_conf_gain,
        "w_multimodal_alignment": drive.w_multimodal_alignment,
        "w_contact_pleasure": drive.w_contact_pleasure,
        "w_curiosity": drive.w_curiosity,
        "w_inner_speech_conf": drive.w_inner_speech_conf,
        "w_uncertainty_increase": drive.w_uncertainty_increase,
        "w_coherence_loss": drive.w_coherence_loss,
        "w_object_conf_loss": drive.w_object_conf_loss,
        "w_speech_conf_loss": drive.w_speech_conf_loss,
        "w_alignment_loss": drive.w_alignment_loss,
        "w_chaotic_touch": drive.w_chaotic_touch,
        "w_instability": drive.w_instability,
    }


def create_emotional_drive(cfg: Any) -> Any:
    from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive, EmotionalDriveConfig

    return EmotionalDrive(EmotionalDriveConfig(**emotional_drive_kwargs(cfg)))


def component_factory_snapshots(cfg: Any) -> list[ComponentFactorySnapshot]:
    """Return pure snapshots for the extracted component constructor kwargs."""
    return [
        ComponentFactorySnapshot("inner_object_system", inner_object_system_kwargs(cfg)),
        ComponentFactorySnapshot("self_core", self_core_kwargs(cfg)),
        ComponentFactorySnapshot("inner_object_visualizer", inner_object_visualizer_kwargs(cfg)),
        ComponentFactorySnapshot("inner_object_open3d_viewer", inner_object_open3d_viewer_kwargs(cfg)),
        ComponentFactorySnapshot("latent_semantic_map", latent_semantic_map_kwargs(cfg)),
        ComponentFactorySnapshot("emotional_drive", emotional_drive_kwargs(cfg)),
    ]
