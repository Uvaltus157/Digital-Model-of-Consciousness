from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_components import (
    component_factory_snapshots,
    emotional_drive_kwargs,
    inner_object_open3d_viewer_kwargs,
    inner_object_system_kwargs,
    inner_object_visualizer_kwargs,
    latent_semantic_map_kwargs,
    self_core_kwargs,
)


def _cfg():
    return SimpleNamespace(
        embodied_dim=64,
        hand_motor_dim=32,
        tactile_dim=16,
        body_state_dim=24,
        leg_control=SimpleNamespace(leg_motor_dim=8),
        object_image=SimpleNamespace(
            enabled=True,
            latent_dim=128,
            hidden_dim=256,
            image_size=64,
            tactile_dim=16,
            point_count=512,
            voxel_res=16,
            num_slots=4,
            max_object_proposals=5,
            proposal_slot_lock=True,
            sleep_freeze_object_slots=False,
            dream_latent_dynamics=True,
            dream_strength=0.025,
            dream_cycle_slots=False,
            dream_slot_cycle_steps=90,
            dream_empty_confidence_threshold=0.05,
            window_name="Inner Object",
            width=640,
            height=480,
        ),
        object_image_open3d=SimpleNamespace(
            enabled=True,
            window_name="Open3D",
            width=800,
            height=600,
            update_every_steps=3,
            point_size=2.0,
            voxel_threshold=0.4,
            max_voxel_points=1000,
            show_voxels=True,
            use_internal_color=True,
            max_slots=4,
            slot_spacing=1.5,
            export_dir="exports",
        ),
        latent_semantic_map=SimpleNamespace(
            enabled=True,
            window_name="Latent Map",
            width=640,
            height=480,
            max_history=100,
            show_every_steps=5,
            delay_ms=1,
            thumbnail_size=32,
            max_thumbnails=10,
            point_radius=3,
            draw_grid=True,
            follow_inner_world_toggle=True,
        ),
        self_core=SimpleNamespace(
            enabled=True,
            object_latent_dim=128,
            workspace_dim=256,
            hidden_dim=128,
            self_dim=64,
        ),
        emotional_drive=SimpleNamespace(
            enabled=True,
            ema_decay=0.98,
            reward_scale=1.0,
            w_gap_fill=1.0,
            w_coherence_gain=1.0,
            w_object_conf_gain=1.0,
            w_multimodal_alignment=1.0,
            w_contact_pleasure=1.0,
            w_curiosity=1.0,
            w_inner_speech_conf=1.0,
            w_uncertainty_increase=1.0,
            w_coherence_loss=1.0,
            w_object_conf_loss=1.0,
            w_speech_conf_loss=1.0,
            w_alignment_loss=1.0,
            w_chaotic_touch=1.0,
            w_instability=1.0,
        ),
    )


def test_inner_object_system_kwargs() -> None:
    kwargs = inner_object_system_kwargs(_cfg())
    assert kwargs["latent_dim"] == 128
    assert kwargs["body_dim"] == 64
    assert kwargs["hand_dim"] == 32
    assert kwargs["leg_dim"] == 8
    assert kwargs["num_slots"] == 4


def test_self_core_kwargs() -> None:
    kwargs = self_core_kwargs(_cfg())
    assert kwargs["body_state_dim"] == 24
    assert kwargs["action_dim"] == 64
    assert kwargs["vestibular_dim"] == 24
    assert kwargs["self_dim"] == 64


def test_visualizer_kwargs_clamp_inner_object_window_size() -> None:
    kwargs = inner_object_visualizer_kwargs(_cfg())
    assert kwargs["width"] == 1520
    assert kwargs["height"] == 1260
    assert kwargs["max_slots"] == 4


def test_open3d_and_latent_map_kwargs() -> None:
    open3d_kwargs = inner_object_open3d_viewer_kwargs(_cfg())
    latent_kwargs = latent_semantic_map_kwargs(_cfg())
    assert open3d_kwargs["window_name"] == "Open3D"
    assert open3d_kwargs["max_slots"] == 4
    assert latent_kwargs["window_name"] == "Latent Map"
    assert latent_kwargs["follow_inner_world_toggle"] is True


def test_emotional_drive_kwargs_and_snapshots() -> None:
    kwargs = emotional_drive_kwargs(_cfg())
    assert kwargs["enabled"] is True
    assert kwargs["w_curiosity"] == 1.0

    snapshots = component_factory_snapshots(_cfg())
    names = [item.component for item in snapshots]
    assert "inner_object_system" in names
    assert "self_core" in names
    assert "emotional_drive" in names
