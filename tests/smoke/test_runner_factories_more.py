from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_memory_factory import memory_factory_snapshot
from src.apps.runner_model_factory import model_factory_snapshot, optimizer_kwargs, resolve_device_name, resolve_runtime_seed
from src.apps.runner_visualizer_factory import inner_world_viz_kwargs, visualizer_factory_snapshot


def _cfg():
    return SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", seed=123),
        train=SimpleNamespace(lr=0.001, weight_decay=0.02),
        replay=SimpleNamespace(capacity=1000, min_ready=32),
        novelty=SimpleNamespace(enabled=True),
        inner_world=SimpleNamespace(width=800, height=600),
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
            max_object_proposals=4,
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
    )


def test_model_factory_snapshot_and_optimizer_kwargs() -> None:
    cfg = _cfg()
    assert resolve_device_name(cfg) == "cpu"
    assert resolve_runtime_seed(cfg) == 123
    assert optimizer_kwargs(cfg) == {"lr": 0.001, "weight_decay": 0.02}
    snap = model_factory_snapshot(cfg, speech_vocab_size=77)
    assert snap.device == "cpu"
    assert snap.seed == 123
    assert snap.text_vocab_size == 77
    assert snap.optimizer_lr == 0.001


def test_memory_factory_snapshot() -> None:
    snap = memory_factory_snapshot(_cfg())
    assert snap.replay_capacity == 1000
    assert snap.replay_min_ready == 32
    assert snap.quality_ema_decay == 0.98
    assert snap.novelty_enabled is True


def test_visualizer_factory_snapshot() -> None:
    cfg = _cfg()
    assert inner_world_viz_kwargs(cfg) == {"width": 800, "height": 600}
    snap = visualizer_factory_snapshot(cfg)
    assert snap.inner_world_kwargs["width"] == 800
    assert snap.inner_object_kwargs["width"] == 1520
    assert snap.latent_semantic_kwargs["window_name"] == "Latent Map"
