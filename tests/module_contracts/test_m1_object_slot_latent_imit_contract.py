
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

class DummyViz:
    def __init__(self):
        self.requested_dream_slot_index = None

class DummySystem:
    from src.modules.m01_object_imagery.imit.m1_object_slot_latent_runtime import M1ObjectSlotLatentImitRuntimeMixin
    _m1_imit_latent_dim = M1ObjectSlotLatentImitRuntimeMixin._m1_imit_latent_dim
    make_m1_object_slot_latent = M1ObjectSlotLatentImitRuntimeMixin.make_m1_object_slot_latent
    request_m1_object_slot_latents = M1ObjectSlotLatentImitRuntimeMixin.request_m1_object_slot_latents
    _m1_select_inner_object_slot = M1ObjectSlotLatentImitRuntimeMixin._m1_select_inner_object_slot
    get_m1_imit_inner_object_proposals = M1ObjectSlotLatentImitRuntimeMixin.get_m1_imit_inner_object_proposals
    m1_object_slot_imit_status = M1ObjectSlotLatentImitRuntimeMixin.m1_object_slot_imit_status
    def __init__(self):
        self.global_step = 12
        self.cfg = SimpleNamespace(object_image=SimpleNamespace(latent_dim=128, num_slots=10))
        self.inner_object_viz = DummyViz()
        self.status_writes = 0
    def write_module_debug_status(self):
        self.status_writes += 1


class DummyProgressiveSystem(DummySystem):
    from src.modules.m01_object_imagery.runtime import ObjectImageryRuntimeMixin

    _run_progressive_inner_object_system = ObjectImageryRuntimeMixin._run_progressive_inner_object_system
    _memory_update_forced_slot = ObjectImageryRuntimeMixin._memory_update_forced_slot
    _attach_slot_4d_playback_tensors = ObjectImageryRuntimeMixin._attach_slot_4d_playback_tensors
    _inner_object_requested_display_slot = ObjectImageryRuntimeMixin._inner_object_requested_display_slot
    _sync_inner_object_requested_slot_from_viz = ObjectImageryRuntimeMixin._sync_inner_object_requested_slot_from_viz
    _m1_imit_shape_kind_for_slot = ObjectImageryRuntimeMixin._m1_imit_shape_kind_for_slot
    _m1_imit_shape_kind_from_latent = ObjectImageryRuntimeMixin._m1_imit_shape_kind_from_latent
    _inner_object_open3d_display_obj = ObjectImageryRuntimeMixin._inner_object_open3d_display_obj
    _inner_object_selected_slot_is_empty = ObjectImageryRuntimeMixin._inner_object_selected_slot_is_empty
    _empty_inner_object_open3d_display_obj = ObjectImageryRuntimeMixin._empty_inner_object_open3d_display_obj
    _scalar_debug_slot = ObjectImageryRuntimeMixin._scalar_debug_slot

    def __init__(self):
        super().__init__()
        from src.modules.m01_object_imagery.models.object_inner_imagery_3d import (
            InnerObjectRepresentation3DSystem,
            ObjectInnerImagery3DConfig,
        )

        self.device = torch.device("cpu")
        self.cfg.object_image = ObjectInnerImagery3DConfig(latent_dim=128, num_slots=10)
        self.inner_object_system = InnerObjectRepresentation3DSystem(self.cfg.object_image)
        self._inner_object_dynamic_debug = {}
        self._inner_object_proposal_target_slots = []
        self._inner_object_proposal_kinds = []
        self._inner_object_proposal_target_names = []

    def _attach_long_dynamic_debug_tensors(self, decoded, ref_tensor):
        return decoded

    def _slot_observation_reconstruction_step(self, *args, **kwargs):
        return None

    def _slot_gaussian_reconstruction_step(self, *args, **kwargs):
        return None

    def _slot_4d_timeline_step(self, *args, **kwargs):
        return None

    def _slot_4d_deformation_step(self, *args, **kwargs):
        return None

    def _slot_4d_playback_step(self, *args, **kwargs):
        return None

    def _slot_4d_open3d_export_step(self, *args, **kwargs):
        return None

    def _slot_4d_jsonrpc_stream_step(self, *args, **kwargs):
        return None

    def _slot_object_memory_step(self, *args, **kwargs):
        return None

def test_m1_object_slot_imit_lives_in_imit():
    assert Path("src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py").exists()
    assert Path("src/modules/m01_object_imagery/imit/__init__.py").exists()
    assert not Path("src/modules/m01_object_imagery/m1_object_slot_latent_runtime.py").exists()

def test_m1_cube_tetra_latents_fill_specific_slots_and_select_slot():
    system = DummySystem()
    state = system.request_m1_object_slot_latents({"kind":"cube_tetra","duration":5,"cube_slot":1,"tetra_slot":2,"selected_slot":2,"auto_select_slot":True})
    assert state["active"] is True
    assert state["layout"] == "imit"
    assert state["selected_slot"] == 2
    assert system.inner_object_viz.requested_dream_slot_index == 2
    assert system._ipc_inner_object_dream_slot_index == 2
    result = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    assert tuple(result["proposals"].shape) == (1, 2, 128)
    assert result["target_slots"] == [1, 2]
    assert result["proposal_kinds"] == ["m1_imit_dynamic_object", "m1_imit_dynamic_object"]
    assert result["target_names"] == ["cube", "tetrahedron"]

def test_m1_imit_direct_z_writes_nonzero_slot_memory():
    system = DummyProgressiveSystem()
    system.request_m1_object_slot_latents({"kind":"cube_tetra","duration":5,"cube_slot":1,"tetra_slot":2,"selected_slot":2,"auto_select_slot":True})
    proposals = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    assert tuple(proposals["proposals"].shape) == (1, 2, 128)

    system._inner_object_proposal_target_slots = list(proposals["target_slots"])
    system._inner_object_proposal_kinds = list(proposals["proposal_kinds"])
    system._inner_object_proposal_target_names = list(proposals["target_names"])

    prev_state = system.inner_object_system.initial_state(batch_size=1, device=system.device)
    decoded = system._run_progressive_inner_object_system(
        prev_state,
        proposals["proposals"],
        torch.zeros(1, 50),
        torch.zeros(1, 12),
        torch.zeros(1, 34),
        torch.zeros(1, 18),
        dream_mode=False,
    )

    z_slots = decoded["z_obj_slots"]
    c_slots = decoded["confidence_slots"]
    assert float(z_slots[:, 1, :].norm().item()) > 0
    assert float(c_slots[:, 1, :].mean().item()) > 0
    assert float(z_slots[:, 2, :].norm().item()) > 0
    assert float(c_slots[:, 2, :].mean().item()) > 0
    assert int(decoded["active_slot_index"].reshape(-1)[0].item()) == 2
    assert int(decoded["semantic_updated_slot"].reshape(-1)[0].item()) == 2
    assert int(decoded["semantic_proposal_count"].reshape(-1)[0].item()) == 2
    assert "point_cloud" in decoded
    assert tuple(decoded["point_cloud"].shape[:2]) == (1, system.cfg.object_image.point_count)
    assert float(decoded["point_cloud"].detach().float().abs().sum().item()) > 0

    system.inner_object_state = decoded
    from src.modules.m08_debug_visual_control.m1_object_slot_imit_status import build_m1_object_slot_imit_status

    status = build_m1_object_slot_imit_status(system)
    assert status["selected_slot"] == 2
    assert status["selected_slot_z_norm"] > 0
    assert status["selected_slot_confidence"] > 0
    assert status["slot_metrics"]["1"]["z_norm"] > 0
    assert status["slot_metrics"]["1"]["confidence"] > 0
    assert status["slot_metrics"]["2"]["z_norm"] > 0
    assert status["slot_metrics"]["2"]["confidence"] > 0

def test_m1_imit_direct_z_writes_slots_even_in_sleep_mode():
    system = DummyProgressiveSystem()
    system.request_m1_object_slot_latents({"kind":"cube_tetra","duration":5,"cube_slot":1,"tetra_slot":2,"selected_slot":2,"auto_select_slot":True})
    proposals = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    system._inner_object_proposal_target_slots = list(proposals["target_slots"])
    system._inner_object_proposal_kinds = list(proposals["proposal_kinds"])
    system._inner_object_proposal_target_names = list(proposals["target_names"])

    decoded = system._run_progressive_inner_object_system(
        system.inner_object_system.initial_state(batch_size=1, device=system.device),
        proposals["proposals"],
        torch.zeros(1, 50),
        torch.zeros(1, 12),
        torch.zeros(1, 34),
        torch.zeros(1, 18),
        dream_mode=True,
    )

    assert float(decoded["z_obj_slots"][:, 1, :].norm().item()) > 0
    assert float(decoded["confidence_slots"][:, 1, :].mean().item()) > 0
    assert float(decoded["z_obj_slots"][:, 2, :].norm().item()) > 0
    assert float(decoded["confidence_slots"][:, 2, :].mean().item()) > 0
    assert int(decoded["active_slot_index"].reshape(-1)[0].item()) == 2
    assert decoded["debug_imit_fallback_shape"] is True
    assert decoded["debug_imit_source"] == "m1_object_slot_imit"
    assert decoded["debug_imit_shape_kind"] == "tetrahedron"

def test_m1_imit_decoded_object_marks_open3d_debug_fallback_shape():
    system = DummyProgressiveSystem()
    system.request_m1_object_slot_latents({"kind":"cube","slot":1,"duration":5})
    proposals = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    system._inner_object_proposal_target_slots = list(proposals["target_slots"])
    system._inner_object_proposal_kinds = list(proposals["proposal_kinds"])
    system._inner_object_proposal_target_names = list(proposals["target_names"])

    decoded = system._run_progressive_inner_object_system(
        system.inner_object_system.initial_state(batch_size=1, device=system.device),
        proposals["proposals"],
        torch.zeros(1, 50),
        torch.zeros(1, 12),
        torch.zeros(1, 34),
        torch.zeros(1, 18),
        dream_mode=True,
    )

    assert decoded["debug_imit_fallback_shape"] is True
    assert decoded["debug_imit_source"] == "m1_object_slot_imit"
    assert decoded["debug_imit_shape_kind"] == "cube"

def test_m1_imit_open3d_fallback_generates_visible_primitives():
    from src.modules.m01_object_imagery.inner_object_open3d_viewer import InnerObjectOpen3DViewerV2, InnerObjectOpen3DViewerV2Config

    cfg = InnerObjectOpen3DViewerV2Config()
    assert cfg.show_long_dynamic_debug is False
    assert cfg.show_slot_snapshots is False
    assert cfg.view_zoom == 0.42
    assert cfg.view_z_far == 100.0
    viewer = InnerObjectOpen3DViewerV2()
    cube_pts, cube_cols, cube_vox, cube_vox_cols = viewer._geometry_from_obj({
        "debug_imit_fallback_shape": True,
        "debug_imit_shape_kind": "cube",
    })
    tetra_pts, tetra_cols, _, _ = viewer._geometry_from_obj({
        "debug_imit_fallback_shape": True,
        "debug_imit_shape_kind": "tetrahedron",
    })
    morph_pts, morph_cols, _, _ = viewer._geometry_from_obj({
        "debug_imit_fallback_shape": True,
        "debug_imit_shape_kind": "morph",
    })

    assert cube_pts.shape[0] > 0
    assert cube_cols.shape == cube_pts.shape
    assert cube_vox.shape == (0, 3)
    assert cube_vox_cols.shape == (0, 3)
    assert tetra_pts.shape[0] > 0
    assert tetra_cols.shape == tetra_pts.shape
    assert morph_pts.shape[0] > 0
    assert morph_cols.shape == morph_pts.shape
    assert not np.allclose(cube_pts[: min(len(cube_pts), len(tetra_pts))], tetra_pts[: min(len(cube_pts), len(tetra_pts))])

def test_inner_object_packet_attaches_existing_4d_playback_preview_tensors():
    system = DummyProgressiveSystem()
    system._slot_4d_latest_metrics = {"frame_count": 8, "motion_norm": 0.12, "temporal_span": 7, "gaussian_count": 128}
    system._slot_4d_deformation_latest_metrics = {"updates": 1, "pred_delta_norm": 0.05, "sample_count": 64, "loss": 0.001}
    system._slot_4d_playback_latest_metrics = {
        "slot_id": 0,
        "frame_count": 1,
        "playback_phase": 0.25,
        "pred_delta_norm": 0.05,
        "render_valid": True,
        "deformation_used": True,
        "preview_fps": 30.0,
    }
    system.slot_4d_playback_renderer = SimpleNamespace(last_preview={
        0: {
            "rgb": torch.ones(16, 16, 3),
            "depth": torch.ones(16, 16, 1) * 0.5,
            "alpha": torch.ones(16, 16, 1),
        }
    })

    obj = system._attach_slot_4d_playback_tensors({
        "z_obj": torch.zeros(1, 128),
        "confidence": torch.ones(1, 1),
    })

    assert float(obj["slot_4d_playback_render_valid"].item()) == 1.0
    assert float(obj["slot_4d_playback_deformation_used"].item()) == 1.0
    assert tuple(obj["slot_4d_playback_rgb"].shape) == (1, 3, 16, 16)
    assert tuple(obj["slot_4d_playback_depth"].shape) == (1, 1, 16, 16)
    assert tuple(obj["slot_4d_playback_alpha"].shape) == (1, 1, 16, 16)

def test_inner_object_open3d_viewer_is_pure_mirror_without_slot_input_state():
    source = Path("src/modules/m01_object_imagery/inner_object_open3d_viewer.py").read_text(encoding="utf-8")
    assert "VisualizerWithKeyCallback" not in source
    assert "register_key_callback" not in source
    assert "requested_slot_index" not in source
    assert "slot_selection_version" not in source

def test_m1_imit_open3d_display_follows_selected_slot():
    system = DummyProgressiveSystem()
    system.request_m1_object_slot_latents({"kind":"cube_tetra","duration":5,"cube_slot":1,"tetra_slot":2,"selected_slot":2,"auto_select_slot":True})
    proposals = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    system._inner_object_proposal_target_slots = list(proposals["target_slots"])
    system._inner_object_proposal_kinds = list(proposals["proposal_kinds"])
    system._inner_object_proposal_target_names = list(proposals["target_names"])

    decoded = system._run_progressive_inner_object_system(
        system.inner_object_system.initial_state(batch_size=1, device=system.device),
        proposals["proposals"],
        torch.zeros(1, 50),
        torch.zeros(1, 12),
        torch.zeros(1, 34),
        torch.zeros(1, 18),
        dream_mode=True,
    )

    system.inner_object_viz.requested_dream_slot_index = 1
    cube_obj = system._inner_object_open3d_display_obj(decoded)
    assert int(cube_obj["active_slot_index"].reshape(-1)[0].item()) == 1
    assert int(cube_obj["open3d_display_slot"].reshape(-1)[0].item()) == 1
    assert cube_obj["debug_imit_fallback_shape"] is True
    assert cube_obj["debug_imit_shape_kind"] == "cube"

    system.inner_object_viz.requested_dream_slot_index = 2
    tetra_obj = system._inner_object_open3d_display_obj(decoded)
    assert int(tetra_obj["active_slot_index"].reshape(-1)[0].item()) == 2
    assert int(tetra_obj["open3d_display_slot"].reshape(-1)[0].item()) == 2
    assert tetra_obj["debug_imit_fallback_shape"] is True
    assert tetra_obj["debug_imit_shape_kind"] == "tetrahedron"

def test_m1_imit_open3d_display_uses_current_frame_requested_slot_marker():
    system = DummyProgressiveSystem()
    system.request_m1_object_slot_latents({"kind":"cube_tetra","duration":5,"cube_slot":1,"tetra_slot":2,"selected_slot":2,"auto_select_slot":True})
    proposals = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    system._inner_object_proposal_target_slots = list(proposals["target_slots"])
    system._inner_object_proposal_kinds = list(proposals["proposal_kinds"])
    system._inner_object_proposal_target_names = list(proposals["target_names"])

    decoded = system._run_progressive_inner_object_system(
        system.inner_object_system.initial_state(batch_size=1, device=system.device),
        proposals["proposals"],
        torch.zeros(1, 50),
        torch.zeros(1, 12),
        torch.zeros(1, 34),
        torch.zeros(1, 18),
        dream_mode=True,
    )

    system.inner_object_viz.requested_dream_slot_index = None
    system._ipc_inner_object_dream_slot_index = None
    decoded["_requested_dream_slot_index"] = 1
    cube_obj = system._inner_object_open3d_display_obj(decoded)
    assert int(cube_obj["open3d_display_slot"].reshape(-1)[0].item()) == 1
    assert cube_obj["debug_imit_shape_kind"] == "cube"

def test_m1_imit_open3d_selected_slot_does_not_inherit_active_tetra_debug_kind():
    system = DummyProgressiveSystem()
    cube_z, _ = system.make_m1_object_slot_latent("cube", device=system.device, dim=128)
    tetra_z, _ = system.make_m1_object_slot_latent("tetrahedron", device=system.device, dim=128)
    state = system.inner_object_system.initial_state(batch_size=1, device=system.device)
    state["z_obj_slots"][:, 1, :] = cube_z
    state["z_obj_slots"][:, 2, :] = tetra_z
    state["confidence_slots"][:, 1, :] = 1.0
    state["confidence_slots"][:, 2, :] = 1.0
    state["active_slot_index"] = torch.tensor([[2]], device=system.device, dtype=torch.long)
    decoded = system.inner_object_system.decode_z(tetra_z, state)
    decoded["z_obj_slots"] = state["z_obj_slots"]
    decoded["confidence_slots"] = state["confidence_slots"]
    decoded["active_slot_index"] = state["active_slot_index"]
    decoded["debug_imit_fallback_shape"] = True
    decoded["debug_imit_source"] = "m1_object_slot_imit"
    decoded["debug_imit_shape_kind"] = "tetrahedron"

    system.inner_object_viz.requested_dream_slot_index = 1
    cube_obj = system._inner_object_open3d_display_obj(decoded)
    assert int(cube_obj["open3d_display_slot"].reshape(-1)[0].item()) == 1
    assert cube_obj["debug_imit_fallback_shape"] is True
    assert cube_obj["debug_imit_shape_kind"] == "cube"

def test_m1_imit_open3d_selected_empty_slot_clears_active_tetra_geometry():
    system = DummyProgressiveSystem()
    tetra_z, _ = system.make_m1_object_slot_latent("tetrahedron", device=system.device, dim=128)
    state = system.inner_object_system.initial_state(batch_size=1, device=system.device)
    state["z_obj_slots"][:, 2, :] = tetra_z
    state["confidence_slots"][:, 2, :] = 1.0
    state["active_slot_index"] = torch.tensor([[2]], device=system.device, dtype=torch.long)
    decoded = system.inner_object_system.decode_z(tetra_z, state)
    decoded["z_obj_slots"] = state["z_obj_slots"]
    decoded["confidence_slots"] = state["confidence_slots"]
    decoded["active_slot_index"] = state["active_slot_index"]
    decoded["debug_imit_fallback_shape"] = True
    decoded["debug_imit_source"] = "m1_object_slot_imit"
    decoded["debug_imit_shape_kind"] = "tetrahedron"

    system.inner_object_viz.requested_dream_slot_index = 5
    empty_obj = system._inner_object_open3d_display_obj(decoded)
    assert int(empty_obj["open3d_display_slot"].reshape(-1)[0].item()) == 5
    assert int(empty_obj["point_cloud"].shape[1]) == 0
    assert float(empty_obj["confidence"].detach().float().sum().item()) == 0.0
    assert float(empty_obj["open3d_display_empty_slot"].detach().float().sum().item()) == 1.0
    assert "debug_imit_shape_kind" not in empty_obj

def test_inner_object_viz_slot_selection_syncs_to_open3d_ipc_slot():
    system = DummyProgressiveSystem()
    system.inner_object_viz.requested_dream_slot_index = 12
    assert system._sync_inner_object_requested_slot_from_viz() == 9
    assert system._ipc_inner_object_dream_slot_index == 9

    system.inner_object_viz.requested_dream_slot_index = None
    assert system._sync_inner_object_requested_slot_from_viz() is None
    assert system._ipc_inner_object_dream_slot_index is None

def test_inner_object_selected_slot_is_not_cleared_outside_dream_mode():
    runtime_source = Path("src/modules/m01_object_imagery/runtime.py").read_text(encoding="utf-8")
    assert "Leaving full sleep immediately cancels dream-only UI state" not in runtime_source

def test_m1_single_morph_status_payload():
    system = DummySystem()
    system.request_m1_object_slot_latents({"kind":"morph","slot":3,"alpha":0.5,"duration":3})
    result = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
    assert result["target_slots"] == [3]
    assert result["target_names"] == ["morph"]
    from src.modules.m08_debug_visual_control.m1_object_slot_imit_status import build_m1_object_slot_imit_status
    status = build_m1_object_slot_imit_status(system)
    assert status["active"] is True
    assert status["layout"] == "imit"
    assert status["is_m1_imit"] is True
    assert status["selected_slot"] == 3

def test_m1_single_cube_tetra_morph_select_expected_slots():
    for kind, slot, name in (("cube", 1, "cube"), ("tetrahedron", 2, "tetrahedron"), ("morph", 3, "morph")):
        system = DummySystem()
        state = system.request_m1_object_slot_latents({"kind": kind, "slot": slot, "duration": 3})
        assert state["selected_slot"] == slot
        assert system.inner_object_viz.requested_dream_slot_index == slot
        assert system._ipc_inner_object_dream_slot_index == slot
        result = system.get_m1_imit_inner_object_proposals(torch.zeros(1, 12))
        assert result["target_slots"] == [slot]
        assert result["target_names"] == [name]

def test_m1_object_slot_imit_runner_uses_imit_mixin():
    source = Path("src/apps/runner.py").read_text(encoding="utf-8")
    assert (
        "from src.modules.m01_object_imagery.imit."
        "m1_object_slot_latent_runtime import M1ObjectSlotLatentImitRuntimeMixin"
    ) in source
    assert "M1ObjectSlotLatentImitRuntimeMixin," in source

def test_m1_object_slot_imit_runtime_hook_and_allowed_kind():
    source = Path("src/modules/m01_object_imagery/runtime.py").read_text(encoding="utf-8")
    assert 'if hasattr(self, "get_m1_imit_inner_object_proposals"):' in source
    assert '"dynamic_source": "m1_object_slot_imit"' in source
    assert '"m1_imit_dynamic_object"' in source
    assert "force_slot_index=slot_index" in source

def test_m1_object_slot_imit_status_payload_registered():
    source = Path("src/modules/m08_debug_visual_control/module_status_runtime.py").read_text(encoding="utf-8")
    assert "from src.modules.m08_debug_visual_control.m1_object_slot_imit_status import build_m1_object_slot_imit_status" in source
    assert '"m1_object_slot_imit": build_m1_object_slot_imit_status(self)' in source
    for key in (
        '"sleep_replay_monitor"',
        '"replay_quality_monitor"',
        '"m5_learning_quality"',
        '"m5_latent_prototype"',
        '"last_module_lab_result"',
    ):
        assert key in source

def test_m1_object_slot_imit_control_panel_window_contract():
    source = Path("src/modules/m08_debug_visual_control/control_panel.py").read_text(encoding="utf-8")
    assert 'self.btn_m1_object_slot_imit = QtWidgets.QPushButton("M1 Object Slot Imit")' in source
    assert '"btn_m1_object_slot_imit"' in source
    assert "self.btn_m1_object_slot_imit.clicked.connect(self.open_m1_object_slot_imit_window)" in source
    assert "def open_m1_object_slot_imit_window(self):" in source
    assert "def refresh_m1_object_slot_imit_window(self):" in source
    assert "self.refresh_m1_object_slot_imit_window()" in source
    assert "Fill cube slot1 + tetra slot2" in source
    assert "Cube → slot1" in source
    assert "Tetra → slot2" in source
    assert "Morph → slot3" in source
    assert "Clear" in source
    assert "m1_object_slot_imit_inject" in source
    assert 'source": "m8_m1_object_slot_imit_window"' in source
    assert '"m1": [' in source
    assert '"btn_m1_object_slot_imit",' in source
    assert '"m8": ["btn_module_debug", "btn_module_lab", "btn_sleep_replay_monitor", "btn_replay_quality_monitor"]' in source
    assert "self._style_plain_status_button(\n                self.btn_m1_object_slot_imit," in source
    assert 'self._window_visible(getattr(self, "m1_object_slot_imit_window", None))' in source
    assert "self.btn_module_lab,\n                self.btn_m1_object_slot_imit," not in source

def test_control_panel_detached_tool_windows_are_top_level():
    source = Path("src/modules/m08_debug_visual_control/control_panel.py").read_text(encoding="utf-8")

    assert "QtWidgets.QDialog(self)" not in source
    for title in (
        "M8 Sleep Replay Monitor",
        "M8 Replay Quality Monitor",
        "M8 M5 Learning Quality Baseline",
        "M8 M1 Object Slot Latent Imitator",
        "M8 M5 Latent Prototype Simulator",
        "M8 Module Lab",
    ):
        assert title in source
