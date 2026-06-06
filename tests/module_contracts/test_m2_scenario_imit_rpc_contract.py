from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import torch


class DummyM2ScenarioSystem:
    from src.modules.m02_event_dream_replay.imit.m2_scenario_runtime import M2ScenarioImitRuntimeMixin

    request_m2_scenario_imit = M2ScenarioImitRuntimeMixin.request_m2_scenario_imit
    m2_scenario_imit_status = M2ScenarioImitRuntimeMixin.m2_scenario_imit_status
    _m2_scenario_device = M2ScenarioImitRuntimeMixin._m2_scenario_device
    _ensure_m2_scenario_gaussian_reconstructor = M2ScenarioImitRuntimeMixin._ensure_m2_scenario_gaussian_reconstructor
    _m2_slot_items = M2ScenarioImitRuntimeMixin._m2_slot_items
    _install_m2_scenario_slot_states = M2ScenarioImitRuntimeMixin._install_m2_scenario_slot_states
    _ensure_m2_scenario_slot_4d_reconstructor = M2ScenarioImitRuntimeMixin._ensure_m2_scenario_slot_4d_reconstructor
    _m2_scenario_make_state = M2ScenarioImitRuntimeMixin._m2_scenario_make_state
    _install_m2_scenario_4d_timeline = M2ScenarioImitRuntimeMixin._install_m2_scenario_4d_timeline
    _publish_m2_scenario_rpc = M2ScenarioImitRuntimeMixin._publish_m2_scenario_rpc
    _clear_m2_scenario_rpc_slots = M2ScenarioImitRuntimeMixin._clear_m2_scenario_rpc_slots
    _sync_m2_scenario_inner_object_slots = M2ScenarioImitRuntimeMixin._sync_m2_scenario_inner_object_slots

    def __init__(self):
        self.device = torch.device("cpu")
        self.global_step = 7
        self.cfg = SimpleNamespace(object_image=SimpleNamespace(
            slot_4d_jsonrpc_host="127.0.0.1",
            slot_4d_jsonrpc_port=8871,
            slot_4d_jsonrpc_sample_points=4096,
            slot_4d_timeline_max_frames=256,
            slot_4d_sample_points=128,
            slot_4d_deformation_hidden_dim=32,
            slot_4d_deformation_lr=0.002,
            slot_4d_deformation_train_steps_per_update=1,
            slot_4d_deformation_min_frames=2,
            slot_4d_deformation_delta_reg_weight=0.0001,
            slot_4d_playback_period_steps=120,
            slot_4d_playback_strength=1.0,
        ))
        self.status_writes = 0
        self.m1_requests = []

    def write_module_debug_status(self):
        self.status_writes += 1

    def request_m1_object_slot_latents(self, payload):
        self.m1_requests.append(dict(payload or {}))
        return {"active": True, **dict(payload or {})}

    def _ensure_slot_4d_jsonrpc_streamer(self):
        from src.modules.m02_event_dream_replay.slot_4d_jsonrpc_stream import Slot4DJsonRpcStreamer

        self.slot_4d_jsonrpc_streamer = Slot4DJsonRpcStreamer(host="127.0.0.1", port=8871, sample_points=4096)
        self.slot_4d_jsonrpc_streamer.start()


def test_m2_scenario_imit_lives_in_m2_imit():
    assert Path("src/modules/m02_event_dream_replay/imit/m2_scenario_runtime.py").exists()
    assert Path("src/modules/m02_event_dream_replay/imit/__init__.py").read_text(encoding="utf-8").strip()


def test_m2_scenario_imit_publishes_cube_tetra_to_jsonrpc_slots():
    system = DummyM2ScenarioSystem()
    state = system.request_m2_scenario_imit({"kind": "cube_tetra", "density": 1, "source": "contract"})

    assert state["active"] is True
    assert state["kind"] == "cube_tetra"
    assert state["items"][0]["slot"] == 0
    assert state["items"][0]["kind"] == "cube"
    assert state["items"][0]["points"] > 0
    assert state["items"][1]["slot"] == 1
    assert state["items"][1]["kind"] == "tetrahedron"
    assert state["items"][1]["points"] > 0
    assert state["rpc"]["updated"] is True
    assert state["rpc"]["slot_0_points"] > 0
    assert state["rpc"]["slot_1_points"] > 0
    assert state["inner_object_slots"]["updated"] is True
    assert state["inner_object_slots"]["kind"] == "cube_tetra"
    assert system.m1_requests[-1]["cube_slot"] == 0
    assert system.m1_requests[-1]["tetra_slot"] == 1
    assert system.status_writes == 1

    status = system.slot_4d_jsonrpc_streamer.status()
    assert status["started"] is True
    assert status["slots"]["0"]["target_name"] == "cube"
    assert status["slots"]["0"]["raw_points"] > 0
    assert status["slots"]["1"]["target_name"] == "tetrahedron"
    assert status["slots"]["1"]["raw_points"] > 0

    both = system.slot_4d_jsonrpc_streamer.reply({
        "id": 1,
        "method": "slot_viewer.get_both_slots",
        "params": {"mode": "deformed"},
    })["result"]
    assert both["slots"][0]["point_count"] > 0
    assert both["slots"][1]["point_count"] > 0

    clear = system.request_m2_scenario_imit({"kind": "clear"})
    assert clear["active"] is False
    assert clear["rpc"]["slot_0_points"] == 0
    assert clear["rpc"]["slot_1_points"] == 0
    assert clear["inner_object_slots"]["updated"] is True
    assert system.m1_requests[-1]["kind"] == "clear"
    assert system.slot_4d_jsonrpc_streamer.status()["slots"] == {}
    system.slot_4d_jsonrpc_streamer.shutdown(timeout=0.2)


def test_m2_scenario_imit_single_cube_rpc_slot0_syncs_inner_object_slot0():
    system = DummyM2ScenarioSystem()
    state = system.request_m2_scenario_imit({"kind": "cube", "slot": 0, "density": 1, "source": "contract"})

    assert state["active"] is True
    assert state["items"][0]["slot"] == 0
    assert state["items"][0]["kind"] == "cube"
    assert state["rpc"]["slot_0_points"] > 0
    assert state["inner_object_slots"]["updated"] is True
    assert state["inner_object_slots"]["selected_slot"] == 0
    assert system.m1_requests[-1]["kind"] == "cube"
    assert system.m1_requests[-1]["slot"] == 0
    assert system.m1_requests[-1]["selected_slot"] == 0
    system.slot_4d_jsonrpc_streamer.shutdown(timeout=0.2)


def test_m2_scenario_imit_4d_cube_move_populates_timeline_deformation_and_rpc():
    system = DummyM2ScenarioSystem()
    state = system.request_m2_scenario_imit({
        "kind": "4d_cube_move",
        "slot": 0,
        "density": 1,
        "frames": 8,
        "amplitude": 0.6,
        "source": "contract",
    })

    assert state["active"] is True
    assert state["kind"] == "4d_cube_move"
    assert state["items"][0]["slot"] == 0
    assert state["items"][0]["frames"] == 8
    assert state["timeline"]["frames"] == 8
    assert state["timeline"]["timeline"]["frame_count"] == 8
    assert state["timeline"]["timeline"]["motion_norm"] > 0.0
    assert state["timeline"]["deformation"]["valid"] is True
    assert state["timeline"]["deformation"]["pred_delta_norm"] > 0.0
    assert state["rpc"]["updated"] is True
    assert state["rpc"]["slot_0_points"] > 0
    assert system.slot_4d_reconstructor.timeline.count(0) == 8
    assert system.slot_4d_jsonrpc_streamer.status()["slots"]["0"]["target_name"] == "cube"
    system.slot_4d_jsonrpc_streamer.shutdown(timeout=0.2)


def test_m2_scenario_imit_runner_ipc_status_and_control_panel_contracts():
    runner = Path("src/apps/runner.py").read_text(encoding="utf-8")
    assert "from src.modules.m02_event_dream_replay.imit.m2_scenario_runtime import M2ScenarioImitRuntimeMixin" in runner
    assert "M2ScenarioImitRuntimeMixin," in runner

    action_runtime = Path("src/modules/m03_self_action_causality/action_runtime.py").read_text(encoding="utf-8")
    assert '"m2_scenario_imit_inject"' in action_runtime
    assert "request_m2_scenario_imit" in action_runtime

    status_runtime = Path("src/modules/m08_debug_visual_control/module_status_runtime.py").read_text(encoding="utf-8")
    assert "from src.modules.m08_debug_visual_control.m2_scenario_imit_status import build_m2_scenario_imit_status" in status_runtime
    assert '"m2_scenario_imit": build_m2_scenario_imit_status(self)' in status_runtime

    panel = Path("src/modules/m08_debug_visual_control/control_panel.py").read_text(encoding="utf-8")
    assert '"btn_event_code",\n        "btn_m2_scenario_imit",\n        "btn_object_open3d_rpc",\n        "btn_object_open3d_step4",\n        "btn_object_open3d_file",' in panel
    assert 'self.btn_m2_scenario_imit = QtWidgets.QPushButton("M2 Scenario Imit")' in panel
    assert "self.btn_m2_scenario_imit.clicked.connect(self.open_m2_scenario_imit_window)" in panel
    assert '"btn_object_open3d_rpc",' not in panel.split('"m2":', 1)[0].split('"m1":', 1)[1]
    assert '"btn_object_open3d_step4",' not in panel.split('"m2":', 1)[0].split('"m1":', 1)[1]
    assert '"btn_object_open3d_file",' not in panel.split('"m2":', 1)[0].split('"m1":', 1)[1]
    assert "m2_scenario_imit_inject" in panel
    assert "Cube slot0 + Tetra slot1" in panel
    assert "4D Cube Move" in panel
    assert "4D Morph" in panel
    assert "4D Circle" in panel
    assert "timeline_frames" in panel
    assert "Open3D RPC" in panel
