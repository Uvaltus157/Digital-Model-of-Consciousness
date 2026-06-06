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
    _publish_m2_scenario_rpc = M2ScenarioImitRuntimeMixin._publish_m2_scenario_rpc
    _clear_m2_scenario_rpc_slots = M2ScenarioImitRuntimeMixin._clear_m2_scenario_rpc_slots

    def __init__(self):
        self.device = torch.device("cpu")
        self.global_step = 7
        self.cfg = SimpleNamespace(object_image=SimpleNamespace(
            slot_4d_jsonrpc_host="127.0.0.1",
            slot_4d_jsonrpc_port=8871,
            slot_4d_jsonrpc_sample_points=4096,
        ))
        self.status_writes = 0

    def write_module_debug_status(self):
        self.status_writes += 1

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
    assert system.slot_4d_jsonrpc_streamer.status()["slots"] == {}
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
    assert '"m2": ["btn_event_code", "btn_m2_scenario_imit"]' in panel
    assert 'self.btn_m2_scenario_imit = QtWidgets.QPushButton("M2 Scenario Imit")' in panel
    assert "self.btn_m2_scenario_imit.clicked.connect(self.open_m2_scenario_imit_window)" in panel
    assert "m2_scenario_imit_inject" in panel
    assert "Cube slot0 + Tetra slot1" in panel
    assert "Open3D RPC" in panel
