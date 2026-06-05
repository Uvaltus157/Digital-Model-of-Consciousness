from __future__ import annotations

from pathlib import Path


def test_m8_module_lab_runtime_bridge_runs_loop_or_all():
    from src.modules.m08_debug_visual_control.module_lab_runtime import run_module_lab_from_payload
    result = run_module_lab_from_payload({"module": "loop"})
    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["kind"] == "module_lab"
    assert result["module"] == "loop"


def test_m8_module_lab_runtime_bridge_runs_behavioral_scenarios():
    from src.modules.m08_debug_visual_control.module_lab_runtime import run_module_lab_from_payload
    result = run_module_lab_from_payload({"module": "scenarios"})
    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["kind"] == "behavioral_scenarios"


def test_m8_sleep_replay_button_sends_full_sensor_state():
    source = Path("src/modules/m08_debug_visual_control/control_panel.py").read_text(encoding="utf-8")
    assert 'QPushButton("Сон / replay mode")' in source
    assert "toggle_sleep_replay_mode" in source
    for field in (
        "input_sensors_enabled=sensors",
        "sleep_sensor_mask=mask",
        "video_sensor_enabled=sensors_enabled",
        "contact_sensor_enabled=sensors_enabled",
        "imu_sensor_enabled=sensors_enabled",
        "sleep_video_cut=enable_sleep",
        "sleep_contact_cut=enable_sleep",
        "sleep_imu_cut=enable_sleep",
    ):
        assert field in source
