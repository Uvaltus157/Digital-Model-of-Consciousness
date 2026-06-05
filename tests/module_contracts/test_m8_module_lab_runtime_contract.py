from __future__ import annotations


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
