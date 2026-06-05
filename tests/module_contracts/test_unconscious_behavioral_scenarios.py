from __future__ import annotations


def test_unconscious_behavioral_scenarios_pass():
    from scripts.module_lab.scenario_unconscious_replay import run_all

    result = run_all()
    assert result["status"] == "ok", result
    names = {item["name"] for item in result["scenarios"]}
    assert "calm_no_replay" in names
    assert "curiosity_replay" in names
    assert "bad_prediction_dream" in names
    assert "object_identity_replay" in names


def test_bad_prediction_has_more_pressure_than_calm():
    from scripts.module_lab.scenario_unconscious_replay import scenario_bad_prediction_dream, scenario_calm_no_replay

    calm = scenario_calm_no_replay()
    bad = scenario_bad_prediction_dream()
    assert bad["dream_pressure"] >= calm["dream_pressure"]
