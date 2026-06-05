from __future__ import annotations

import torch


class DummySystem:
    def __init__(self):
        self.global_step = 1
        self.video_sensor_enabled = False
        self.contact_sensor_enabled = False
        self.imu_sensor_enabled = False
        self.latest_out = {
            "affect": {
                "stress_latent": torch.tensor([[0.10]]),
                "panic_latent": torch.tensor([[0.05]]),
                "curiosity_latent": torch.tensor([[0.20]]),
            },
            "event_dream_replay": {
                "replay_context": torch.ones(1, 256),
                "replay_gate": torch.tensor([[0.1]]),
            },
            "sleep_motor_guard": {"blocked": True},
        }

    def is_full_sleep_mode(self):
        return True

    def sensor_state_label(self):
        return "sleep"

    def input_sensors_enabled_dict(self):
        return {"video": False, "contact": False, "imu": False}


def test_m11_affect_visibility_adds_deltas_and_ranges():
    from src.modules.m08_debug_visual_control.sleep_replay_monitor_status import build_sleep_replay_monitor_status

    sys = DummySystem()
    first = build_sleep_replay_monitor_status(sys)
    assert "m11_delta" in first
    assert "m11_range" in first
    assert "m11_activity" in first
    assert first["m11_activity"]["samples"] == 1

    sys.global_step = 2
    sys.latest_out["affect"]["stress_latent"] = torch.tensor([[0.14]])
    sys.latest_out["affect"]["panic_latent"] = torch.tensor([[0.08]])
    sys.latest_out["affect"]["curiosity_latent"] = torch.tensor([[0.30]])

    second = build_sleep_replay_monitor_status(sys)

    assert second["m11_delta"]["stress"] > 0
    assert second["m11_delta"]["panic"] > 0
    assert second["m11_delta"]["curiosity"] > 0
    assert second["m11_activity"]["change_score"] > 0
    assert second["m11_activity"]["trend"] in ("↑", "↓", "→", "↕")
    assert second["m11_range"]["stress_min"] <= second["m11_range"]["stress_max"]

    repeated = build_sleep_replay_monitor_status(sys)
    assert repeated["m11_delta"] == second["m11_delta"]
    assert repeated["m11_activity"] == second["m11_activity"]

    sys.latest_out["affect"]["curiosity_latent"] = torch.tensor([[0.45]])
    changed_same_step = build_sleep_replay_monitor_status(sys)
    assert changed_same_step["m11_delta"]["curiosity"] > 0
