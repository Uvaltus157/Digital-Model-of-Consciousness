from __future__ import annotations

from types import SimpleNamespace

import torch


class DummySystem:
    def __init__(self) -> None:
        self.global_step = 42
        self.video_sensor_enabled = False
        self.contact_sensor_enabled = False
        self.imu_sensor_enabled = False
        self.latest_out = {
            "affect": {
                "stress_latent": torch.tensor([[0.4]]),
                "panic_latent": torch.tensor([[0.2]]),
                "curiosity_latent": torch.tensor([[0.7]]),
            },
            "autobiographical_memory": {
                "retrieval_relevance": torch.tensor([[0.5]]),
                "retrieved_episode_count": torch.tensor([[3.0]]),
                "summary": "episode",
            },
            "long_dynamic_memory": {
                "identity_token": "obj:test",
                "dynamic_memory_gate": torch.tensor([[0.8]]),
                "identity_stability": torch.tensor([[0.9]]),
                "identity_novelty": torch.tensor([[0.1]]),
            },
            "event_dream_replay": {
                "replay_context": torch.ones(1, 256),
                "replay_gate": torch.tensor([[1.0]]),
                "dream_pressure": torch.tensor([[0.75]]),
                "event_salience": torch.tensor([[0.55]]),
                "should_replay": torch.tensor([[1.0]]),
                "replay_source": "test",
            },
            "sleep_motor_guard": {
                "blocked": True,
                "blocked_motor_norm": 1.25,
                "blocked_keys": ["embodied_targets", "hand_ctrl"],
                "stage": "main",
            },
        }

    def is_full_sleep_mode(self):
        return True

    def sensor_state_label(self):
        return "sleep"

    def input_sensors_enabled_dict(self):
        return {"video": False, "contact": False, "imu": False}


def test_build_sleep_replay_monitor_status_compact_payload():
    from src.modules.m08_debug_visual_control.sleep_replay_monitor_status import build_sleep_replay_monitor_status

    payload = build_sleep_replay_monitor_status(DummySystem())
    assert payload["global_step"] == 42
    assert payload["full_sleep"] is True
    assert payload["sensor_state"] == "sleep"
    assert payload["m1"]["video_on"] is False
    assert payload["m11"]["curiosity"] == 0.7
    assert payload["m13"]["summary"] == "episode"
    assert payload["m4"]["token"] == "obj:test"
    assert payload["m2"]["replay_gate"] == 1.0
    assert payload["m5"]["seed_norm"] > 0
    assert payload["m3"]["sleep_blocked"] is True


def test_monitor_prefers_fresh_runtime_values_and_probe_seed_over_stale_trace():
    from src.modules.m08_debug_visual_control.sleep_replay_monitor_status import build_sleep_replay_monitor_status

    sys = DummySystem()
    sys.latest_out["unconscious_loop_trace"] = {
        "m11": {"curiosity": 0.0},
        "m2": {"dream_pressure": 0.0},
        "m5_seed": {"seed_gate": 0.0, "seed_norm": 0.0},
    }
    sys._event_dream_next_focus_seed = torch.ones(1, 256)
    sys._event_dream_next_focus_gate = torch.tensor([[0.75]])
    sys.latest_out["event_dream_replay"]["dream_pressure"] = torch.tensor([[0.9]])

    payload = build_sleep_replay_monitor_status(sys)

    assert payload["m11"]["curiosity"] == 0.7
    assert payload["m2"]["dream_pressure"] == 0.9
    assert payload["m5"]["seed_gate"] == 0.75
    assert payload["m5"]["seed_norm"] > 0
