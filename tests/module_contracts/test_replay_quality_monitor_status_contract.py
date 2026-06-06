from __future__ import annotations

import torch


class DummyQualitySystem:
    def __init__(self):
        self.global_step = 1
        self.latest_out = {
            "affect": {
                "stress_latent": torch.tensor([[0.5]]),
                "panic_latent": torch.tensor([[0.2]]),
                "relief_latent": torch.tensor([[0.1]]),
                "curiosity_latent": torch.tensor([[0.4]]),
                "valence": torch.tensor([[0.0]]),
                "coherence_latent": torch.tensor([[0.3]]),
                "expected_affect_delta": torch.tensor([[0.0]]),
            },
            "autobiographical_memory": {
                "retrieval_relevance": torch.tensor([[0.7]]),
                "retrieved_episode_count": torch.tensor([[5.0]]),
                "summary": "test episode",
            },
            "long_dynamic_memory": {
                "identity_token": "obj:test",
                "dynamic_memory_gate": torch.tensor([[0.8]]),
                "identity_stability": torch.tensor([[0.9]]),
                "identity_novelty": torch.tensor([[0.1]]),
                "selected_sentence": "stable object",
            },
            "event_dream_replay": {
                "replay_gate": torch.tensor([[0.6]]),
                "should_replay": torch.tensor([[1.0]]),
                "dream_pressure": torch.tensor([[0.8]]),
                "event_salience": torch.tensor([[0.5]]),
                "replay_source": "test",
                "selected_episode_summary": "selected episode",
                "selected_identity_token": "obj:test",
                "next_focus_context_seed": torch.ones(1, 256),
                "next_focus_context_seed_gate": torch.tensor([[0.6]]),
            },
        }

    def is_full_sleep_mode(self):
        return True

    def sensor_state_label(self):
        return "sleep"


def test_replay_quality_monitor_payload_and_deltas():
    from src.modules.m08_debug_visual_control.replay_quality_monitor_status import build_replay_quality_monitor_status

    sys = DummyQualitySystem()
    first = build_replay_quality_monitor_status(sys)

    expected_top = {
        "global_step", "full_sleep", "sensor_state", "quality_score", "quality_ema",
        "verdict", "selected_episode_summary", "selected_identity_token",
        "selected_identity_sentence", "replay_source", "m2", "affect", "m13",
        "m4", "m5", "dream_probe", "history", "samples",
    }
    assert expected_top <= set(first)
    assert first["selected_identity_token"] == "obj:test"
    assert first["selected_episode_summary"] == "selected episode"
    assert first["m2"]["replay_gate"] == 0.6
    assert first["m5"]["seed_norm"] > 0
    assert 0.0 <= first["quality_score"] <= 1.0
    assert first["verdict"] in ("idle", "weak", "replaying", "integrating")
    assert {
        "replay_gate", "should_replay", "dream_pressure", "event_salience",
        "dream_pressure_delta", "dream_pressure_trend",
    } <= set(first["m2"])
    assert {
        "stress", "stress_delta", "panic", "panic_delta", "relief",
        "relief_delta", "curiosity", "valence", "valence_delta", "coherence",
        "coherence_delta", "expected_affect_delta",
    } <= set(first["affect"])

    sys.global_step = 2
    sys.latest_out["event_dream_replay"]["dream_pressure"] = torch.tensor([[0.6]])
    sys.latest_out["affect"]["relief_latent"] = torch.tensor([[0.2]])
    sys.latest_out["affect"]["stress_latent"] = torch.tensor([[0.4]])
    second = build_replay_quality_monitor_status(sys)

    assert second["m2"]["dream_pressure_delta"] < 0
    assert second["affect"]["relief_delta"] > 0
    assert second["affect"]["stress_delta"] < 0
    assert second["verdict"] == "integrating"
    assert len(second["history"]) >= 2

    repeated = build_replay_quality_monitor_status(sys)
    assert repeated["m2"]["dream_pressure_delta"] == second["m2"]["dream_pressure_delta"]
    assert repeated["affect"]["relief_delta"] == second["affect"]["relief_delta"]


def test_replay_quality_monitor_uses_runtime_seed_bus_and_is_read_only():
    from src.modules.m08_debug_visual_control.replay_quality_monitor_status import build_replay_quality_monitor_status

    sys = DummyQualitySystem()
    sys.latest_out["event_dream_replay"]["replay_gate"] = torch.tensor([[0.0]])
    sys.latest_out["event_dream_replay"].pop("next_focus_context_seed", None)
    sys.latest_out["event_dream_replay"].pop("next_focus_context_seed_gate", None)
    sys._event_dream_next_focus_seed = torch.ones(1, 256)
    sys._event_dream_next_focus_gate = torch.tensor([[0.75]])
    before_ids = {k: id(v) for k, v in sys.latest_out.items()}

    payload = build_replay_quality_monitor_status(sys)

    assert payload["m5"]["seed_gate"] == 0.75
    assert payload["m5"]["seed_norm"] > 0.0
    assert payload["quality_score"] > 0.0
    assert payload["verdict"] in ("weak", "replaying", "integrating")
    assert before_ids == {k: id(v) for k, v in sys.latest_out.items()}
