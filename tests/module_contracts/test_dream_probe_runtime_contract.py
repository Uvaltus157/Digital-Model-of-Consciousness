from __future__ import annotations

import torch
import pytest


class DummyProbeSystem:
    from src.modules.m02_event_dream_replay.dream_probe_runtime import DreamProbeRuntimeMixin

    request_dream_probe = DreamProbeRuntimeMixin.request_dream_probe
    apply_dream_probe_to_out = DreamProbeRuntimeMixin.apply_dream_probe_to_out
    _inject_probe_replay_seed = DreamProbeRuntimeMixin._inject_probe_replay_seed

    def __init__(self):
        self.global_step = 10
        self.latest_out = {
            "focus_context": torch.ones(1, 256) * 0.25,
        }
        self.status_writes = 0

    def write_module_debug_status(self):
        self.status_writes += 1


def test_curiosity_probe_changes_values_without_focus_mutation():
    sys = DummyProbeSystem()
    sys.request_dream_probe({"kind": "curiosity", "intensity": 0.8, "duration": 3})

    out = {
        "values": {
            "curiosity": torch.tensor([[0.1]]),
            "coherence": torch.tensor([[0.9]]),
        },
        "focus_context": torch.zeros(1, 256),
    }
    old_focus = out["focus_context"].clone()
    out2 = sys.apply_dream_probe_to_out(out)

    assert out2["dream_probe"]["kind"] == "curiosity"
    assert float(out2["values"]["curiosity"].item()) >= 0.7
    assert torch.allclose(out2["focus_context"], old_focus)
    assert sys._dream_probe_state["remaining"] == 2


def test_stress_probe_lowers_confidence_for_m11_uncertainty():
    sys = DummyProbeSystem()
    sys.request_dream_probe({"kind": "stress", "intensity": 0.9, "duration": 2})

    out = {
        "values": {"coherence": torch.tensor([[0.95]])},
        "object_imagery": {"object_confidence": torch.tensor([[0.95]])},
        "preconscious_reflection_out": {"model_confidence": torch.tensor([[0.95]])},
    }
    out2 = sys.apply_dream_probe_to_out(out)

    assert out2["dream_probe"]["kind"] == "stress"
    assert float(out2["values"]["coherence"].item()) < 0.2
    assert float(out2["object_imagery"]["object_confidence"].item()) < 0.2
    assert float(out2["preconscious_reflection_out"]["model_confidence"].item()) < 0.2


def test_replay_seed_probe_uses_existing_m5_seed_bus():
    sys = DummyProbeSystem()
    state = sys.request_dream_probe({"kind": "replay_seed", "intensity": 0.6, "duration": 4})

    assert state["kind"] == "replay_seed"
    assert torch.is_tensor(sys._event_dream_next_focus_seed)
    assert torch.is_tensor(sys._event_dream_next_focus_gate)
    assert tuple(sys._event_dream_next_focus_seed.shape) == (1, 256)
    assert float(sys._event_dream_next_focus_gate.item()) == pytest.approx(0.6)
    assert sys.status_writes == 1


def test_probe_invalidates_cached_m11_packet():
    sys = DummyProbeSystem()
    sys.request_dream_probe({"kind": "curiosity", "intensity": 0.8, "duration": 3})
    out = {
        "values": {"curiosity": torch.tensor([[0.1]])},
        "emotion": {"_emotion_cache_reusable": True},
        "affect": {"curiosity_latent": torch.tensor([[0.1]])},
    }

    sys.apply_dream_probe_to_out(out)

    assert out["emotion"]["_emotion_cache_reusable"] is False
    assert "affect" not in out
