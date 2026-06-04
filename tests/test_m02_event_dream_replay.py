from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from src.modules.m02_event_dream_replay.event_dream_replay import (
    EventDreamReplay,
    EventDreamReplayConfig,
)
from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin


def test_m2_event_dream_replay_builds_replay_packet_from_event_and_m13() -> None:
    event_memory = SimpleNamespace(
        events=deque([
            {
                "slot": 1,
                "confidence": 0.8,
                "delta_norm": 0.7,
                "action_norm": 0.4,
                "contact_norm": 0.6,
                "vision_strength": 0.5,
                "touch_strength": 0.7,
                "dream_mode": False,
                "event_code": torch.tensor([[1.0, 0.8, 0.7, 0.4, 0.6, 0.5, 0.7, 0.0]]),
                "semantic_sentence": "SENT SUBJ=OBJ_001 VERB=touch_changes",
                "kind": "contact_transition",
                "slot_token": "OBJ_001",
            }
        ], maxlen=8),
        last_event=None,
    )
    replay = EventDreamReplay(EventDreamReplayConfig(replay_context_dim=16, replay_threshold=0.2))
    out = {
        "focus_context": torch.randn(1, 16),
        "affect": {
            "panic_latent": torch.tensor([[0.1]]),
            "stress_latent": torch.tensor([[0.2]]),
            "curiosity_latent": torch.tensor([[0.8]]),
        },
        "metacognition": {"doubt": torch.tensor([[0.3]])},
        "thought_chain": {
            "no_viable_chain": torch.tensor([[0.0]]),
            "predicted_affect_delta": torch.tensor([[0.1]]),
        },
        "autobiographical_memory": {
            "retrieved_context": torch.randn(1, 16),
            "retrieval_relevance": torch.tensor([[0.5]]),
            "summary": "step=7 source=m15_focus",
        },
    }

    packet = replay.compute(out=out, event_memory=event_memory, dream_mode=False)

    assert packet["replay_context"].shape == (1, 16)
    assert packet["replay_gate"].shape == (1, 1)
    assert packet["event_salience"].shape == (1, 1)
    assert packet["dream_pressure"].shape == (1, 1)
    assert packet["should_replay"].shape == (1, 1)
    assert float(packet["should_replay"].item()) == 1.0
    assert packet["selected_event_sentence"] == "SENT SUBJ=OBJ_001 VERB=touch_changes"
    assert packet["selected_event_kind"] == "contact_transition"
    assert packet["selected_event_slot_token"] == "OBJ_001"
    assert packet["selected_episode_summary"] == "step=7 source=m15_focus"
    assert packet["replay_source"] == "m02_event_memory"


def test_m2_event_dream_replay_blends_m4_identity_context() -> None:
    replay = EventDreamReplay(EventDreamReplayConfig(
        replay_context_dim=4,
        replay_threshold=0.0,
        use_m13_context=False,
        use_m4_context=True,
        m4_context_weight=0.20,
        use_event_memory=False,
    ))
    out = {
        "focus_context": torch.zeros(1, 4),
        "affect": {"curiosity_latent": torch.tensor([[0.1]])},
        "long_dynamic_memory": {
            "dynamic_identity_context": torch.ones(1, 4),
            "identity_stability": torch.tensor([[0.7]]),
            "identity_novelty": torch.tensor([[0.3]]),
            "dynamic_memory_gate": torch.tensor([[1.0]]),
            "identity_token": "OBJ_009",
            "selected_sentence": "SENT SUBJ=OBJ_009 VERB=changed",
        },
    }

    packet = replay.compute(out=out, event_memory=None, dream_mode=False)

    assert torch.allclose(packet["replay_context"], torch.full((1, 4), 0.20))
    assert packet["selected_identity_token"] == "OBJ_009"
    assert packet["selected_identity_sentence"] == "SENT SUBJ=OBJ_009 VERB=changed"
    assert abs(float(packet["identity_stability"].item()) - 0.7) < 1e-6
    assert abs(float(packet["identity_novelty"].item()) - 0.3) < 1e-6
    assert float(packet["dynamic_memory_gate"].item()) == 1.0


def test_event_dream_runtime_default_uses_m5_seed_boundary_not_focus_blend() -> None:
    class Dummy(EventDreamReplayRuntimeMixin):
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                self_core=SimpleNamespace(focus_context_dim=4),
                event_dream_replay=SimpleNamespace(),
            )

    dummy = Dummy()

    dummy.ensure_event_dream_replay_ready()

    assert dummy.event_dream_replay.cfg.seed_to_m5_boundary is True
    assert dummy.event_dream_replay.cfg.blend_replay_into_focus is False
