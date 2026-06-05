from __future__ import annotations

import torch

from scripts.module_lab.module_fixture_factory import (
    assert_gate,
    assert_tensor,
    make_fake_event_memory,
    make_unconscious_loop_out,
)


def test_m02_event_dream_replay_contract_with_m13_and_m4():
    from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig

    cfg = EventDreamReplayConfig(
        replay_context_dim=256,
        event_code_dim=8,
        blend_replay_into_focus=False,
        use_m13_context=True,
        use_m4_context=True,
        seed_to_m5_boundary=True,
    )
    m2 = EventDreamReplay(cfg)
    out = make_unconscious_loop_out()

    packet = m2.compute(out=out, event_memory=make_fake_event_memory(), dream_mode=True)

    for key in ("replay_context", "replay_gate", "event_salience", "dream_pressure", "should_replay", "replay_source", "selected_episode_summary"):
        assert key in packet, f"missing {key}"

    assert_tensor("replay_context", packet["replay_context"], (1, 256))
    assert_gate("replay_gate", packet["replay_gate"], 0.0, 1.0)
    assert_gate("dream_pressure", packet["dream_pressure"], 0.0, 1.0)
    assert_gate("should_replay", packet["should_replay"], 0.0, 1.0)

    assert "selected_identity_token" in packet, "M2 must expose M4 identity token"
    assert "dynamic_memory_gate" in packet, "M2 must expose M4 dynamic memory gate"
    assert packet["replay_source"] == "m02_event_memory"


def test_m02_dream_pressure_increases_with_affect():
    from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig

    cfg = EventDreamReplayConfig(
        replay_context_dim=256,
        event_code_dim=8,
        blend_replay_into_focus=False,
        use_m13_context=True,
        use_m4_context=True,
        seed_to_m5_boundary=True,
    )
    m2 = EventDreamReplay(cfg)

    low = make_unconscious_loop_out()
    low["affect"]["panic_latent"] = torch.tensor([[0.0]])
    low["affect"]["stress_latent"] = torch.tensor([[0.0]])
    low_packet = m2.compute(out=low, event_memory=None, dream_mode=True)

    high = make_unconscious_loop_out()
    high["affect"]["panic_latent"] = torch.tensor([[0.9]])
    high["affect"]["stress_latent"] = torch.tensor([[0.9]])
    high_packet = m2.compute(out=high, event_memory=None, dream_mode=True)

    assert float(high_packet["dream_pressure"].item()) >= float(low_packet["dream_pressure"].item())
