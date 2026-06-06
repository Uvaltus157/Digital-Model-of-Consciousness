from __future__ import annotations

import torch

from scripts.module_lab.module_fixture_factory import assert_tensor, make_unconscious_loop_out


class DummyEventDreamRuntime:
    """Small local object to test EventDreamReplayRuntimeMixin without full runner."""

    from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin

    class Cfg:
        class EventDreamReplay:
            enabled = True
            replay_context_dim = 256
            event_code_dim = 8
            replay_threshold = 0.35
            focus_blend = 0.15
            blend_replay_into_focus = False
            use_m13_context = True
            use_m4_context = True
            m4_context_weight = 0.20
            use_event_memory = False
            max_recent_events_scan = 16
            seed_to_m5_boundary = True
            seed_gate_gain = 1.0
            apply_stage = "pre_observe"
            seed_only_in_sleep = True

        event_dream_replay = EventDreamReplay()

    cfg = Cfg()
    device = torch.device("cpu")
    event_latent_memory = None
    global_step = 1

    def is_full_sleep_mode(self):
        return True

    ensure_event_dream_replay_ready = EventDreamReplayRuntimeMixin.ensure_event_dream_replay_ready
    compute_event_dream_replay = EventDreamReplayRuntimeMixin.compute_event_dream_replay
    get_event_dream_focus_seed = EventDreamReplayRuntimeMixin.get_event_dream_focus_seed
    get_m5_focus_seed = EventDreamReplayRuntimeMixin.get_m5_focus_seed
    _store_event_dream_m5_seed = EventDreamReplayRuntimeMixin._store_event_dream_m5_seed
    _event_dream_stage_allowed = EventDreamReplayRuntimeMixin._event_dream_stage_allowed


def test_m02_runtime_seed_bus_contract():
    runtime = DummyEventDreamRuntime()
    out = make_unconscious_loop_out()

    packet = runtime.compute_event_dream_replay({}, out)
    assert packet is not None
    assert "event_dream_replay" in out

    seed, gate = runtime.get_m5_focus_seed(stage="pre_observe")
    assert_tensor("M2 seed", seed, (1, 256))
    assert_tensor("M2 seed gate", gate)

    # main stage is disabled by default in this contract.
    seed2, gate2 = runtime.get_m5_focus_seed(stage="main")
    assert seed2 is None and gate2 is None


def test_m5_latent_prototype_seed_has_priority_over_m2_seed():
    runtime = DummyEventDreamRuntime()
    out = make_unconscious_loop_out()
    runtime.compute_event_dream_replay({}, out)

    prototype_seed = torch.ones(1, 256) * 0.25
    prototype_gate = torch.tensor([[0.9]], dtype=torch.float32)

    def prototype_focus_seed(stage="model_step"):
        assert stage == "pre_observe"
        return prototype_seed, prototype_gate

    runtime.get_m5_latent_prototype_focus_seed = prototype_focus_seed

    seed, gate = runtime.get_m5_focus_seed(stage="pre_observe")
    assert seed is prototype_seed
    assert gate is prototype_gate
