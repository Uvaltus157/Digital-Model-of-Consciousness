from __future__ import annotations

import torch

from scripts.module_lab.module_fixture_factory import (
    FakePassportManager,
    assert_gate,
    assert_tensor,
    make_fake_event_memory,
    make_fake_inner_object,
    make_fake_m5_out,
)


def test_unconscious_loop_contract_m11_m13_m4_m2_m5_boundary():
    from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig
    from src.modules.m04_long_dynamic_memory.long_dynamic_memory import LongDynamicMemory, LongDynamicMemoryConfig
    from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary
    from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive, EmotionalDriveConfig
    from src.modules.m13_autobiographical_memory.autobiographical_memory import AutobiographicalMemory, AutobiographicalMemoryConfig

    # Fake M5 output: M1 must not be fed to M2 directly in this contract.
    out = make_fake_m5_out()
    obs = {"tactile": torch.ones(1, 16) * 0.05}
    assert_tensor("M5 embodied_targets for M3", out["embodied_targets"], (1, 15))
    assert_tensor("M5 hand_ctrl for M3", out["hand_ctrl"], (1, 8))

    # M11 evaluates M5 output.
    m11 = EmotionalDrive(EmotionalDriveConfig())
    emotion = m11.compute(out, obs)
    out["emotion"] = emotion
    out["affect"] = emotion["affect"]

    # M13 retrieves episodic context.
    m13 = AutobiographicalMemory(AutobiographicalMemoryConfig(memory_dim=256, max_episodes=16, retrieval_topk=1))
    m13.write_episode(obs=obs, out=out, global_step=1)
    out["autobiographical_memory"] = m13.retrieve(out)

    # M4 derives object identity from inner_object/z_obj, not raw M1.
    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=256))
    inner_object = make_fake_inner_object()
    out["inner_object"] = inner_object
    out["long_dynamic_memory"] = m4.compute(
        out=out,
        obj=inner_object,
        passport_manager=FakePassportManager(context_dim=256),
        event_memory=None,
        dream_mode=True,
        global_step=1,
    )

    # M2 selects replay context from M11 + M13 + M4 + current M5 focus.
    m2 = EventDreamReplay(EventDreamReplayConfig(
        replay_context_dim=256,
        event_code_dim=8,
        blend_replay_into_focus=False,
        use_m13_context=True,
        use_m4_context=True,
        seed_to_m5_boundary=True,
    ))
    packet = m2.compute(out=out, event_memory=make_fake_event_memory(), dream_mode=True)

    assert_tensor("M2 replay_context", packet["replay_context"], (1, 256))
    assert_gate("M2 replay_gate", packet["replay_gate"], 0.0, 1.0)
    assert "selected_identity_token" in packet
    assert "selected_episode_summary" in packet
    assert "embodied_targets" not in packet
    assert "hand_ctrl" not in packet

    # M2 replay_context enters M5 only through FocusFeedbackBoundary.
    boundary = FocusFeedbackBoundary(focus_context_dim=256, workspace_seed_dim=256, thought_dim=192)
    workspace_seed = torch.randn(1, 256)
    boundary_packet = boundary(
        workspace_seed=workspace_seed,
        focus_context_seed=packet["replay_context"],
        focus_context_seed_gate=packet["replay_gate"],
    )

    assert_tensor("boundary workspace_seed", boundary_packet["workspace_seed"], (1, 256))
    assert_tensor("boundary preconscious_delta", boundary_packet["preconscious_delta"], (1, 192))
    assert_gate("boundary total_gate", boundary_packet["total_gate"], 0.0, 0.35)


def test_forbidden_direct_m1_to_m2_is_not_used_in_contract():
    # The test intentionally builds M2 input from M5 out + M11/M13/M4,
    # not from raw obs/M1. This is the architectural invariant.
    from scripts.module_lab.module_fixture_factory import make_unconscious_loop_out
    out = make_unconscious_loop_out()
    assert "left" not in out
    assert "right" not in out
    assert "depth" not in out
