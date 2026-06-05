from __future__ import annotations

import torch


def test_sleep_motor_guard_blocks_executable_motor_outputs():
    from src.modules.m03_self_action_causality.sleep_motor_guard import block_motor_outputs_for_sleep

    out = {
        "embodied_targets": torch.ones(1, 15),
        "hand_ctrl": torch.ones(1, 8) * 0.5,
        "leg_ctrl": torch.ones(1, 18) * 0.25,
        "action_ids": torch.tensor([2]),
    }

    blocked = block_motor_outputs_for_sleep(out, sleep_mode=True, stage="test")

    assert torch.allclose(blocked["embodied_targets"], torch.zeros_like(blocked["embodied_targets"]))
    assert torch.allclose(blocked["hand_ctrl"], torch.zeros_like(blocked["hand_ctrl"]))
    assert torch.allclose(blocked["leg_ctrl"], torch.zeros_like(blocked["leg_ctrl"]))
    assert "imagined_embodied_targets" in blocked
    assert "imagined_hand_ctrl" in blocked
    assert "imagined_leg_ctrl" in blocked
    assert "imagined_action_ids" in blocked
    assert blocked["sleep_motor_guard"]["blocked"] is True
    assert blocked["sleep_motor_guard"]["blocked_motor_norm"] > 0.0


def test_sleep_motor_guard_keeps_awake_outputs():
    from src.modules.m03_self_action_causality.sleep_motor_guard import block_motor_outputs_for_sleep

    embodied = torch.ones(1, 15)
    hand = torch.ones(1, 8) * 0.5
    out = {
        "embodied_targets": embodied.clone(),
        "hand_ctrl": hand.clone(),
    }

    awake = block_motor_outputs_for_sleep(out, sleep_mode=False, stage="test")

    assert torch.allclose(awake["embodied_targets"], embodied)
    assert torch.allclose(awake["hand_ctrl"], hand)
    assert awake["sleep_motor_guard"]["blocked"] is False
