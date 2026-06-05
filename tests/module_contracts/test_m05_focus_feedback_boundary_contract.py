from __future__ import annotations

import torch

from scripts.module_lab.module_fixture_factory import assert_gate, assert_tensor


def test_m05_focus_feedback_boundary_contract():
    from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary

    boundary = FocusFeedbackBoundary(focus_context_dim=256, workspace_seed_dim=256, thought_dim=192)

    workspace_seed = torch.randn(1, 256)
    focus_seed = torch.randn(1, 256)
    gate = torch.tensor([[0.2]])

    packet = boundary(
        workspace_seed=workspace_seed,
        focus_context_seed=focus_seed,
        focus_context_seed_gate=gate,
    )

    for key in ("active", "workspace_seed", "external_gate", "learned_gate", "total_gate", "workspace_delta", "preconscious_delta", "seed_norm"):
        assert key in packet, f"missing {key}"

    assert_tensor("workspace_seed", packet["workspace_seed"], (1, 256))
    assert_tensor("preconscious_delta", packet["preconscious_delta"], (1, 192))
    assert_gate("total_gate", packet["total_gate"], 0.0, 0.35)

    pre = torch.randn(1, 192)
    pre2 = boundary.apply_preconscious_seed(pre, packet)
    assert_tensor("pre2", pre2, (1, 192))


def test_m05_focus_feedback_boundary_zero_gate_no_runaway():
    from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary

    boundary = FocusFeedbackBoundary(focus_context_dim=256, workspace_seed_dim=256, thought_dim=192)

    workspace_seed = torch.randn(1, 256)
    packet = boundary(
        workspace_seed=workspace_seed,
        focus_context_seed=None,
        focus_context_seed_gate=None,
    )

    assert torch.allclose(packet["workspace_seed"], workspace_seed)
    assert float(packet["total_gate"].detach().cpu().item()) == 0.0
