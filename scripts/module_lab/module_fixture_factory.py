from __future__ import annotations

"""
Shared tensor fixtures for DMoC module debug labs.

The goal is to debug modules in isolation without MuJoCo:
    fake inputs -> module -> output contract -> behavior checks.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch


DEFAULT_DEVICE = torch.device("cpu")


def seed_all(seed: int = 7) -> None:
    torch.manual_seed(int(seed))


def finite_tensor(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and bool(torch.isfinite(x).all().item())


def assert_tensor(name: str, value: Any, shape: Optional[tuple[int, ...]] = None) -> None:
    assert torch.is_tensor(value), f"{name} must be a torch.Tensor, got {type(value)!r}"
    assert finite_tensor(value), f"{name} contains NaN/Inf"
    if shape is not None:
        assert tuple(value.shape) == tuple(shape), f"{name} shape {tuple(value.shape)} != {shape}"


def assert_gate(name: str, value: Any, min_value: float = 0.0, max_value: float = 1.0) -> None:
    assert_tensor(name, value)
    vmin = float(value.detach().float().min().cpu().item())
    vmax = float(value.detach().float().max().cpu().item())
    assert vmin >= min_value - 1e-6, f"{name} min {vmin} < {min_value}"
    assert vmax <= max_value + 1e-6, f"{name} max {vmax} > {max_value}"


def randn(batch: int = 1, dim: int = 256, scale: float = 0.25, device: torch.device | str = DEFAULT_DEVICE) -> torch.Tensor:
    return torch.randn(batch, dim, device=device) * float(scale)


def scalar(value: float, device: torch.device | str = DEFAULT_DEVICE) -> torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32, device=device)


def make_fake_obs(
    *,
    batch: int = 1,
    image_h: int = 32,
    image_w: int = 48,
    tactile_dim: int = 16,
    body_state_dim: int = 32,
    hand_motor_dim: int = 8,
    embodied_dim: int = 15,
    device: torch.device | str = DEFAULT_DEVICE,
) -> Dict[str, torch.Tensor]:
    seed_all()
    return {
        "left": torch.rand(batch, 3, image_h, image_w, device=device),
        "right": torch.rand(batch, 3, image_h, image_w, device=device),
        "depth": torch.rand(batch, 1, image_h, image_w, device=device),
        "pose": randn(batch, 7, device=device),
        "body_state": randn(batch, body_state_dim, device=device),
        "tactile": randn(batch, tactile_dim, scale=0.10, device=device),
        "hand_motor": randn(batch, hand_motor_dim, scale=0.10, device=device),
        "embodied_action": randn(batch, embodied_dim, scale=0.10, device=device),
        "object_state": randn(batch, 9, scale=0.10, device=device),
        "reward": scalar(0.0, device=device),
        "done": scalar(0.0, device=device),
    }


def make_fake_m5_out(
    *,
    batch: int = 1,
    focus_dim: int = 256,
    workspace_dim: int = 256,
    object_dim: int = 128,
    thought_dim: int = 192,
    memory_dim: int = 256,
    coherence: float = 0.55,
    curiosity: float = 0.45,
    device: torch.device | str = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    seed_all()
    focus_context = randn(batch, focus_dim, device=device)
    workspace_out = randn(batch, workspace_dim, device=device)
    object_repr = randn(batch, object_dim, device=device)
    thought_candidate = randn(batch, thought_dim, device=device)
    memory_context = randn(batch, memory_dim, device=device)

    return {
        "focus_context": focus_context,
        "workspace_out": workspace_out,
        "object_repr": object_repr,
        "object_imagery": {
            "object_confidence": scalar(0.62, device=device),
        },
        "preconscious_thoughts": {
            "thought_candidate": thought_candidate,
            "workspace_seed": randn(batch, thought_dim, device=device),
            "candidate_delta": randn(batch, thought_dim, scale=0.05, device=device),
        },
        "preconscious_reflection_out": {
            "reflection": randn(batch, 128, device=device),
            "model_confidence": scalar(0.58, device=device),
        },
        "memory": {
            "memory_context": memory_context,
        },
        "values": {
            "coherence": scalar(coherence, device=device),
            "curiosity": scalar(curiosity, device=device),
            "value_latent": randn(batch, 64, device=device),
        },
        "focus": {
            "focus_idx": torch.zeros(batch, dtype=torch.long, device=device),
        },
        "action_logits": randn(batch, 8, device=device),
        "action_ids": torch.zeros(batch, dtype=torch.long, device=device),
        "embodied_targets": randn(batch, 15, scale=0.05, device=device),
        "hand_ctrl": randn(batch, 8, scale=0.05, device=device),
        "imagined": {
            "imagined_value": scalar(0.15, device=device),
            "imagined_touch": randn(batch, 16, scale=0.05, device=device),
        },
    }


def make_fake_affect(
    *,
    panic: float = 0.30,
    stress: float = 0.25,
    curiosity: float = 0.70,
    expected_delta: float = 0.10,
    device: torch.device | str = DEFAULT_DEVICE,
) -> Dict[str, torch.Tensor]:
    return {
        "affect_latents": torch.tensor(
            [[0.10, 0.50, 0.05, stress, 0.10, panic, 0.35, 0.25, curiosity, 0.40, 0.55, expected_delta]],
            dtype=torch.float32,
            device=device,
        ),
        "valence": scalar(0.10, device=device),
        "arousal": scalar(0.50, device=device),
        "pain_latent": scalar(0.05, device=device),
        "panic_latent": scalar(panic, device=device),
        "stress_latent": scalar(stress, device=device),
        "fear_latent": scalar(0.10, device=device),
        "curiosity_latent": scalar(curiosity, device=device),
        "discovery_latent": scalar(0.40, device=device),
        "coherence_latent": scalar(0.55, device=device),
        "comfort_latent": scalar(0.35, device=device),
        "relief_latent": scalar(0.25, device=device),
        "expected_affect_delta": scalar(expected_delta, device=device),
        "intrinsic_reward": scalar(0.02, device=device),
    }


def make_fake_m13_memory(
    *,
    batch: int = 1,
    dim: int = 256,
    relevance: float = 0.55,
    device: torch.device | str = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    return {
        "retrieved_context": randn(batch, dim, device=device),
        "retrieval_relevance": scalar(relevance, device=device),
        "retrieved_episode_count": scalar(3.0, device=device),
        "summary": "fake autobiographical episode",
    }


def make_fake_m4_identity(
    *,
    batch: int = 1,
    dim: int = 256,
    stability: float = 0.70,
    novelty: float = 0.20,
    gate: float = 0.80,
    device: torch.device | str = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    return {
        "dynamic_identity_context": randn(batch, dim, device=device),
        "dynamic_memory_gate": scalar(gate, device=device),
        "identity_token": "obj:test",
        "identity_stability": scalar(stability, device=device),
        "identity_similarity": scalar(1.0 - novelty, device=device),
        "identity_novelty": scalar(novelty, device=device),
        "passport_slot": scalar(1.0, device=device),
        "selected_sentence": "same object identity",
    }


def make_fake_inner_object(
    *,
    batch: int = 1,
    obj_dim: int = 128,
    device: torch.device | str = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    return {
        "z_obj": randn(batch, obj_dim, device=device),
        "slot_token": "slot:test",
        "object_confidence": scalar(0.65, device=device),
    }


class FakePassportManager:
    """Minimal fake for LongDynamicMemory tests."""

    def __init__(self, context_dim: int = 256, device: torch.device | str = DEFAULT_DEVICE) -> None:
        self.context_dim = int(context_dim)
        self.device = torch.device(device)
        self.passports = {
            "obj:test": {
                "token": "obj:test",
                "replay_z": randn(1, self.context_dim, device=self.device),
                "confidence_ema": 0.72,
                "last_similarity": 0.82,
                "dynamic_score_ema": 0.35,
                "lives_in_slot": 1,
                "last_sentence": "same object identity",
                "episode_summary": "fake passport episode",
            }
        }

    def observe(self, obj: Dict[str, Any], *, event_memory=None, dream_mode: bool = False, global_step: int = 0) -> Dict[str, Any]:
        del obj, event_memory, dream_mode, global_step
        return {
            "passport_token": "obj:test",
            "passport_confidence_ema": 0.72,
            "passport_similarity": 0.82,
            "passport_dynamic_score": 0.35,
            "passport_count": float(len(self.passports)),
            "passport_slot": 1.0,
            "passport_created": False,
            "passport_sentence": "same object identity",
            "passport_episode_summary": "fake passport episode",
            "passport_source": "fake_passport",
        }

    def select_for_replay(self, token: str = "") -> Dict[str, Any]:
        return self.passports.get(token or "obj:test", self.passports["obj:test"])


def make_fake_event_memory() -> SimpleNamespace:
    events = [
        {
            "sentence": "contact with object changed prediction",
            "kind": "contact_prediction_delta",
            "delta_norm": 0.75,
            "contact_norm": 0.55,
            "action_norm": 0.25,
            "touch_strength": 0.50,
            "event_code": torch.tensor([[0.75, 0.55, 0.25, 0.50, 0.10, 0.20, 0.30, 0.40]], dtype=torch.float32),
        }
    ]
    return SimpleNamespace(events=events, last_event=events[-1])


def make_unconscious_loop_out(device: torch.device | str = DEFAULT_DEVICE) -> Dict[str, Any]:
    out = make_fake_m5_out(device=device)
    out["affect"] = make_fake_affect(device=device)
    out["autobiographical_memory"] = make_fake_m13_memory(device=device)
    out["long_dynamic_memory"] = make_fake_m4_identity(device=device)
    return out
