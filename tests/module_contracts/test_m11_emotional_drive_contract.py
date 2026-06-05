from __future__ import annotations

import torch

from scripts.module_lab.module_fixture_factory import assert_tensor, make_fake_m5_out


def test_m11_emotional_drive_affect_contract():
    from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive, EmotionalDriveConfig

    m11 = EmotionalDrive(EmotionalDriveConfig())
    out = make_fake_m5_out(coherence=0.75, curiosity=0.60)
    obs = {"tactile": torch.ones(1, 16) * 0.05}

    emotion = m11.compute(out, obs)
    assert "affect" in emotion
    affect = emotion["affect"]

    required = (
        "affect_latents",
        "valence",
        "arousal",
        "pain_latent",
        "stress_latent",
        "fear_latent",
        "panic_latent",
        "comfort_latent",
        "relief_latent",
        "curiosity_latent",
        "discovery_latent",
        "coherence_latent",
        "expected_affect_delta",
        "intrinsic_reward",
    )
    for key in required:
        assert key in affect, f"missing affect.{key}"
        assert_tensor(f"affect.{key}", affect[key])
