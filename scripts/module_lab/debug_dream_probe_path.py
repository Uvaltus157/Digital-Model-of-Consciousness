from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.module_lab.module_fixture_factory import make_fake_m5_out, make_fake_obs
from src.modules.m02_event_dream_replay.dream_probe_runtime import DreamProbeRuntimeMixin
from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().reshape(-1)[0].cpu().item())


class DummySystem(DreamProbeRuntimeMixin):
    def __init__(self) -> None:
        self.global_step = 1
        self.latest_out = make_fake_m5_out()


def affect_for(out: dict, obs: dict) -> dict:
    return EmotionalDrive().compute(out, obs)["affect"]


def main() -> None:
    obs = make_fake_obs()

    curiosity_before = affect_for(make_fake_m5_out(curiosity=0.1), obs)
    curiosity_system = DummySystem()
    curiosity_out = make_fake_m5_out(curiosity=0.1)
    curiosity_system.request_dream_probe({"kind": "curiosity", "intensity": 0.85, "duration": 80})
    curiosity_system.apply_dream_probe_to_out(curiosity_out, obs)
    curiosity_after = affect_for(curiosity_out, obs)

    stress_before = affect_for(make_fake_m5_out(coherence=0.95), obs)
    stress_system = DummySystem()
    stress_out = make_fake_m5_out(coherence=0.95)
    stress_system.request_dream_probe({"kind": "stress", "intensity": 0.85, "duration": 80})
    stress_system.apply_dream_probe_to_out(stress_out, obs)
    stress_after = affect_for(stress_out, obs)

    seed_system = DummySystem()
    seed_system.request_dream_probe({"kind": "replay_seed", "intensity": 0.75, "duration": 60})
    seed_gate = scalar(seed_system._event_dream_next_focus_gate)
    seed_norm = float(seed_system._event_dream_next_focus_seed.norm().item())

    result = {
        "curiosity_before": scalar(curiosity_before["curiosity_latent"]),
        "curiosity_after": scalar(curiosity_after["curiosity_latent"]),
        "stress_before": scalar(stress_before["stress_latent"]),
        "stress_after": scalar(stress_after["stress_latent"]),
        "panic_before": scalar(stress_before["panic_latent"]),
        "panic_after": scalar(stress_after["panic_latent"]),
        "replay_seed_gate": seed_gate,
        "replay_seed_norm": seed_norm,
    }
    assert result["curiosity_after"] > result["curiosity_before"], result
    assert result["stress_after"] > result["stress_before"] or result["panic_after"] > result["panic_before"], result
    assert result["replay_seed_gate"] > 0.0 and result["replay_seed_norm"] > 0.0, result
    print(result)


if __name__ == "__main__":
    main()
