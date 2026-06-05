from __future__ import annotations

"""
Run DMoC module lab without MuJoCo.

Examples:
    python scripts/module_lab/run_module_lab.py --module all
    python scripts/module_lab/run_module_lab.py --module m02
    python scripts/module_lab/run_module_lab.py --module loop
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.module_lab.module_fixture_factory import (
    FakePassportManager,
    assert_gate,
    assert_tensor,
    make_fake_affect,
    make_fake_event_memory,
    make_fake_inner_object,
    make_fake_m13_memory,
    make_fake_m4_identity,
    make_fake_m5_out,
    make_unconscious_loop_out,
    scalar,
)


def f(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().reshape(-1)[0].cpu().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def lab_m11() -> Dict[str, Any]:
    from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive, EmotionalDriveConfig

    out = make_fake_m5_out(coherence=0.75, curiosity=0.55)
    obs = {"tactile": torch.ones(1, 16) * 0.05}
    m11 = EmotionalDrive(EmotionalDriveConfig())
    emotion = m11.compute(out, obs)

    assert "affect" in emotion
    affect = emotion["affect"]
    for key in (
        "affect_latents", "valence", "arousal", "pain_latent", "stress_latent",
        "fear_latent", "panic_latent", "comfort_latent", "relief_latent",
        "curiosity_latent", "discovery_latent", "coherence_latent",
        "expected_affect_delta", "intrinsic_reward",
    ):
        assert key in affect, f"M11 affect missing {key}"
        assert_tensor(f"M11 affect.{key}", affect[key])

    return {
        "module": "M11",
        "valence": emotion["emotional_valence"],
        "arousal": emotion["emotional_arousal"],
        "stress": f(affect["stress_latent"]),
        "panic": f(affect["panic_latent"]),
        "curiosity": f(affect["curiosity_latent"]),
        "status": "ok",
    }


def lab_m13() -> Dict[str, Any]:
    from src.modules.m13_autobiographical_memory.autobiographical_memory import AutobiographicalMemory, AutobiographicalMemoryConfig

    out = make_fake_m5_out()
    out["affect"] = make_fake_affect()

    m13 = AutobiographicalMemory(AutobiographicalMemoryConfig(memory_dim=256, max_episodes=16, retrieval_topk=1))
    empty = m13.retrieve(out)
    assert_tensor("M13 empty.retrieved_context", empty["retrieved_context"], (1, 256))

    m13.write_episode(obs={}, out=out, global_step=1)
    got = m13.retrieve(out)
    assert_tensor("M13 retrieved_context", got["retrieved_context"], (1, 256))
    assert_gate("M13 retrieval_relevance", got["retrieval_relevance"], -1.0, 1.0)
    assert_tensor("M13 retrieved_episode_count", got["retrieved_episode_count"])
    assert "summary" in got

    return {
        "module": "M13",
        "episode_count": f(got["retrieved_episode_count"]),
        "retrieval_relevance": f(got["retrieval_relevance"]),
        "summary": got.get("summary", ""),
        "status": "ok",
    }


def lab_m4() -> Dict[str, Any]:
    from src.modules.m04_long_dynamic_memory.long_dynamic_memory import LongDynamicMemory, LongDynamicMemoryConfig

    out = make_fake_m5_out()
    obj = make_fake_inner_object()
    passport = FakePassportManager(context_dim=256)
    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=256))

    packet = m4.compute(out=out, obj=obj, passport_manager=passport, event_memory=None, dream_mode=True, global_step=1)
    assert_tensor("M4 dynamic_identity_context", packet["dynamic_identity_context"], (1, 256))
    assert_gate("M4 dynamic_memory_gate", packet["dynamic_memory_gate"], 0.0, 1.0)
    for key in ("identity_token", "identity_stability", "identity_novelty", "passport_slot", "selected_sentence"):
        assert key in packet, f"M4 missing {key}"
    assert_gate("M4 identity_stability", packet["identity_stability"], 0.0, 1.0)
    assert_gate("M4 identity_novelty", packet["identity_novelty"], 0.0, 1.0)

    return {
        "module": "M4",
        "identity_token": packet.get("identity_token"),
        "dynamic_memory_gate": f(packet["dynamic_memory_gate"]),
        "identity_stability": f(packet["identity_stability"]),
        "identity_novelty": f(packet["identity_novelty"]),
        "status": "ok",
    }


def lab_m02() -> Dict[str, Any]:
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

    assert_tensor("M2 replay_context", packet["replay_context"], (1, 256))
    assert_gate("M2 replay_gate", packet["replay_gate"], 0.0, 1.0)
    assert_gate("M2 dream_pressure", packet["dream_pressure"], 0.0, 1.0)
    for key in ("event_salience", "should_replay", "replay_source", "selected_identity_token", "dynamic_memory_gate", "selected_episode_summary"):
        assert key in packet, f"M2 missing {key}"
    assert_gate("M2 event_salience", packet["event_salience"], 0.0, 1.0)
    assert_gate("M2 should_replay", packet["should_replay"], 0.0, 1.0)

    return {
        "module": "M2",
        "replay_gate": f(packet["replay_gate"]),
        "dream_pressure": f(packet["dream_pressure"]),
        "event_salience": f(packet["event_salience"]),
        "should_replay": f(packet["should_replay"]),
        "source": packet.get("replay_source", ""),
        "identity": packet.get("selected_identity_token", ""),
        "status": "ok",
    }


def lab_m05_boundary() -> Dict[str, Any]:
    from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary

    b = 1
    boundary = FocusFeedbackBoundary(focus_context_dim=256, workspace_seed_dim=256, thought_dim=192)
    workspace_seed = torch.randn(b, 256)
    focus_seed = torch.randn(b, 256)
    gate = torch.tensor([[0.1]])

    packet = boundary(
        workspace_seed=workspace_seed,
        focus_context_seed=focus_seed,
        focus_context_seed_gate=gate,
    )

    for key in ("active", "workspace_seed", "external_gate", "learned_gate", "total_gate", "workspace_delta", "preconscious_delta", "seed_norm"):
        assert key in packet, f"M5Boundary missing {key}"
    assert_tensor("M5Boundary workspace_seed", packet["workspace_seed"], (b, 256))
    assert_tensor("M5Boundary preconscious_delta", packet["preconscious_delta"], (b, 192))
    assert_gate("M5Boundary total_gate", packet["total_gate"], 0.0, 0.35)

    pre = torch.randn(b, 192)
    pre2 = boundary.apply_preconscious_seed(pre, packet)
    assert_tensor("M5Boundary applied_preconscious", pre2, (b, 192))

    changed = float((packet["workspace_seed"] - workspace_seed).abs().mean().detach().cpu().item())

    return {
        "module": "M5 FocusFeedbackBoundary",
        "total_gate": f(packet["total_gate"]),
        "learned_gate": f(packet["learned_gate"]),
        "workspace_delta_mean_abs": changed,
        "status": "ok",
    }


def lab_loop() -> Dict[str, Any]:
    """Contract test for M11 + M13 + M4 + M2 + M5 boundary."""
    m11 = lab_m11()
    m13 = lab_m13()
    m4 = lab_m4()
    m2 = lab_m02()
    m5b = lab_m05_boundary()

    assert m2["replay_gate"] >= 0.0
    assert m5b["total_gate"] >= 0.0

    return {
        "module": "unconscious_loop",
        "chain": "M1 -> M5 -> M11 -> M2 -> FocusFeedbackBoundary -> M5; M4/M13 -> M2; M5 -> M3",
        "m11": m11,
        "m13": m13,
        "m4": m4,
        "m2": m2,
        "m5_boundary": m5b,
        "status": "ok",
    }


LABS = {
    "m11": lab_m11,
    "m13": lab_m13,
    "m4": lab_m4,
    "m02": lab_m02,
    "m2": lab_m02,
    "m05": lab_m05_boundary,
    "m5": lab_m05_boundary,
    "boundary": lab_m05_boundary,
    "loop": lab_loop,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="all", choices=["all", *LABS.keys()])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.module == "all":
        result = {name: func() for name, func in LABS.items() if name in ("m11", "m13", "m4", "m02", "m05", "loop")}
    else:
        result = LABS[args.module]()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
