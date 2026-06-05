from __future__ import annotations

"""
Behavioral scenarios for the unconscious sleep/replay loop.

Run:
    python scripts/module_lab/scenario_unconscious_replay.py --json
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
    make_fake_event_memory,
    make_fake_inner_object,
    make_fake_m13_memory,
    make_fake_m4_identity,
    make_fake_m5_out,
    scalar,
)


def f(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().reshape(-1)[0].cpu().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def compute_m2_packet(
    *,
    panic: float,
    stress: float,
    curiosity: float,
    m13_relevance: float,
    m4_gate: float,
    dream_mode: bool = True,
    include_event_memory: bool = True,
) -> Dict[str, Any]:
    from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig

    out = make_fake_m5_out(curiosity=curiosity)
    out["affect"] = {
        "panic_latent": scalar(panic),
        "stress_latent": scalar(stress),
        "curiosity_latent": scalar(curiosity),
        "expected_affect_delta": scalar(0.10),
    }
    out["autobiographical_memory"] = make_fake_m13_memory(relevance=m13_relevance)
    out["long_dynamic_memory"] = make_fake_m4_identity(gate=m4_gate)

    m2 = EventDreamReplay(EventDreamReplayConfig(
        replay_context_dim=256,
        event_code_dim=8,
        blend_replay_into_focus=False,
        use_m13_context=True,
        use_m4_context=True,
        seed_to_m5_boundary=True,
    ))
    event_memory = make_fake_event_memory() if include_event_memory else None
    return m2.compute(out=out, event_memory=event_memory, dream_mode=dream_mode)


def scenario_calm_no_replay() -> Dict[str, Any]:
    packet = compute_m2_packet(
        panic=0.0,
        stress=0.0,
        curiosity=0.0,
        m13_relevance=0.0,
        m4_gate=0.0,
        dream_mode=False,
        include_event_memory=False,
    )
    return {
        "name": "calm_no_replay",
        "dream_pressure": f(packet["dream_pressure"]),
        "event_salience": f(packet["event_salience"]),
        "should_replay": f(packet["should_replay"]),
        "expect": "low replay pressure",
        "pass": f(packet["dream_pressure"]) < 0.50,
    }


def scenario_curiosity_replay() -> Dict[str, Any]:
    packet = compute_m2_packet(
        panic=0.1,
        stress=0.2,
        curiosity=0.9,
        m13_relevance=0.7,
        m4_gate=0.8,
        dream_mode=True,
    )
    return {
        "name": "curiosity_replay",
        "dream_pressure": f(packet["dream_pressure"]),
        "event_salience": f(packet["event_salience"]),
        "should_replay": f(packet["should_replay"]),
        "identity": packet.get("selected_identity_token", ""),
        "expect": "high curiosity + M13/M4 context activates replay",
        "pass": f(packet["dream_pressure"]) >= 0.35 and f(packet["event_salience"]) >= 0.25,
    }


def scenario_bad_prediction_dream() -> Dict[str, Any]:
    packet = compute_m2_packet(
        panic=0.9,
        stress=0.9,
        curiosity=0.3,
        m13_relevance=0.5,
        m4_gate=0.6,
        dream_mode=True,
    )
    return {
        "name": "bad_prediction_dream",
        "dream_pressure": f(packet["dream_pressure"]),
        "event_salience": f(packet["event_salience"]),
        "should_replay": f(packet["should_replay"]),
        "expect": "panic/stress increases dream pressure",
        "pass": f(packet["dream_pressure"]) >= 0.50,
    }


def scenario_m4_identity_context() -> Dict[str, Any]:
    from src.modules.m04_long_dynamic_memory.long_dynamic_memory import LongDynamicMemory, LongDynamicMemoryConfig

    out = make_fake_m5_out()
    obj = make_fake_inner_object()
    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=256))
    packet = m4.compute(
        out=out,
        obj=obj,
        passport_manager=FakePassportManager(context_dim=256),
        event_memory=None,
        dream_mode=True,
        global_step=1,
    )
    return {
        "name": "object_identity_replay",
        "identity_token": packet.get("identity_token", ""),
        "dynamic_memory_gate": f(packet["dynamic_memory_gate"]),
        "identity_stability": f(packet["identity_stability"]),
        "identity_novelty": f(packet["identity_novelty"]),
        "expect": "stable object identity is available for M2",
        "pass": bool(packet.get("identity_token")) and f(packet["dynamic_memory_gate"]) > 0.0,
    }


SCENARIOS = [
    scenario_calm_no_replay,
    scenario_curiosity_replay,
    scenario_bad_prediction_dream,
    scenario_m4_identity_context,
]


def run_all() -> Dict[str, Any]:
    results = [fn() for fn in SCENARIOS]
    return {
        "status": "ok" if all(r["pass"] for r in results) else "fail",
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
