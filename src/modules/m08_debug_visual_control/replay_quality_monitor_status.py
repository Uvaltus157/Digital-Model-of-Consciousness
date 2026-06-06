from __future__ import annotations

"""Replay Quality Monitor status payload.

This is a read-only diagnostic layer. It estimates whether sleep/replay is
processing useful experience by watching selected M2/M13/M4 items and affect
deltas over live runtime steps.
"""

from typing import Any, Dict

import torch


def _dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _scalar(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return round(float(x.detach().float().reshape(-1)[0].cpu().item()), 6)
        if x is None:
            return float(default)
        return round(float(x), 6)
    except Exception:
        return float(default)


def _norm(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return round(float(x.detach().float().norm().cpu().item()), 6)
        return float(default)
    except Exception:
        return float(default)


def _text(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _trend(delta: float, eps: float = 0.002) -> str:
    if delta > eps:
        return "↑"
    if delta < -eps:
        return "↓"
    return "→"


def _seed_gate_and_norm(system: Any, m2: Dict[str, Any]) -> tuple[float, float]:
    seed = getattr(system, "_event_dream_next_focus_seed", None)
    if seed is None:
        seed = m2.get("next_focus_context_seed", m2.get("replay_context"))
    gate = getattr(system, "_event_dream_next_focus_gate", None)
    if gate is None:
        gate = m2.get("next_focus_context_seed_gate", m2.get("replay_gate"))
    last_probe_seed = _dict(getattr(system, "_dream_probe_last_seed", {}))
    return _scalar(gate, 0.0), _norm(seed, 0.0)


def build_replay_quality_monitor_status(system: Any) -> Dict[str, Any]:
    out = _dict(getattr(system, "latest_out", {}) or {})
    affect = _dict(out.get("affect"))
    emotion = _dict(out.get("emotion"))
    m13 = _dict(out.get("autobiographical_memory"))
    m4 = _dict(out.get("long_dynamic_memory"))
    m2 = _dict(out.get("event_dream_replay"))
    dream_probe = _dict(out.get("dream_probe", getattr(system, "_dream_probe_state", {}) or {}) or {})

    step = int(getattr(system, "global_step", 0))
    full_sleep = bool(system.is_full_sleep_mode()) if hasattr(system, "is_full_sleep_mode") else False
    sensor_state = system.sensor_state_label() if hasattr(system, "sensor_state_label") else "unknown"

    dream_pressure = _scalar(m2.get("dream_pressure"), 0.0)
    replay_gate = _scalar(m2.get("replay_gate"), 0.0)
    event_salience = _scalar(m2.get("event_salience"), 0.0)
    should_replay = _scalar(m2.get("should_replay"), 0.0)
    stress = _scalar(affect.get("stress_latent"), 0.0)
    panic = _scalar(affect.get("panic_latent"), 0.0)
    relief = _scalar(affect.get("relief_latent"), 0.0)
    curiosity = _scalar(affect.get("curiosity_latent"), _scalar(emotion.get("curiosity"), 0.0))
    valence = _scalar(affect.get("valence"), _scalar(emotion.get("emotional_valence"), 0.0))
    coherence = _scalar(affect.get("coherence_latent"), _scalar(out.get("values", {}).get("coherence") if isinstance(out.get("values"), dict) else None, 0.0))
    expected_affect_delta = _scalar(affect.get("expected_affect_delta"), 0.0)

    seed_gate, seed_norm = _seed_gate_and_norm(system, m2)

    selected_summary = _text(
        m2.get("selected_episode_summary", m13.get("summary", m13.get("last_summary", "")))
    )
    selected_identity = _text(
        m2.get("selected_identity_token", m4.get("identity_token", ""))
    )
    selected_sentence = _text(
        m2.get("selected_identity_sentence", m4.get("selected_sentence", ""))
    )
    replay_source = _text(m2.get("replay_source", ""))

    state = getattr(system, "_replay_quality_monitor_state", None)
    if not isinstance(state, dict):
        state = {
            "last_step": None,
            "prev": {},
            "last_delta": {},
            "samples": 0,
            "quality_ema": 0.0,
            "history": [],
        }
        setattr(system, "_replay_quality_monitor_state", state)

    current = {
        "dream_pressure": dream_pressure,
        "stress": stress,
        "panic": panic,
        "relief": relief,
        "curiosity": curiosity,
        "valence": valence,
        "coherence": coherence,
        "replay_gate": replay_gate,
        "event_salience": event_salience,
        "seed_norm": seed_norm,
    }

    last_step = state.get("last_step")
    if last_step == step:
        delta = dict(state.get("last_delta", {}) or {})
    else:
        prev = state.get("prev", {}) if isinstance(state.get("prev"), dict) else {}
        delta = {k: float(current[k] - float(prev.get(k, current[k]))) for k in current}
        state["prev"] = dict(current)
        state["last_delta"] = dict(delta)
        state["last_step"] = step
        state["samples"] = int(state.get("samples", 0) or 0) + 1

    pressure_improvement = max(0.0, -float(delta.get("dream_pressure", 0.0)))
    stress_improvement = max(0.0, -float(delta.get("stress", 0.0)))
    relief_gain = max(0.0, float(delta.get("relief", 0.0)))
    coherence_gain = max(0.0, float(delta.get("coherence", 0.0)))

    m13_relevance = _scalar(m13.get("retrieval_relevance"), 0.0)
    m4_gate = _scalar(m4.get("dynamic_memory_gate"), 0.0)
    m4_stability = _scalar(m4.get("identity_stability"), 0.0)
    m4_novelty = _scalar(m4.get("identity_novelty"), 0.0)

    quality_score = (
        0.20 * min(1.0, replay_gate)
        + 0.15 * min(1.0, event_salience)
        + 0.15 * min(1.0, m13_relevance)
        + 0.12 * min(1.0, m4_gate)
        + 0.10 * min(1.0, seed_gate)
        + 0.10 * min(1.0, seed_norm / 10.0)
        + 0.08 * min(1.0, pressure_improvement * 4.0)
        + 0.05 * min(1.0, relief_gain * 4.0)
        + 0.03 * min(1.0, stress_improvement * 4.0)
        + 0.02 * min(1.0, coherence_gain * 4.0)
    )
    quality_score = float(max(0.0, min(1.0, quality_score)))
    quality_ema = 0.90 * float(state.get("quality_ema", 0.0) or 0.0) + 0.10 * quality_score
    state["quality_ema"] = quality_ema

    if replay_gate <= 0.02 and seed_gate <= 0.02:
        verdict = "idle"
    elif pressure_improvement > 0.002 or relief_gain > 0.002 or coherence_gain > 0.002:
        verdict = "integrating"
    elif replay_gate > 0.25 or seed_gate > 0.25:
        verdict = "replaying"
    else:
        verdict = "weak"

    row = {
        "step": step,
        "verdict": verdict,
        "quality": quality_score,
        "pressure": dream_pressure,
        "pressure_delta": float(delta.get("dream_pressure", 0.0)),
        "identity": selected_identity,
        "source": replay_source,
    }
    history = list(state.get("history", []) or [])
    if not history or history[-1].get("step") != step:
        history.append(row)
        history = history[-20:]
    state["history"] = history
    setattr(system, "_replay_quality_monitor_state", state)

    return {
        "global_step": step,
        "full_sleep": bool(full_sleep),
        "sensor_state": str(sensor_state),
        "quality_score": quality_score,
        "quality_ema": float(quality_ema),
        "verdict": verdict,
        "selected_episode_summary": selected_summary,
        "selected_identity_token": selected_identity,
        "selected_identity_sentence": selected_sentence,
        "replay_source": replay_source,
        "m2": {
            "replay_gate": replay_gate,
            "should_replay": should_replay,
            "dream_pressure": dream_pressure,
            "event_salience": event_salience,
            "dream_pressure_delta": float(delta.get("dream_pressure", 0.0)),
            "dream_pressure_trend": _trend(float(delta.get("dream_pressure", 0.0))),
        },
        "affect": {
            "stress": stress,
            "panic": panic,
            "relief": relief,
            "curiosity": curiosity,
            "valence": valence,
            "coherence": coherence,
            "expected_affect_delta": expected_affect_delta,
            "stress_delta": float(delta.get("stress", 0.0)),
            "panic_delta": float(delta.get("panic", 0.0)),
            "relief_delta": float(delta.get("relief", 0.0)),
            "coherence_delta": float(delta.get("coherence", 0.0)),
            "valence_delta": float(delta.get("valence", 0.0)),
        },
        "m13": {
            "relevance": m13_relevance,
            "episodes": _scalar(m13.get("episode_count", m13.get("retrieved_episode_count")), 0.0),
            "summary": _text(m13.get("summary", "")),
        },
        "m4": {
            "identity_token": _text(m4.get("identity_token", "")),
            "dynamic_memory_gate": m4_gate,
            "identity_stability": m4_stability,
            "identity_novelty": m4_novelty,
            "selected_sentence": _text(m4.get("selected_sentence", "")),
        },
        "m5": {
            "seed_gate": seed_gate,
            "seed_norm": seed_norm,
        },
        "dream_probe": dict(dream_probe),
        "history": history,
        "samples": int(state.get("samples", 0) or 0),
    }


__all__ = ["build_replay_quality_monitor_status"]
