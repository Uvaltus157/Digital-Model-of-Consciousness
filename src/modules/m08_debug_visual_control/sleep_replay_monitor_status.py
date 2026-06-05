from __future__ import annotations

"""Compact status payload for the M8 Sleep Replay Monitor window."""

from typing import Any, Dict

import torch


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


def _dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}



def _trend(delta: float, eps: float = 0.002) -> str:
    if delta > eps:
        return "↑"
    if delta < -eps:
        return "↓"
    return "→"


def _init_affect_visibility_state(system: Any) -> Dict[str, Any]:
    state = getattr(system, "_sleep_replay_m11_visibility_state", None)
    if not isinstance(state, dict):
        state = {
            "prev": {},
            "mins": {},
            "maxs": {},
            "samples": 0,
        }
        setattr(system, "_sleep_replay_m11_visibility_state", state)
    return state


def _attach_m11_visibility(system: Any, payload: Dict[str, Any]) -> None:
    m11 = payload.get("m11", {})
    if not isinstance(m11, dict):
        return

    keys = ("valence", "arousal", "stress", "panic", "curiosity")
    current = {k: float(m11.get(k, 0.0) or 0.0) for k in keys}

    state = _init_affect_visibility_state(system)
    step = int(payload.get("global_step", 0))
    previous_current = state.get("prev", {}) if isinstance(state.get("prev"), dict) else {}
    same_values = all(float(previous_current.get(key, value)) == value for key, value in current.items())
    if int(state.get("last_step", -1)) == step and same_values:
        payload["m11_delta"] = dict(state.get("last_delta", {}))
        payload["m11_range"] = dict(state.get("last_range", {}))
        payload["m11_activity"] = dict(state.get("last_activity", {}))
        return

    prev = state.get("prev", {}) if isinstance(state.get("prev"), dict) else {}
    mins = state.get("mins", {}) if isinstance(state.get("mins"), dict) else {}
    maxs = state.get("maxs", {}) if isinstance(state.get("maxs"), dict) else {}
    samples = int(state.get("samples", 0) or 0) + 1

    delta = {}
    for key, value in current.items():
        old = float(prev.get(key, value))
        delta[key] = float(value - old)
        mins[key] = min(float(mins.get(key, value)), value)
        maxs[key] = max(float(maxs.get(key, value)), value)

    state["prev"] = dict(current)
    state["mins"] = dict(mins)
    state["maxs"] = dict(maxs)
    state["samples"] = samples

    affect_keys = ("stress", "panic", "curiosity")
    change_score = float(sum(abs(delta[k]) for k in affect_keys))
    signed_score = float(sum(delta[k] for k in affect_keys))
    if change_score < 0.002:
        trend = "→"
    elif signed_score > 0.002:
        trend = "↑"
    elif signed_score < -0.002:
        trend = "↓"
    else:
        trend = "↕"

    payload["m11_delta"] = {
        "valence": float(delta["valence"]),
        "arousal": float(delta["arousal"]),
        "stress": float(delta["stress"]),
        "panic": float(delta["panic"]),
        "curiosity": float(delta["curiosity"]),
    }
    payload["m11_range"] = {
        "stress_min": float(mins.get("stress", current["stress"])),
        "stress_max": float(maxs.get("stress", current["stress"])),
        "panic_min": float(mins.get("panic", current["panic"])),
        "panic_max": float(maxs.get("panic", current["panic"])),
        "curiosity_min": float(mins.get("curiosity", current["curiosity"])),
        "curiosity_max": float(maxs.get("curiosity", current["curiosity"])),
    }
    payload["m11_activity"] = {
        "change_score": change_score,
        "trend": trend,
        "stress_trend": _trend(delta["stress"]),
        "panic_trend": _trend(delta["panic"]),
        "curiosity_trend": _trend(delta["curiosity"]),
        "samples": samples,
    }
    state["last_step"] = step
    state["last_delta"] = dict(payload["m11_delta"])
    state["last_range"] = dict(payload["m11_range"])
    state["last_activity"] = dict(payload["m11_activity"])


def build_sleep_replay_monitor_status(system: Any) -> Dict[str, Any]:
    """Build a JSON-safe live monitor payload from the current runtime state."""
    out = _dict(getattr(system, "latest_out", {}) or {})
    trace = _dict(out.get("unconscious_loop_trace"))
    affect = _dict(out.get("affect"))
    emotion = _dict(out.get("emotion"))
    m13 = _dict(out.get("autobiographical_memory"))
    m4 = _dict(out.get("long_dynamic_memory"))
    m2 = _dict(out.get("event_dream_replay"))
    focus_feedback = _dict(out.get("focus_feedback"))
    attention = _dict(out.get("attention"))
    sleep_motor = _dict(out.get("sleep_motor_guard"))

    full_sleep = bool(system.is_full_sleep_mode()) if hasattr(system, "is_full_sleep_mode") else False
    sensor_state = system.sensor_state_label() if hasattr(system, "sensor_state_label") else "unknown"
    sensors = system.input_sensors_enabled_dict() if hasattr(system, "input_sensors_enabled_dict") else {
        "video": bool(getattr(system, "video_sensor_enabled", True)),
        "contact": bool(getattr(system, "contact_sensor_enabled", True)),
        "imu": bool(getattr(system, "imu_sensor_enabled", True)),
    }

    trace_m11 = _dict(trace.get("m11"))
    trace_m13 = _dict(trace.get("m13"))
    trace_m4 = _dict(trace.get("m4"))
    trace_m2 = _dict(trace.get("m2"))
    trace_m5 = _dict(trace.get("m5_seed"))
    trace_m3 = _dict(trace.get("m3"))

    seed = getattr(system, "_event_dream_next_focus_seed", None)
    if seed is None:
        seed = m2.get("next_focus_context_seed", m2.get("replay_context"))
    seed_gate = getattr(system, "_event_dream_next_focus_gate", None)
    if seed_gate is None:
        seed_gate = m2.get("next_focus_context_seed_gate", m2.get("replay_gate"))
    last_probe_seed = _dict(getattr(system, "_dream_probe_last_seed", {}))

    payload = {
        "global_step": int(getattr(system, "global_step", 0)),
        "full_sleep": bool(full_sleep),
        "sensor_state": str(sensor_state),
        "sensors": {
            "video": bool(sensors.get("video", True)),
            "contact": bool(sensors.get("contact", True)),
            "imu": bool(sensors.get("imu", True)),
        },
        "m1": {
            "state": str(sensor_state),
            "video_on": bool(sensors.get("video", True)),
            "contact_on": bool(sensors.get("contact", True)),
            "imu_on": bool(sensors.get("imu", True)),
        },
        "m11": {
            "valence": _scalar(emotion.get("emotional_valence", affect.get("valence")), trace_m11.get("valence", 0.0)),
            "arousal": _scalar(emotion.get("emotional_arousal", affect.get("arousal")), trace_m11.get("arousal", 0.0)),
            "stress": _scalar(affect.get("stress_latent"), trace_m11.get("stress", 0.0)),
            "panic": _scalar(affect.get("panic_latent"), trace_m11.get("panic", 0.0)),
            "curiosity": _scalar(affect.get("curiosity_latent"), trace_m11.get("curiosity", 0.0)),
            "comfort": _scalar(affect.get("comfort_latent"), trace_m11.get("comfort", 0.0)),
            "relief": _scalar(affect.get("relief_latent"), trace_m11.get("relief", 0.0)),
        },
        "m13": {
            "relevance": float(trace_m13.get("relevance", _scalar(m13.get("retrieval_relevance"), 0.0))),
            "episodes": float(trace_m13.get("episodes", _scalar(m13.get("episode_count", m13.get("retrieved_episode_count")), 0.0))),
            "summary": _text(trace_m13.get("summary", m13.get("summary", m13.get("last_summary", "")))),
        },
        "m4": {
            "token": _text(trace_m4.get("token", m4.get("identity_token", ""))),
            "gate": float(trace_m4.get("gate", _scalar(m4.get("dynamic_memory_gate"), 0.0))),
            "stability": float(trace_m4.get("stability", _scalar(m4.get("identity_stability"), 0.0))),
            "novelty": float(trace_m4.get("novelty", _scalar(m4.get("identity_novelty"), 0.0))),
            "sentence": _text(trace_m4.get("sentence", m4.get("selected_sentence", ""))),
        },
        "m2": {
            "replay_gate": _scalar(m2.get("replay_gate"), trace_m2.get("replay_gate", 0.0)),
            "should_replay": _scalar(m2.get("should_replay"), trace_m2.get("should_replay", 0.0)),
            "dream_pressure": _scalar(m2.get("dream_pressure"), trace_m2.get("dream_pressure", 0.0)),
            "event_salience": _scalar(m2.get("event_salience"), trace_m2.get("event_salience", 0.0)),
            "source": _text(m2.get("replay_source"), _text(trace_m2.get("source", ""))),
            "identity": _text(m2.get("selected_identity_token"), _text(trace_m2.get("identity", ""))),
        },
        "m5": {
            "seed_gate": _scalar(seed_gate, last_probe_seed.get("seed_gate", trace_m5.get("seed_gate", 0.0))),
            "seed_norm": _norm(seed, last_probe_seed.get("seed_norm", trace_m5.get("seed_norm", 0.0))),
            "feedback_gate": _scalar(focus_feedback.get("total_gate", attention.get("focus_feedback_gate")), trace_m5.get("feedback_gate", 0.0)),
            "feedback_active": _scalar(focus_feedback.get("active"), trace_m5.get("feedback_active", 0.0)),
        },
        "m3": {
            "sleep_blocked": bool(trace_m3.get("sleep_blocked", sleep_motor.get("blocked", False))),
            "blocked_norm": float(trace_m3.get("blocked_norm", _scalar(sleep_motor.get("blocked_motor_norm"), 0.0))),
            "stage": _text(trace_m3.get("stage", sleep_motor.get("stage", ""))),
            "blocked_keys": list(sleep_motor.get("blocked_keys", []) or []),
        },
        "dream_probe": dict(out.get("dream_probe", getattr(system, "_dream_probe_state", {}) or {}) or {}),
        "trace_present": bool(trace),
    }
    _attach_m11_visibility(system, payload)
    return payload


__all__ = ["build_sleep_replay_monitor_status"]
