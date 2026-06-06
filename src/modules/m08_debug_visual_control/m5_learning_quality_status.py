from __future__ import annotations

"""M5 Learning Quality Baseline status.

Read-only diagnostic payload for answering:

    "Is M5 merely receiving signals, or is the world model actually learning?"

This builder does not train, mutate model outputs, create replay seeds, or modify
M5. It only summarizes existing runtime statistics and keeps a lightweight
baseline/trend state across status polls.
"""

from typing import Any, Dict, Iterable
import math

import torch


def _dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _scalar(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _norm(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().norm().cpu().item())
        return float(default)
    except Exception:
        return float(default)


def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _first_scalar_dict(dicts: Iterable[Dict[str, Any]], keys: Iterable[str], default: float = 0.0) -> float:
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for key in keys:
            if key in d:
                return _scalar(d.get(key), default)
    return float(default)


def _first_norm_dict(dicts: Iterable[Dict[str, Any]], keys: Iterable[str], default: float = 0.0) -> float:
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for key in keys:
            if key in d:
                return _norm(d.get(key), default)
    return float(default)


def _trend(delta: float, eps: float = 1e-4, lower_is_better: bool = True) -> str:
    if abs(delta) <= eps:
        return "→"
    improved = delta < 0.0 if lower_is_better else delta > 0.0
    return "improving" if improved else "worse"


def _quality_from_improvements(loss_delta: float, pred_delta: float, recon_delta: float, coherence_delta: float, seed_response: float) -> float:
    score = 0.0
    score += 0.30 * min(1.0, max(0.0, -loss_delta))
    score += 0.25 * min(1.0, max(0.0, -pred_delta))
    score += 0.20 * min(1.0, max(0.0, -recon_delta))
    score += 0.15 * min(1.0, max(0.0, coherence_delta))
    score += 0.10 * min(1.0, max(0.0, seed_response / 5.0))
    return float(max(0.0, min(1.0, score)))


def build_m5_learning_quality_status(system: Any) -> Dict[str, Any]:
    out = _dict(getattr(system, "latest_out", {}) or {})
    focus_feedback = _dict(out.get("focus_feedback"))
    attention = _dict(out.get("attention"))
    event_dream = _dict(out.get("event_dream_replay"))
    latent_prototype = _dict(out.get("m5_latent_prototype"))
    object_decoder = _dict(getattr(system, "latest_object_decoder_stats", {}) or {})
    long_memory = _dict(getattr(system, "latest_long_dynamic_memory_stats", {}) or {})
    replay_quality = _dict(getattr(system, "_replay_quality_monitor_state", {}) or {})

    step = int(getattr(system, "global_step", 0) or 0)
    train_steps = int(getattr(system, "train_steps", 0) or 0)
    training_enabled = bool(getattr(system, "training_enabled", False))
    cfg_train_enabled = bool(getattr(getattr(system, "cfg", None), "train", None).enabled) if getattr(getattr(system, "cfg", None), "train", None) is not None else False
    full_sleep = bool(system.is_full_sleep_mode()) if hasattr(system, "is_full_sleep_mode") else False

    train_loss = _finite(_scalar(getattr(system, "last_train_loss", 0.0), 0.0), 0.0)
    last_train_reason = str(getattr(system, "last_train_reason", ""))
    last_train_error = str(getattr(system, "last_train_error", ""))

    candidate_dicts = [out, focus_feedback, attention, event_dream, object_decoder, long_memory]
    prediction_error = _first_scalar_dict(
        candidate_dicts,
        (
            "prediction_error",
            "pred_error",
            "next_obs_error",
            "obs_prediction_error",
            "world_model_prediction_error",
            "wm_prediction_error",
            "pred_loss",
            "prediction_loss",
        ),
        default=0.0,
    )
    reconstruction_error = _first_scalar_dict(
        candidate_dicts,
        (
            "reconstruction_error",
            "recon_error",
            "recon",
            "reconstruction_loss",
            "object_reconstruction_error",
            "decoder_recon",
        ),
        default=0.0,
    )
    latent_coherence = _first_scalar_dict(
        candidate_dicts,
        (
            "latent_coherence",
            "coherence",
            "workspace_coherence",
            "model_coherence",
            "reward_proxy",
        ),
        default=0.0,
    )

    focus_context = out.get("focus_context")
    workspace = out.get("workspace_out", out.get("workspace"))
    obs_embed = out.get("obs_embed", out.get("latent"))
    seed_gate = _scalar(event_dream.get("next_focus_context_seed_gate", event_dream.get("replay_gate")), 0.0)
    seed_norm = _norm(event_dream.get("next_focus_context_seed", event_dream.get("replay_context")), 0.0)
    if seed_norm <= 0.0 and latent_prototype:
        seed_gate = _scalar(latent_prototype.get("next_focus_context_seed_gate", latent_prototype.get("gate")), 0.0)
        seed_norm = _norm(latent_prototype.get("next_focus_context_seed"), 0.0)
    focus_norm = _norm(focus_context, 0.0)
    workspace_norm = _norm(workspace, 0.0)
    obs_embed_norm = _norm(obs_embed, 0.0)
    feedback_gate = _scalar(focus_feedback.get("total_gate", attention.get("focus_feedback_gate")), 0.0)

    identity_stability = _first_scalar_dict([out.get("long_dynamic_memory", {}) if isinstance(out.get("long_dynamic_memory"), dict) else {}, long_memory], ("identity_stability", "stability"), 0.0)
    identity_novelty = _first_scalar_dict([out.get("long_dynamic_memory", {}) if isinstance(out.get("long_dynamic_memory"), dict) else {}, long_memory], ("identity_novelty", "novelty"), 0.0)
    object_recon = _first_scalar_dict([object_decoder, long_memory], ("recon", "loss", "reconstruction_error"), 0.0)

    state = getattr(system, "_m5_learning_quality_state", None)
    if not isinstance(state, dict):
        state = {
            "last_step": None,
            "last_train_steps": None,
            "baseline": {},
            "prev": {},
            "last_delta": {},
            "samples": 0,
            "quality_ema": 0.0,
            "history": [],
        }
        setattr(system, "_m5_learning_quality_state", state)

    current = {
        "train_loss": train_loss,
        "prediction_error": prediction_error,
        "reconstruction_error": reconstruction_error,
        "latent_coherence": latent_coherence,
        "seed_norm": seed_norm,
        "seed_gate": seed_gate,
        "feedback_gate": feedback_gate,
        "identity_stability": identity_stability,
        "identity_novelty": identity_novelty,
        "object_recon": object_recon,
    }

    if not isinstance(state.get("baseline"), dict) or not state.get("baseline"):
        state["baseline"] = dict(current)

    last_step = state.get("last_step")
    last_train_steps = state.get("last_train_steps")
    if last_step == step and last_train_steps == train_steps:
        delta = dict(state.get("last_delta", {}) or {})
    else:
        prev = state.get("prev", {}) if isinstance(state.get("prev"), dict) else {}
        delta = {k: float(current[k] - float(prev.get(k, current[k]))) for k in current}
        state["prev"] = dict(current)
        state["last_delta"] = dict(delta)
        state["last_step"] = step
        state["last_train_steps"] = train_steps
        state["samples"] = int(state.get("samples", 0) or 0) + 1

    baseline = state.get("baseline", {}) if isinstance(state.get("baseline"), dict) else {}
    baseline_delta = {k: float(current[k] - float(baseline.get(k, current[k]))) for k in current}

    seed_response = float(seed_gate * min(1.0, seed_norm / 10.0) * min(1.0, max(feedback_gate, 0.0) + 0.25))
    learning_quality = _quality_from_improvements(
        float(delta.get("train_loss", 0.0)),
        float(delta.get("prediction_error", 0.0)),
        float(delta.get("reconstruction_error", 0.0)),
        float(delta.get("latent_coherence", 0.0)),
        seed_response,
    )
    quality_ema = 0.92 * float(state.get("quality_ema", 0.0) or 0.0) + 0.08 * learning_quality
    state["quality_ema"] = float(quality_ema)

    if last_train_error:
        verdict = "training_error"
    elif seed_response > 0.05 and train_steps <= 0:
        verdict = "seed_reactive_untrained"
    elif train_steps <= 0 and train_loss <= 0.0:
        verdict = "untrained_or_no_data"
    elif learning_quality > 0.20 or quality_ema > 0.15:
        verdict = "improving"
    elif train_steps > 0:
        verdict = "tracking"
    else:
        verdict = "idle"

    row = {
        "step": step,
        "train_steps": train_steps,
        "verdict": verdict,
        "train_loss": train_loss,
        "prediction_error": prediction_error,
        "reconstruction_error": reconstruction_error,
        "latent_coherence": latent_coherence,
        "seed_response": seed_response,
        "quality": learning_quality,
    }
    history = list(state.get("history", []) or [])
    if not history or history[-1].get("step") != step or history[-1].get("train_steps") != train_steps:
        history.append(row)
        history = history[-30:]
    state["history"] = history
    setattr(system, "_m5_learning_quality_state", state)

    return {
        "global_step": step,
        "train_steps": train_steps,
        "training_enabled": training_enabled,
        "cfg_train_enabled": cfg_train_enabled,
        "full_sleep": full_sleep,
        "verdict": verdict,
        "learning_quality": float(learning_quality),
        "learning_quality_ema": float(quality_ema),
        "last_train_reason": last_train_reason,
        "last_train_error": last_train_error,
        "m5_loss": {
            "train_loss": train_loss,
            "train_loss_delta": float(delta.get("train_loss", 0.0)),
            "train_loss_from_baseline": float(baseline_delta.get("train_loss", 0.0)),
            "train_loss_trend": _trend(float(delta.get("train_loss", 0.0)), lower_is_better=True),
            "prediction_error": prediction_error,
            "prediction_error_delta": float(delta.get("prediction_error", 0.0)),
            "prediction_error_from_baseline": float(baseline_delta.get("prediction_error", 0.0)),
            "prediction_error_trend": _trend(float(delta.get("prediction_error", 0.0)), lower_is_better=True),
            "reconstruction_error": reconstruction_error,
            "reconstruction_error_delta": float(delta.get("reconstruction_error", 0.0)),
            "reconstruction_error_from_baseline": float(baseline_delta.get("reconstruction_error", 0.0)),
            "reconstruction_error_trend": _trend(float(delta.get("reconstruction_error", 0.0)), lower_is_better=True),
        },
        "m5_latent": {
            "latent_coherence": latent_coherence,
            "latent_coherence_delta": float(delta.get("latent_coherence", 0.0)),
            "latent_coherence_from_baseline": float(baseline_delta.get("latent_coherence", 0.0)),
            "focus_norm": focus_norm,
            "workspace_norm": workspace_norm,
            "obs_embed_norm": obs_embed_norm,
        },
        "m5_seed_response": {
            "seed_gate": seed_gate,
            "seed_norm": seed_norm,
            "feedback_gate": feedback_gate,
            "seed_response": seed_response,
            "seed_norm_delta": float(delta.get("seed_norm", 0.0)),
        },
        "object_identity_proxy": {
            "object_recon": object_recon,
            "object_recon_delta": float(delta.get("object_recon", 0.0)),
            "identity_stability": identity_stability,
            "identity_stability_delta": float(delta.get("identity_stability", 0.0)),
            "identity_novelty": identity_novelty,
            "identity_novelty_delta": float(delta.get("identity_novelty", 0.0)),
        },
        "baseline": dict(baseline),
        "current": dict(current),
        "delta": dict(delta),
        "history": history,
        "samples": int(state.get("samples", 0) or 0),
        "note": "Read-only baseline. Before M5 is trained, this confirms signal response and metrics availability, not semantic understanding.",
    }


__all__ = ["build_m5_learning_quality_status"]
