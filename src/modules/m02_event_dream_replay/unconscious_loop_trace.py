from __future__ import annotations

"""
Live trace for the strictly unconscious DMoC loop.

Target architecture:

    M1 -> M5 -> M11 -> M2 -> FocusFeedbackBoundary -> M5
          |             ^
          v             |
         M4 ------------+
                       M13 -> M2

This file only observes runtime packets. It does not change architecture.
"""

from typing import Any, Dict, Optional

import torch


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
            return float(x.detach().float().norm(dim=-1).reshape(-1)[0].cpu().item())
        return float(default)
    except Exception:
        return float(default)


def build_unconscious_loop_trace_packet(out: Dict, *, sleep_mode: bool = False, sensor_state: str = "") -> Dict[str, Any]:
    emotion = out.get("emotion", {}) if isinstance(out.get("emotion"), dict) else {}
    affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
    m13 = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}
    m4 = out.get("long_dynamic_memory", {}) if isinstance(out.get("long_dynamic_memory"), dict) else {}
    m2 = out.get("event_dream_replay", {}) if isinstance(out.get("event_dream_replay"), dict) else {}
    m5_feedback = out.get("focus_feedback", {}) if isinstance(out.get("focus_feedback"), dict) else {}
    attention = out.get("attention", {}) if isinstance(out.get("attention"), dict) else {}
    sleep_motor = out.get("sleep_motor_guard", {}) if isinstance(out.get("sleep_motor_guard"), dict) else {}

    seed = m2.get("next_focus_context_seed", m2.get("replay_context"))
    seed_gate = m2.get("next_focus_context_seed_gate", m2.get("replay_gate"))

    return {
        "sleep": bool(sleep_mode),
        "sensor_state": str(sensor_state),
        "m11": {
            "valence": _scalar(emotion.get("emotional_valence"), _scalar(affect.get("valence"), 0.0)),
            "arousal": _scalar(emotion.get("emotional_arousal"), _scalar(affect.get("arousal"), 0.0)),
            "stress": _scalar(affect.get("stress_latent"), 0.0),
            "panic": _scalar(affect.get("panic_latent"), 0.0),
            "curiosity": _scalar(affect.get("curiosity_latent"), 0.0),
            "comfort": _scalar(affect.get("comfort_latent"), 0.0),
            "relief": _scalar(affect.get("relief_latent"), 0.0),
        },
        "m13": {
            "relevance": _scalar(m13.get("retrieval_relevance"), 0.0),
            "episodes": _scalar(m13.get("episode_count", m13.get("retrieved_episode_count")), 0.0),
            "summary": str(m13.get("summary", m13.get("last_summary", ""))),
        },
        "m4": {
            "token": str(m4.get("identity_token", "")),
            "gate": _scalar(m4.get("dynamic_memory_gate"), 0.0),
            "stability": _scalar(m4.get("identity_stability"), 0.0),
            "novelty": _scalar(m4.get("identity_novelty"), 0.0),
            "sentence": str(m4.get("selected_sentence", "")),
        },
        "m2": {
            "replay_gate": _scalar(m2.get("replay_gate"), 0.0),
            "should_replay": _scalar(m2.get("should_replay"), 0.0),
            "dream_pressure": _scalar(m2.get("dream_pressure"), 0.0),
            "event_salience": _scalar(m2.get("event_salience"), 0.0),
            "source": str(m2.get("replay_source", "")),
            "identity": str(m2.get("selected_identity_token", "")),
        },
        "m5_seed": {
            "seed_norm": _norm(seed, 0.0),
            "seed_gate": _scalar(seed_gate, 0.0),
            "feedback_gate": _scalar(
                m5_feedback.get("total_gate", attention.get("focus_feedback_gate")),
                0.0,
            ),
            "feedback_active": _scalar(m5_feedback.get("active"), 0.0),
        },
        "m3": {
            "sleep_blocked": bool(sleep_motor.get("blocked", False)),
            "stage": str(sleep_motor.get("stage", "")),
            "blocked_norm": _scalar(sleep_motor.get("blocked_motor_norm"), 0.0),
        },
    }


def format_unconscious_loop_trace(packet: Dict[str, Any], *, step: int = 0) -> str:
    m11 = packet["m11"]
    m13 = packet["m13"]
    m4 = packet["m4"]
    m2 = packet["m2"]
    seed = packet["m5_seed"]
    m3 = packet["m3"]
    return (
        f"[unconscious_loop step={int(step)}] "
        f"sleep={int(packet['sleep'])} state={packet['sensor_state']} | "
        f"m11: val={m11['valence']:.2f} ar={m11['arousal']:.2f} "
        f"stress={m11['stress']:.2f} panic={m11['panic']:.2f} cur={m11['curiosity']:.2f} | "
        f"m13: rel={m13['relevance']:.2f} eps={m13['episodes']:.0f} | "
        f"m4: token={m4['token']} gate={m4['gate']:.2f} stab={m4['stability']:.2f} nov={m4['novelty']:.2f} | "
        f"m2: gate={m2['replay_gate']:.2f} should={m2['should_replay']:.0f} "
        f"pressure={m2['dream_pressure']:.2f} sal={m2['event_salience']:.2f} src={m2['source']} | "
        f"m5_seed: gate={seed['seed_gate']:.2f} norm={seed['seed_norm']:.2f} fb={seed['feedback_gate']:.3f} | "
        f"m3_sleep_block={int(m3['sleep_blocked'])}"
    )


class UnconsciousLoopTraceRuntimeMixin:
    def _unconscious_trace_cfg(self) -> Any:
        return getattr(getattr(self, "cfg", None), "event_dream_replay", None)

    def _unconscious_trace_enabled(self) -> bool:
        cfg = self._unconscious_trace_cfg()
        return bool(getattr(cfg, "unconscious_trace_enabled", True))

    def build_unconscious_loop_trace_packet(self, out: Dict, obs: Optional[Dict] = None) -> Dict[str, Any]:
        del obs
        sleep_mode = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        try:
            state = self.sensor_state_label() if hasattr(self, "sensor_state_label") else ""
        except Exception:
            state = ""
        packet = build_unconscious_loop_trace_packet(out, sleep_mode=sleep_mode, sensor_state=state)
        out["unconscious_loop_trace"] = packet
        return packet

    def maybe_print_unconscious_loop_trace(self, out: Dict, obs: Optional[Dict] = None) -> None:
        if not self._unconscious_trace_enabled():
            return
        cfg = self._unconscious_trace_cfg()
        every = int(getattr(cfg, "unconscious_trace_every_steps", getattr(cfg, "print_every_steps", 30)))
        if every <= 0:
            return
        step = int(getattr(self, "global_step", 0))
        if step % every != 0:
            return
        packet = self.build_unconscious_loop_trace_packet(out, obs)
        print(format_unconscious_loop_trace(packet, step=step))


__all__ = [
    "UnconsciousLoopTraceRuntimeMixin",
    "build_unconscious_loop_trace_packet",
    "format_unconscious_loop_trace",
]
