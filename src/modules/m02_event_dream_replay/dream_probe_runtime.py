from __future__ import annotations

"""
Live diagnostic stimulus for sleep/replay debugging.

This is not a training fixture and not a MuJoCo simulator. It is a short-lived
runtime probe controlled by M8 IPC buttons. The probe perturbs internal M5/M11
inputs so the Sleep Replay Monitor can show visible deltas.

Rules:
    - does not feed raw M1 into M2
    - does not mutate out["focus_context"] directly
    - can create a temporary M2/M5 seed through the existing seed bus
"""

from typing import Any, Dict, Optional

import torch


def _clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    return max(float(lo), min(float(hi), v))


def _device_from_out(out: Optional[Dict[str, Any]]) -> torch.device:
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v):
                return v.device
            if isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        return vv.device
    return torch.device("cpu")


def _scalar_tensor(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32, device=device)


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


class DreamProbeRuntimeMixin:
    def request_dream_probe(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = dict(payload or {})
        kind = str(payload.get("kind", payload.get("probe", "curiosity"))).lower().strip()
        if kind in ("stop", "clear", "off", "none"):
            self._dream_probe_state = {
                "active": False,
                "kind": "clear",
                "remaining": 0,
                "duration": 0,
                "intensity": 0.0,
                "source": str(payload.get("source", "ipc")),
                "started_step": int(getattr(self, "global_step", 0)),
            }
            print("[dream_probe] cleared")
            if hasattr(self, "write_module_debug_status"):
                self.write_module_debug_status()
            return dict(self._dream_probe_state)

        duration = max(1, int(_clamp(payload.get("duration", 60), 1, 500, 60)))
        intensity = _clamp(payload.get("intensity", 0.75), 0.0, 1.5, 0.75)

        if kind in ("replay", "seed", "replay_seed", "m5_seed"):
            kind = "replay_seed"
        elif kind in ("stress", "panic", "fear", "uncertainty"):
            kind = "stress"
        elif kind in ("mixed", "dream", "pulse"):
            kind = "mixed"
        else:
            kind = "curiosity"

        state = {
            "active": True,
            "kind": kind,
            "remaining": int(duration),
            "duration": int(duration),
            "intensity": float(intensity),
            "source": str(payload.get("source", "ipc")),
            "started_step": int(getattr(self, "global_step", 0)),
            "last_pulse": 0.0,
        }
        self._dream_probe_state = state

        if kind == "replay_seed":
            self._inject_probe_replay_seed(float(intensity), source="dream_probe_request")

        print(
            f"[dream_probe] requested kind={kind} intensity={intensity:.3f} "
            f"duration={duration} source={state['source']}"
        )
        print(f"[dream_probe][state] {self._dream_probe_state}")
        if hasattr(self, "write_module_debug_status"):
            self.write_module_debug_status()
        return dict(state)

    def _inject_probe_replay_seed(self, intensity: float, *, source: str = "dream_probe") -> None:
        latest = getattr(self, "latest_out", {}) or {}
        device = _device_from_out(latest)
        seed = None
        for key in ("focus_context", "workspace_out", "obs_embed"):
            value = latest.get(key) if isinstance(latest, dict) else None
            if torch.is_tensor(value):
                seed = value.detach().clone().float()
                break
        if seed is None:
            seed = torch.zeros(1, 256, dtype=torch.float32, device=device)
            seed[:, 0] = float(intensity)
        if seed.ndim == 1:
            seed = seed.unsqueeze(0)
        if seed.shape[-1] != 256:
            fixed = torch.zeros(seed.shape[0], 256, dtype=seed.dtype, device=seed.device)
            n = min(int(seed.shape[-1]), 256)
            fixed[:, :n] = seed[:, :n]
            seed = fixed

        gate = _scalar_tensor(float(intensity), seed.device)
        self._event_dream_next_focus_seed = seed
        self._event_dream_next_focus_gate = gate
        self._dream_probe_last_seed = {
            "seed_gate": float(intensity),
            "seed_norm": float(seed.detach().float().norm().cpu().item()),
            "source": str(source),
        }

    def apply_dream_probe_to_out(self, out: Dict[str, Any], obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        del obs
        if not isinstance(out, dict):
            return out

        state = getattr(self, "_dream_probe_state", None)
        if not isinstance(state, dict) or not bool(state.get("active", False)):
            return out

        remaining = int(state.get("remaining", 0) or 0)
        duration = max(1, int(state.get("duration", 1) or 1))
        if remaining <= 0:
            state["active"] = False
            state["remaining"] = 0
            return out

        intensity = float(state.get("intensity", 0.75) or 0.75)
        phase = max(0.0, min(1.0, float(remaining) / float(duration)))
        pulse = float(intensity * phase)
        kind = str(state.get("kind", "curiosity"))

        device = _device_from_out(out)
        values = out.setdefault("values", {})
        if not isinstance(values, dict):
            values = {}
            out["values"] = values

        object_imagery = out.setdefault("object_imagery", {})
        if not isinstance(object_imagery, dict):
            object_imagery = {}
            out["object_imagery"] = object_imagery

        reflection = out.setdefault("preconscious_reflection_out", {})
        if not isinstance(reflection, dict):
            reflection = {}
            out["preconscious_reflection_out"] = reflection

        if kind in ("curiosity", "mixed"):
            current_curiosity = _scalar(values.get("curiosity"), 0.0)
            values["curiosity"] = _scalar_tensor(max(current_curiosity, pulse), device)

        if kind in ("stress", "mixed"):
            # Raise uncertainty seen by M11 without bypassing M11:
            # low coherence/object/self confidence -> higher stress/fear/panic.
            low_conf = max(0.0, 1.0 - pulse)
            current_coh = _scalar(values.get("coherence"), 1.0)
            values["coherence"] = _scalar_tensor(min(current_coh, low_conf), device)
            object_imagery["object_confidence"] = _scalar_tensor(low_conf, device)
            reflection["model_confidence"] = _scalar_tensor(low_conf, device)
            if "self_core" not in out or not isinstance(out.get("self_core"), dict):
                out["self_core"] = {}
            out["self_core"]["self_confidence"] = _scalar_tensor(low_conf, device)

        if kind == "replay_seed":
            self._inject_probe_replay_seed(pulse, source="dream_probe_apply")
            current_curiosity = _scalar(values.get("curiosity"), 0.0)
            values["curiosity"] = _scalar_tensor(max(current_curiosity, min(1.0, 0.25 + pulse)), device)

        # M11 may have been computed earlier in the same life step for M9/M15.
        # The probe changes M11 inputs, so its reusable packet is now stale.
        if isinstance(out.get("emotion"), dict):
            out["emotion"]["_emotion_cache_reusable"] = False
        out.pop("affect", None)

        state["remaining"] = int(max(0, remaining - 1))
        state["last_pulse"] = float(pulse)
        state["active"] = bool(state["remaining"] > 0)

        out["dream_probe"] = {
            "active": bool(state.get("active", False)),
            "kind": kind,
            "remaining": int(state.get("remaining", 0)),
            "duration": int(duration),
            "intensity": float(intensity),
            "pulse": float(pulse),
            "source": str(state.get("source", "")),
        }
        self._dream_probe_state = state
        print(
            f"[dream_probe][apply] step={getattr(self, 'global_step', 0)} "
            f"kind={kind} pulse={pulse:.4f} remaining={state['remaining']}"
        )
        if kind in ("curiosity", "mixed", "replay_seed"):
            print(
                f"[dream_probe][values] curiosity={_scalar(values.get('curiosity')):.4f} "
                f"coherence={_scalar(values.get('coherence')):.4f}"
            )
        if kind in ("stress", "mixed"):
            print(
                f"[dream_probe][stress_inputs] coherence={_scalar(values.get('coherence')):.4f} "
                f"object_conf={_scalar(object_imagery.get('object_confidence')):.4f} "
                f"model_conf={_scalar(reflection.get('model_confidence')):.4f} "
                f"self_conf={_scalar(out['self_core'].get('self_confidence')):.4f}"
            )
        return out


__all__ = ["DreamProbeRuntimeMixin"]
