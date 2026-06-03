from __future__ import annotations

"""
Runtime bridge for EnergyResonator.

Intended place in life_runtime.py:

    out["inner_object"] = self.compute_inner_object_image(obs, out)
    self._compute_long_dynamic_memory(obs, out)
    self.compute_energy_resonator(obs, out)
    out["self_core"] = self.compute_self_core(obs, out)

This makes the social-affect process available before M9 self-binding.
"""

from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.modules.m11_motivational_homeostasis.energy_resonator import (
    EnergyResonator,
    EnergyResonatorConfig,
)


def _cfg_value(root: Any, dotted: str, default: Any) -> Any:
    cur = root
    for part in dotted.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


class EnergyResonatorRuntimeMixin:
    def ensure_energy_resonator_ready(self) -> None:
        if hasattr(self, "energy_resonator") and self.energy_resonator is not None:
            return

        cfg = EnergyResonatorConfig(
            enabled=bool(_cfg_value(getattr(self, "cfg", None), "energy_resonator.enabled", True)),
            blend_into_focus_context=bool(
                _cfg_value(getattr(self, "cfg", None), "energy_resonator.blend_into_focus_context", False)
            ),
            focus_blend_weight=float(
                _cfg_value(getattr(self, "cfg", None), "energy_resonator.focus_blend_weight", 0.015)
            ),
            default_mode=str(_cfg_value(getattr(self, "cfg", None), "energy_resonator.default_mode", "animal")),
            drive=float(_cfg_value(getattr(self, "cfg", None), "energy_resonator.drive", 0.65)),
            empathy=float(_cfg_value(getattr(self, "cfg", None), "energy_resonator.empathy", 0.45)),
            tolerance=float(_cfg_value(getattr(self, "cfg", None), "energy_resonator.tolerance", 0.45)),
            cost=float(_cfg_value(getattr(self, "cfg", None), "energy_resonator.cost", 0.45)),
            her_need=float(_cfg_value(getattr(self, "cfg", None), "energy_resonator.her_need", 0.70)),
            decay=float(_cfg_value(getattr(self, "cfg", None), "energy_resonator.decay", 0.035)),
        )
        self.energy_resonator = EnergyResonator(cfg)
        print(
            "[energy_resonator] initialized | "
            f"enabled={cfg.enabled} mode={cfg.default_mode} "
            f"blend_into_focus_context={cfg.blend_into_focus_context}"
        )

    def _energy_resonator_mode_from_runtime(self, obs: Dict, out: Dict) -> str:
        """
        Runtime policy for now.

        If config/env/control later provides a mode, use it.
        Otherwise:
        - sleep/dream-like mode can be conscious, because external sensors are lower.
        - default is cfg default.
        """
        del obs
        cfg_mode = str(_cfg_value(getattr(self, "cfg", None), "energy_resonator.default_mode", "animal"))
        explicit = getattr(self, "_energy_resonator_mode", None)
        if explicit in ("animal", "conscious"):
            return str(explicit)

        # If M5/M2 already marks dream/replay, keep the process conscious by default.
        replay = out.get("event_dream_replay") if isinstance(out, dict) else None
        if isinstance(replay, dict) and bool(replay.get("dream_mode", False)):
            return "conscious"

        return cfg_mode if cfg_mode in ("animal", "conscious") else "animal"

    def _blend_energy_resonator_into_focus_context(self, out: Dict, latent: Any) -> None:
        if torch is None:
            return
        if not bool(getattr(self.energy_resonator.cfg, "blend_into_focus_context", False)):
            return

        focus = out.get("focus_context")
        if not torch.is_tensor(focus) or not torch.is_tensor(latent):
            return

        if focus.ndim > 2:
            focus_flat = focus.reshape(focus.shape[0], -1)
        else:
            focus_flat = focus

        batch, dim = int(focus_flat.shape[0]), int(focus_flat.shape[-1])
        latent_flat = latent
        if latent_flat.ndim > 2:
            latent_flat = latent_flat.reshape(latent_flat.shape[0], -1)
        if latent_flat.shape[0] != batch:
            latent_flat = latent_flat[:1].repeat(batch, 1)

        if latent_flat.shape[-1] < dim:
            pad = torch.zeros(batch, dim - latent_flat.shape[-1], dtype=focus_flat.dtype, device=focus_flat.device)
            latent_padded = torch.cat([latent_flat.to(focus_flat.device, focus_flat.dtype), pad], dim=-1)
        else:
            latent_padded = latent_flat[:, :dim].to(focus_flat.device, focus_flat.dtype)

        w = float(getattr(self.energy_resonator.cfg, "focus_blend_weight", 0.015))
        out["focus_context"] = focus_flat + latent_padded * w
        out["focus_context_energy_resonator_blended"] = True

    def compute_energy_resonator(self, obs: Dict, out: Dict) -> Dict | None:
        self.ensure_energy_resonator_ready()
        if not self.energy_resonator.cfg.enabled:
            return None

        mode = self._energy_resonator_mode_from_runtime(obs, out)
        step = self.energy_resonator.step(mode=mode)
        device = getattr(self, "device", None)
        latent = self.energy_resonator.latent_vector(step, device=device)

        packet = step.to_dict()
        packet["latent_dim"] = 12
        packet["source"] = "m11_energy_resonator_experimental"

        out["energy_resonator"] = packet
        out["other_agent_focus"] = packet["m5_other_agent_focus"]
        out["energy_resonator_latent"] = latent

        # M11 bridge: expose social affect as a nested packet without replacing
        # existing emotional_drive output.
        out["social_affect"] = {
            "resonance_index": step.resonance_index,
            "conscious_alignment": step.conscious_alignment,
            "he_negative_cost": step.he_negative_cost,
            "he_empathic_positive": step.he_empathic_positive,
            "she_received_positive": step.she_received_positive,
            "latent": latent,
        }

        self._blend_energy_resonator_into_focus_context(out, latent)
        return packet

    def maybe_print_energy_resonator_trace(self, out: Dict) -> None:
        packet = out.get("energy_resonator") if isinstance(out, dict) else None
        if not isinstance(packet, dict):
            return
        step = int(packet.get("step", 0))
        print_every = int(_cfg_value(getattr(self, "cfg", None), "energy_resonator.print_every_steps", 50))
        global_step = int(getattr(self, "global_step", step))
        if global_step % max(1, print_every) != 0:
            return
        print(
            f"[energy_resonator step={global_step}] "
            f"mode={packet.get('mode')} "
            f"action={packet.get('action_intensity', 0.0):.3f} "
            f"she+={packet.get('she_received_positive', 0.0):.3f} "
            f"he_empathy+={packet.get('he_empathic_positive', 0.0):.3f} "
            f"cost-={packet.get('he_negative_cost', 0.0):.3f} "
            f"resonance={packet.get('resonance_index', 0.0):.3f}"
        )


__all__ = ["EnergyResonatorRuntimeMixin"]
