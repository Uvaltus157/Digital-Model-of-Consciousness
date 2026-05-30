from __future__ import annotations

import torch

from src.modules.m04_long_dynamic_memory.long_dynamic_memory import (
    LongDynamicMemory,
    LongDynamicMemoryConfig,
)


class LongDynamicMemoryRuntimeMixin:
    def ensure_long_dynamic_memory_ready(self) -> None:
        if hasattr(self, "long_dynamic_memory_controller") and self.long_dynamic_memory_controller is not None:
            return
        cfg_obj = getattr(self.cfg, "long_dynamic_memory", None)
        self.long_dynamic_memory_controller = LongDynamicMemory(LongDynamicMemoryConfig(
            enabled=bool(getattr(cfg_obj, "enabled", True)),
            context_dim=int(getattr(cfg_obj, "context_dim", getattr(self.cfg.self_core, "focus_context_dim", 256))),
            focus_blend=float(getattr(cfg_obj, "focus_blend", 0.18)),
            blend_into_focus=bool(getattr(cfg_obj, "blend_into_focus", True)),
            stability_threshold=float(getattr(cfg_obj, "stability_threshold", 0.12)),
            novelty_threshold=float(getattr(cfg_obj, "novelty_threshold", 0.35)),
            use_passport_manager=bool(getattr(cfg_obj, "use_passport_manager", True)),
            use_event_memory=bool(getattr(cfg_obj, "use_event_memory", True)),
        ))
        print("[long_dynamic_memory] controller initialized")

    def compute_long_dynamic_memory(self, obs: dict, out: dict):
        del obs
        cfg_obj = getattr(self.cfg, "long_dynamic_memory", None)
        if not bool(getattr(cfg_obj, "enabled", True)):
            return None
        obj = out.get("inner_object")
        if not isinstance(obj, dict):
            return None
        self.ensure_long_dynamic_memory_ready()
        if hasattr(self, "_ensure_dynamic_object_passports"):
            try:
                self._ensure_dynamic_object_passports()
            except Exception:
                pass
        dream_mode = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        packet = self.long_dynamic_memory_controller.compute(
            out=out,
            obj=obj,
            passport_manager=getattr(self, "dynamic_object_passports", None),
            event_memory=getattr(self, "event_latent_memory", None),
            dream_mode=dream_mode,
            global_step=int(getattr(self, "global_step", 0)),
        )
        out["long_dynamic_memory"] = packet

        # Mirror key identity fields back into inner_object for existing
        # visualizers/debug code that already reads passport_* fields there.
        try:
            ref = obj.get("z_obj")
            device = ref.device if torch.is_tensor(ref) else getattr(self, "device", "cpu")
            dtype = ref.dtype if torch.is_tensor(ref) else torch.float32
            for key in ("identity_stability", "identity_similarity", "identity_novelty", "identity_dynamic_score", "passport_count", "passport_slot", "passport_created"):
                value = packet.get(key)
                if torch.is_tensor(value):
                    obj[f"m4_{key}"] = value.to(device=device, dtype=dtype)
            obj["m4_identity_token"] = str(packet.get("identity_token", ""))
            obj["m4_identity_source"] = str(packet.get("identity_source", ""))
            obj["m4_selected_sentence"] = str(packet.get("selected_sentence", ""))
            out["inner_object"] = obj
        except Exception:
            pass
        return packet

    def blend_long_dynamic_memory_into_focus(self, out: dict):
        packet = out.get("long_dynamic_memory") if isinstance(out.get("long_dynamic_memory"), dict) else None
        if not isinstance(packet, dict):
            return None
        focus = out.get("focus_context")
        context = packet.get("dynamic_identity_context")
        gate = packet.get("dynamic_memory_gate")
        cfg = self.long_dynamic_memory_controller.cfg if hasattr(self, "long_dynamic_memory_controller") else None
        if cfg is None or not bool(getattr(cfg, "blend_into_focus", True)):
            return None
        if not (torch.is_tensor(focus) and torch.is_tensor(context) and torch.is_tensor(gate)):
            return None
        try:
            gate_value = float(gate.detach().reshape(-1)[0].cpu().item())
            if gate_value <= 0.0:
                packet["blended_into_focus"] = torch.tensor([0.0], device=focus.device)
                return packet
            if context.shape[-1] != focus.shape[-1]:
                if context.shape[-1] > focus.shape[-1]:
                    context = context[..., : focus.shape[-1]]
                else:
                    pad = torch.zeros(*context.shape[:-1], focus.shape[-1] - context.shape[-1], device=focus.device, dtype=focus.dtype)
                    context = torch.cat([context.to(focus.device, focus.dtype), pad], dim=-1)
            out["pre_long_dynamic_focus_context"] = focus
            out["focus_context"] = focus + float(cfg.focus_blend) * gate_value * context.to(focus.device, focus.dtype)
            out["focus_context_source"] = "m04_long_dynamic_memory"
            packet["blended_into_focus"] = torch.tensor([1.0], device=focus.device)
            return packet
        except Exception as e:
            if not hasattr(self, "_long_dynamic_focus_blend_warned"):
                print(f"[long_dynamic_memory] focus blend skipped: {e}")
                self._long_dynamic_focus_blend_warned = True
            return None

    def maybe_print_long_dynamic_memory_trace(self, out: dict) -> None:
        cfg_obj = getattr(self.cfg, "long_dynamic_memory", None)
        every = int(getattr(cfg_obj, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        packet = out.get("long_dynamic_memory")
        if not isinstance(packet, dict):
            print(f"[long_dynamic_memory step={self.global_step}] no long_dynamic_memory output")
            return

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[long_dynamic_memory step={self.global_step}] "
            f"token={packet.get('identity_token', '')} "
            f"gate={f(packet.get('dynamic_memory_gate')):.3f} "
            f"stability={f(packet.get('identity_stability')):.3f} "
            f"novelty={f(packet.get('identity_novelty')):.3f} "
            f"passports={f(packet.get('passport_count')):.0f} "
            f"blend={f(packet.get('blended_into_focus')):.0f} | "
            f"{packet.get('selected_sentence', '')}"
        )


__all__ = ["LongDynamicMemoryRuntimeMixin"]
