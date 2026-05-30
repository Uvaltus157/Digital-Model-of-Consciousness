from __future__ import annotations

import torch

from src.modules.m10_global_conscious_broadcast.broadcast_gate import (
    GlobalBroadcastConfig,
    GlobalConsciousBroadcastGate,
)


class GlobalBroadcastRuntimeMixin:
    def ensure_global_broadcast_ready(self) -> None:
        if hasattr(self, "global_broadcast_gate") and self.global_broadcast_gate is not None:
            return

        cfg = GlobalBroadcastConfig(
            enabled=bool(getattr(getattr(self.cfg, "global_broadcast", None), "enabled", True)),
            focus_context_dim=int(getattr(self.cfg.self_core, "focus_context_dim", 256)),
            affect_latent_dim=int(getattr(self.cfg.self_core, "affect_latent_dim", 12)),
            thought_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "thought_dim", 128)),
            plan_context_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "plan_context_dim", 256)),
            hidden_dim=int(getattr(getattr(self.cfg, "global_broadcast", None), "hidden_dim", 256)),
            broadcast_threshold=float(getattr(getattr(self.cfg, "global_broadcast", None), "broadcast_threshold", 0.35)),
        )
        self.global_broadcast_gate = GlobalConsciousBroadcastGate(cfg).to(self.device)
        try:
            if hasattr(self, "module_training_gate"):
                self.module_training_gate.apply()
                self.rebuild_optimizer_from_trainable_modules()
            else:
                self.optimizer.add_param_group({"params": self.global_broadcast_gate.parameters()})
        except Exception as e:
            print(f"[global_broadcast] lazy optimizer attach skipped: {e}")
        print("[global_broadcast] lazy initialized")

    def compute_global_broadcast(self, obs: dict, out: dict):
        del obs
        if not bool(getattr(getattr(self.cfg, "global_broadcast", None), "enabled", True)):
            return None

        focus_context = out.get("focus_context")
        if not torch.is_tensor(focus_context):
            return None

        self.ensure_global_broadcast_ready()
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        thought_chain = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        plan_context = thought_chain.get("plan_context")
        if not torch.is_tensor(plan_context):
            plan_context = out.get("plan_context")

        broadcast = self.global_broadcast_gate(
            focus_context=focus_context,
            raw_focus_context=out.get("raw_focus_context"),
            active_thought=thought_chain.get("active_thought"),
            plan_context=plan_context,
            affect_latents=affect.get("affect_latents"),
            best_chain_score=thought_chain.get("best_chain_score"),
            predicted_affect_delta=thought_chain.get("predicted_affect_delta"),
            no_viable_chain=thought_chain.get("no_viable_chain"),
            panic_trigger=thought_chain.get("panic_trigger"),
        )
        out["broadcast"] = broadcast
        out["pre_broadcast_focus_context"] = focus_context
        out["focus_context"] = broadcast["broadcast_latent"]
        out["focus_context_source"] = "m10_global_broadcast"
        return broadcast

    def maybe_print_global_broadcast_trace(self, out: dict) -> None:
        cfg = getattr(self.cfg, "global_broadcast", None)
        every = int(getattr(cfg, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        bc = out.get("broadcast")
        if not isinstance(bc, dict):
            print(f"[global_broadcast step={self.global_step}] no broadcast output")
            return

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[global_broadcast step={self.global_step}] "
            f"source={bc.get('selected_source', '')} "
            f"priority={f(bc.get('priority')):.3f} "
            f"gate={f(bc.get('broadcast_gate')):.3f} "
            f"urgency={f(bc.get('urgency')):.3f} "
            f"allowed={f(bc.get('broadcast_allowed')):.0f}"
        )


__all__ = ["GlobalBroadcastRuntimeMixin"]
