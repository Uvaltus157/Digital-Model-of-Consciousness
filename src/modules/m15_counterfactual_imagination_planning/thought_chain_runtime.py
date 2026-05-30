from __future__ import annotations

import torch

from src.modules.m15_counterfactual_imagination_planning.thought_chain_controller import (
    ThoughtChainController,
    ThoughtChainControllerConfig,
)


class ThoughtChainRuntimeMixin:
    def ensure_thought_chain_ready(self) -> None:
        if hasattr(self, "thought_chain_controller") and self.thought_chain_controller is not None:
            return

        self_bound_dim = (
            int(getattr(self.cfg.self_core, "self_dim", 128))
            + int(getattr(self.cfg.self_core, "focus_context_dim", 256))
            + int(getattr(self.cfg.self_core, "affect_latent_dim", 12))
        )
        cfg = ThoughtChainControllerConfig(
            enabled=bool(getattr(getattr(self.cfg, "thought_chain", None), "enabled", True)),
            self_bound_context_dim=self_bound_dim,
            subjective_affect_dim=16,
            focus_context_dim=int(getattr(self.cfg.self_core, "focus_context_dim", 256)),
            affect_latent_dim=int(getattr(self.cfg.self_core, "affect_latent_dim", 12)),
            hidden_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "hidden_dim", 256)),
            thought_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "thought_dim", 128)),
            plan_context_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "plan_context_dim", 256)),
            chain_len=int(getattr(getattr(self.cfg, "thought_chain", None), "chain_len", 4)),
        )
        self.thought_chain_controller = ThoughtChainController(cfg).to(self.device)
        try:
            if hasattr(self, "module_training_gate"):
                self.module_training_gate.apply()
                self.rebuild_optimizer_from_trainable_modules()
            else:
                self.optimizer.add_param_group({"params": self.thought_chain_controller.parameters()})
        except Exception as e:
            print(f"[thought_chain] lazy optimizer attach skipped: {e}")
        print("[thought_chain] lazy initialized")

    def compute_thought_chain(self, obs: dict, out: dict, *, pre_self_binding: bool = True):
        del obs
        if not bool(getattr(getattr(self.cfg, "thought_chain", None), "enabled", True)):
            return None

        focus_context = out.get("focus_context")
        if not torch.is_tensor(focus_context):
            return None

        self.ensure_thought_chain_ready()

        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}

        thought_chain = self.thought_chain_controller(
            focus_context=focus_context,
            affect_latents=affect.get("affect_latents"),
            self_bound_context=None if pre_self_binding else self_core.get("self_bound_context"),
            subjective_affect_state=None if pre_self_binding else self_core.get("subjective_affect_state"),
        )
        thought_chain["source_present"] = {
            "focus_context": torch.tensor([1.0 if torch.is_tensor(focus_context) else 0.0], device=self.device),
            "affect_latents": torch.tensor([1.0 if torch.is_tensor(affect.get("affect_latents")) else 0.0], device=self.device),
            "self_bound_context": torch.tensor([1.0 if (not pre_self_binding and torch.is_tensor(self_core.get("self_bound_context"))) else 0.0], device=self.device),
            "subjective_affect_state": torch.tensor([1.0 if (not pre_self_binding and torch.is_tensor(self_core.get("subjective_affect_state"))) else 0.0], device=self.device),
        }
        thought_chain["stage"] = "pre_self_binding" if pre_self_binding else "post_self_binding"

        out["thought_chain"] = thought_chain
        out["active_thought"] = thought_chain.get("active_thought_packet", {})
        out["plan_context"] = thought_chain["plan_context"]

        enhanced_focus = thought_chain.get("enhanced_focus_context")
        if pre_self_binding and torch.is_tensor(enhanced_focus):
            out["raw_focus_context"] = focus_context
            out["focus_context"] = enhanced_focus
            out["focus_context_source"] = "m15_enhanced_best_chain"
        return thought_chain

    def maybe_print_thought_chain_trace(self, out: dict) -> None:
        cfg = getattr(self.cfg, "thought_chain", None)
        every = int(getattr(cfg, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        tc = out.get("thought_chain")
        if not isinstance(tc, dict):
            print(f"[thought_chain step={self.global_step}] no thought_chain output")
            return

        metrics = tc.get("thought_chain_metrics", {}) if isinstance(tc.get("thought_chain_metrics"), dict) else {}

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[thought_chain step={self.global_step}] "
            f"stage={tc.get('stage', '')} "
            f"best={f(tc.get('best_chain_score')):.3f} "
            f"affect_delta={f(tc.get('predicted_affect_delta')):.3f} "
            f"panic={f(tc.get('panic_trigger')):.0f} "
            f"no_viable={f(tc.get('no_viable_chain')):.0f} "
            f"stability={f(metrics.get('stability')):.3f} "
            f"urgency={f(metrics.get('urgency')):.3f} "
            f"planning_readiness={f(metrics.get('planning_readiness')):.3f} "
            f"plan_norm={f(tc.get('plan_context').norm(dim=-1).mean() if torch.is_tensor(tc.get('plan_context')) else None):.3f}"
        )


__all__ = ["ThoughtChainRuntimeMixin"]
