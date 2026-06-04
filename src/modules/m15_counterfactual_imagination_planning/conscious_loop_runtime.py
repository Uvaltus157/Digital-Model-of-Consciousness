from __future__ import annotations

"""
ConsciousLoopRuntimeMixin v2

More correct self-reflective consciousness loop:

    M5 focus_context
        -> M10 global broadcast
        -> M9 self binding
        -> M11 affect / M12 optional metacognitive gate
        -> M15 post-self thought chain
        -> gated next_focus_context_seed
        -> M5 FocusFeedbackBoundary
        -> workspace_seed + preconscious_seed

This does not inject raw self_state into M5.
"""

from typing import Any, Dict, Optional, Tuple

import torch


def _cfg_value(root: Any, dotted: str, default: Any) -> Any:
    cur = root
    for part in dotted.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


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


class ConsciousLoopRuntimeMixin:
    def ensure_conscious_loop_ready(self) -> None:
        if bool(getattr(self, "_conscious_loop_ready", False)):
            return
        self._conscious_loop_ready = True
        self._conscious_loop_next_focus_seed = None
        self._conscious_loop_next_focus_gate = None
        self._conscious_loop_seed_step = -1
        self._conscious_loop_last_packet = None
        print("[conscious_loop] initialized v2 | M15 post-self -> M5 FocusFeedbackBoundary")

    def _conscious_loop_cfg(self) -> Any:
        return getattr(getattr(self, "cfg", None), "conscious_loop", None)

    def _conscious_loop_enabled(self) -> bool:
        return bool(_cfg_value(getattr(self, "cfg", None), "conscious_loop.enabled", True))

    def get_conscious_loop_focus_seed(self, stage: str = "model_step") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.ensure_conscious_loop_ready()
        if not self._conscious_loop_enabled():
            return None, None

        cfg = self._conscious_loop_cfg()
        apply_stage = str(getattr(cfg, "apply_stage", "both"))
        if apply_stage not in ("both", "pre_observe", "main"):
            apply_stage = "both"
        if apply_stage != "both" and str(stage) != apply_stage:
            return None, None

        seed = getattr(self, "_conscious_loop_next_focus_seed", None)
        gate = getattr(self, "_conscious_loop_next_focus_gate", None)
        if not torch.is_tensor(seed):
            return None, None
        return seed.detach(), gate.detach() if torch.is_tensor(gate) else gate

    def _conscious_loop_gate_from_outputs(self, out: Dict, post_chain: Dict) -> torch.Tensor:
        cfg = self._conscious_loop_cfg()
        base_gain = float(getattr(cfg, "feedback_gain", 0.22))
        min_gate = float(getattr(cfg, "min_gate", 0.00))
        max_gate = float(getattr(cfg, "max_gate", 0.22))
        require_self_binding = bool(getattr(cfg, "require_self_binding", True))
        use_metacognition_gate = bool(getattr(cfg, "use_metacognition_gate", True))

        sc = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        meta = out.get("metacognition", {}) if isinstance(out.get("metacognition"), dict) else {}

        focus_binding = _scalar(sc.get("focus_binding_score"), 0.0)
        affect_binding = _scalar(sc.get("affect_binding_score"), 0.0)
        continuity = _scalar(sc.get("self_continuity_score"), 0.0)
        agency = _scalar(sc.get("agency_score"), 0.0)

        best_score = _scalar(post_chain.get("best_chain_score"), 0.0)
        thought_gate = _scalar(post_chain.get("focus_update_gate"), 0.0)
        no_viable = _scalar(post_chain.get("no_viable_chain"), 0.0)
        panic = _scalar(post_chain.get("panic_trigger"), 0.0)

        if use_metacognition_gate and isinstance(meta, dict):
            doubt = _scalar(meta.get("doubt"), 0.0)
            meta_conf = _scalar(meta.get("metacognitive_confidence"), 0.5)
        else:
            doubt = 0.0
            meta_conf = 0.5

        if require_self_binding and focus_binding <= 0.01:
            gate_value = 0.0
        else:
            self_relevance = (
                0.45 * focus_binding
                + 0.22 * affect_binding
                + 0.20 * continuity
                + 0.13 * agency
            )
            chain_quality = 0.60 * best_score + 0.40 * thought_gate
            safety = max(0.0, 1.0 - 0.55 * no_viable - 0.45 * panic - 0.30 * doubt)
            meta_factor = 0.65 + 0.35 * meta_conf
            gate_value = base_gain * self_relevance * chain_quality * safety * meta_factor

        gate_value = max(min_gate, min(max_gate, float(gate_value)))
        device = None
        focus = out.get("focus_context")
        if torch.is_tensor(focus):
            device = focus.device
        return torch.tensor([[gate_value]], dtype=torch.float32, device=device)

    def compute_conscious_loop_feedback(self, obs: Dict, out: Dict) -> Optional[Dict]:
        """
        Run late in life_step, after M9 and after current affect is available.

        It calls M15 with pre_self_binding=False and stores the enhanced focus
        seed for the next M5 call. M5 receives the seed through its
        FocusFeedbackBoundary, not by direct raw latent concatenation.
        """
        del obs
        self.ensure_conscious_loop_ready()
        if not self._conscious_loop_enabled():
            return None
        if not hasattr(self, "compute_thought_chain"):
            return None
        if not torch.is_tensor(out.get("focus_context")):
            return None

        sc = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        if not torch.is_tensor(sc.get("self_bound_context")):
            return None

        pre_self_chain = out.get("thought_chain") if isinstance(out.get("thought_chain"), dict) else None
        if pre_self_chain is not None:
            out["pre_self_thought_chain"] = pre_self_chain

        post_chain = self.compute_thought_chain({}, out, pre_self_binding=False)
        if not isinstance(post_chain, dict):
            return None

        seed = post_chain.get("enhanced_focus_context")
        if not torch.is_tensor(seed):
            return None

        gate = self._conscious_loop_gate_from_outputs(out, post_chain).detach()
        seed = seed.detach()

        self._conscious_loop_next_focus_seed = seed
        self._conscious_loop_next_focus_gate = gate
        self._conscious_loop_seed_step = int(getattr(self, "global_step", -1))

        packet = {
            "enabled": True,
            "version": "v2_focus_feedback_boundary",
            "stage": "post_self_after_affect",
            "loop": "M5->M10->M9->M11/M12->M15->M5",
            "source": "m15_post_self_enhanced_focus_context",
            "seed_step": self._conscious_loop_seed_step,
            "feedback_gate": gate,
            "seed_norm": seed.norm(dim=-1, keepdim=True).detach() if seed.ndim >= 2 else seed.norm().reshape(1, 1).detach(),
            "focus_binding_score": sc.get("focus_binding_score"),
            "affect_binding_score": sc.get("affect_binding_score"),
            "self_continuity_score": sc.get("self_continuity_score"),
            "agency_score": sc.get("agency_score"),
            "post_self_thought_chain": post_chain,
            "next_focus_context_seed": seed,
            "target_m5_boundary": "FocusFeedbackBoundary(workspace_seed + preconscious_seed)",
        }

        out["conscious_loop"] = packet
        out["next_focus_context_seed"] = seed
        out["next_focus_context_seed_gate"] = gate
        self._conscious_loop_last_packet = packet
        return packet

    def maybe_print_conscious_loop_trace(self, out: Dict) -> None:
        cfg = self._conscious_loop_cfg()
        every = int(getattr(cfg, "print_every_steps", 30))
        if every <= 0:
            return
        if int(getattr(self, "global_step", 0)) % every != 0:
            return

        packet = out.get("conscious_loop") if isinstance(out, dict) else None
        if not isinstance(packet, dict):
            seed = getattr(self, "_conscious_loop_next_focus_seed", None)
            print(
                f"[conscious_loop v2 step={getattr(self, 'global_step', 0)}] "
                f"no post-self packet | seed_pending={int(torch.is_tensor(seed))}"
            )
            return

        print(
            f"[conscious_loop v2 step={getattr(self, 'global_step', 0)}] "
            f"gate={_scalar(packet.get('feedback_gate'), 0.0):.4f} "
            f"seed_norm={_scalar(packet.get('seed_norm'), 0.0):.3f} "
            f"focus_bind={_scalar(packet.get('focus_binding_score'), 0.0):.3f} "
            f"affect_bind={_scalar(packet.get('affect_binding_score'), 0.0):.3f} "
            f"target=FocusFeedbackBoundary"
        )


__all__ = ["ConsciousLoopRuntimeMixin"]
