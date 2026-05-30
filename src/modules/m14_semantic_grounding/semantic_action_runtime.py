from __future__ import annotations

import torch

from src.modules.m14_semantic_grounding.semantic_action_grounding import (
    SemanticActionGrounding,
    SemanticActionGroundingConfig,
)


class SemanticActionRuntimeMixin:
    def ensure_semantic_action_ready(self) -> None:
        if hasattr(self, "semantic_action_grounding") and self.semantic_action_grounding is not None:
            return
        cfg_obj = getattr(self.cfg, "semantic_action", None)
        cfg = SemanticActionGroundingConfig(
            enabled=bool(getattr(cfg_obj, "enabled", True)),
            hold_threshold=float(getattr(cfg_obj, "hold_threshold", 0.70)),
            verify_threshold=float(getattr(cfg_obj, "verify_threshold", 0.55)),
            high_doubt_threshold=float(getattr(cfg_obj, "high_doubt_threshold", 0.50)),
            min_action_scale=float(getattr(cfg_obj, "min_action_scale", 0.05)),
            soft_hold_scale=float(getattr(cfg_obj, "soft_hold_scale", 0.25)),
            explore_threshold=float(getattr(cfg_obj, "explore_threshold", 0.35)),
            positive_delta_threshold=float(getattr(cfg_obj, "positive_delta_threshold", 0.10)),
            emergency_threshold=float(getattr(cfg_obj, "emergency_threshold", 0.50)),
        )
        self.semantic_action_grounding = SemanticActionGrounding(cfg)
        print("[semantic_action] initialized")

    def compute_conscious_action(self, obs: dict, out: dict):
        del obs
        cfg_obj = getattr(self.cfg, "semantic_action", None)
        if not bool(getattr(cfg_obj, "enabled", True)):
            return None
        self.ensure_semantic_action_ready()
        manual = bool(getattr(self, "_ipc_manual_actions_enabled", False))
        action = self.semantic_action_grounding.compute(out, manual_override=manual)
        out["conscious_action"] = action
        out["semantic_action"] = action

        # M14 bridge: do not replace policy heads yet. Only soften unsafe or
        # doubtful conscious actions after M12 has requested hold/verify, while
        # preserving full semantic intent metadata for downstream modules.
        scale = action.get("applied_action_scale")
        if torch.is_tensor(scale) and not manual:
            for key in ("embodied_targets", "hand_ctrl", "leg_ctrl"):
                value = out.get(key)
                if torch.is_tensor(value):
                    try:
                        out[key] = value * scale.to(value.device, value.dtype)
                    except Exception:
                        pass
            action["applied_to_policy"] = True
        else:
            action["applied_to_policy"] = False
        return action

    def maybe_print_semantic_action_trace(self, out: dict) -> None:
        cfg_obj = getattr(self.cfg, "semantic_action", None)
        every = int(getattr(cfg_obj, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        action = out.get("conscious_action")
        if not isinstance(action, dict):
            print(f"[semantic_action step={self.global_step}] no conscious_action output")
            return

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[semantic_action step={self.global_step}] "
            f"intent={action.get('semantic_intent', '')} "
            f"target={action.get('target_source', '')} "
            f"scale={f(action.get('applied_action_scale')):.3f} "
            f"inhibition={f(action.get('action_inhibition')):.3f} "
            f"verify={f(action.get('verify_before_action')):.0f} "
            f"emergency={f(action.get('emergency_mode')):.0f} "
            f"outcome={f(action.get('expected_outcome')):.3f} "
            f"reason={action.get('reason', '')} | "
            f"{action.get('goal_text', '')}"
        )


__all__ = ["SemanticActionRuntimeMixin"]
