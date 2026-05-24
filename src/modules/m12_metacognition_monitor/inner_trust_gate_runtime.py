from __future__ import annotations

import torch

from src.modules.m12_metacognition_monitor.inner_trust_gate import InnerTrustGate, InnerTrustGateConfig


class InnerTrustGateRuntimeMixin:
    """
    Runtime glue for trust-gating inner action intention.

    It can either:
        - only report recommended trust/alpha; or
        - apply safe low-alpha blend to out["embodied_targets"] / out["hand_ctrl"].
    """

    def _ensure_inner_trust_gate(self) -> None:
        if hasattr(self, "inner_trust_gate") and self.inner_trust_gate is not None:
            return

        cfg = getattr(self.cfg, "inner_trust_gate", None)
        self.inner_trust_gate = InnerTrustGate(InnerTrustGateConfig(
            enabled=bool(getattr(cfg, "enabled", True)),
            min_success_ema=float(getattr(cfg, "min_success_ema", 0.55)),
            max_error_ema=float(getattr(cfg, "max_error_ema", 0.18)),
            min_action_confidence=float(getattr(cfg, "min_action_confidence", 0.25)),
            max_alpha=float(getattr(cfg, "max_alpha", 0.08)),
            warmup_steps=int(getattr(cfg, "warmup_steps", 500)),
            allow_policy_blend=bool(getattr(cfg, "allow_policy_blend", False)),
            apply_to_policy=bool(getattr(cfg, "apply_to_policy", False)),
        ))

    def update_inner_trust_gate(self, obj: dict, out: dict | None = None) -> dict:
        try:
            cfg = getattr(self.cfg, "inner_trust_gate", None)
            if not bool(getattr(cfg, "enabled", True)):
                return obj
            if not isinstance(obj, dict):
                return obj

            self._ensure_inner_trust_gate()
            gate = self.inner_trust_gate.compute(obj)
            if not gate:
                return obj

            ref = obj.get("z_obj")
            device = ref.device if torch.is_tensor(ref) else getattr(self, "device", "cpu")
            dtype = ref.dtype if torch.is_tensor(ref) else torch.float32

            for k, v in gate.items():
                if isinstance(v, (float, int, bool)):
                    obj[k] = torch.tensor([[float(v)]], device=device, dtype=dtype)
                else:
                    obj[k] = v

            # Optional real blending. Default config keeps this disabled.
            apply = bool(getattr(cfg, "apply_to_policy", False))
            allowed = bool(gate.get("inner_trust_allowed", False))
            alpha = float(gate.get("inner_trust_alpha", 0.0))

            if out is not None and apply and allowed and alpha > 0.0:
                body = obj.get("inner_action_body")
                hand = obj.get("inner_action_hand")

                blend_trace = {
                    "alpha": float(alpha),
                    "body_before_norm": 0.0,
                    "body_intent_norm": 0.0,
                    "body_after_norm": 0.0,
                    "hand_before_norm": 0.0,
                    "hand_intent_norm": 0.0,
                    "hand_after_norm": 0.0,
                }

                if torch.is_tensor(body) and "embodied_targets" in out and torch.is_tensor(out["embodied_targets"]):
                    blend_trace["body_before_norm"] = float(out["embodied_targets"].detach().float().norm(dim=-1).mean().cpu().item())
                    body2 = body.to(out["embodied_targets"].device, out["embodied_targets"].dtype)
                    blend_trace["body_intent_norm"] = float(body2.detach().float().norm(dim=-1).mean().cpu().item())
                    out["embodied_targets"] = (1.0 - alpha) * out["embodied_targets"] + alpha * body2
                    blend_trace["body_after_norm"] = float(out["embodied_targets"].detach().float().norm(dim=-1).mean().cpu().item())

                if torch.is_tensor(hand) and "hand_ctrl" in out and torch.is_tensor(out["hand_ctrl"]):
                    blend_trace["hand_before_norm"] = float(out["hand_ctrl"].detach().float().norm(dim=-1).mean().cpu().item())
                    hand2 = hand.to(out["hand_ctrl"].device, out["hand_ctrl"].dtype)
                    blend_trace["hand_intent_norm"] = float(hand2.detach().float().norm(dim=-1).mean().cpu().item())
                    out["hand_ctrl"] = (1.0 - alpha) * out["hand_ctrl"] + alpha * hand2
                    blend_trace["hand_after_norm"] = float(out["hand_ctrl"].detach().float().norm(dim=-1).mean().cpu().item())

                obj["inner_action_blend_trace"] = blend_trace
                obj["inner_trust_applied_to_policy"] = torch.tensor([[1.0]], device=device, dtype=dtype)
            else:
                obj["inner_action_blend_trace"] = {
                    "alpha": float(alpha),
                    "applied": False,
                    "reason": str(gate.get("inner_trust_reason", "")),
                }
                obj["inner_trust_applied_to_policy"] = torch.tensor([[0.0]], device=device, dtype=dtype)

            return obj

        except Exception as e:
            if not hasattr(self, "_inner_trust_gate_warned"):
                print(f"[inner_trust_gate] update failed: {e}")
                self._inner_trust_gate_warned = True
            return obj
