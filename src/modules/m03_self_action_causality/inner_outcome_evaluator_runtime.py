from __future__ import annotations

import torch

from src.modules.m03_self_action_causality.inner_outcome_evaluator import InnerOutcomeEvaluator, InnerOutcomeEvaluatorConfig


class InnerOutcomeEvaluatorRuntimeMixin:
    """
    Runtime glue for evaluating internal scenario outcomes.

    It compares previous selected inner_mind_z with current z_obj and writes
    outcome metrics into inner_object.
    """

    def _ensure_inner_outcome_evaluator(self) -> None:
        if hasattr(self, "inner_outcome_evaluator") and self.inner_outcome_evaluator is not None:
            return

        cfg_eval = getattr(self.cfg, "inner_outcome_evaluator", None)
        self.inner_outcome_evaluator = InnerOutcomeEvaluator(InnerOutcomeEvaluatorConfig(
            enabled=bool(getattr(cfg_eval, "enabled", True)),
            success_error_threshold=float(getattr(cfg_eval, "success_error_threshold", 0.12)),
            ema_decay=float(getattr(cfg_eval, "ema_decay", 0.97)),
            reward_scale=float(getattr(cfg_eval, "reward_scale", 0.10)),
            store_prev_prediction=bool(getattr(cfg_eval, "store_prev_prediction", True)),
        ))

    def update_inner_outcome_evaluator(self, obj: dict) -> dict:
        try:
            cfg_eval = getattr(self.cfg, "inner_outcome_evaluator", None)
            if not bool(getattr(cfg_eval, "enabled", True)):
                return obj

            if not isinstance(obj, dict):
                return obj

            self._ensure_inner_outcome_evaluator()
            res = self.inner_outcome_evaluator.evaluate(obj)
            if isinstance(res, dict) and res:
                device = getattr(self, "device", "cpu")
                ref = obj.get("z_obj")
                if torch.is_tensor(ref):
                    device = ref.device
                    dtype = ref.dtype
                else:
                    dtype = torch.float32

                for k, v in list(res.items()):
                    if isinstance(v, (float, int, bool)):
                        obj[k] = torch.tensor([[float(v)]], device=device, dtype=dtype)
                    else:
                        obj[k] = v
            return obj
        except Exception as e:
            if not hasattr(self, "_inner_outcome_evaluator_warned"):
                print(f"[inner_outcome_evaluator] update failed: {e}")
                self._inner_outcome_evaluator_warned = True
            return obj
