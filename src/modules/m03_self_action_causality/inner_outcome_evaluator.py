from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class InnerOutcomeEvaluatorConfig:
    """
    Step 3 after InnerActionDecoder.

    The system has:
        inner_mind_z            = selected imagined latent future
        inner_action_body/hand  = intention proposal

    Now it needs feedback:
        did the next real inner z move toward the imagined future?

    This module compares previous selected scenario_z with the next observed z_obj.
    It does not control action directly.
    """
    enabled: bool = True
    success_error_threshold: float = 0.12
    ema_decay: float = 0.97
    reward_scale: float = 0.10
    store_prev_prediction: bool = True


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


class InnerOutcomeEvaluator:
    """
    Feedback loop for inner coded planning.

    It answers:
        "Was my selected scenario_z a good prediction/intention?"

    The comparison:
        prev_predicted_z = inner_mind_z from previous step
        current_z        = current object latent z_obj
        outcome_error    = ||current_z - prev_predicted_z||

    Later this can become a training/reward signal for:
        InnerScenarioMind
        InnerActionDecoder
        NeuralEventDecoder
    """

    def __init__(self, cfg: Optional[InnerOutcomeEvaluatorConfig] = None):
        self.cfg = cfg or InnerOutcomeEvaluatorConfig()
        self.prev_prediction: Optional[Dict[str, Any]] = None
        self.error_ema: float = 0.0
        self.success_ema: float = 0.0
        self.count: int = 0
        self.last_result: Dict[str, Any] = {}

    def _norm_error(self, z_now: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        if z_now.ndim == 1:
            z_now = z_now.unsqueeze(0)
        if z_pred.ndim == 1:
            z_pred = z_pred.unsqueeze(0)

        if z_now.shape[-1] != z_pred.shape[-1]:
            d = min(z_now.shape[-1], z_pred.shape[-1])
            z_now = z_now[..., :d]
            z_pred = z_pred[..., :d]

        return (z_now.detach().float() - z_pred.detach().float()).norm(dim=-1).mean()

    def evaluate(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        z_now = obj.get("z_obj") if isinstance(obj, dict) else None
        if not torch.is_tensor(z_now):
            return {}

        result: Dict[str, Any] = {}

        if self.prev_prediction is not None:
            z_pred = self.prev_prediction.get("z_pred")
            if torch.is_tensor(z_pred):
                z_pred = z_pred.to(device=z_now.device, dtype=z_now.dtype)
                err_t = self._norm_error(z_now, z_pred)
                err = float(err_t.detach().cpu().item())
                success = 1.0 if err <= float(self.cfg.success_error_threshold) else 0.0

                if self.count <= 0:
                    self.error_ema = err
                    self.success_ema = success
                else:
                    d = float(self.cfg.ema_decay)
                    self.error_ema = d * self.error_ema + (1.0 - d) * err
                    self.success_ema = d * self.success_ema + (1.0 - d) * success

                self.count += 1
                reward = (float(self.cfg.success_error_threshold) - err) * float(self.cfg.reward_scale)

                result.update({
                    "inner_outcome_active": True,
                    "inner_outcome_error": err,
                    "inner_outcome_success": success,
                    "inner_outcome_error_ema": self.error_ema,
                    "inner_outcome_success_ema": self.success_ema,
                    "inner_outcome_reward": reward,
                    "inner_outcome_prev_sentence": str(self.prev_prediction.get("sentence", "")),
                    "inner_outcome_prev_slot_token": str(self.prev_prediction.get("slot_token", "")),
                })

        # Store current selected scenario_z as prediction for next step.
        if bool(getattr(self.cfg, "store_prev_prediction", True)):
            z_pred_next = obj.get("inner_mind_z")
            if torch.is_tensor(z_pred_next):
                self.prev_prediction = {
                    "z_pred": z_pred_next.detach().cpu(),
                    "sentence": str(obj.get("inner_mind_selected_sentence", "")),
                    "slot_token": str(obj.get("inner_mind_selected_slot_token", "")),
                    "score": _to_float(obj.get("inner_mind_selected_score"), 0.0),
                }

        self.last_result = result
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "prev_prediction": self.prev_prediction,
            "error_ema": float(self.error_ema),
            "success_ema": float(self.success_ema),
            "count": int(self.count),
            "last_result": self.last_result,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        cfg = state.get("cfg", {})
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if hasattr(self.cfg, k):
                    try:
                        setattr(self.cfg, k, v)
                    except Exception:
                        pass
        self.prev_prediction = state.get("prev_prediction", None)
        self.error_ema = float(state.get("error_ema", 0.0) or 0.0)
        self.success_ema = float(state.get("success_ema", 0.0) or 0.0)
        self.count = int(state.get("count", 0) or 0)
        self.last_result = state.get("last_result", {})
