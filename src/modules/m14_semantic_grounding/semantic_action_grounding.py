from __future__ import annotations

"""
M14 Semantic Action Grounding.

Architecture role:
    M14 translates the conscious branch state into action-level semantic intent.
    It does not replace low-level action heads. It receives M9/M7/M10/M11/M12/M13/M15
    products and builds an action packet:

        why act / why hold,
        what semantic intent is active,
        which source/target it is grounded in,
        how much the low-level policy should be allowed through.

The first bridge still only softens already proposed actions, but the contract is
now ready for later symbolic/semantic action selection.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class SemanticActionGroundingConfig:
    enabled: bool = True
    hold_threshold: float = 0.70
    verify_threshold: float = 0.55
    high_doubt_threshold: float = 0.50
    min_action_scale: float = 0.05
    soft_hold_scale: float = 0.25
    explore_threshold: float = 0.35
    positive_delta_threshold: float = 0.10
    emergency_threshold: float = 0.50


def _scalar_tensor(value, *, device: torch.device, default: float = 0.0) -> torch.Tensor:
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return torch.tensor([[default]], dtype=torch.float32, device=device)
            return value.detach().float().reshape(1, -1)[:, 0:1].to(device)
        if value is None:
            return torch.tensor([[default]], dtype=torch.float32, device=device)
        return torch.tensor([[float(value)]], dtype=torch.float32, device=device)
    except Exception:
        return torch.tensor([[default]], dtype=torch.float32, device=device)


def _device_from_out(out: Dict) -> torch.device:
    for value in out.values():
        if torch.is_tensor(value):
            return value.device
        if isinstance(value, dict):
            for nested in value.values():
                if torch.is_tensor(nested):
                    return nested.device
    return torch.device("cpu")


def _text_from(value, default: str = "") -> str:
    try:
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


class SemanticActionGrounding:
    INTENT_ORDER = ("hold", "verify", "avoid", "explore", "approach", "continue")

    def __init__(self, cfg: Optional[SemanticActionGroundingConfig] = None) -> None:
        self.cfg = cfg or SemanticActionGroundingConfig()

    def _semantic_scores(
        self,
        *,
        action_inhibition: torch.Tensor,
        verification_need: torch.Tensor,
        doubt: torch.Tensor,
        emergency_mode: torch.Tensor,
        curiosity: torch.Tensor,
        comfort: torch.Tensor,
        predicted_delta: torch.Tensor,
        best_chain_score: torch.Tensor,
        broadcast_urgency: torch.Tensor,
    ) -> torch.Tensor:
        c = self.cfg
        hold_score = torch.clamp(action_inhibition, 0.0, 1.0)
        verify_score = torch.clamp(verification_need * torch.maximum(doubt, torch.tensor([[float(c.high_doubt_threshold)]], device=doubt.device)), 0.0, 1.0)
        avoid_score = torch.clamp(emergency_mode + 0.35 * torch.relu(-predicted_delta), 0.0, 1.0)
        explore_score = torch.clamp(curiosity + 0.25 * broadcast_urgency + 0.15 * torch.relu(predicted_delta), 0.0, 1.0)
        approach_score = torch.clamp(best_chain_score * torch.relu(predicted_delta + 0.25) + 0.15 * comfort, 0.0, 1.0)
        continue_score = torch.clamp((1.0 - action_inhibition) * (0.50 + 0.50 * best_chain_score), 0.0, 1.0)
        return torch.cat([hold_score, verify_score, avoid_score, explore_score, approach_score, continue_score], dim=-1)

    def _choose_intent(self, scores: torch.Tensor) -> tuple[str, torch.Tensor, torch.Tensor]:
        idx = torch.argmax(scores, dim=-1)
        confidence = torch.max(scores, dim=-1, keepdim=True).values
        label = self.INTENT_ORDER[int(idx.detach().cpu().reshape(-1)[0].item())]
        return label, idx, confidence

    def _goal_text(
        self,
        *,
        semantic_intent: str,
        reason: str,
        target_source: str,
        memory_summary: str,
        selected_source: str,
    ) -> str:
        target = target_source or selected_source or "current_focus"
        if semantic_intent == "avoid":
            return f"avoid or reduce risk from {target}"
        if semantic_intent == "verify":
            return f"verify {target} before acting"
        if semantic_intent == "hold":
            return f"hold action because {reason}"
        if semantic_intent == "explore":
            return f"explore {target} while monitoring affect"
        if semantic_intent == "approach":
            return f"approach/use {target} because predicted outcome is positive"
        if memory_summary:
            return f"continue current action with autobiographical context: {memory_summary[:96]}"
        return "continue current action"

    def compute(self, out: Dict, *, manual_override: bool = False) -> Dict[str, torch.Tensor | str | bool]:
        device = _device_from_out(out)
        c = self.cfg
        meta = out.get("metacognition", {}) if isinstance(out.get("metacognition"), dict) else {}
        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        bc = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}
        memory = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}
        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        inner_speech = out.get("inner_speech", {}) if isinstance(out.get("inner_speech"), dict) else {}

        action_hold = _scalar_tensor(meta.get("action_hold"), device=device)
        verification_need = _scalar_tensor(meta.get("verification_need"), device=device)
        doubt = _scalar_tensor(meta.get("doubt"), device=device)
        should_hold = _scalar_tensor(meta.get("should_hold_action"), device=device)
        should_verify = _scalar_tensor(meta.get("should_verify"), device=device)
        high_doubt = _scalar_tensor(meta.get("high_doubt"), device=device)
        panic = _scalar_tensor(affect.get("panic_latent"), device=device)
        fear = _scalar_tensor(affect.get("fear_latent"), device=device)
        stress = _scalar_tensor(affect.get("stress_latent"), device=device)
        curiosity = _scalar_tensor(affect.get("curiosity_latent"), device=device)
        comfort = _scalar_tensor(affect.get("comfort_latent"), device=device)
        no_viable = _scalar_tensor(tc.get("no_viable_chain"), device=device)
        predicted_delta = _scalar_tensor(tc.get("predicted_affect_delta"), device=device)
        best_chain_score = _scalar_tensor(tc.get("best_chain_score"), device=device)
        broadcast_urgency = _scalar_tensor(bc.get("urgency"), device=device)
        broadcast_priority = _scalar_tensor(bc.get("priority"), device=device)
        memory_relevance = _scalar_tensor(memory.get("retrieval_relevance"), device=device)
        self_agency = _scalar_tensor(self_core.get("agency_score"), device=device)
        report_confidence = _scalar_tensor(inner_speech.get("confidence"), device=device)

        verify_and_doubt = torch.clamp(verification_need * torch.maximum(doubt, high_doubt), 0.0, 1.0)
        danger_signal = torch.clamp(torch.maximum(panic, torch.maximum(fear, stress)), 0.0, 1.0)
        emergency_mode = torch.clamp(torch.maximum(danger_signal, no_viable), 0.0, 1.0)
        action_inhibition = torch.clamp(
            torch.maximum(should_hold, action_hold)
            + 0.55 * verify_and_doubt
            + 0.65 * emergency_mode,
            0.0,
            1.0,
        )
        action_scale = torch.clamp(1.0 - action_inhibition, min=float(c.min_action_scale), max=1.0)
        soft_hold = torch.clamp(torch.minimum(action_scale, torch.tensor([[float(c.soft_hold_scale)]], device=device)), 0.0, 1.0)
        allow_action = (action_inhibition < float(c.hold_threshold)).float()
        verify_before_action = torch.maximum(should_verify, (verification_need > float(c.verify_threshold)).float())

        if bool(manual_override):
            reason = "manual_override_preserved"
            apply_scale = torch.ones_like(action_scale)
        elif float(emergency_mode.reshape(-1)[0].item()) > float(c.emergency_threshold):
            reason = "emergency_or_no_viable_chain"
            apply_scale = soft_hold
        elif float(verify_before_action.reshape(-1)[0].item()) > 0.5 and float(doubt.reshape(-1)[0].item()) > float(c.high_doubt_threshold):
            reason = "verify_before_action"
            apply_scale = soft_hold
        elif float(action_hold.reshape(-1)[0].item()) > float(c.hold_threshold):
            reason = "metacognitive_hold"
            apply_scale = soft_hold
        else:
            reason = "action_allowed"
            apply_scale = action_scale

        semantic_scores = self._semantic_scores(
            action_inhibition=action_inhibition,
            verification_need=verification_need,
            doubt=doubt,
            emergency_mode=emergency_mode,
            curiosity=curiosity,
            comfort=comfort,
            predicted_delta=predicted_delta,
            best_chain_score=best_chain_score,
            broadcast_urgency=broadcast_urgency,
        )
        semantic_intent, semantic_intent_idx, semantic_confidence = self._choose_intent(semantic_scores)

        selected_source = _text_from(bc.get("selected_source"), "")
        memory_summary = _text_from(memory.get("summary", memory.get("last_summary", "")), "")
        decoded_report = _text_from(out.get("decoded_report", inner_speech.get("text", "")), "")
        target_source = selected_source or ("autobiographical_memory" if memory_summary else "current_focus")
        goal_text = self._goal_text(
            semantic_intent=semantic_intent,
            reason=reason,
            target_source=target_source,
            memory_summary=memory_summary,
            selected_source=selected_source,
        )

        expected_outcome = torch.clamp(
            0.35 * torch.relu(predicted_delta)
            + 0.25 * best_chain_score
            + 0.20 * broadcast_priority
            + 0.10 * memory_relevance
            + 0.10 * self_agency
            - 0.35 * emergency_mode
            - 0.20 * doubt,
            -1.0,
            1.0,
        )
        grounding_confidence = torch.clamp(
            0.30 * semantic_confidence
            + 0.25 * best_chain_score
            + 0.20 * broadcast_priority
            + 0.15 * self_agency
            + 0.10 * report_confidence,
            0.0,
            1.0,
        )

        return {
            "semantic_intent": semantic_intent,
            "semantic_intent_idx": semantic_intent_idx,
            "semantic_intent_scores": semantic_scores,
            "semantic_confidence": semantic_confidence,
            "grounding_confidence": grounding_confidence,
            "goal_text": goal_text,
            "target_source": target_source,
            "selected_source": selected_source,
            "memory_summary": memory_summary,
            "decoded_report": decoded_report,
            "expected_outcome": expected_outcome,
            "action_scale": action_scale,
            "applied_action_scale": apply_scale,
            "action_inhibition": action_inhibition,
            "allow_action": allow_action,
            "verify_before_action": verify_before_action,
            "emergency_mode": emergency_mode,
            "manual_override_preserved": bool(manual_override),
            "predicted_affect_delta": predicted_delta,
            "best_chain_score": best_chain_score,
            "broadcast_urgency": broadcast_urgency,
            "memory_relevance": memory_relevance,
            "selected_chain_id": torch.zeros(1, dtype=torch.long, device=device),
            "reason": reason,
        }


__all__ = [
    "SemanticActionGrounding",
    "SemanticActionGroundingConfig",
]
