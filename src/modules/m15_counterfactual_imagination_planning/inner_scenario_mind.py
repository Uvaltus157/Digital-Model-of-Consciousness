from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class InnerScenarioMindConfig:
    """
    First implementation of internal thinking/planning over the coded world.

    It operates on scenario_z, not on RGB.
    """
    enabled: bool = True
    max_candidates: int = 4
    rollout_steps: int = 6
    use_neural_decoder: bool = True
    use_deterministic_decoder: bool = True
    select_by: str = "score"

    score_confidence_weight: float = 1.0
    score_change_weight: float = 0.35
    score_contact_weight: float = 0.25
    score_stability_weight: float = 0.50
    score_uncertainty_penalty: float = 0.40


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


class InnerScenarioMind:
    """
    Internal coded-world simulator.

    Input:
        event_latent_memory
        optional NeuralEventDecoder

    Output:
        selected scenario_z candidate.
    """

    def __init__(self, cfg: Optional[InnerScenarioMindConfig] = None):
        self.cfg = cfg or InnerScenarioMindConfig()
        self.last_candidates: List[Dict[str, Any]] = []
        self.last_selected: Optional[Dict[str, Any]] = None
        self.tick: int = 0

    def _events_from_memory(self, event_memory: Any) -> List[Dict[str, Any]]:
        try:
            events = list(getattr(event_memory, "events", []) or [])
            return events[-max(1, int(self.cfg.max_candidates)):]
        except Exception:
            return []

    def _candidate_from_event(
        self,
        event: Dict[str, Any],
        *,
        neural_decoder: Any = None,
        device=None,
        dtype=torch.float32,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(event, dict):
            return None

        z0 = event.get("z_before")
        z1 = event.get("z_after")
        if not torch.is_tensor(z0) and not torch.is_tensor(z1):
            return None

        if torch.is_tensor(z0):
            z0 = z0.detach().to(device=device, dtype=dtype)
            if z0.ndim == 1:
                z0 = z0.unsqueeze(0)
        if torch.is_tensor(z1):
            z1 = z1.detach().to(device=device, dtype=dtype)
            if z1.ndim == 1:
                z1 = z1.unsqueeze(0)

        z_pred = None
        used = "deterministic"
        if bool(self.cfg.use_neural_decoder) and neural_decoder is not None:
            try:
                pred = neural_decoder.decode_event(event, device=device, dtype=dtype)
                if pred and torch.is_tensor(pred.get("neural_pred_z_after")):
                    z_pred = pred["neural_pred_z_after"].detach()
                    used = "neural"
            except Exception:
                z_pred = None

        if z_pred is None:
            z_pred = z1 if torch.is_tensor(z1) else z0
            used = "deterministic"

        if z0 is None:
            z0 = z_pred

        steps = max(1, int(self.cfg.rollout_steps))
        z_seq = []
        for i in range(steps):
            a = float(i + 1) / float(steps)
            z_seq.append(((1.0 - a) * z0 + a * z_pred).detach())
        z_sequence = torch.cat(z_seq, dim=0)

        sentence = str(
            event.get("semantic_sentence", "")
            or event.get("semantic_code_sentence", "")
            or event.get("sentence", "")
            or ""
        )
        roles = event.get("sentence_roles", event.get("roles", {}))
        if not isinstance(roles, dict):
            roles = {}

        return {
            "source": "event",
            "decoder": used,
            "z_sequence": z_sequence,
            "z_start": z0.detach(),
            "z_end": z_pred.detach(),
            "sentence_chain": [sentence],
            "roles": roles,
            "event": event,
            "slot_token": str(event.get("slot_token", "")),
            "slot": int(event.get("slot", 0) or 0),
        }

    def _score_candidate(self, cand: Dict[str, Any]) -> float:
        ev = cand.get("event", {})
        roles = cand.get("roles", {})
        z_seq = cand.get("z_sequence")

        conf = _f(ev.get("confidence"), 0.0)
        contact = max(_f(ev.get("contact_norm"), 0.0), _f(ev.get("touch_strength"), 0.0))
        delta = _f(ev.get("delta_norm"), 0.0)

        stability = 0.0
        try:
            if torch.is_tensor(z_seq) and z_seq.shape[0] > 1:
                dz = z_seq[1:] - z_seq[:-1]
                stability = 1.0 / (1.0 + float(dz.float().norm(dim=-1).mean().detach().cpu().item()))
        except Exception:
            stability = 0.0

        context = str(roles.get("context", "latent") if isinstance(roles, dict) else "latent")
        context_bonus = 0.15 if context == "contact" else (0.10 if context == "self_action" else (0.05 if context == "dream" else 0.0))

        score = (
            float(self.cfg.score_confidence_weight) * conf
            + float(self.cfg.score_change_weight) * delta
            + float(self.cfg.score_contact_weight) * contact
            + float(self.cfg.score_stability_weight) * stability
            + context_bonus
        )
        if delta > 0.05 and conf < 0.10:
            score -= float(self.cfg.score_uncertainty_penalty)
        return float(score)

    def think(self, *, event_memory: Any, neural_decoder: Any = None, device=None, dtype=torch.float32) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        self.tick += 1
        events = self._events_from_memory(event_memory)
        candidates: List[Dict[str, Any]] = []

        for ev in reversed(events):
            if len(candidates) >= int(self.cfg.max_candidates):
                break
            cand = self._candidate_from_event(ev, neural_decoder=neural_decoder, device=device, dtype=dtype)
            if cand is None:
                continue
            cand["score"] = self._score_candidate(cand)
            candidates.append(cand)

        self.last_candidates = candidates
        if not candidates:
            self.last_selected = None
            return {
                "inner_mind_active": False,
                "inner_mind_candidate_count": 0,
                "inner_mind_tick": self.tick,
            }

        selected = candidates[0] if str(self.cfg.select_by).lower().strip() == "latest" else max(candidates, key=lambda c: float(c.get("score", 0.0)))
        self.last_selected = selected

        z_seq = selected.get("z_sequence")
        if torch.is_tensor(z_seq) and z_seq.numel() > 0:
            idx = (self.tick - 1) % max(1, int(z_seq.shape[0]))
            z_now = z_seq[idx:idx + 1].detach()
        else:
            idx = 0
            z_now = None

        return {
            "inner_mind_active": True,
            "inner_mind_tick": self.tick,
            "inner_mind_candidate_count": len(candidates),
            "inner_mind_selected_score": float(selected.get("score", 0.0)),
            "inner_mind_selected_decoder": str(selected.get("decoder", "")),
            "inner_mind_selected_sentence": (selected.get("sentence_chain") or [""])[0],
            "inner_mind_selected_slot_token": str(selected.get("slot_token", "")),
            "inner_mind_selected_slot": int(selected.get("slot", 0)),
            "inner_mind_cursor": int(idx),
            "inner_mind_z": z_now,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "tick": int(self.tick),
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
        self.tick = int(state.get("tick", 0) or 0)
