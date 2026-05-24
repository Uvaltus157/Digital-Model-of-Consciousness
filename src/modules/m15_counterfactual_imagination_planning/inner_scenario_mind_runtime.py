from __future__ import annotations

import torch

from src.modules.m15_counterfactual_imagination_planning.inner_scenario_mind import InnerScenarioMind, InnerScenarioMindConfig


class InnerScenarioMindRuntimeMixin:
    """
    Runtime glue for first-order coded-world thinking.
    """

    def _ensure_inner_scenario_mind(self) -> None:
        if hasattr(self, "inner_scenario_mind") and self.inner_scenario_mind is not None:
            return

        cfg_mind = getattr(self.cfg, "inner_scenario_mind", None)
        self.inner_scenario_mind = InnerScenarioMind(InnerScenarioMindConfig(
            enabled=bool(getattr(cfg_mind, "enabled", True)),
            max_candidates=int(getattr(cfg_mind, "max_candidates", 4)),
            rollout_steps=int(getattr(cfg_mind, "rollout_steps", 6)),
            use_neural_decoder=bool(getattr(cfg_mind, "use_neural_decoder", True)),
            use_deterministic_decoder=bool(getattr(cfg_mind, "use_deterministic_decoder", True)),
            select_by=str(getattr(cfg_mind, "select_by", "score")),
            score_confidence_weight=float(getattr(cfg_mind, "score_confidence_weight", 1.0)),
            score_change_weight=float(getattr(cfg_mind, "score_change_weight", 0.35)),
            score_contact_weight=float(getattr(cfg_mind, "score_contact_weight", 0.25)),
            score_stability_weight=float(getattr(cfg_mind, "score_stability_weight", 0.50)),
            score_uncertainty_penalty=float(getattr(cfg_mind, "score_uncertainty_penalty", 0.40)),
        ))

    def update_inner_scenario_mind(self, obj: dict) -> dict:
        try:
            cfg_mind = getattr(self.cfg, "inner_scenario_mind", None)
            if not bool(getattr(cfg_mind, "enabled", True)):
                return obj
            if not hasattr(self, "event_latent_memory") or self.event_latent_memory is None:
                return obj

            self._ensure_inner_scenario_mind()

            ref = obj.get("z_obj") if isinstance(obj, dict) else None
            device = ref.device if torch.is_tensor(ref) else getattr(self, "device", "cpu")
            dtype = ref.dtype if torch.is_tensor(ref) else torch.float32

            thought = self.inner_scenario_mind.think(
                event_memory=self.event_latent_memory,
                neural_decoder=getattr(self, "neural_event_decoder", None),
                device=device,
                dtype=dtype,
            )
            if isinstance(thought, dict) and thought:
                obj.update(thought)
            return obj
        except Exception as e:
            if not hasattr(self, "_inner_scenario_mind_warned"):
                print(f"[inner_scenario_mind] update failed: {e}")
                self._inner_scenario_mind_warned = True
            return obj
