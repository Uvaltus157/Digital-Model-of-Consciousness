from __future__ import annotations

"""
Energy Resonator — experimental bivalent social-affect process.

This is a small architecture-safe simulator for the two-agent process:

- agent_he / self agent acts.
- agent_she / other agent receives positive affect.
- animal mode: he acts for his own positive emotion; her positive emotion is a side-effect.
- conscious mode: he keeps her positive emotion as the main goal and tolerates negative cost.
- resonance: he receives positive emotion from her positive emotion while tolerating cost.

The module is intentionally pure Python + optional torch tensors.
It does not require MuJoCo, Open3D, PyQt, or the main world model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class EmotionState:
    positive: float = 0.0
    negative: float = 0.0
    energy: float = 0.5

    @property
    def valence(self) -> float:
        return float(self.positive - self.negative)


@dataclass
class AgentState:
    name: str
    emotion: EmotionState = field(default_factory=EmotionState)
    goal: str = ""
    drive: float = 0.5
    tolerance: float = 0.5
    empathy: float = 0.5


@dataclass
class EnergyResonatorConfig:
    enabled: bool = True

    # If False, the process only writes out["energy_resonator"] and does not
    # change out["focus_context"]. This is the safest default for integration.
    blend_into_focus_context: bool = False

    # Small value because focus_context may be used by many downstream modules.
    focus_blend_weight: float = 0.015

    default_mode: str = "animal"  # "animal" or "conscious"
    drive: float = 0.65
    empathy: float = 0.45
    tolerance: float = 0.45
    cost: float = 0.45
    her_need: float = 0.70
    decay: float = 0.035


@dataclass
class EnergyResonatorStep:
    step: int
    mode: str
    action_intensity: float
    she_received_positive: float
    he_self_positive: float
    he_empathic_positive: float
    he_negative_cost: float
    conscious_alignment: float
    resonance_index: float
    m5_other_agent_focus: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "mode": self.mode,
            "action_intensity": self.action_intensity,
            "she_received_positive": self.she_received_positive,
            "he_self_positive": self.he_self_positive,
            "he_empathic_positive": self.he_empathic_positive,
            "he_negative_cost": self.he_negative_cost,
            "conscious_alignment": self.conscious_alignment,
            "resonance_index": self.resonance_index,
            "m5_other_agent_focus": dict(self.m5_other_agent_focus),
        }


class EnergyResonator:
    """
    Pure process model.

    The class stores internal state so the process has continuity between life steps.
    It can be called from runtime and converted to:
        out["energy_resonator"]
        out["other_agent_focus"]
        out["energy_resonator_latent"]
    """

    def __init__(self, cfg: Optional[EnergyResonatorConfig] = None) -> None:
        self.cfg = cfg or EnergyResonatorConfig()
        self.step_count = 0
        self.he = AgentState(
            name="он",
            emotion=EmotionState(positive=0.15, negative=0.05, energy=0.75),
            goal="получить собственную положительную эмоцию",
            drive=self.cfg.drive,
            tolerance=self.cfg.tolerance,
            empathy=self.cfg.empathy,
        )
        self.she = AgentState(
            name="она",
            emotion=EmotionState(positive=0.10, negative=0.10, energy=0.55),
            goal="хочет быть главной целью его воздействия",
        )

    def step(
        self,
        *,
        mode: Optional[str] = None,
        drive: Optional[float] = None,
        empathy: Optional[float] = None,
        tolerance: Optional[float] = None,
        cost: Optional[float] = None,
        her_need: Optional[float] = None,
        decay: Optional[float] = None,
    ) -> EnergyResonatorStep:
        self.step_count += 1

        mode = str(mode or self.cfg.default_mode)
        drive = clamp(self.cfg.drive if drive is None else drive)
        empathy = clamp(self.cfg.empathy if empathy is None else empathy)
        tolerance = clamp(self.cfg.tolerance if tolerance is None else tolerance)
        cost = clamp(self.cfg.cost if cost is None else cost)
        her_need = clamp(self.cfg.her_need if her_need is None else her_need)
        decay = clamp(self.cfg.decay if decay is None else decay, 0.0, 0.25)

        self.he.drive = drive
        self.he.empathy = empathy
        self.he.tolerance = tolerance

        self.he.emotion.positive *= (1.0 - decay)
        self.he.emotion.negative *= (1.0 - decay * 0.75)
        self.she.emotion.positive *= (1.0 - decay)
        self.she.emotion.negative *= (1.0 - decay)

        if mode == "conscious":
            conscious_alignment = clamp(0.35 + 0.65 * tolerance)
            goal_boost = her_need * conscious_alignment
            endurance = clamp(tolerance - cost * 0.25)
            action_intensity = clamp((drive * 0.45 + goal_boost * 0.65) * (0.55 + endurance * 0.65))

            she_received_positive = action_intensity * (0.50 + 0.50 * conscious_alignment)
            he_self_positive = action_intensity * drive * 0.28
            he_empathic_positive = she_received_positive * empathy * (0.35 + conscious_alignment * 0.85)
            he_negative_cost = action_intensity * cost * (1.0 - tolerance * 0.55)

            self.he.goal = "поддерживать её положительную эмоцию как главную цель"
            self.she.goal = "получать осознанно направленную положительную эмоцию"
        else:
            mode = "animal"
            conscious_alignment = 0.0
            pain_brake = clamp(cost * 1.25)
            action_intensity = clamp(drive * (1.0 - pain_brake * 0.55))

            she_received_positive = action_intensity * (0.58 + 0.15 * drive)
            he_self_positive = action_intensity * drive * 0.72
            he_empathic_positive = self.she.emotion.positive * empathy * 0.10
            he_negative_cost = action_intensity * cost * 0.45

            self.he.goal = "получить собственную положительную эмоцию"
            self.she.goal = "хочет, чтобы её положительная эмоция стала целью"

        self.she.emotion.positive = clamp(self.she.emotion.positive + she_received_positive * 0.18)
        self.she.emotion.negative = clamp(self.she.emotion.negative + max(0.0, her_need - conscious_alignment) * 0.025)

        self.he.emotion.positive = clamp(
            self.he.emotion.positive + he_self_positive * 0.12 + he_empathic_positive * 0.18
        )
        self.he.emotion.negative = clamp(self.he.emotion.negative + he_negative_cost * 0.15)

        self.he.emotion.energy = clamp(
            self.he.emotion.energy
            + self.he.emotion.positive * 0.025
            - self.he.emotion.negative * 0.020
            - action_intensity * cost * 0.015
        )
        self.she.emotion.energy = clamp(
            self.she.emotion.energy
            + self.she.emotion.positive * 0.020
            - self.she.emotion.negative * 0.012
        )

        resonance_index = clamp(
            (self.she.emotion.positive * 0.45)
            + (he_empathic_positive * 0.35)
            + (conscious_alignment * 0.25)
            - (self.he.emotion.negative * 0.20)
        )

        m5_other_agent_focus = {
            "self_agent": "он",
            "other_agent": "она",
            "mode": mode,
            "current_focus": "other_agent_positive_emotion"
            if mode == "conscious"
            else "self_positive_emotion",
            "other_observed_positive": round(self.she.emotion.positive, 4),
            "other_observed_negative": round(self.she.emotion.negative, 4),
            "self_positive": round(self.he.emotion.positive, 4),
            "self_negative": round(self.he.emotion.negative, 4),
            "self_tolerating_negative": bool(mode == "conscious" and self.he.emotion.negative > 0.10),
            "conscious_alignment": round(conscious_alignment, 4),
            "resonance_index": round(resonance_index, 4),
            "interpretation": (
                "он удерживает её положительную эмоцию как главную цель"
                if mode == "conscious"
                else "он действует автоматически ради собственной положительной эмоции"
            ),
        }

        return EnergyResonatorStep(
            step=self.step_count,
            mode=mode,
            action_intensity=float(action_intensity),
            she_received_positive=float(she_received_positive),
            he_self_positive=float(he_self_positive),
            he_empathic_positive=float(he_empathic_positive),
            he_negative_cost=float(he_negative_cost),
            conscious_alignment=float(conscious_alignment),
            resonance_index=float(resonance_index),
            m5_other_agent_focus=m5_other_agent_focus,
        )

    def latent_vector(self, step: EnergyResonatorStep, *, device: Any = None) -> Any:
        """
        Return a 12-dim social-affect latent.

        Shape if torch is available:
            [1, 12]
        """
        values = [
            self.he.emotion.positive,
            self.he.emotion.negative,
            self.he.emotion.energy,
            self.she.emotion.positive,
            self.she.emotion.negative,
            self.she.emotion.energy,
            step.action_intensity,
            step.she_received_positive,
            step.he_self_positive,
            step.he_empathic_positive,
            step.he_negative_cost,
            step.resonance_index,
        ]
        if torch is None:
            return values
        return torch.tensor([values], dtype=torch.float32, device=device)


def run_headless_demo(steps: int = 80, switch_at: int = 40) -> None:
    model = EnergyResonator()
    for i in range(steps):
        mode = "animal" if i < switch_at else "conscious"
        s = model.step(mode=mode)
        if i % 5 == 0 or i in (switch_at - 1, switch_at):
            print(
                f"step={s.step:03d} mode={s.mode:9s} "
                f"action={s.action_intensity:.3f} "
                f"she+={s.she_received_positive:.3f} "
                f"he_self+={s.he_self_positive:.3f} "
                f"he_empathy+={s.he_empathic_positive:.3f} "
                f"he_cost-={s.he_negative_cost:.3f} "
                f"resonance={s.resonance_index:.3f} | "
                f"{s.m5_other_agent_focus['interpretation']}"
            )


__all__ = [
    "AgentState",
    "EmotionState",
    "EnergyResonator",
    "EnergyResonatorConfig",
    "EnergyResonatorStep",
    "run_headless_demo",
]
