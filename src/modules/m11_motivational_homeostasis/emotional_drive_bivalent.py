from __future__ import annotations

"""
emotional_drive.py

Модуль эмоционального стимулирующего переживания.

Идея:
    Агент получает положительное внутреннее переживание, когда исследование среды
    заполняет пустоты во внутреннем представлении.

Положительный drive возникает от:
    - снижения неопределённости;
    - роста coherence;
    - роста object confidence;
    - согласования object/workspace/thought/memory;
    - осмысленного тактильного контакта;
    - роста уверенности внутренней речи, если M7 уже предоставил отчёт.

Выход:
    out["emotion"]["intrinsic_reward"]
    out["emotion"]["emotional_valence"]
    out["emotion"]["emotional_arousal"]
    out["emotion"]["affect"]
    out["affect"] is attached by life_runtime for M9/M15 consumers.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class EmotionalDriveConfig:
    enabled: bool = True
    ema_decay: float = 0.985

    w_gap_fill: float = 1.25
    w_coherence_gain: float = 0.75
    w_object_conf_gain: float = 0.75
    w_multimodal_alignment: float = 0.55
    w_contact_pleasure: float = 0.35
    w_curiosity: float = 0.25
    w_inner_speech_conf: float = 0.35

    # Negative emotion terms.
    w_uncertainty_increase: float = 1.10
    w_coherence_loss: float = 0.85
    w_object_conf_loss: float = 0.75
    w_speech_conf_loss: float = 0.55
    w_alignment_loss: float = 0.70
    w_chaotic_touch: float = 0.45
    w_instability: float = 0.40
    w_over_arousal_penalty: float = 0.15

    reward_scale: float = 0.15
    max_intrinsic_reward: float = 1.0
    min_intrinsic_reward: float = -0.25
    contact_threshold: float = 0.05
    reward_progress_not_static_confidence: bool = True


def _scalar(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float(default)
        return float(x.detach().float().cpu().reshape(-1)[0].item())
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _norm(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float(default)
        return float(x.detach().float().norm(dim=-1).mean().cpu().item())
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return float(default)
        return float(np.linalg.norm(arr.reshape(-1)))
    except Exception:
        return float(default)


def _cosine(a, b, default: float = 0.0) -> float:
    if a is None or b is None:
        return float(default)
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().reshape(-1).numpy()
    else:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu().reshape(-1).numpy()
    else:
        b = np.asarray(b, dtype=np.float32).reshape(-1)

    n = min(a.size, b.size)
    if n <= 0:
        return float(default)
    a = a[:n]
    b = b[:n]
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-8:
        return float(default)
    return float(np.dot(a, b) / den)


def _first_dict(out: Dict, *keys: str) -> Dict:
    for key in keys:
        value = out.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _device_from(out: Dict) -> torch.device:
    for v in out.values():
        if isinstance(v, torch.Tensor):
            return v.device
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, torch.Tensor):
                    return vv.device
    return torch.device("cpu")


def _t(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32, device=device)


class EmotionalDrive:
    def __init__(self, cfg: Optional[EmotionalDriveConfig] = None) -> None:
        self.cfg = cfg or EmotionalDriveConfig()

        self.ema_coherence: Optional[float] = None
        self.ema_object_conf: Optional[float] = None
        self.ema_uncertainty: Optional[float] = None
        self.ema_touch: Optional[float] = None
        self.ema_alignment: Optional[float] = None
        self.ema_speech_conf: Optional[float] = None

        self.prev_object_norm: Optional[float] = None
        self.prev_workspace_norm: Optional[float] = None
        self.prev_valence: Optional[float] = None

    def _ema_update(self, attr: str, value: float) -> float:
        old = getattr(self, attr)
        if old is None:
            setattr(self, attr, float(value))
            return 0.0
        decay = self.cfg.ema_decay
        new = decay * old + (1.0 - decay) * float(value)
        setattr(self, attr, new)
        return float(value - old)

    def _build_affect_packet(
        self,
        *,
        device: torch.device,
        emotional_valence: float,
        emotional_arousal: float,
        intrinsic_reward: float,
        meaning_progress: float,
        misunderstanding: float,
        gap_fill_reward: float,
        confusion_increase: float,
        coherence: float,
        coherence_progress: float,
        coherence_loss: float,
        object_progress: float,
        speech_progress: float,
        alignment_progress: float,
        alignment_loss: float,
        contact_pleasure: float,
        chaotic_touch: float,
        instability: float,
        uncertainty: float,
        curiosity: float,
        touch_sum: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Structured affect packet used by later M9/M15 stages.

        Scalar legacy emotion fields remain unchanged. This nested packet is the
        new architectural contract: M10/M11 affect is a first-class input to
        self-binding and thought-chain planning.
        """
        pain = float(np.clip(chaotic_touch + 0.35 * instability + 0.05 * max(0.0, touch_sum), 0.0, 1.0))
        stress = float(np.clip(uncertainty + confusion_increase + instability, 0.0, 1.0))
        fear = float(np.clip(0.55 * uncertainty + 0.35 * coherence_loss + 0.25 * instability, 0.0, 1.0))
        panic = float(np.clip(stress * fear + emotional_arousal * max(0.0, -emotional_valence), 0.0, 1.0))
        comfort = float(np.clip(contact_pleasure + 0.35 * coherence + max(0.0, emotional_valence), 0.0, 1.0))
        relief = float(np.clip(gap_fill_reward + coherence_progress + object_progress + speech_progress + alignment_progress, 0.0, 1.0))
        discovery = float(np.clip(np.tanh(max(0.0, meaning_progress) + curiosity), 0.0, 1.0))
        coherence_latent = float(np.clip(coherence, 0.0, 1.0))
        expected_affect_delta = float(np.tanh(meaning_progress - misunderstanding))

        parts = [
            emotional_valence,
            emotional_arousal,
            pain,
            stress,
            fear,
            panic,
            comfort,
            relief,
            curiosity,
            discovery,
            coherence_latent,
            expected_affect_delta,
        ]
        affect_latents = torch.tensor([parts], dtype=torch.float32, device=device)
        return {
            "affect_latents": affect_latents,
            "valence": _t(emotional_valence, device),
            "arousal": _t(emotional_arousal, device),
            "pain_latent": _t(pain, device),
            "stress_latent": _t(stress, device),
            "fear_latent": _t(fear, device),
            "panic_latent": _t(panic, device),
            "comfort_latent": _t(comfort, device),
            "relief_latent": _t(relief, device),
            "curiosity_latent": _t(curiosity, device),
            "discovery_latent": _t(discovery, device),
            "coherence_latent": _t(coherence_latent, device),
            "expected_affect_delta": _t(expected_affect_delta, device),
            "intrinsic_reward": _t(intrinsic_reward, device),
            "alignment_loss": _t(alignment_loss, device),
        }

    def compute(self, out: Dict, obs: Optional[Dict] = None) -> Dict[str, torch.Tensor | float | Dict[str, torch.Tensor]]:
        if not self.cfg.enabled:
            return self._zeros(out)

        obs = obs or {}

        values = out.get("values", {}) if isinstance(out.get("values"), dict) else {}
        reflection = _first_dict(out, "preconscious_reflection_out", "model_reflection", "reflection_out")
        memory = out.get("memory", {}) if isinstance(out.get("memory"), dict) else {}
        object_imagery = out.get("object_imagery", {}) if isinstance(out.get("object_imagery"), dict) else {}
        report = _first_dict(out, "inner_speech", "conscious_report", "m7_inner_speech", "symbolic_report")
        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}

        coherence = _scalar(values.get("coherence"), 0.0)
        curiosity = _scalar(values.get("curiosity"), 0.0)
        self_conf = max(
            _scalar(reflection.get("model_confidence"), 0.0),
            _scalar(reflection.get("confidence"), 0.0),
            _scalar(self_core.get("self_confidence"), 0.0),
            _scalar(self_core.get("agency_score"), 0.0),
        )
        object_conf = _scalar(object_imagery.get("object_confidence"), 0.0)
        speech_conf = _scalar(report.get("confidence"), 0.0)

        tactile = obs.get("tactile")
        if tactile is None:
            touch_sum = 0.0
        elif isinstance(tactile, torch.Tensor):
            touch_sum = float(tactile.detach().float().abs().sum().cpu().item())
        else:
            touch_sum = float(np.abs(np.asarray(tactile, dtype=np.float32)).sum())

        object_repr = out.get("object_repr")
        workspace = out.get("workspace_out")
        preconscious = out.get("preconscious_thoughts", {}) if isinstance(out.get("preconscious_thoughts"), dict) else {}
        thought = preconscious.get("thought_candidate")
        memory_context = memory.get("memory_context")

        align_ow = _cosine(object_repr, workspace)
        align_ot = _cosine(object_repr, thought)
        align_wm = _cosine(workspace, memory_context)
        multimodal_alignment = float(np.clip((align_ow + align_ot + align_wm + 3.0) / 6.0, 0.0, 1.0))

        uncertainty = float(np.clip(1.0 - np.mean([coherence, object_conf, speech_conf, self_conf]), 0.0, 1.0))

        coherence_gain = self._ema_update("ema_coherence", coherence)
        object_conf_gain = self._ema_update("ema_object_conf", object_conf)
        speech_conf_gain = self._ema_update("ema_speech_conf", speech_conf)
        uncertainty_delta = self._ema_update("ema_uncertainty", uncertainty)
        alignment_gain = self._ema_update("ema_alignment", multimodal_alignment)
        self._ema_update("ema_touch", touch_sum)

        # Positive changes: understanding becomes clearer.
        gap_fill_reward = max(0.0, -uncertainty_delta)
        coherence_progress = max(0.0, coherence_gain)
        object_progress = max(0.0, object_conf_gain)
        speech_progress = max(0.0, speech_conf_gain)
        alignment_progress = max(0.0, alignment_gain)

        # Negative changes: misunderstanding grows.
        confusion_increase = max(0.0, uncertainty_delta)
        coherence_loss = max(0.0, -coherence_gain)
        object_conf_loss = max(0.0, -object_conf_gain)
        speech_conf_loss = max(0.0, -speech_conf_gain)
        alignment_loss = max(0.0, -alignment_gain)

        if touch_sum > self.cfg.contact_threshold:
            # Pleasant if the contact is coherent; unpleasant if contact happens
            # while the model cannot integrate it into the current object/world state.
            contact_pleasure = float(np.tanh(touch_sum) * (0.3 + 0.7 * coherence))
            chaotic_touch = float(np.tanh(touch_sum) * (1.0 - coherence) * (0.5 + 0.5 * uncertainty))
        else:
            contact_pleasure = 0.0
            chaotic_touch = 0.0

        object_norm = _norm(object_repr)
        workspace_norm = _norm(workspace)
        instability = 0.0
        if self.prev_object_norm is not None:
            instability += abs(object_norm - self.prev_object_norm)
        if self.prev_workspace_norm is not None:
            instability += abs(workspace_norm - self.prev_workspace_norm)
        self.prev_object_norm = object_norm
        self.prev_workspace_norm = workspace_norm
        instability = float(np.tanh(instability * 0.1))

        if self.cfg.reward_progress_not_static_confidence:
            meaning_progress = (
                self.cfg.w_gap_fill * gap_fill_reward
                + self.cfg.w_coherence_gain * coherence_progress
                + self.cfg.w_object_conf_gain * object_progress
                + self.cfg.w_inner_speech_conf * speech_progress
                + self.cfg.w_multimodal_alignment * alignment_progress
            )
        else:
            meaning_progress = (
                self.cfg.w_gap_fill * gap_fill_reward
                + self.cfg.w_coherence_gain * coherence
                + self.cfg.w_object_conf_gain * object_conf
                + self.cfg.w_inner_speech_conf * speech_conf
                + self.cfg.w_multimodal_alignment * multimodal_alignment
            )

        misunderstanding = (
            self.cfg.w_uncertainty_increase * confusion_increase
            + self.cfg.w_coherence_loss * coherence_loss
            + self.cfg.w_object_conf_loss * object_conf_loss
            + self.cfg.w_speech_conf_loss * speech_conf_loss
            + self.cfg.w_alignment_loss * alignment_loss
            + self.cfg.w_chaotic_touch * chaotic_touch
        )

        positive = meaning_progress + self.cfg.w_contact_pleasure * contact_pleasure + self.cfg.w_curiosity * curiosity
        negative = misunderstanding + self.cfg.w_instability * instability

        raw_valence = positive - negative
        emotional_valence = float(np.tanh(raw_valence))
        emotional_arousal = float(np.clip(abs(raw_valence) + curiosity * 0.25 + touch_sum * 0.05, 0.0, 1.0))

        over_arousal_penalty = self.cfg.w_over_arousal_penalty * max(0.0, emotional_arousal - 0.85)
        intrinsic_reward = self.cfg.reward_scale * (emotional_valence - over_arousal_penalty)
        intrinsic_reward = float(np.clip(intrinsic_reward, self.cfg.min_intrinsic_reward, self.cfg.max_intrinsic_reward))

        device = _device_from(out)
        reward_tensor = torch.tensor([intrinsic_reward], dtype=torch.float32, device=device)
        affect = self._build_affect_packet(
            device=device,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            intrinsic_reward=intrinsic_reward,
            meaning_progress=float(meaning_progress),
            misunderstanding=float(misunderstanding),
            gap_fill_reward=float(gap_fill_reward),
            confusion_increase=float(confusion_increase),
            coherence=float(coherence),
            coherence_progress=float(coherence_progress),
            coherence_loss=float(coherence_loss),
            object_progress=float(object_progress),
            speech_progress=float(speech_progress),
            alignment_progress=float(alignment_progress),
            alignment_loss=float(alignment_loss),
            contact_pleasure=float(contact_pleasure),
            chaotic_touch=float(chaotic_touch),
            instability=float(instability),
            uncertainty=float(uncertainty),
            curiosity=float(curiosity),
            touch_sum=float(touch_sum),
        )

        return {
            "intrinsic_reward": reward_tensor,
            "emotional_valence": emotional_valence,
            "emotional_arousal": emotional_arousal,
            "affect": affect,
            "meaning_progress": float(meaning_progress),
            "gap_fill_reward": float(gap_fill_reward),
            "misunderstanding": float(misunderstanding),
            "confusion_increase": float(confusion_increase),
            "coherence_loss": float(coherence_loss),
            "object_conf_loss": float(object_conf_loss),
            "speech_conf_loss": float(speech_conf_loss),
            "alignment_loss": float(alignment_loss),
            "chaotic_touch": float(chaotic_touch),
            "multimodal_alignment": float(multimodal_alignment),
            "alignment_progress": float(alignment_progress),
            "contact_pleasure": float(contact_pleasure),
            "coherence_progress": float(coherence_progress),
            "object_conf_progress": float(object_progress),
            "speech_conf_progress": float(speech_progress),
            "instability": float(instability),
            "uncertainty": float(uncertainty),
            "touch_sum": float(touch_sum),
            "curiosity": float(curiosity),
        }

    def _zeros(self, out: Dict) -> Dict[str, torch.Tensor | float | Dict[str, torch.Tensor]]:
        device = _device_from(out)
        affect = self._build_affect_packet(
            device=device,
            emotional_valence=0.0,
            emotional_arousal=0.0,
            intrinsic_reward=0.0,
            meaning_progress=0.0,
            misunderstanding=0.0,
            gap_fill_reward=0.0,
            confusion_increase=0.0,
            coherence=0.0,
            coherence_progress=0.0,
            coherence_loss=0.0,
            object_progress=0.0,
            speech_progress=0.0,
            alignment_progress=0.0,
            alignment_loss=0.0,
            contact_pleasure=0.0,
            chaotic_touch=0.0,
            instability=0.0,
            uncertainty=1.0,
            curiosity=0.0,
            touch_sum=0.0,
        )
        return {
            "intrinsic_reward": torch.zeros(1, dtype=torch.float32, device=device),
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
            "affect": affect,
            "meaning_progress": 0.0,
            "gap_fill_reward": 0.0,
            "misunderstanding": 0.0,
            "confusion_increase": 0.0,
            "coherence_loss": 0.0,
            "object_conf_loss": 0.0,
            "speech_conf_loss": 0.0,
            "alignment_loss": 0.0,
            "chaotic_touch": 0.0,
            "multimodal_alignment": 0.0,
            "alignment_progress": 0.0,
            "contact_pleasure": 0.0,
            "coherence_progress": 0.0,
            "object_conf_progress": 0.0,
            "speech_conf_progress": 0.0,
            "instability": 0.0,
            "uncertainty": 1.0,
            "touch_sum": 0.0,
            "curiosity": 0.0,
        }
