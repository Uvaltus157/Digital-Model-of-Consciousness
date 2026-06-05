from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch

def _scalar(value:float)->torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32)

def _randn(batch:int=1, dim:int=256, scale:float=0.25)->torch.Tensor:
    torch.manual_seed(202)
    return torch.randn(batch, dim) * float(scale)

def make_m02_fake_affect(*, panic:float=0.25, stress:float=0.30, curiosity:float=0.70)->Dict[str, torch.Tensor]:
    return {
        "panic_latent": _scalar(panic),
        "stress_latent": _scalar(stress),
        "curiosity_latent": _scalar(curiosity),
        "expected_affect_delta": _scalar(0.10),
        "valence": _scalar(0.05),
        "arousal": _scalar(0.50),
    }

def make_m02_fake_m13(*, relevance:float=0.55)->Dict[str, Any]:
    return {
        "retrieved_context": _randn(1,256),
        "retrieval_relevance": _scalar(relevance),
        "retrieved_episode_count": _scalar(3.0),
        "summary": "imitated M13 episode",
    }

def make_m02_fake_m4(*, gate:float=0.80, stability:float=0.70, novelty:float=0.20)->Dict[str, Any]:
    return {
        "dynamic_identity_context": _randn(1,256),
        "dynamic_memory_gate": _scalar(gate),
        "identity_token": "obj:imit",
        "identity_stability": _scalar(stability),
        "identity_similarity": _scalar(1.0-novelty),
        "identity_novelty": _scalar(novelty),
        "passport_slot": _scalar(1.0),
        "selected_sentence": "imitated stable object identity",
    }

def make_m02_fake_event_memory()->List[Dict[str, Any]]:
    return [{
        "sentence": "imitated contact changed prediction",
        "kind": "contact_prediction_delta",
        "delta_norm": 0.75,
        "contact_norm": 0.55,
        "action_norm": 0.25,
        "touch_strength": 0.50,
        "event_code": torch.tensor([[0.75,0.55,0.25,0.50,0.10,0.20,0.30,0.40]], dtype=torch.float32),
    }]

def make_m02_input(*, panic:float=0.25, stress:float=0.30, curiosity:float=0.70, m13_relevance:float=0.55, m4_gate:float=0.80, dream_mode:bool=True)->Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    out = {
        "focus_context": _randn(1,256),
        "affect": make_m02_fake_affect(panic=panic, stress=stress, curiosity=curiosity),
        "autobiographical_memory": make_m02_fake_m13(relevance=m13_relevance),
        "long_dynamic_memory": make_m02_fake_m4(gate=m4_gate),
    }
    return out, make_m02_fake_event_memory(), bool(dream_mode)

def make_m02_calm_input():
    return make_m02_input(panic=0.0, stress=0.0, curiosity=0.0, m13_relevance=0.0, m4_gate=0.0, dream_mode=False)

def make_m02_stress_input():
    return make_m02_input(panic=0.9, stress=0.9, curiosity=0.35, m13_relevance=0.55, m4_gate=0.65, dream_mode=True)

__all__=["make_m02_input","make_m02_calm_input","make_m02_stress_input","make_m02_fake_affect","make_m02_fake_m13","make_m02_fake_m4","make_m02_fake_event_memory"]
