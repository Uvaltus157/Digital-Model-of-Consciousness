from __future__ import annotations
from typing import Any, Dict
import torch

def _randn(batch:int, dim:int, scale:float=0.25)->torch.Tensor:
    torch.manual_seed(1313)
    return torch.randn(batch, dim) * float(scale)

def _scalar(value:float)->torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32)

def make_m13_out()->Dict[str,Any]:
    return {
        "focus_context": _randn(1,256),
        "affect": {"affect_latents": _randn(1,12), "valence": _scalar(0.10), "arousal": _scalar(0.50), "stress_latent": _scalar(0.25), "panic_latent": _scalar(0.10), "curiosity_latent": _scalar(0.70)},
        "long_dynamic_memory": {"identity_token": "obj:imit", "dynamic_identity_context": _randn(1,256)},
        "event_dream_replay": {"replay_context": _randn(1,256)},
    }

def make_m13_obs()->Dict[str, torch.Tensor]:
    return {"reward": _scalar(0.0), "done": _scalar(0.0)}

def make_m13_input()->Dict[str,Any]:
    return {"obs": make_m13_obs(), "out": make_m13_out(), "global_step": 1}

__all__=["make_m13_out","make_m13_obs","make_m13_input"]
