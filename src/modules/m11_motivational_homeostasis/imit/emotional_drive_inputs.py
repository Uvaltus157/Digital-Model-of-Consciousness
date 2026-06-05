from __future__ import annotations
from typing import Any, Dict
import torch

def _randn(batch:int, dim:int, scale:float=0.25)->torch.Tensor:
    torch.manual_seed(1111)
    return torch.randn(batch, dim) * float(scale)

def _scalar(value:float)->torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32)

def make_m11_m5_out(*, coherence:float=0.65, curiosity:float=0.45)->Dict[str,Any]:
    return {
        "focus_context": _randn(1,256),
        "workspace_out": _randn(1,256),
        "object_repr": _randn(1,128),
        "object_imagery": {"object_confidence": _scalar(0.62)},
        "preconscious_thoughts": {"thought_candidate": _randn(1,192), "workspace_seed": _randn(1,192), "candidate_delta": _randn(1,192,0.05)},
        "preconscious_reflection_out": {"reflection": _randn(1,128), "model_confidence": _scalar(0.58)},
        "memory": {"memory_context": _randn(1,256)},
        "values": {"coherence": _scalar(coherence), "curiosity": _scalar(curiosity), "value_latent": _randn(1,64)},
        "imagined": {"imagined_value": _scalar(0.15), "imagined_touch": _randn(1,16,0.05)},
    }

def make_m11_obs()->Dict[str, torch.Tensor]:
    return {"tactile": torch.ones(1,16)*0.05}

def make_m11_input()->Dict[str,Any]:
    return {"out": make_m11_m5_out(), "obs": make_m11_obs()}

__all__=["make_m11_m5_out","make_m11_obs","make_m11_input"]
