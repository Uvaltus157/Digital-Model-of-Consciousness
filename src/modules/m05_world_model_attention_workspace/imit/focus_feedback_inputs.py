from __future__ import annotations
from typing import Dict
import torch

def make_m05_focus_feedback_input(*, batch:int=1, seed_gate:float=0.10)->Dict[str, torch.Tensor]:
    torch.manual_seed(505)
    return {
        "workspace_seed": torch.randn(batch,256),
        "focus_context_seed": torch.randn(batch,256),
        "focus_context_seed_gate": torch.tensor([[float(seed_gate)]], dtype=torch.float32).expand(batch,1),
        "preconscious_seed": torch.randn(batch,192),
    }

def make_m05_zero_seed_input(*, batch:int=1)->Dict[str, object]:
    torch.manual_seed(506)
    return {"workspace_seed": torch.randn(batch,256), "focus_context_seed": None, "focus_context_seed_gate": None, "preconscious_seed": torch.randn(batch,192)}

__all__=["make_m05_focus_feedback_input","make_m05_zero_seed_input"]
