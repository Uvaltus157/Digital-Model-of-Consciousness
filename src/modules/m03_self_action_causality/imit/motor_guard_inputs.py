from __future__ import annotations
from typing import Dict
import torch

def make_m03_motor_output(*, batch:int=1)->Dict[str, torch.Tensor]:
    return {
        "embodied_targets": torch.ones(batch,15),
        "hand_ctrl": torch.ones(batch,8)*0.5,
        "leg_ctrl": torch.ones(batch,18)*0.25,
        "action_ids": torch.zeros(batch, dtype=torch.long),
    }

__all__=["make_m03_motor_output"]
