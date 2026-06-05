from __future__ import annotations
from typing import Dict
import torch

def _randn(batch:int, dim:int, scale:float=0.25)->torch.Tensor:
    torch.manual_seed(101)
    return torch.randn(batch, dim) * float(scale)

def make_m01_fake_obs(*, batch:int=1, image_h:int=32, image_w:int=48, tactile_dim:int=16, body_state_dim:int=32, hand_motor_dim:int=8, embodied_dim:int=15)->Dict[str, torch.Tensor]:
    torch.manual_seed(101)
    return {
        "left": torch.rand(batch,3,image_h,image_w),
        "right": torch.rand(batch,3,image_h,image_w),
        "depth": torch.rand(batch,1,image_h,image_w),
        "pose": _randn(batch,7),
        "body_state": _randn(batch,body_state_dim),
        "tactile": _randn(batch,tactile_dim,0.10),
        "hand_motor": _randn(batch,hand_motor_dim,0.10),
        "embodied_action": _randn(batch,embodied_dim,0.10),
        "object_state": _randn(batch,9,0.10),
        "reward": torch.zeros(batch,1),
        "done": torch.zeros(batch,1),
    }

__all__=["make_m01_fake_obs"]
