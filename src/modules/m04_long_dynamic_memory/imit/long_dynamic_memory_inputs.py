from __future__ import annotations
from typing import Any, Dict
import torch

def _randn(batch:int=1, dim:int=256, scale:float=0.25)->torch.Tensor:
    torch.manual_seed(404)
    return torch.randn(batch, dim) * float(scale)

def _scalar(value:float)->torch.Tensor:
    return torch.tensor([[float(value)]], dtype=torch.float32)

def make_m04_fake_out()->Dict[str, Any]:
    return {"focus_context": _randn(1,256), "event_dream_replay": {"replay_context": _randn(1,256), "event_salience": _scalar(0.50)}}

def make_m04_inner_object()->Dict[str, Any]:
    return {"z_obj": _randn(1,128), "slot_token": "slot:imit", "object_confidence": _scalar(0.65)}

class FakeM4PassportManager:
    def __init__(self, context_dim:int=256)->None:
        self.context_dim=int(context_dim)
        self.passports={"obj:imit": {"token":"obj:imit","replay_z":_randn(1,self.context_dim),"confidence_ema":0.72,"last_similarity":0.82,"dynamic_score_ema":0.35,"lives_in_slot":1,"last_sentence":"imitated stable object identity","episode_summary":"imitated passport episode"}}
    def observe(self, obj:Dict[str,Any], *, event_memory=None, dream_mode:bool=False, global_step:int=0)->Dict[str,Any]:
        del obj,event_memory,dream_mode,global_step
        return {"passport_token":"obj:imit","passport_confidence_ema":0.72,"passport_similarity":0.82,"passport_dynamic_score":0.35,"passport_count":float(len(self.passports)),"passport_slot":1.0,"passport_created":False,"passport_sentence":"imitated stable object identity","passport_episode_summary":"imitated passport episode","passport_source":"imit_passport"}
    def select_for_replay(self, token:str="")->Dict[str,Any]:
        return self.passports.get(token or "obj:imit", self.passports["obj:imit"])

def make_m04_input()->Dict[str, Any]:
    return {"out": make_m04_fake_out(), "obj": make_m04_inner_object(), "passport_manager": FakeM4PassportManager(256), "event_memory": None, "dream_mode": True, "global_step": 1}

__all__=["FakeM4PassportManager","make_m04_fake_out","make_m04_inner_object","make_m04_input"]
