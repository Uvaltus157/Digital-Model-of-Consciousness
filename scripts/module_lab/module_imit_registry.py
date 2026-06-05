from __future__ import annotations
from typing import Callable, Dict

from src.modules.m01_object_imagery.imit.sensor_inputs import make_m01_fake_obs
from src.modules.m02_event_dream_replay.imit.event_dream_replay_inputs import make_m02_input
from src.modules.m03_self_action_causality.imit.motor_guard_inputs import make_m03_motor_output
from src.modules.m04_long_dynamic_memory.imit.long_dynamic_memory_inputs import make_m04_input
from src.modules.m05_world_model_attention_workspace.imit.focus_feedback_inputs import make_m05_focus_feedback_input
from src.modules.m11_motivational_homeostasis.imit.emotional_drive_inputs import make_m11_input
from src.modules.m13_autobiographical_memory.imit.autobiographical_memory_inputs import make_m13_input

IMIT_REGISTRY: Dict[str, Callable] = {
    "m01": make_m01_fake_obs,
    "m02": make_m02_input,
    "m03": make_m03_motor_output,
    "m04": make_m04_input,
    "m05": make_m05_focus_feedback_input,
    "m11": make_m11_input,
    "m13": make_m13_input,
}

def get_imitator(module_name:str)->Callable:
    key=str(module_name).lower().strip()
    if key not in IMIT_REGISTRY:
        raise KeyError(f"unknown imitator {module_name!r}; available={sorted(IMIT_REGISTRY)}")
    return IMIT_REGISTRY[key]

__all__=["IMIT_REGISTRY","get_imitator"]
