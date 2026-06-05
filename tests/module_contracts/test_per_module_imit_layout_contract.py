from __future__ import annotations
import torch

def test_per_module_imit_registry_imports():
    from scripts.module_lab.module_imit_registry import IMIT_REGISTRY
    assert {"m01","m02","m03","m04","m05","m11","m13"}.issubset(set(IMIT_REGISTRY))

def test_m02_imitator_returns_processed_internal_inputs_not_raw_m1():
    from src.modules.m02_event_dream_replay.imit.event_dream_replay_inputs import make_m02_input
    out, event_memory, dream_mode = make_m02_input()
    assert "focus_context" in out
    assert "affect" in out
    assert "autobiographical_memory" in out
    assert "long_dynamic_memory" in out
    assert isinstance(event_memory, list)
    assert dream_mode is True
    assert "left" not in out and "right" not in out and "depth" not in out

def test_m04_imitator_uses_inner_object_not_raw_sensors():
    from src.modules.m04_long_dynamic_memory.imit.long_dynamic_memory_inputs import make_m04_input
    packet = make_m04_input()
    assert "obj" in packet and "z_obj" in packet["obj"]
    assert "out" in packet and "focus_context" in packet["out"]
    assert "left" not in packet["out"] and "right" not in packet["out"]

def test_m05_imitator_returns_focus_seed_boundary_inputs():
    from src.modules.m05_world_model_attention_workspace.imit.focus_feedback_inputs import make_m05_focus_feedback_input
    packet = make_m05_focus_feedback_input()
    assert tuple(packet["workspace_seed"].shape) == (1,256)
    assert tuple(packet["focus_context_seed"].shape) == (1,256)
    assert tuple(packet["focus_context_seed_gate"].shape) == (1,1)
    assert tuple(packet["preconscious_seed"].shape) == (1,192)
    assert bool(torch.isfinite(packet["workspace_seed"]).all().item())
