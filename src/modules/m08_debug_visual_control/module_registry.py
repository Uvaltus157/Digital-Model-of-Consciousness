from __future__ import annotations

"""Canonical M1-M15 registry for Module Debug UI.

The debug window should not duplicate module names, order, or colors in several
places. This registry is the lightweight source of truth for the visual module
mnemoscheme.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModuleRegistryEntry:
    module_id: str
    package: str
    title: str
    short_title: str
    color_group: str
    layer: int
    description: str


MODULE_REGISTRY: List[ModuleRegistryEntry] = [
    ModuleRegistryEntry("M1", "m01_object_imagery", "M1_OBJECT_IMAGERY", "Object Imagery", "perception", 1, "Sensors, fusion, object slots, latent object and 2D/3D/4D reconstruction."),
    ModuleRegistryEntry("M2", "m02_event_dream_replay", "M2_EVENT_DREAM_REPLAY", "Event Dream Replay", "event", 2, "Slot changes become events, sentences, episodes and dream replay."),
    ModuleRegistryEntry("M3", "m03_self_action_causality", "M3_SELF_ACTION_CAUSALITY", "Action Causality", "action", 2, "Intent, action heads, dynamic control and causal attribution."),
    ModuleRegistryEntry("M4", "m04_long_dynamic_memory", "M4_LONG_DYNAMIC_MEMORY", "Long Dynamic Memory", "memory", 2, "Object identity, temporal object memory, restore/reuse and memory context."),
    ModuleRegistryEntry("M5", "m05_world_model_attention_workspace", "M5_WORLD_MODEL_ATTENTION_WORKSPACE", "World Model Workspace", "world_model", 0, "World model, RSSM, novelty, attention, planner and active workspace."),
    ModuleRegistryEntry("M6", "m06_learning_sleep_consolidation", "M6_LEARNING_SLEEP_CONSOLIDATION", "Learning Sleep", "learning", 4, "Training, sleep mode, dream replay training and memory consolidation."),
    ModuleRegistryEntry("M7", "m07_inner_speech_thoughts", "M7_INNER_SPEECH_THOUGHTS", "Inner Speech", "language", 5, "Latent semantic map, thought decoder and subjective stream."),
    ModuleRegistryEntry("M8", "m08_debug_visual_control", "M8_DEBUG_VISUAL_CONTROL", "Debug Visual Control", "debug", 5, "Module debug window, status IPC, visual diagnostics and manual control."),
    ModuleRegistryEntry("M9", "m09_self_core", "M9_SELF_CORE", "Self Core", "self", 3, "Body ownership, agency, egocentric frame, temporal self and self-causation."),
    ModuleRegistryEntry("M10", "m10_global_conscious_broadcast", "M10_GLOBAL_CONSCIOUS_BROADCAST", "Conscious Broadcast", "broadcast", 3, "Candidates, competition, selector, broadcast gate and broadcast packet."),
    ModuleRegistryEntry("M11", "m11_motivational_homeostasis", "M11_MOTIVATIONAL_HOMEOSTASIS", "Motivation", "drive", 3, "Curiosity, safety, mastery, fatigue, valence, arousal and urge state."),
    ModuleRegistryEntry("M12", "m12_metacognition_monitor", "M12_METACOGNITION_MONITOR", "Metacognition", "meta", 3, "Object, memory, action and self confidence monitors with uncertainty state."),
    ModuleRegistryEntry("M13", "m13_autobiographical_memory", "M13_AUTOBIOGRAPHICAL_MEMORY", "Autobiographical Memory", "memory", 4, "Self episodes, personal timeline, discovery/mistake tags and recall."),
    ModuleRegistryEntry("M14", "m14_semantic_grounding", "M14_SEMANTIC_GROUNDING", "Semantic Grounding", "language", 4, "Word-to-object/action/sensation/self grounding and concept memory."),
    ModuleRegistryEntry("M15", "m15_counterfactual_imagination_planning", "M15_COUNTERFACTUAL_IMAGINATION_PLANNING", "Imagination Planning", "planning", 4, "Future scenarios, counterfactual rollout, outcome evaluation and candidate plans."),
]

MODULE_BY_ID: Dict[str, ModuleRegistryEntry] = {entry.module_id: entry for entry in MODULE_REGISTRY}
MODULE_BY_PACKAGE: Dict[str, ModuleRegistryEntry] = {entry.package: entry for entry in MODULE_REGISTRY}


def module_registry_as_dicts() -> List[dict]:
    return [entry.__dict__.copy() for entry in MODULE_REGISTRY]


def get_module_entry(module_id: str) -> ModuleRegistryEntry:
    key = str(module_id).upper()
    return MODULE_BY_ID[key]
