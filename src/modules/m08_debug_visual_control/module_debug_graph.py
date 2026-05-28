from __future__ import annotations

"""Module Debug graph definitions derived from the M1-M15 registry.

This file is the bridge between the canonical architecture registry and the
PyQt module-debug mnemoscheme. It intentionally keeps the legacy training keys
for the currently controllable modules while exposing a full M1-M15 graph for
architecture/debug views.
"""

from typing import Dict, List, Tuple

from .module_registry import MODULE_BY_ID, MODULE_REGISTRY

# Colors are kept here so the registry describes meaning/order while this file
# owns the concrete debug-window visual palette.
GROUP_COLORS: Dict[str, str] = {
    "perception": "#66B5FF",
    "event": "#59C6FF",
    "action": "#4FD788",
    "memory": "#35E3C4",
    "world_model": "#4F9DFF",
    "learning": "#FFD36D",
    "language": "#FF7DD1",
    "debug": "#8DBDFF",
    "self": "#C981FF",
    "broadcast": "#A18BFF",
    "drive": "#FFB86B",
    "meta": "#D2A8FF",
    "planning": "#FF9F7D",
}

# Runtime training keys currently understood by the runner/status payload.
# These aliases let the debug UI stay useful while the architecture moves to M1-M15.
MODULE_ID_TO_RUNTIME_KEY: Dict[str, str] = {
    "M1": "object_imagery",
    "M3": "action_heads",
    "M4": "long_dynamic_memory",
    "M5": "world_model",
    "M7": "inner_speech",
    "M9": "self_core",
}

RUNTIME_KEY_TO_MODULE_ID: Dict[str, str] = {v: k for k, v in MODULE_ID_TO_RUNTIME_KEY.items()}

# Extra legacy/runtime cards that are not exact M1-M15 modules but are still
# useful controls in the current runner.
LEGACY_RUNTIME_MODULES: List[Tuple[str, str, str, str]] = [
    ("core_model", "Core Model", "Integration", "#A18BFF"),
    ("leg_control", "Leg Control", "Action", "#56E2AE"),
]


def _entry_color(module_id: str) -> str:
    entry = MODULE_BY_ID[module_id]
    return GROUP_COLORS.get(entry.color_group, "#8DBDFF")


def legacy_training_modules() -> List[Tuple[str, str, str, str]]:
    """Return currently controllable training cards, using registry labels."""
    ordered_ids = ["M5", "M1", "M4", "M9", "M7"]
    result: List[Tuple[str, str, str, str]] = []
    for module_id in ordered_ids:
        entry = MODULE_BY_ID[module_id]
        runtime_key = MODULE_ID_TO_RUNTIME_KEY[module_id]
        result.append((runtime_key, f"{module_id} {entry.short_title}", entry.title, _entry_color(module_id)))

    # Preserve current non-M module controls.
    result.insert(3, ("core_model", "M5 Core Model", "M5_WORLD_MODEL_ATTENTION_WORKSPACE", "#A18BFF"))
    result.insert(5, ("action_heads", "M3 Action Heads", "M3_SELF_ACTION_CAUSALITY", _entry_color("M3")))
    result.insert(6, ("leg_control", "M3 Leg Control", "M3_SELF_ACTION_CAUSALITY", "#56E2AE"))
    return result


def m15_architecture_modules() -> List[Tuple[str, str, str, str]]:
    """Return one card for every M1-M15 architecture module."""
    result: List[Tuple[str, str, str, str]] = []
    for entry in MODULE_REGISTRY:
        key = MODULE_ID_TO_RUNTIME_KEY.get(entry.module_id, entry.module_id.lower())
        label = f"{entry.module_id} {entry.short_title}"
        result.append((key, label, entry.title, GROUP_COLORS.get(entry.color_group, "#8DBDFF")))
    return result


LEGACY_TRAINING_MODULES: List[Tuple[str, str, str, str]] = legacy_training_modules()
M15_ARCHITECTURE_MODULES: List[Tuple[str, str, str, str]] = m15_architecture_modules()

LEGACY_TRAINING_EDGES: List[Tuple[str, str, str]] = [
    ("world_model", "core_model", "latent state"),
    ("world_model", "object_imagery", "object slots"),
    ("long_dynamic_memory", "self_core", "dynamic object"),
    ("long_dynamic_memory", "object_imagery", "z_dynamic slot"),
    ("world_model", "long_dynamic_memory", "temporal context"),
    ("object_imagery", "long_dynamic_memory", "z_static stream"),
    ("object_imagery", "self_core", "object percept"),
    ("core_model", "action_heads", "intent"),
    ("action_heads", "leg_control", "motor"),
    ("action_heads", "self_core", "agency"),
    ("core_model", "self_core", "workspace"),
    ("self_core", "inner_speech", "report"),
    ("world_model", "inner_speech", "semantics"),
    ("self_core", "action_heads", "self-guided"),
]

M15_ARCHITECTURE_EDGES: List[Tuple[str, str, str]] = [
    ("world_model", "object_imagery", "focus/features"),
    ("object_imagery", "m2", "slot→event"),
    ("object_imagery", "long_dynamic_memory", "object state"),
    ("object_imagery", "self_core", "body/object evidence"),
    ("action_heads", "self_core", "agency evidence"),
    ("long_dynamic_memory", "self_core", "memory context"),
    ("self_core", "m10", "self_state"),
    ("m11", "m10", "drive salience"),
    ("m12", "m10", "confidence/doubt"),
    ("world_model", "m10", "attention/novelty"),
    ("m10", "m13", "meaningful content"),
    ("m2", "m13", "events→episodes"),
    ("m13", "self_core", "continuity"),
    ("m13", "m14", "personal meaning"),
    ("m14", "inner_speech", "grounded words"),
    ("m10", "inner_speech", "broadcast→thought"),
    ("m15", "action_heads", "candidate plan"),
]

LEGACY_POSITIONS: Dict[str, Tuple[float, float]] = {
    "world_model": (0.12, 0.20),
    "object_imagery": (0.12, 0.62),
    "long_dynamic_memory": (0.30, 0.62),
    "core_model": (0.39, 0.20),
    "action_heads": (0.66, 0.20),
    "leg_control": (0.88, 0.20),
    "self_core": (0.48, 0.62),
    "inner_speech": (0.80, 0.62),
}

M15_POSITIONS: Dict[str, Tuple[float, float]] = {
    "world_model": (0.50, 0.08),
    "object_imagery": (0.13, 0.24),
    "m2": (0.31, 0.24),
    "action_heads": (0.49, 0.24),
    "long_dynamic_memory": (0.67, 0.24),
    "m10": (0.85, 0.24),
    "self_core": (0.13, 0.47),
    "m11": (0.31, 0.47),
    "m12": (0.49, 0.47),
    "m13": (0.67, 0.47),
    "m14": (0.85, 0.47),
    "m6": (0.18, 0.72),
    "inner_speech": (0.38, 0.72),
    "m8": (0.58, 0.72),
    "m15": (0.78, 0.72),
}

DEFAULT_FLAGS: Dict[str, bool] = {key: True for key, *_ in LEGACY_TRAINING_MODULES}

COLLECTIVE_PRESETS: List[Tuple[str, Dict[str, bool]]] = [
    ("Perception", {"world_model": True, "object_imagery": True, "core_model": False, "long_dynamic_memory": True, "action_heads": False, "leg_control": False, "self_core": False, "inner_speech": False}),
    ("Dynamic Object Memory", {"world_model": True, "object_imagery": True, "long_dynamic_memory": True, "core_model": False, "action_heads": False, "leg_control": False, "self_core": False, "inner_speech": False}),
    ("World + Core", {"world_model": True, "object_imagery": True, "core_model": True, "long_dynamic_memory": True, "action_heads": False, "leg_control": False, "self_core": False, "inner_speech": False}),
    ("Action Stack", {"world_model": False, "object_imagery": False, "long_dynamic_memory": False, "core_model": True, "action_heads": True, "leg_control": True, "self_core": False, "inner_speech": False}),
    ("Self Loop", {"world_model": True, "object_imagery": True, "long_dynamic_memory": True, "core_model": True, "action_heads": True, "leg_control": False, "self_core": True, "inner_speech": True}),
    ("Language / Report", {"world_model": True, "object_imagery": True, "core_model": True, "long_dynamic_memory": True, "action_heads": False, "leg_control": False, "self_core": True, "inner_speech": True}),
    ("All modules", {k: True for k, *_ in LEGACY_TRAINING_MODULES}),
    ("Freeze all", {k: False for k, *_ in LEGACY_TRAINING_MODULES}),
]
