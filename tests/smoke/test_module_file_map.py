from __future__ import annotations

import json
from pathlib import Path


EXPECTED_MODULES = [
    "M1_OBJECT_IMAGERY",
    "M2_EVENT_DREAM_REPLAY",
    "M3_SELF_ACTION_CAUSALITY",
    "M4_LONG_DYNAMIC_MEMORY",
    "M5_WORLD_MODEL_ATTENTION_WORKSPACE",
    "M6_LEARNING_SLEEP_CONSOLIDATION",
    "M7_INNER_SPEECH_THOUGHTS",
    "M8_DEBUG_VISUAL_CONTROL",
    "M9_SELF_CORE",
    "M10_GLOBAL_CONSCIOUS_BROADCAST",
    "M11_MOTIVATIONAL_HOMEOSTASIS",
    "M12_METACOGNITION_MONITOR",
    "M13_AUTOBIOGRAPHICAL_MEMORY",
    "M14_SEMANTIC_GROUNDING",
    "M15_COUNTERFACTUAL_IMAGINATION_PLANNING",
]

ALLOWED_GROUPS = set(EXPECTED_MODULES) | {
    "apps",
    "platform_mujoco_world",
    "platform_scene_builder",
    "platform_ipc",
    "platform_gui",
    "shared",
    "experiments",
    "legacy",
    "unclassified",
}

ALLOWED_ROLES = {
    "state",
    "runtime",
    "models",
    "memory",
    "debug",
    "visualizer",
    "training",
    "config",
    "schema",
    "entrypoint",
    "platform",
    "experiment",
    "legacy",
    "unknown",
}

ALLOWED_STATUSES = {
    "keep",
    "move_later",
    "split_later",
    "wrap_later",
    "legacy_later",
    "remove_later",
    "unclassified",
    "moved_with_wrapper",
}

ALLOWED_CONFIDENCE = {"high", "medium", "low", "unknown"}


def _load_map() -> dict:
    path = Path("docs/architecture/module_file_map.json")
    assert path.exists(), "docs/architecture/module_file_map.json is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def test_module_file_map_is_valid_json_and_has_required_top_level_keys() -> None:
    data = _load_map()
    assert data["repository"] == "Uvaltus157/Digital-Model-of-Consciousness"
    assert isinstance(data.get("version"), int)
    assert isinstance(data.get("modules"), dict)
    assert isinstance(data.get("files"), list)
    assert isinstance(data.get("unclassified"), list)
    assert isinstance(data.get("questions"), list)


def test_module_file_map_contains_all_m1_to_m15_modules() -> None:
    data = _load_map()
    modules = data["modules"]
    assert list(modules.keys()) == EXPECTED_MODULES
    for module_name in EXPECTED_MODULES:
        item = modules[module_name]
        assert "target_dir" in item
        assert item["target_dir"].startswith("src/modules/m")
        assert isinstance(item.get("files"), list)


def test_module_file_map_entries_have_required_schema() -> None:
    data = _load_map()
    required = {
        "old_path",
        "target_group",
        "target_path",
        "role",
        "status",
        "confidence",
        "reason",
        "risk",
    }
    for entry in data["files"]:
        assert required.issubset(entry), f"missing keys in map entry: {entry}"
        assert entry["target_group"] in ALLOWED_GROUPS
        assert entry["role"] in ALLOWED_ROLES
        assert entry["status"] in ALLOWED_STATUSES
        assert entry["confidence"] in ALLOWED_CONFIDENCE
        assert entry["old_path"].strip()
        assert entry["target_path"].strip()
        assert entry["reason"].strip()
        assert entry["risk"].strip()


def test_module_file_map_rules_are_explicit() -> None:
    data = _load_map()
    rules = data.get("rules", {})
    assert "no_file_moves_in_this_step" in rules
    assert "next_step_requires_compatibility_wrappers" in rules
    assert rules["next_step_requires_compatibility_wrappers"] is True
