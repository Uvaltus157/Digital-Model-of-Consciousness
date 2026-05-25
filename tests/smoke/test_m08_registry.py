from __future__ import annotations

from src.modules.m08_debug_visual_control.module_debug_graph import (
    DEFAULT_FLAGS,
    LEGACY_TRAINING_MODULES,
    M15_ARCHITECTURE_MODULES,
    M15_ARCHITECTURE_EDGES,
)
from src.modules.m08_debug_visual_control.module_registry import MODULE_BY_ID, MODULE_REGISTRY
from src.modules.m08_debug_visual_control.status_schema import ModuleDebugStatus, ModuleDebugStatusPayload


def test_registry_contains_m1_to_m15_in_order() -> None:
    ids = [entry.module_id for entry in MODULE_REGISTRY]
    assert ids == [f"M{i}" for i in range(1, 16)]
    assert MODULE_BY_ID["M1"].package == "m01_object_imagery"
    assert MODULE_BY_ID["M15"].package == "m15_counterfactual_imagination_planning"


def test_debug_graph_uses_registry_architecture_cards() -> None:
    keys = [key for key, _label, _title, _color in M15_ARCHITECTURE_MODULES]
    assert len(M15_ARCHITECTURE_MODULES) == 15
    assert "object_imagery" in keys
    assert "world_model" in keys
    assert "m10" in keys
    assert "m15" in keys
    assert len(M15_ARCHITECTURE_EDGES) > 0


def test_legacy_training_cards_keep_current_runtime_keys() -> None:
    keys = [key for key, _label, _title, _color in LEGACY_TRAINING_MODULES]
    assert "world_model" in keys
    assert "object_imagery" in keys
    assert "long_dynamic_memory" in keys
    assert "self_core" in keys
    assert "inner_speech" in keys
    assert "action_heads" in keys
    assert set(DEFAULT_FLAGS).issuperset(keys)


def test_status_schema_is_json_friendly() -> None:
    status = ModuleDebugStatus(
        module_id="M8",
        title="M8_DEBUG_VISUAL_CONTROL",
        active=True,
        confidence=0.75,
        extra={"source": "test"},
    )
    payload = ModuleDebugStatusPayload(
        ready=True,
        updated_at=123.0,
        modules={"M8": status},
        trainable_counts={"active": 1},
    )
    data = payload.to_dict()
    assert data["ready"] is True
    assert data["modules"]["M8"]["module_id"] == "M8"
    assert data["modules"]["M8"]["confidence"] == 0.75
