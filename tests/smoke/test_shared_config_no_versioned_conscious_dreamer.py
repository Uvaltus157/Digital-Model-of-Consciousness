from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "shared" / "config.py"
FORBIDDEN_IMPORT_MODULE_PARTS = (
    "conscious_dreamer_object_imagery",
    "conscious_dreamer_inner_speech",
    "conscious_dreamer_memory_thought",
    "conscious_dreamer_full",
)
FORBIDDEN_IMPORTED_NAMES = {
    "ConsciousDreamerV23Config",
    "ConsciousDreamerV22Config",
    "ConsciousDreamerV21Config",
}


def _runner_cfg():
    return SimpleNamespace(
        mujoco_world=SimpleNamespace(height=72, width=128),
        action_dim=24,
        embodied_dim=15,
        hand_motor_dim=44,
        tactile_dim=42,
        body_state_dim=83,
    )


def test_shared_config_does_not_import_versioned_conscious_dreamer_layers():
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue

        module = node.module or ""
        if any(part in module for part in FORBIDDEN_IMPORT_MODULE_PARTS):
            offenders.append(f"imports from {module}")

        for alias in node.names:
            if alias.name in FORBIDDEN_IMPORTED_NAMES:
                offenders.append(f"imports {alias.name}")

    assert not offenders, "src/shared/config.py must use canonical ConsciousDreamer API:\n" + "\n".join(offenders)


def test_legacy_v23_config_alias_matches_canonical_helper():
    from src.shared.config import make_v23_config_from_unified
    from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified

    cfg = _runner_cfg()

    legacy = make_v23_config_from_unified(cfg)
    canonical = make_conscious_dreamer_config_from_unified(cfg)

    assert type(legacy) is type(canonical)
    assert legacy.data.image_height == canonical.data.image_height == 72
    assert legacy.data.image_width == canonical.data.image_width == 128
    assert legacy.data.action_dim == canonical.data.action_dim == 24
    assert legacy.data.embodied_dim == canonical.data.embodied_dim == 15
    assert legacy.data.hand_motor_dim == canonical.data.hand_motor_dim == 44
    assert legacy.data.tactile_dim == canonical.data.tactile_dim == 42
    assert legacy.data.body_state_dim == canonical.data.body_state_dim == 83
    assert legacy.object_imagery.image_size == canonical.object_imagery.image_size == 96
