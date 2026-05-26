from __future__ import annotations

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2] / "src" / "apps"
FORBIDDEN_IMPORT_MODULE_PARTS = (
    "conscious_dreamer_inner_speech",
    "conscious_dreamer_object_imagery",
    "conscious_dreamer_memory_thought",
    "conscious_dreamer_full",
)
FORBIDDEN_IMPORTED_NAMES = {
    "ConsciousDreamerV2",
    "ConsciousDreamerV21",
    "ConsciousDreamerV22",
    "ConsciousDreamerV23",
    "ConsciousDreamerV2_3",
    "ConsciousDreamerV21Config",
    "ConsciousDreamerV22Config",
    "ConsciousDreamerV23Config",
    "make_v22_config_from_world",
    "make_v23_config_from_unified",
}
FORBIDDEN_CALL_NAMES = {
    "ConsciousDreamerV2",
    "ConsciousDreamerV21",
    "ConsciousDreamerV22",
    "ConsciousDreamerV23",
    "ConsciousDreamerV2_3",
    "make_v22_config_from_world",
    "make_v23_config_from_unified",
}


def _iter_app_python_files():
    for path in sorted(APP_ROOT.glob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_app_layer_does_not_import_versioned_conscious_dreamer_layers():
    offenders: list[str] = []
    for path in _iter_app_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(part in module for part in FORBIDDEN_IMPORT_MODULE_PARTS):
                    offenders.append(f"{path.relative_to(APP_ROOT.parent.parent)} imports from {module}")
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORTED_NAMES:
                        offenders.append(f"{path.relative_to(APP_ROOT.parent.parent)} imports {alias.name}")
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in FORBIDDEN_CALL_NAMES:
                    offenders.append(f"{path.relative_to(APP_ROOT.parent.parent)} calls {name}")

    assert not offenders, "App-level code must use canonical ConsciousDreamer API:\n" + "\n".join(offenders)
