from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _import_from_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def test_canonical_conscious_dreamer_config_helper_is_decoupled_from_shared_config():
    path = ROOT / "src" / "shared" / "conscious_dreamer_config.py"
    modules = _import_from_modules(path)

    assert "src.shared.config" not in modules
    assert "src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery" not in modules
    assert "src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_inner_speech" not in modules
    assert "src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought" not in modules
    assert "src.shared.model_dimensions" in modules
    assert "src.modules.m05_world_model_attention_workspace.models.conscious_dreamer" in modules


def test_shared_model_dimensions_helper_has_no_m5_model_imports():
    path = ROOT / "src" / "shared" / "model_dimensions.py"
    modules = _import_from_modules(path)

    assert not any("conscious_dreamer" in module for module in modules)
