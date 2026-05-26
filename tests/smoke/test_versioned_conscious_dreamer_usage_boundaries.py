from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SMOKE_ROOT = PROJECT_ROOT / "tests" / "smoke"
INTERNAL_M5_MODELS_ROOT = SRC_ROOT / "modules" / "m05_world_model_attention_workspace" / "models"

VERSIONED_MODULE_PARTS = (
    "conscious_dreamer_object_imagery",
    "conscious_dreamer_inner_speech",
    "conscious_dreamer_memory_thought",
    "conscious_dreamer_full",
)
VERSIONED_NAMES = {
    "ConsciousDreamerV2",
    "ConsciousDreamerV21",
    "ConsciousDreamerV22",
    "ConsciousDreamerV23",
    "ConsciousDreamerV2_3",
    "ConsciousDreamerV21Config",
    "ConsciousDreamerV22Config",
    "ConsciousDreamerV23Config",
    "v22_cfg",
    "v23_cfg",
    "make_v22_config_from_world",
    "make_v23_config_from_unified",
    "create_v23_config",
    "create_conscious_dreamer_v23",
}

ALLOWED_COMPAT_NAMES_BY_FILE = {
    Path("src/apps/runner_model_factory.py"): {
        "create_v23_config",
        "create_conscious_dreamer_v23",
    },
    Path("src/apps/runner_unified_init.py"): {
        "v23_cfg",
    },
    Path("src/apps/unified_conscious_viewer.py"): {
        "v22_cfg",
        "v23_cfg",
    },
    Path("src/shared/config.py"): {
        "make_v23_config_from_unified",
    },
}


def _python_files() -> list[Path]:
    src_files = sorted(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)
    smoke_files = sorted(path for path in SMOKE_ROOT.glob("*.py") if path.name != "__init__.py")
    return src_files + smoke_files


def _relative(path: Path) -> Path:
    return path.relative_to(PROJECT_ROOT)


def _is_internal_m5_model(path: Path) -> bool:
    return path.is_relative_to(INTERNAL_M5_MODELS_ROOT)


def _is_allowed_compat_test(path: Path) -> bool:
    if not path.is_relative_to(SMOKE_ROOT):
        return False
    name = path.name
    return (
        "canonical" in name
        or "compatibility" in name
        or name == "test_shared_config_no_versioned_conscious_dreamer.py"
        or name == Path(__file__).name
    )


def _allowed_compat_names(path: Path) -> set[str]:
    return ALLOWED_COMPAT_NAMES_BY_FILE.get(_relative(path), set())


def _is_allowed(path: Path, name: str) -> bool:
    return _is_internal_m5_model(path) or _is_allowed_compat_test(path) or name in _allowed_compat_names(path)


def _record(offenders: list[str], path: Path, node: ast.AST, reason: str) -> None:
    rel = _relative(path)
    line = getattr(node, "lineno", "?")
    offenders.append(f"{rel}:{line}: {reason}")


def test_versioned_conscious_dreamer_names_stay_behind_m5_boundary():
    offenders: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(part in alias.name for part in VERSIONED_MODULE_PARTS) and not _is_allowed(path, alias.name):
                        _record(offenders, path, node, f"imports versioned M5 module {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(part in module for part in VERSIONED_MODULE_PARTS) and not _is_allowed(path, module):
                    _record(offenders, path, node, f"imports from versioned M5 module {module}")

                for alias in node.names:
                    if alias.name in VERSIONED_NAMES and not _is_allowed(path, alias.name):
                        _record(offenders, path, node, f"imports versioned M5 name {alias.name}")

            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name in VERSIONED_NAMES and not _is_allowed(path, node.name):
                    _record(offenders, path, node, f"defines external versioned M5 name {node.name}")

            elif isinstance(node, ast.Name):
                if node.id in VERSIONED_NAMES and not _is_allowed(path, node.id):
                    _record(offenders, path, node, f"uses versioned M5 name {node.id}")

            elif isinstance(node, ast.Attribute):
                if node.attr in VERSIONED_NAMES and not _is_allowed(path, node.attr):
                    _record(offenders, path, node, f"uses versioned M5 attribute {node.attr}")

    assert not offenders, "Versioned ConsciousDreamer names escaped the allowed boundary:\n" + "\n".join(offenders)
