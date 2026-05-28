# Codex Task: M5 Guard Core Facade Boundary

## Context

The previous step added the semantic core facade:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py
```

The old implementation file still exists:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
```

For now, `conscious_dreamer_full.py` remains the implementation/compatibility file. New M5 code should import the core layer through the facade:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
```

not directly from:

```python
conscious_dreamer_full.py
```

## Goal

Add a smoke test that protects this boundary:

```text
new M5 internal code -> conscious_dreamer_core.py facade
compatibility implementation -> conscious_dreamer_full.py
```

If direct imports from `conscious_dreamer_full.py` remain in internal M5 files, replace them with imports from `conscious_dreamer_core.py`.

This is still not a file rename and not a code move.

## Important rules

Do **not** delete `conscious_dreamer_full.py`.

Do **not** move implementation code yet.

Do **not** rename files in git.

Do **not** delete compatibility aliases:

```text
ConsciousDreamerV2
ConsciousDreamerV2Config
```

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** change runtime logic, model architecture, forward methods, losses, checkpoint loading, state dict keys, or config values.

## Search targets

Search for imports from:

```text
src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full
```

and relative imports equivalent to:

```python
from .conscious_dreamer_full import ...
from src.modules...conscious_dreamer_full import ...
```

Also search raw text for:

```text
conscious_dreamer_full
ConsciousDreamerV2
ConsciousDreamerV2Config
ConsciousDreamerCore
ConsciousDreamerCoreConfig
```

## Required cleanup

If any M5 file imports `ConsciousDreamerCore` or `ConsciousDreamerCoreConfig` from `conscious_dreamer_full.py`, switch it to:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
```

Allowed to keep direct imports from `conscious_dreamer_full.py` only in:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py
tests/smoke/test_conscious_dreamer_core_facade.py
tests/smoke/test_conscious_dreamer_core_semantic_base.py
```

If some other compatibility test needs it, add a narrow allowlist entry and explain why.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_core_import_boundary.py
```

The test should parse Python files with `ast`.

It should fail if direct imports from `conscious_dreamer_full.py` appear outside the allowed files.

Suggested logic:

```python
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]

ALLOWED = {
    ROOT / "src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py",
    ROOT / "tests/smoke/test_conscious_dreamer_core_facade.py",
    ROOT / "tests/smoke/test_conscious_dreamer_core_semantic_base.py",
    ROOT / "tests/smoke/test_conscious_dreamer_core_import_boundary.py",
}

def test_no_direct_core_imports_from_full_outside_facade():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.endswith("conscious_dreamer_full"):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "Import core from conscious_dreamer_core.py facade, not conscious_dreamer_full.py:\\n" + "\\n".join(offenders)
```

Extend it to also scan `tests/smoke/*.py`, but allow explicitly listed compatibility tests.

## Required behavior check

The new test should also verify the facade still points to the full implementation:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full import (
    ConsciousDreamerCore as FullConsciousDreamerCore,
    ConsciousDreamerCoreConfig as FullConsciousDreamerCoreConfig,
)

assert ConsciousDreamerCore is FullConsciousDreamerCore
assert ConsciousDreamerCoreConfig is FullConsciousDreamerCoreConfig
```

## What not to change

Do not modify:

```text
docs/
docs/html/
README*
```

Do not rename:

```text
conscious_dreamer_full.py
conscious_dreamer_core.py
```

Do not remove imports needed by `conscious_dreamer_core.py`.

Do not remove compatibility tests.

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_core_import_boundary.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py

pytest tests/smoke/test_conscious_dreamer_core_import_boundary.py
pytest tests/smoke/test_conscious_dreamer_core_facade.py
pytest tests/smoke/test_conscious_dreamer_core_semantic_base.py
pytest tests/smoke/test_conscious_dreamer_semantic_classes_primary.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. `conscious_dreamer_core.py` is the only normal facade into the base core.
2. `conscious_dreamer_full.py` remains as implementation/compatibility file.
3. New internal code does not import core directly from `conscious_dreamer_full.py`.
4. Smoke tests pass.
5. No runtime behavior changes.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- direct full imports replaced with core facade where needed
- full implementation file preserved
- no docs touched
- no runtime behavior changed
```
