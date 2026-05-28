# Codex Task: M5 Migrate Versioned Imports to Compatibility Facade

## Context

The previous cleanup step introduced a dedicated compatibility facade:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

The canonical public API remains:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

Old V names should now be imported from the compatibility facade:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV21,
    ConsciousDreamerV22,
    ConsciousDreamerV23,
    ConsciousDreamerV2_3,
)
```

## Goal

Move all versioned imports in tests/internal code to the compatibility facade.

The canonical module `conscious_dreamer.py` may keep old V attributes temporarily for backward compatibility, but new project code and smoke tests should not import V names from it anymore.

This task creates an import-boundary test that enforces:

```text
canonical module  -> canonical / semantic names
compat module     -> old V names
implementation    -> may define aliases internally
```

## Important rules

Do **not** remove old V names yet.

Do **not** delete compatibility aliases.

Do **not** change runtime logic, state dict keys, checkpoint loading, forward methods, model layers, losses, or config values.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

## Search targets

Search for imports of these names:

```text
ConsciousDreamerV2
ConsciousDreamerV21
ConsciousDreamerV22
ConsciousDreamerV23
ConsciousDreamerV2_3
ConsciousDreamerV2Config
ConsciousDreamerV21Config
ConsciousDreamerV22Config
ConsciousDreamerV23Config
```

from this module:

```text
src.modules.m05_world_model_attention_workspace.models.conscious_dreamer
```

Forbidden pattern for new code/tests:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamerV23,
)
```

Preferred replacement:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
)
```

## Required cleanup

Update smoke tests and any internal project imports so they use:

```text
conscious_dreamer_compat.py
```

for old V names.

Likely tests to inspect/update:

```text
tests/smoke/test_conscious_dreamer_canonical_api.py
tests/smoke/test_conscious_dreamer_compat_facade.py
tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
```

Do not weaken existing behavior checks. If a test currently checks that `ConsciousDreamer` maps to the latest implementation, keep that check but compare against semantic aliases or compat facade exports.

Example:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)

assert ConsciousDreamer is ConsciousDreamerObjectImagery
assert ConsciousDreamerConfig is ConsciousDreamerObjectImageryConfig
```

If a test must check old V aliases, import them from:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
)
```

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
```

The test should parse Python files with `ast`.

It should fail if a Python file imports V names from:

```text
src.modules.m05_world_model_attention_workspace.models.conscious_dreamer
```

Allowed files:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py
tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
```

If a compatibility test genuinely needs to verify the old import path still exists, allow that file narrowly and explain in a comment. Prefer not to keep such tests unless necessary.

Suggested test logic:

```python
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VERSIONED_NAMES = {
    "ConsciousDreamerV2",
    "ConsciousDreamerV21",
    "ConsciousDreamerV22",
    "ConsciousDreamerV23",
    "ConsciousDreamerV2_3",
    "ConsciousDreamerV2Config",
    "ConsciousDreamerV21Config",
    "ConsciousDreamerV22Config",
    "ConsciousDreamerV23Config",
}

CANONICAL_MODULE = "src.modules.m05_world_model_attention_workspace.models.conscious_dreamer"

ALLOWED = {
    ROOT / "src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py",
    ROOT / "tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py",
}

def _iter_py_files():
    for root in [ROOT / "src", ROOT / "tests"]:
        for path in root.rglob("*.py"):
            yield path

def test_versioned_names_are_imported_from_compat_not_canonical():
    offenders = []
    for path in _iter_py_files():
        if path in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == CANONICAL_MODULE:
                bad_names = [alias.name for alias in node.names if alias.name in VERSIONED_NAMES]
                if bad_names:
                    offenders.append(f"{path.relative_to(ROOT)} imports {bad_names} from canonical module")
    assert not offenders, (
        "Import versioned ConsciousDreamer names from conscious_dreamer_compat.py, "
        "not conscious_dreamer.py:\\n" + "\\n".join(offenders)
    )
```

## Optional behavior check

In the same test, verify compat imports still work:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
)

assert ConsciousDreamerV23 is ConsciousDreamerObjectImagery
```

## What not to do

Do not remove these attributes from `conscious_dreamer.py` yet, even if they are now discouraged:

```text
ConsciousDreamerV23
ConsciousDreamerV23Config
ConsciousDreamerV2_3
```

This task changes project imports, not external compatibility.

Do not modify:

```text
docs/
docs/html/
README*
```

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py

pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. Old V imports in tests/project code come from `conscious_dreamer_compat.py`.
2. Canonical `conscious_dreamer.py` remains focused on canonical names.
3. Old V attributes are still preserved for backward compatibility.
4. A smoke test prevents new V imports from the canonical module.
5. Runtime behavior does not change.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- versioned imports migrated to compat facade
- canonical API unchanged
- old V attributes preserved
- no docs touched
- no runtime behavior changed
```
