# Codex Task: M5 Add Semantic Core Module Facade

## Context

The M5 cleanup has moved class names toward semantic names:

```text
ConsciousDreamerCore
  -> ConsciousDreamerMemoryThought
    -> ConsciousDreamerInnerSpeech
      -> ConsciousDreamerObjectImagery
        -> ConsciousDreamer
```

The old version names should still work as aliases:

```text
ConsciousDreamerV2  = ConsciousDreamerCore
ConsciousDreamerV21 = ConsciousDreamerMemoryThought
ConsciousDreamerV22 = ConsciousDreamerInnerSpeech
ConsciousDreamerV23 = ConsciousDreamerObjectImagery
```

The remaining naming problem is file-level: the base/core implementation still lives in:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
```

Do not move implementation yet. This task only adds a semantic facade module and switches internal imports to it.

## Goal

Create a new semantic module:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py
```

This module should re-export the core/base classes from `conscious_dreamer_full.py`.

Then update M5 internal imports so new code imports core through:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
```

The old file `conscious_dreamer_full.py` remains untouched as the implementation location for now.

## Important rules

Do **not** delete `conscious_dreamer_full.py`.

Do **not** move implementation code yet.

Do **not** rename files in git yet.

Do **not** delete `ConsciousDreamerV2` or `ConsciousDreamerV2Config`.

Do **not** touch documentation.

Do **not** change runtime logic, model architecture, losses, checkpoint loading, state dict keys, or config values.

## Required changes

### 1. Add `conscious_dreamer_core.py`

Create:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py
```

Content should be a compatibility-safe facade:

```python
from __future__ import annotations

\"\"\"Semantic core facade for the base M5 ConsciousDreamer implementation.

The implementation still lives in `conscious_dreamer_full.py` for backward
compatibility. New internal M5 code should import the core layer from this file.
\"\"\"

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
    ConsciousDreamerV2,
    ConsciousDreamerV2Config,
)

__all__ = [
    "ConsciousDreamerCore",
    "ConsciousDreamerCoreConfig",
    "ConsciousDreamerV2",
    "ConsciousDreamerV2Config",
]
```

If `conscious_dreamer_full.py` does not yet expose `ConsciousDreamerCore`, complete the previous semantic-core step first.

### 2. Update `conscious_dreamer_memory_thought.py`

If it imports core directly from `conscious_dreamer_full.py`, change it to import from:

```text
conscious_dreamer_core.py
```

Allowed:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
```

Not preferred anymore:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
```

### 3. Update `conscious_dreamer.py`

If it exports `ConsciousDreamerCore` / `ConsciousDreamerCoreConfig`, import them from:

```text
conscious_dreamer_core.py
```

not from `conscious_dreamer_full.py`.

### 4. Update `models/__init__.py`

If package-level exports include core names, import them from:

```text
conscious_dreamer_core.py
```

not from `conscious_dreamer_full.py`.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_core_facade.py
```

The test should verify:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
    ConsciousDreamerV2,
    ConsciousDreamerV2Config,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_full import (
    ConsciousDreamerCore as FullConsciousDreamerCore,
    ConsciousDreamerCoreConfig as FullConsciousDreamerCoreConfig,
    ConsciousDreamerV2 as FullConsciousDreamerV2,
    ConsciousDreamerV2Config as FullConsciousDreamerV2Config,
)

assert ConsciousDreamerCore is FullConsciousDreamerCore
assert ConsciousDreamerCoreConfig is FullConsciousDreamerCoreConfig
assert ConsciousDreamerV2 is FullConsciousDreamerV2
assert ConsciousDreamerV2Config is FullConsciousDreamerV2Config
```

Also verify package-level exports, if available:

```python
from src.modules.m05_world_model_attention_workspace.models import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
```

### Optional AST check

If easy, add AST assertions that:

```text
conscious_dreamer_memory_thought.py
conscious_dreamer.py
models/__init__.py
```

do not import `ConsciousDreamerCore` directly from `conscious_dreamer_full.py`.

They should import core through `conscious_dreamer_core.py`.

Do not forbid imports of `conscious_dreamer_full.py` everywhere yet, because compatibility tests and the facade itself may still need it.

## What not to do

Do not edit docs:

```text
docs/
docs/html/
README*
```

Do not delete:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
```

Do not replace all repository references to `conscious_dreamer_full.py` yet. This task is only about establishing the semantic facade and switching M5 internal core imports.

## Commands to run

Run:

```bash
python3 -m py_compile \\
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py \\
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py \\
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py \\
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \\
  src/modules/m05_world_model_attention_workspace/models/__init__.py \\
  tests/smoke/test_conscious_dreamer_core_facade.py

pytest tests/smoke/test_conscious_dreamer_core_facade.py
pytest tests/smoke/test_conscious_dreamer_core_semantic_base.py
pytest tests/smoke/test_conscious_dreamer_semantic_classes_primary.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. A semantic module `conscious_dreamer_core.py` exists.
2. New internal M5 imports can use `conscious_dreamer_core.py`.
3. `conscious_dreamer_full.py` remains as implementation and compatibility file.
4. No runtime behavior changes.
5. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- semantic core facade added
- old full implementation file preserved
- M5 internal core imports switched to facade
- no docs touched
- no runtime behavior changed
```
