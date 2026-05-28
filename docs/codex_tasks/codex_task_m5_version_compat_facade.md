# Codex Task: M5 Add Version Compatibility Facade

## Context

The M5 naming cleanup has reached this state:

```text
ConsciousDreamerCore
  -> ConsciousDreamerMemoryThought
    -> ConsciousDreamerInnerSpeech
      -> ConsciousDreamerObjectImagery
        -> ConsciousDreamer
```

Old version names are still required for backward compatibility:

```text
ConsciousDreamerV2
ConsciousDreamerV21
ConsciousDreamerV22
ConsciousDreamerV23
ConsciousDreamerV2_3
```

The previous tasks introduced semantic class names, semantic bases, and a core facade. The next safe step is to centralize old versioned exports in one compatibility facade.

## Goal

Create a dedicated compatibility module:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

This file should be the single normal place that re-exports historical V-names.

The canonical API remains:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
)
```

The compatibility API becomes:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV21,
    ConsciousDreamerV22,
    ConsciousDreamerV23,
)
```

Do not remove versioned aliases from the implementation files yet. This step only adds a central facade and updates public compatibility imports to use it.

## Important rules

Do **not** rename files.

Do **not** delete old V names.

Do **not** change model logic.

Do **not** change state dict keys, checkpoint loading, forward methods, losses, or config values.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** edit app/shared runtime code unless a tiny import compatibility fix is needed by tests.

## Required changes

### 1. Add `conscious_dreamer_compat.py`

Create:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

Suggested content:

```python
from __future__ import annotations

\"\"\"Backward-compatible versioned exports for historical M5 names.

New code should use semantic/canonical names:
    ConsciousDreamer
    ConsciousDreamerConfig
    ConsciousDreamerCore
    ConsciousDreamerMemoryThought
    ConsciousDreamerInnerSpeech
    ConsciousDreamerObjectImagery

This module exists so older imports of V2/V21/V22/V23 names keep working while
the implementation moves to semantic class names.
\"\"\"

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerV2,
    ConsciousDreamerV2Config,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought import (
    ConsciousDreamerV21,
    ConsciousDreamerV21Config,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_inner_speech import (
    ConsciousDreamerV22,
    ConsciousDreamerV22Config,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
)

ConsciousDreamerV2_3 = ConsciousDreamerV23

__all__ = [
    "ConsciousDreamerV2",
    "ConsciousDreamerV2Config",
    "ConsciousDreamerV21",
    "ConsciousDreamerV21Config",
    "ConsciousDreamerV22",
    "ConsciousDreamerV22Config",
    "ConsciousDreamerV23",
    "ConsciousDreamerV23Config",
    "ConsciousDreamerV2_3",
]
```

### 2. Update `conscious_dreamer.py`

If `conscious_dreamer.py` still imports versioned V23 names directly from implementation files for compatibility exports, switch those imports to the new compat facade.

Canonical semantic imports should remain semantic, for example:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
    ConsciousDreamerV2_3,
)
```

Keep:

```python
ConsciousDreamer = ConsciousDreamerObjectImagery
ConsciousDreamerConfig = ConsciousDreamerObjectImageryConfig
```

### 3. Update `models/__init__.py`

If package-level exports include old V names, import them from:

```text
conscious_dreamer_compat.py
```

not directly from implementation files.

Keep canonical exports unchanged:

```python
ConsciousDreamer
ConsciousDreamerConfig
```

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_compat_facade.py
```

The test should verify:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV2Config,
    ConsciousDreamerV21,
    ConsciousDreamerV21Config,
    ConsciousDreamerV22,
    ConsciousDreamerV22Config,
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
    ConsciousDreamerV2_3,
)

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought import (
    ConsciousDreamerMemoryThought,
    ConsciousDreamerMemoryThoughtConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_inner_speech import (
    ConsciousDreamerInnerSpeech,
    ConsciousDreamerInnerSpeechConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)

assert ConsciousDreamerV2 is ConsciousDreamerCore
assert ConsciousDreamerV2Config is ConsciousDreamerCoreConfig
assert ConsciousDreamerV21 is ConsciousDreamerMemoryThought
assert ConsciousDreamerV21Config is ConsciousDreamerMemoryThoughtConfig
assert ConsciousDreamerV22 is ConsciousDreamerInnerSpeech
assert ConsciousDreamerV22Config is ConsciousDreamerInnerSpeechConfig
assert ConsciousDreamerV23 is ConsciousDreamerObjectImagery
assert ConsciousDreamerV23Config is ConsciousDreamerObjectImageryConfig
assert ConsciousDreamerV2_3 is ConsciousDreamerV23
```

Also verify canonical API still works:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
)
assert ConsciousDreamer is ConsciousDreamerObjectImagery
assert ConsciousDreamerConfig is ConsciousDreamerObjectImageryConfig
```

### Optional AST boundary check

If easy, add an AST check that the following files do not import versioned names directly from implementation files:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py
src/modules/m05_world_model_attention_workspace/models/__init__.py
```

They should import old V names through:

```text
conscious_dreamer_compat.py
```

Allowed direct versioned imports remain inside:

```text
conscious_dreamer_compat.py
implementation files that define the aliases
compatibility tests
```

## What not to do

Do not remove versioned aliases from implementation files yet.

Do not change these files unless explicitly required:

```text
src/apps/
src/shared/
src/platform/
```

Do not edit docs.

Do not rename:

```text
conscious_dreamer_full.py
conscious_dreamer_core.py
conscious_dreamer_memory_thought.py
conscious_dreamer_inner_speech.py
conscious_dreamer_object_imagery.py
```

## Commands to run

Run:

```bash
python3 -m py_compile \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/__init__.py \
  tests/smoke/test_conscious_dreamer_compat_facade.py

pytest tests/smoke/test_conscious_dreamer_compat_facade.py
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

1. `conscious_dreamer_compat.py` exists.
2. Old V names have a single compatibility facade.
3. Canonical public API remains unchanged.
4. Semantic internal names remain primary.
5. Tests pass.
6. Runtime behavior does not change.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- versioned compatibility facade added
- canonical API unchanged
- semantic internal names preserved
- old V names preserved
- no docs touched
- no runtime behavior changed
```
