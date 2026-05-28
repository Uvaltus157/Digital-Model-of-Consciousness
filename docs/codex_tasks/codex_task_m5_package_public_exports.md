# Codex Task: M5 Clean Package-Level Public Exports

## Context

The M5 cleanup now has three API layers:

### Canonical public API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

### Semantic internal API

```python
ConsciousDreamerCore
ConsciousDreamerMemoryThought
ConsciousDreamerInnerSpeech
ConsciousDreamerObjectImagery
```

### Historical compatibility API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV21,
    ConsciousDreamerV22,
    ConsciousDreamerV23,
    ConsciousDreamerV2_3,
)
```

The previous step should have moved versioned imports in project code/tests to `conscious_dreamer_compat.py`.

## Goal

Make the package-level `models/__init__.py` clean and intentional.

Package-level imports should expose canonical and semantic names, not old V names.

Preferred:

```python
from src.modules.m05_world_model_attention_workspace.models import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    ConsciousDreamerCore,
    ConsciousDreamerMemoryThought,
    ConsciousDreamerInnerSpeech,
    ConsciousDreamerObjectImagery,
)
```

Versioned names should be imported only from:

```python
src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat
```

## Important rules

Do **not** delete old V aliases from implementation files.

Do **not** delete `conscious_dreamer_compat.py`.

Do **not** change runtime logic.

Do **not** change model architecture, forward methods, losses, state dict keys, checkpoint loading, or config values.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

## Target file

Main target:

```text
src/modules/m05_world_model_attention_workspace/models/__init__.py
```

Possible test files only:

```text
tests/smoke/
```

## Required `models/__init__.py` behavior

Package-level `models` should export:

```python
ConsciousDreamer
ConsciousDreamerConfig
ConsciousDreamerLatest
ConsciousDreamerLatestConfig
make_conscious_dreamer_config_from_world

ConsciousDreamerCore
ConsciousDreamerCoreConfig
ConsciousDreamerMemoryThought
ConsciousDreamerMemoryThoughtConfig
ConsciousDreamerInnerSpeech
ConsciousDreamerInnerSpeechConfig
ConsciousDreamerObjectImagery
ConsciousDreamerObjectImageryConfig

CONSCIOUS_DREAMER_MODEL_VERSION
CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER
```

Package-level `models/__init__.py` should **not** export these old V names through `__all__`:

```python
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

Those old V names belong in:

```text
conscious_dreamer_compat.py
```

## If package-level V imports currently exist

If `models/__init__.py` currently imports V names, remove those imports unless they are needed internally.

If tests still import versioned names from package-level `models`, update those tests to import from:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (...)
```

Do not weaken behavioral assertions.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_package_exports.py
```

The test should verify package-level canonical and semantic exports:

```python
from src.modules.m05_world_model_attention_workspace import models

def test_package_exports_canonical_and_semantic_names():
    assert hasattr(models, "ConsciousDreamer")
    assert hasattr(models, "ConsciousDreamerConfig")
    assert hasattr(models, "make_conscious_dreamer_config_from_world")

    assert hasattr(models, "ConsciousDreamerCore")
    assert hasattr(models, "ConsciousDreamerCoreConfig")
    assert hasattr(models, "ConsciousDreamerMemoryThought")
    assert hasattr(models, "ConsciousDreamerMemoryThoughtConfig")
    assert hasattr(models, "ConsciousDreamerInnerSpeech")
    assert hasattr(models, "ConsciousDreamerInnerSpeechConfig")
    assert hasattr(models, "ConsciousDreamerObjectImagery")
    assert hasattr(models, "ConsciousDreamerObjectImageryConfig")
```

Verify canonical mapping:

```python
from src.modules.m05_world_model_attention_workspace.models import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)

assert ConsciousDreamer is ConsciousDreamerObjectImagery
assert ConsciousDreamerConfig is ConsciousDreamerObjectImageryConfig
```

Verify `__all__` does not expose old V names:

```python
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

def test_package_all_does_not_export_versioned_names():
    exported = set(getattr(models, "__all__", ()))
    assert not (exported & VERSIONED_NAMES)
```

Verify old V names remain available from compat:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
)

assert ConsciousDreamerV23 is ConsciousDreamerObjectImagery
```

## Optional AST check

If easy, extend the test to parse `models/__init__.py` and fail if it imports old V names from implementation modules.

Allowed:

```text
models/__init__.py imports canonical/semantic names
conscious_dreamer_compat.py imports old V names
```

Not preferred:

```python
from ...conscious_dreamer_compat import ConsciousDreamerV23
```

inside `models/__init__.py`, unless there is a strong compatibility reason.

## What not to do

Do not remove versioned attributes from:

```text
conscious_dreamer_compat.py
implementation files
```

Do not remove old aliases inside implementation files.

Do not edit:

```text
docs/
docs/html/
README*
```

Do not change app/shared runtime behavior.

## Commands to run

Run:

```bash
python3 -m py_compile \
  src/modules/m05_world_model_attention_workspace/models/__init__.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  tests/smoke/test_conscious_dreamer_package_exports.py

pytest tests/smoke/test_conscious_dreamer_package_exports.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. Package-level `models` exports canonical and semantic names.
2. Package-level `models.__all__` does not expose old V names.
3. Old V names remain available from `conscious_dreamer_compat.py`.
4. Runtime behavior does not change.
5. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- package-level exports cleaned
- versioned names kept in compat facade
- canonical API unchanged
- no docs touched
- no runtime behavior changed
```
