# Codex Task: M5 Clean Canonical Module Exports

## Context

M5 now has three API layers:

### 1. Canonical public API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

### 2. Semantic internal API

```python
ConsciousDreamerCore
ConsciousDreamerMemoryThought
ConsciousDreamerInnerSpeech
ConsciousDreamerObjectImagery
```

### 3. Historical compatibility API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV21,
    ConsciousDreamerV22,
    ConsciousDreamerV23,
    ConsciousDreamerV2_3,
)
```

The previous step should have cleaned package-level exports so `models.__all__` exposes canonical/semantic names, while old V names live in `conscious_dreamer_compat.py`.

The next safe cleanup is to make the canonical module’s `__all__` clean too.

## Goal

Update:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py
```

so its `__all__` exposes only canonical and semantic names, not old V names.

Old V attributes may still exist on the module for backward compatibility, but they should not be part of the canonical module’s public `__all__`.

## Important rules

Do **not** remove old V attributes from `conscious_dreamer.py` yet.

Do **not** delete `conscious_dreamer_compat.py`.

Do **not** delete old V aliases from implementation files.

Do **not** change runtime logic.

Do **not** change model architecture, forward methods, losses, state dict keys, checkpoint loading, or config values.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

## Required behavior

### `conscious_dreamer.py`

Its `__all__` should include canonical/semantic names such as:

```python
__all__ = [
    "ConsciousDreamer",
    "ConsciousDreamerConfig",
    "ConsciousDreamerLatest",
    "ConsciousDreamerLatestConfig",
    "ConsciousDreamerCore",
    "ConsciousDreamerCoreConfig",
    "ConsciousDreamerMemoryThought",
    "ConsciousDreamerMemoryThoughtConfig",
    "ConsciousDreamerInnerSpeech",
    "ConsciousDreamerInnerSpeechConfig",
    "ConsciousDreamerObjectImagery",
    "ConsciousDreamerObjectImageryConfig",
    "make_conscious_dreamer_config_from_world",
    "CONSCIOUS_DREAMER_MODEL_VERSION",
    "CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER",
]
```

Its `__all__` should **not** include:

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

However, for now, direct attributes may still exist:

```python
hasattr(conscious_dreamer, "ConsciousDreamerV23") == True
```

Those attributes should point to compatibility aliases, preferably imported from `conscious_dreamer_compat.py`.

## Required cleanup

Search tests and code for imports of V names from canonical module:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamerV23,
)
```

Replace those imports with:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
)
```

Do not weaken assertions. If a test checks that canonical `ConsciousDreamer` is the latest model, compare it against semantic names:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamer
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import ConsciousDreamerObjectImagery

assert ConsciousDreamer is ConsciousDreamerObjectImagery
```

If a test checks old V compatibility, import old V names from `conscious_dreamer_compat.py`.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
```

The test should verify:

```python
from src.modules.m05_world_model_attention_workspace.models import conscious_dreamer

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

def test_canonical_module_all_does_not_export_versioned_names():
    exported = set(getattr(conscious_dreamer, "__all__", ()))
    assert not (exported & VERSIONED_NAMES)
```

Verify canonical names are exported:

```python
def test_canonical_module_all_exports_canonical_and_semantic_names():
    exported = set(conscious_dreamer.__all__)
    expected = {
        "ConsciousDreamer",
        "ConsciousDreamerConfig",
        "ConsciousDreamerLatest",
        "ConsciousDreamerLatestConfig",
        "ConsciousDreamerCore",
        "ConsciousDreamerCoreConfig",
        "ConsciousDreamerMemoryThought",
        "ConsciousDreamerMemoryThoughtConfig",
        "ConsciousDreamerInnerSpeech",
        "ConsciousDreamerInnerSpeechConfig",
        "ConsciousDreamerObjectImagery",
        "ConsciousDreamerObjectImageryConfig",
        "make_conscious_dreamer_config_from_world",
    }
    assert expected <= exported
```

Verify compatibility attrs still exist but are not public exports:

```python
def test_canonical_module_keeps_temporary_versioned_attributes_for_compatibility():
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import ConsciousDreamerV23
    assert conscious_dreamer.ConsciousDreamerV23 is ConsciousDreamerV23
    assert "ConsciousDreamerV23" not in conscious_dreamer.__all__
```

Also verify canonical mapping:

```python
def test_canonical_module_maps_to_object_imagery_semantic_layer():
    from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
        ConsciousDreamerObjectImagery,
        ConsciousDreamerObjectImageryConfig,
    )
    assert conscious_dreamer.ConsciousDreamer is ConsciousDreamerObjectImagery
    assert conscious_dreamer.ConsciousDreamerConfig is ConsciousDreamerObjectImageryConfig
```

## Optional AST check

If easy, add an AST test that no project file outside `conscious_dreamer.py` imports V names from canonical module.

This may already be covered by:

```text
tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
```

If it is already covered, do not duplicate.

## What not to do

Do not remove old V attrs from `conscious_dreamer.py` yet.

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
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  tests/smoke/test_conscious_dreamer_canonical_exports_clean.py

pytest tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
pytest tests/smoke/test_conscious_dreamer_package_exports.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. `conscious_dreamer.py.__all__` is clean and canonical.
2. Old V names are not public canonical exports.
3. Old V attrs still exist temporarily for backward compatibility.
4. Old V imports are centralized through `conscious_dreamer_compat.py`.
5. Smoke tests pass.
6. Runtime behavior does not change.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- canonical module __all__ cleaned
- old V attributes preserved temporarily
- versioned compatibility remains in compat facade
- no docs touched
- no runtime behavior changed
```
