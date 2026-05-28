# Codex Task: M5 Remove Versioned Attributes from Canonical Module

## Context

M5 now has separated API layers:

### Canonical public API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

### Semantic internal API

```text
ConsciousDreamerCore
  -> ConsciousDreamerMemoryThought
    -> ConsciousDreamerInnerSpeech
      -> ConsciousDreamerObjectImagery
        -> ConsciousDreamer
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

Previous tasks made old V names compatibility-only and moved versioned model aliases into:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

The canonical module `conscious_dreamer.py` may still temporarily expose old V attributes for compatibility. Now remove those temporary versioned attributes from the canonical module.

## Goal

Make `conscious_dreamer.py` strictly canonical/semantic.

After this task:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamer
```

works.

But this should no longer work:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamerV23
```

Old V names must still work through:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import ConsciousDreamerV23
```

## Important rules

Do **not** delete `conscious_dreamer_compat.py`.

Do **not** delete old V names from `conscious_dreamer_compat.py`.

Do **not** delete factory aliases yet:

```text
create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified
```

Do **not** delete runtime cfg compatibility assignments yet:

```text
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

Do **not** change runtime logic, model architecture, checkpoint loading, state dict keys, losses, or training behavior.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

## Required changes

### 1. Update `conscious_dreamer.py`

Remove imports/assignments/exports for old V names from canonical module:

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

The canonical module should keep only canonical/semantic names, for example:

```python
ConsciousDreamer
ConsciousDreamerConfig
ConsciousDreamerLatest
ConsciousDreamerLatestConfig

ConsciousDreamerCore
ConsciousDreamerCoreConfig
ConsciousDreamerMemoryThought
ConsciousDreamerMemoryThoughtConfig
ConsciousDreamerInnerSpeech
ConsciousDreamerInnerSpeechConfig
ConsciousDreamerObjectImagery
ConsciousDreamerObjectImageryConfig

make_conscious_dreamer_config_from_world

CONSCIOUS_DREAMER_MODEL_VERSION
CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER
```

Its `__all__` should already contain only canonical/semantic names; keep it that way.

### 2. Update tests that expected temporary V attrs

Some earlier tests may check:

```python
hasattr(conscious_dreamer, "ConsciousDreamerV23")
```

or:

```python
conscious_dreamer.ConsciousDreamerV23 is ...
```

Update those tests. The new expectation is:

```python
assert not hasattr(conscious_dreamer, "ConsciousDreamerV23")
```

and compatibility should be checked through:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import ConsciousDreamerV23
```

Do not weaken compatibility checks. Move them to `conscious_dreamer_compat.py`.

Likely tests to inspect:

```text
tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
tests/smoke/test_conscious_dreamer_api_integrity.py
tests/smoke/test_conscious_dreamer_compat_facade.py
tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
```

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_canonical_has_no_versioned_attrs.py
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

def test_canonical_module_has_no_versioned_attributes():
    for name in VERSIONED_NAMES:
        assert not hasattr(conscious_dreamer, name), name
```

Verify canonical names still exist:

```python
def test_canonical_module_still_exports_canonical_names():
    assert conscious_dreamer.ConsciousDreamer is conscious_dreamer.ConsciousDreamerObjectImagery
    assert conscious_dreamer.ConsciousDreamerConfig is conscious_dreamer.ConsciousDreamerObjectImageryConfig
    assert "ConsciousDreamer" in conscious_dreamer.__all__
    assert "ConsciousDreamerV23" not in conscious_dreamer.__all__
```

Verify compatibility facade still works:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
    ConsciousDreamerV23Config,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)

assert ConsciousDreamerV23 is ConsciousDreamerObjectImagery
assert ConsciousDreamerV23Config is ConsciousDreamerObjectImageryConfig
```

### Optional AST check

Add an AST check that `conscious_dreamer.py` contains no imports of old V names from `conscious_dreamer_compat.py`.

It is now forbidden for the canonical module to import compatibility V names.

## Required import-boundary update

Update or add an AST test to enforce:

1. No project file imports old V names from canonical module.
2. Canonical module itself does not expose old V names.
3. Old V names are imported from `conscious_dreamer_compat.py` only in compatibility tests or code that explicitly needs compatibility.

If existing boundary tests already cover (1), only add (2).

## What not to do

Do not remove old V names from `conscious_dreamer_compat.py`.

Do not remove old V references from compatibility tests unless replacing them with compat imports.

Do not edit docs.

Do not change runtime behavior.

Do not start MuJoCo/viewer.

## Commands to run

Run:

```bash
python3 -m py_compile \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  tests/smoke/test_conscious_dreamer_canonical_has_no_versioned_attrs.py

pytest tests/smoke/test_conscious_dreamer_canonical_has_no_versioned_attrs.py
pytest tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
pytest tests/smoke
```

## Expected result

After this task:

1. `conscious_dreamer.py` is canonical/semantic only.
2. Old V names are no longer attributes of canonical module.
3. Old V names remain available through `conscious_dreamer_compat.py`.
4. Tests and internal code no longer import old V names from canonical module.
5. Runtime behavior does not change.
6. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Alias cleanup:
- canonical module versioned attrs removed
- compatibility facade still provides old V names
- tests updated to import V names from compat only

Notes:
- factory aliases not removed
- runtime cfg aliases not removed
- no docs touched
- no runtime behavior changed
```
