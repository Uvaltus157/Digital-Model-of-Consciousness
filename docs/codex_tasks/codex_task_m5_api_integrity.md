# Codex Task: M5 Final ConsciousDreamer API Integrity Check

## Context

The M5 cleanup has established three clear API layers:

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

The previous steps added:

- semantic aliases;
- semantic class definitions;
- semantic bases;
- `conscious_dreamer_core.py` facade;
- `conscious_dreamer_compat.py` facade;
- cleaned package/canonical exports;
- boundary tests for versioned imports.

Now we need one final integrity test that checks the whole M5 API structure in one place.

## Goal

Add a final smoke test that validates the complete M5 ConsciousDreamer API contract.

This task should not change runtime behavior.

This task should not rename files.

This task should not remove old V aliases.

## Target

Add:

```text
tests/smoke/test_conscious_dreamer_api_integrity.py
```

This test should be the final high-level M5 API contract test.

## Required checks

### 1. Canonical API maps to final semantic layer

Verify:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    ConsciousDreamerLatest,
    ConsciousDreamerLatestConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_object_imagery import (
    ConsciousDreamerObjectImagery,
    ConsciousDreamerObjectImageryConfig,
)

assert ConsciousDreamer is ConsciousDreamerObjectImagery
assert ConsciousDreamerConfig is ConsciousDreamerObjectImageryConfig
assert ConsciousDreamerLatest is ConsciousDreamer
assert ConsciousDreamerLatestConfig is ConsciousDreamerConfig
```

### 2. Semantic inheritance chain is correct

Verify:

```python
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

assert issubclass(ConsciousDreamerMemoryThought, ConsciousDreamerCore)
assert issubclass(ConsciousDreamerInnerSpeech, ConsciousDreamerMemoryThought)
assert issubclass(ConsciousDreamerObjectImagery, ConsciousDreamerInnerSpeech)

assert issubclass(ConsciousDreamerMemoryThoughtConfig, ConsciousDreamerCoreConfig)
assert issubclass(ConsciousDreamerInnerSpeechConfig, ConsciousDreamerMemoryThoughtConfig)
assert issubclass(ConsciousDreamerObjectImageryConfig, ConsciousDreamerInnerSpeechConfig)
```

### 3. Compatibility facade maps old V names to semantic classes

Verify:

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

### 4. Package-level exports are canonical/semantic only

Verify:

```python
from src.modules.m05_world_model_attention_workspace import models

CANONICAL_AND_SEMANTIC = {
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

for name in CANONICAL_AND_SEMANTIC:
    assert hasattr(models, name)
```

And versioned names should not be in `models.__all__`:

```python
VERSIONED = {
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

assert not (set(getattr(models, "__all__", ())) & VERSIONED)
```

### 5. Canonical config factories return canonical config type

Verify both factories:

```python
from types import SimpleNamespace

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified

world_cfg = make_conscious_dreamer_config_from_world(
    image_height=64,
    image_width=96,
    body_state_dim=83,
    tactile_dim=42,
    hand_motor_dim=44,
    embodied_dim=15,
    action_dim=24,
)

assert isinstance(world_cfg, ConsciousDreamerConfig)

runner_cfg = SimpleNamespace(
    mujoco_world=SimpleNamespace(height=72, width=128),
    action_dim=24,
    embodied_dim=15,
    hand_motor_dim=44,
    tactile_dim=42,
    body_state_dim=83,
)

unified_cfg = make_conscious_dreamer_config_from_unified(runner_cfg)
assert isinstance(unified_cfg, ConsciousDreamerConfig)
```

### 6. Version metadata is present

Verify:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    CONSCIOUS_DREAMER_MODEL_VERSION,
    CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER,
)

assert isinstance(CONSCIOUS_DREAMER_MODEL_VERSION, str)
assert CONSCIOUS_DREAMER_MODEL_VERSION.startswith("M5_CONSCIOUS_DREAMER")
assert isinstance(CONSCIOUS_DREAMER_IMPLEMENTATION_LAYER, str)
```

## Optional AST checks

If easy, include a light AST check that:

1. `src/apps` does not import versioned names from canonical module.
2. `src/shared` does not import versioned implementation files.
3. `models/__init__.py` does not export versioned names through `__all__`.

If these are already covered by existing smoke tests, do not duplicate too much logic. The purpose of this final test is high-level contract validation.

## What not to change

Do not edit:

```text
docs/
docs/html/
README*
```

Do not delete compatibility aliases.

Do not rename files.

Do not change runtime logic.

Do not change model internals.

Do not touch checkpoint logic.

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_api_integrity.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py \
  src/modules/m05_world_model_attention_workspace/models/__init__.py \
  src/shared/conscious_dreamer_config.py

pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
pytest tests/smoke/test_conscious_dreamer_package_exports.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_conscious_dreamer_core_import_boundary.py
pytest tests/smoke/test_conscious_dreamer_core_facade.py
pytest tests/smoke/test_conscious_dreamer_core_semantic_base.py
pytest tests/smoke/test_conscious_dreamer_semantic_classes_primary.py
pytest tests/smoke/test_conscious_dreamer_internal_semantic_bases.py
pytest tests/smoke/test_conscious_dreamer_internal_semantic_aliases.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. There is one final high-level M5 API integrity test.
2. Canonical API is verified.
3. Semantic inheritance chain is verified.
4. Compatibility V aliases are verified.
5. Package exports are verified.
6. Config factories are verified.
7. Smoke tests pass.
8. Runtime behavior does not change.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- final M5 API integrity test added
- canonical/semantic/compat layers verified
- no docs touched
- no runtime behavior changed
```
