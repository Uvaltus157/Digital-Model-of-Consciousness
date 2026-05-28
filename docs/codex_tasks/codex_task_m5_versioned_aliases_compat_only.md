# Codex Task: M5 Move Versioned Model Aliases to Compat Facade Only

## Context

M5 now has a clean API split:

### Canonical public API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
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

The next cleanup step is to stop defining old V aliases inside implementation files.

The old V model names should live only in:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

`conscious_dreamer.py` may still expose temporary compatibility attributes imported from the compat facade, but the implementation files should become semantic-only.

## Goal

Move versioned model aliases out of implementation files and into `conscious_dreamer_compat.py`.

After this task:

```text
conscious_dreamer_full.py              -> defines ConsciousDreamerCore only
conscious_dreamer_memory_thought.py    -> defines ConsciousDreamerMemoryThought only
conscious_dreamer_inner_speech.py      -> defines ConsciousDreamerInnerSpeech only
conscious_dreamer_object_imagery.py    -> defines ConsciousDreamerObjectImagery only
conscious_dreamer_compat.py            -> defines old V2/V21/V22/V23 aliases
```

## Important rules

Do **not** delete the compatibility facade.

Do **not** remove old V names from `conscious_dreamer_compat.py`.

Do **not** change runtime logic, model architecture, state dict keys, checkpoint loading, forward methods, losses, or config values.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** remove factory aliases in this task:

```text
create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified
```

This task is only about model class aliases.

## Required changes

### 1. Update `conscious_dreamer_compat.py`

Make the compat module import semantic classes and assign V aliases there.

Preferred pattern:

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

ConsciousDreamerV2 = ConsciousDreamerCore
ConsciousDreamerV2Config = ConsciousDreamerCoreConfig

ConsciousDreamerV21 = ConsciousDreamerMemoryThought
ConsciousDreamerV21Config = ConsciousDreamerMemoryThoughtConfig

ConsciousDreamerV22 = ConsciousDreamerInnerSpeech
ConsciousDreamerV22Config = ConsciousDreamerInnerSpeechConfig

ConsciousDreamerV23 = ConsciousDreamerObjectImagery
ConsciousDreamerV23Config = ConsciousDreamerObjectImageryConfig

ConsciousDreamerV2_3 = ConsciousDreamerV23
```

Keep `__all__` exporting all V names.

### 2. Update implementation files

Remove versioned alias assignments from:

```text
conscious_dreamer_full.py
conscious_dreamer_memory_thought.py
conscious_dreamer_inner_speech.py
conscious_dreamer_object_imagery.py
```

Examples to remove from implementation files:

```python
ConsciousDreamerV2 = ConsciousDreamerCore
ConsciousDreamerV2Config = ConsciousDreamerCoreConfig

ConsciousDreamerV21 = ConsciousDreamerMemoryThought
ConsciousDreamerV21Config = ConsciousDreamerMemoryThoughtConfig

ConsciousDreamerV22 = ConsciousDreamerInnerSpeech
ConsciousDreamerV22Config = ConsciousDreamerInnerSpeechConfig

ConsciousDreamerV23 = ConsciousDreamerObjectImagery
ConsciousDreamerV23Config = ConsciousDreamerObjectImageryConfig
```

Also remove V names from those files' `__all__`.

Keep semantic names exported from implementation files.

### 3. Update `conscious_dreamer.py`

If `conscious_dreamer.py` still exposes temporary V attributes, import them from:

```text
conscious_dreamer_compat.py
```

not from implementation files.

Canonical `__all__` should remain clean and should not include V names.

### 4. Update tests

Some older smoke tests may still import V names directly from implementation files, for example:

```python
from ...conscious_dreamer_object_imagery import ConsciousDreamerV23
```

Update those tests to import old V names from:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV23,
)
```

Do not weaken assertions.

If a test checks that `ConsciousDreamerV23 is ConsciousDreamerObjectImagery`, keep the assertion but import `ConsciousDreamerV23` from compat.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_versioned_aliases_in_compat_only.py
```

The test should verify:

### Behavior

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

### Import boundary

The test should parse the four implementation files with `ast` and verify that they do **not** export or define old V names.

Implementation files:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py
```

For these files, fail if:

1. `__all__` contains any V names.
2. There is an assignment target named any of:

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

3. There is a class definition with any of those names.

Allowed exception: if a compatibility comment mentions V names as plain text, do not fail on comments.

## What not to do

Do not remove V attributes from `conscious_dreamer.py` yet.

Do not remove V aliases from `conscious_dreamer_compat.py`.

Do not delete compatibility tests.

Do not touch docs.

Do not change runtime behavior.

## Commands to run

Run:

```bash
python3 -m py_compile \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py \
  tests/smoke/test_conscious_dreamer_versioned_aliases_in_compat_only.py

pytest tests/smoke/test_conscious_dreamer_versioned_aliases_in_compat_only.py
pytest tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke
```

## Expected result

After this task:

1. Implementation files export semantic names only.
2. Old V model names exist only through `conscious_dreamer_compat.py` and temporary canonical-module attributes.
3. Tests that need old V names import them from compat.
4. Runtime behavior does not change.
5. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Alias cleanup:
- V aliases moved to compat facade
- implementation files semantic-only
- old V compatibility still available through compat
- no factory aliases removed
- no docs touched
- no runtime behavior changed
```
