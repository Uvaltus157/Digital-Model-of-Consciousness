# Codex Task: M5 Strict Cleanup — Remove All Compatibility Bridges

## User decision

Compatibility bridges are no longer wanted.

The project should be clean and use only canonical/semantic M5 names.

This is a breaking cleanup. Old `V2/V21/V22/V23` names and old `v23` factory/config aliases should be removed from the codebase.

## Final desired state

### Canonical public API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

### Semantic internal chain

```text
ConsciousDreamerCore
  -> ConsciousDreamerMemoryThought
    -> ConsciousDreamerInnerSpeech
      -> ConsciousDreamerObjectImagery
        -> ConsciousDreamer
```

### Normal runtime config attribute

```python
self.model_cfg
```

## Names to remove completely from project code

Remove these names from normal code, exports, aliases, and tests, except in the final forbidden-name smoke test itself where they appear as strings:

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

create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified

v22_cfg
v23_cfg
```

## Files/modules to remove if they exist

Delete compatibility modules if they exist:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
src/apps/runner_model_factory_compat.py
src/shared/conscious_dreamer_config_compat.py
```

Use `git rm` if available.

## Hard rules

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** change model logic, forward methods, losses, state dict keys, checkpoint loading logic, training logic, MuJoCo code, viewer code, or runner loop behavior.

Do **not** rename implementation files in this task.

Do **not** keep compatibility bridges.

Do **not** keep old alias tests that assert old names still exist.

## Required code changes

### 1. M5 implementation files

Implementation files should define and export only semantic names:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
  ConsciousDreamerCore
  ConsciousDreamerCoreConfig

src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
  ConsciousDreamerMemoryThought
  ConsciousDreamerMemoryThoughtConfig

src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py
  ConsciousDreamerInnerSpeech
  ConsciousDreamerInnerSpeechConfig

src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py
  ConsciousDreamerObjectImagery
  ConsciousDreamerObjectImageryConfig
```

Remove any class definitions, aliases, imports, or `__all__` entries for old V names.

### 2. Canonical module

`src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py` should export only canonical and semantic names:

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

It must not import, assign, expose, or export old V names.

### 3. Package exports

`src/modules/m05_world_model_attention_workspace/models/__init__.py` should export canonical and semantic names only.

No old V names in `__all__`.

### 4. Runner factory

`src/apps/runner_model_factory.py` should contain only canonical factory functions:

```python
create_conscious_dreamer_config()
create_conscious_dreamer()
```

Remove:

```python
create_v23_config()
create_conscious_dreamer_v23()
```

Update all imports/tests to use canonical names.

### 5. Shared config

`src/shared/conscious_dreamer_config.py` should contain only:

```python
make_conscious_dreamer_config_from_unified()
```

`src/shared/config.py` should not define or export:

```python
make_v23_config_from_unified()
```

Update all imports/tests to use:

```python
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified
```

### 6. Runtime config attributes

Remove assignments:

```python
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

and replace all reads of `self.v22_cfg` / `self.v23_cfg` with:

```python
self.model_cfg
```

Normal runtime code should use `self.model_cfg` only.

Likely files to inspect:

```text
src/apps/runner_unified_init.py
src/apps/unified_conscious_viewer.py
```

### 7. Tests

Update or delete old compatibility tests that assert old names still exist.

Remove or rewrite tests such as:

```text
test_conscious_dreamer_compat_facade.py
test_conscious_dreamer_versioned_imports_use_compat.py
test_conscious_dreamer_versioned_aliases_in_compat_only.py
test_conscious_dreamer_factory_aliases_compat_only.py
test_conscious_dreamer_factory_aliases_in_compat_modules.py
test_conscious_dreamer_canonical_has_no_versioned_attrs.py
test_conscious_dreamer_alias_retirement_audit.py
```

If they exist and are now obsolete, either delete them or rewrite them to assert the strict cleanup rule.

Do not leave contradictory tests.

## Required new final smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_no_legacy_aliases.py
```

This test should parse Python files with `ast`.

### Test scope

Scan:

```text
src/
tests/
```

Skip:

```text
docs/
README*
```

Allow the test file itself to contain forbidden strings.

### Forbidden names

```python
FORBIDDEN_NAMES = {
    "ConsciousDreamerV2",
    "ConsciousDreamerV21",
    "ConsciousDreamerV22",
    "ConsciousDreamerV23",
    "ConsciousDreamerV2_3",
    "ConsciousDreamerV2Config",
    "ConsciousDreamerV21Config",
    "ConsciousDreamerV22Config",
    "ConsciousDreamerV23Config",
    "create_v23_config",
    "create_conscious_dreamer_v23",
    "make_v23_config_from_unified",
    "v22_cfg",
    "v23_cfg",
}
```

### AST checks

Fail if any scanned Python file contains:

1. Class definitions with forbidden names.
2. Function definitions with forbidden names.
3. Assignment targets with forbidden names.
4. Imported names with forbidden names.
5. Calls to forbidden names.
6. Attribute access using `v22_cfg` or `v23_cfg`.

Do not use raw text grep for all checks, because the new test itself will contain forbidden strings.

### Behavior checks

The same test should verify clean canonical behavior:

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

Verify no compatibility modules exist:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

assert not (ROOT / "src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py").exists()
assert not (ROOT / "src/apps/runner_model_factory_compat.py").exists()
assert not (ROOT / "src/shared/conscious_dreamer_config_compat.py").exists()
```

Verify canonical factory functions work:

```python
from types import SimpleNamespace
from src.apps.runner_model_factory import create_conscious_dreamer_config
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified

cfg = SimpleNamespace(
    runtime=SimpleNamespace(device="cpu", seed=123),
    train=SimpleNamespace(lr=1e-4, weight_decay=0.0),
    mujoco_world=SimpleNamespace(height=32, width=48),
    action_dim=24,
    embodied_dim=15,
    hand_motor_dim=44,
    tactile_dim=42,
    body_state_dim=83,
)

model_cfg_1 = create_conscious_dreamer_config(cfg, speech_vocab_size=128)
model_cfg_2 = make_conscious_dreamer_config_from_unified(cfg)

assert isinstance(model_cfg_1, ConsciousDreamerConfig)
assert isinstance(model_cfg_2, ConsciousDreamerConfig)
```

## Update existing integrity/runtime tests

Make sure these tests use canonical/semantic names only:

```text
tests/smoke/test_conscious_dreamer_api_integrity.py
tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
tests/smoke/test_conscious_dreamer_package_exports.py
tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
```

If any of them imports old V names, replace with semantic/canonical names or delete obsolete assertions.

## Commands to run

Run:

```bash
python3 -m py_compile \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py \
  src/modules/m05_world_model_attention_workspace/models/__init__.py \
  src/apps/runner_model_factory.py \
  src/apps/runner_unified_init.py \
  src/apps/unified_conscious_viewer.py \
  src/shared/config.py \
  src/shared/conscious_dreamer_config.py \
  tests/smoke/test_conscious_dreamer_no_legacy_aliases.py

pytest tests/smoke/test_conscious_dreamer_no_legacy_aliases.py
pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
pytest tests/smoke/test_conscious_dreamer_package_exports.py
pytest tests/smoke
```

## Expected result

After this task:

1. No old V model aliases remain in code.
2. No old V factory/config aliases remain in code.
3. No `self.v22_cfg` / `self.v23_cfg` remain in code.
4. No compatibility facade modules remain.
5. Canonical/semantic M5 API remains working.
6. Smoke tests pass.
7. Runtime behavior does not change except old legacy import paths intentionally no longer work.

## Final report format

Report back exactly in this format:

```text
Changed files:
- ...

Deleted files:
- ...

Tests:
- python3 -m py_compile ... -> passed/failed
- pytest tests/smoke/test_conscious_dreamer_no_legacy_aliases.py -> passed/failed
- pytest tests/smoke/test_conscious_dreamer_api_integrity.py -> passed/failed
- pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py -> passed/failed
- pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py -> passed/failed
- pytest tests/smoke/test_conscious_dreamer_package_exports.py -> passed/failed
- pytest tests/smoke -> passed/failed

Final M5 cleanup status:
- canonical API: clean
- semantic internal chain: clean
- legacy model V aliases: removed
- legacy factory aliases: removed
- runtime cfg aliases v22/v23: removed
- compatibility facade modules: removed
- docs touched: no
- runtime behavior changed: only old legacy aliases/imports intentionally removed

Unexpected legacy usages:
- none / list
```
