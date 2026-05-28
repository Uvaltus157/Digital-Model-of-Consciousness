# Codex Task: Finish M5 ConsciousDreamer Cleanup in One Pass

## Goal

Finish the M5 ConsciousDreamer naming/API cleanup now. Do not split into more small tasks.

The final desired state:

```text
Canonical public API:
  src.modules.m05_world_model_attention_workspace.models.conscious_dreamer
    ConsciousDreamer
    ConsciousDreamerConfig
    make_conscious_dreamer_config_from_world

Semantic internal API:
  ConsciousDreamerCore
    -> ConsciousDreamerMemoryThought
      -> ConsciousDreamerInnerSpeech
        -> ConsciousDreamerObjectImagery
          -> ConsciousDreamer

Historical model compatibility API:
  src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat
    ConsciousDreamerV2
    ConsciousDreamerV21
    ConsciousDreamerV22
    ConsciousDreamerV23
    ConsciousDreamerV2_3

Historical factory/config compatibility API:
  src.apps.runner_model_factory_compat
    create_v23_config
    create_conscious_dreamer_v23

  src.shared.conscious_dreamer_config_compat
    make_v23_config_from_unified

Normal runtime config attribute:
  self.model_cfg

Temporary runtime compatibility bridges:
  self.v22_cfg = self.model_cfg
  self.v23_cfg = self.model_cfg
```

## Hard rules

Do not touch docs:

```text
docs/
docs/html/
README*
```

Do not change model logic, checkpoint logic, training logic, losses, state dict keys, MuJoCo code, viewers, or runner loop behavior.

Do not rename implementation files in this task.

Do not remove runtime compatibility assignments:

```python
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

## Do these remaining changes in one pass

### 1. Finish factory/config alias relocation

Create if missing:

```text
src/apps/runner_model_factory_compat.py
src/shared/conscious_dreamer_config_compat.py
```

Move deprecated factory/config aliases there:

```python
# src/apps/runner_model_factory_compat.py
create_v23_config()
create_conscious_dreamer_v23()

# src/shared/conscious_dreamer_config_compat.py
make_v23_config_from_unified()
```

Remove these deprecated aliases from canonical modules:

```text
src/apps/runner_model_factory.py
src/shared/config.py
```

Canonical modules should expose only:

```python
create_conscious_dreamer_config
create_conscious_dreamer
make_conscious_dreamer_config_from_unified
```

Update tests/imports accordingly.

### 2. Finish canonical module cleanup

Ensure:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py
```

does NOT expose old V attrs:

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

Old V names must exist only through:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

Canonical module should contain canonical/semantic names only.

### 3. Finish runtime cfg alias boundary

Normal code should use:

```python
self.model_cfg
```

Only keep old attributes as compatibility assignments:

```python
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

Replace other internal reads/usages of `self.v22_cfg` / `self.v23_cfg` with `self.model_cfg`, unless they are explicitly part of compatibility/checkpoint handling.

### 4. Keep model class aliases only in compat facade

Implementation files should be semantic-only:

```text
conscious_dreamer_full.py              -> ConsciousDreamerCore
conscious_dreamer_memory_thought.py    -> ConsciousDreamerMemoryThought
conscious_dreamer_inner_speech.py      -> ConsciousDreamerInnerSpeech
conscious_dreamer_object_imagery.py    -> ConsciousDreamerObjectImagery
```

Old model V aliases belong only in:

```text
conscious_dreamer_compat.py
```

Update tests that import V names to import from compat.

## Required final smoke tests

Add or update these tests as needed. If some already exist, update them instead of duplicating contradictory tests.

```text
tests/smoke/test_conscious_dreamer_factory_aliases_in_compat_modules.py
tests/smoke/test_conscious_dreamer_canonical_has_no_versioned_attrs.py
tests/smoke/test_conscious_dreamer_runtime_cfg_alias_boundary.py
tests/smoke/test_conscious_dreamer_versioned_aliases_in_compat_only.py
tests/smoke/test_conscious_dreamer_api_integrity.py
tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
```

The tests should enforce:

1. Canonical module has no V attrs.
2. Package `models.__all__` has no V names.
3. V model names are available only from `conscious_dreamer_compat.py`.
4. Deprecated factory names are available only from new compat modules.
5. Normal project code does not import/call deprecated factory aliases.
6. Normal runtime code uses `self.model_cfg`, with only compatibility assignments for `v22_cfg/v23_cfg`.
7. Canonical model can instantiate on CPU.
8. Canonical model can run one synthetic CPU `model.step(...)` if the existing model step supports it in smoke time. If too slow, keep construction/initial_state smoke and report why step was not kept in smoke.

## Commands to run

Run these exact checks or their updated equivalents:

```bash
python3 -m py_compile \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_core.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py \
  src/apps/runner_model_factory.py \
  src/apps/runner_model_factory_compat.py \
  src/shared/config.py \
  src/shared/conscious_dreamer_config.py \
  src/shared/conscious_dreamer_config_compat.py

pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
pytest tests/smoke/test_conscious_dreamer_factory_aliases_in_compat_modules.py
pytest tests/smoke/test_conscious_dreamer_canonical_has_no_versioned_attrs.py
pytest tests/smoke/test_conscious_dreamer_runtime_cfg_alias_boundary.py
pytest tests/smoke/test_conscious_dreamer_versioned_aliases_in_compat_only.py
pytest tests/smoke
```

## If tests fail

Do not broaden the refactor.

Fix only the failing boundary.

If full synthetic step is too slow or fails because it touches heavy state, keep it as construction + initial_state smoke and report exactly why full step was not retained.

## Final report format

Report back exactly in this format:

```text
Changed files:
- ...

Tests:
- python3 -m py_compile ... -> passed/failed
- pytest tests/smoke/... -> passed/failed
- pytest tests/smoke -> passed/failed

Final M5 cleanup status:
- canonical API: clean
- semantic internal chain: clean
- model V aliases: compat facade only
- factory aliases: compat modules only
- runtime cfg aliases: compatibility assignment only
- docs touched: no
- runtime behavior changed: no

Remaining aliases intentionally kept:
- model V aliases in conscious_dreamer_compat.py
- factory aliases in *_compat.py
- self.v22_cfg/self.v23_cfg compatibility assignments

Unexpected usages:
- none / list
```
