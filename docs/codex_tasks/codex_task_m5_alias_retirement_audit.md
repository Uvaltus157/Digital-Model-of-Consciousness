# Codex Task: M5 Alias Retirement Audit and Internal Usage Cleanup

## Context

M5 naming/API cleanup is now structurally complete:

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

The old compatibility aliases are still intentionally present:

```text
ConsciousDreamerV2/V21/V22/V23/V2_3
create_v23_config()
create_conscious_dreamer_v23()
make_v23_config_from_unified()
self.v22_cfg
self.v23_cfg
```

Now we begin the final cleanup safely: audit and remove **internal project usage** of aliases where possible, while keeping the alias definitions for backward compatibility.

## Goal

Do an alias retirement audit and cleanup internal usage.

This task should:

1. Find all remaining alias usages.
2. Replace internal usage with canonical names where safe.
3. Keep compatibility alias definitions.
4. Add smoke tests that prevent new internal usage of deprecated aliases.
5. Do **not** delete aliases yet.

This is a preparation step before any real alias deletion.

## Important rules

Do **not** delete compatibility aliases yet.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** change runtime logic, model architecture, checkpoint loading, state dict keys, losses, or training behavior.

Do **not** break backward compatibility.

## Alias groups

### Group A: versioned model aliases

These must remain available through:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

Names:

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

Internal project code should use semantic/canonical names instead:

```text
ConsciousDreamerCore
ConsciousDreamerMemoryThought
ConsciousDreamerInnerSpeech
ConsciousDreamerObjectImagery
ConsciousDreamer
ConsciousDreamerConfig
```

### Group B: factory aliases

Alias definitions should remain only in their compatibility locations:

```text
src/apps/runner_model_factory.py
  create_v23_config()
  create_conscious_dreamer_v23()

src/shared/config.py
  make_v23_config_from_unified()
```

Internal project code should use:

```text
create_conscious_dreamer_config()
create_conscious_dreamer()
make_conscious_dreamer_config_from_unified()
```

### Group C: runtime attribute aliases

These may remain temporarily:

```text
self.v22_cfg
self.v23_cfg
```

Allowed only as compatibility assignments/read compatibility in runner/legacy viewer/checkpoint code.

New code should use:

```text
self.model_cfg
```

## Required search

Search the repo for:

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

Classify each usage into:

```text
allowed alias definition
allowed compatibility test
allowed compatibility facade
allowed temporary runtime compatibility assignment
should be replaced by canonical API
unexpected usage
```

## Required cleanup

Replace internal usages where safe:

### Factory aliases

Replace project calls/imports:

```python
create_v23_config(...)
```

with:

```python
create_conscious_dreamer_config(...)
```

Replace:

```python
create_conscious_dreamer_v23(...)
```

with:

```python
create_conscious_dreamer(...)
```

Replace:

```python
make_v23_config_from_unified(...)
```

with:

```python
make_conscious_dreamer_config_from_unified(...)
```

Do not remove the old functions; only stop using them internally.

### Runtime cfg aliases

If code reads:

```python
self.v23_cfg
self.v22_cfg
```

and it is not explicitly checkpoint/backward compatibility logic, replace with:

```python
self.model_cfg
```

Keep assignments like:

```python
self.v23_cfg = self.model_cfg
self.v22_cfg = self.model_cfg
```

only as temporary compatibility bridges.

### Versioned model names

If internal code outside compatibility tests/facades imports or uses old V model names, replace with semantic/canonical names.

Examples:

```python
ConsciousDreamerV23
```

should become:

```python
ConsciousDreamerObjectImagery
```

or, at app/runtime boundary:

```python
ConsciousDreamer
```

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
```

The test should parse Python files with `ast`.

It should enforce that deprecated aliases are not used internally outside allowlisted locations.

### Suggested allowlists

Allowed files for versioned V names:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py
tests/smoke/*compat*
tests/smoke/*canonical*
tests/smoke/*semantic*
tests/smoke/*versioned*
tests/smoke/*alias*
tests/smoke/test_conscious_dreamer_api_integrity.py
```

Allowed files for factory alias definitions:

```text
src/apps/runner_model_factory.py
src/shared/config.py
tests/smoke/*compat*
tests/smoke/*alias*
```

Allowed files for `v22_cfg` / `v23_cfg` temporary compatibility:

```text
src/apps/runner_unified_init.py
src/apps/unified_conscious_viewer.py
tests/smoke/*compat*
tests/smoke/*alias*
```

If the actual code needs a slightly different allowlist, keep it narrow and comment why.

### Test requirements

The test should fail on:

1. Calls to deprecated factory aliases outside allowlist.
2. Imports of deprecated factory aliases outside allowlist.
3. Reads/writes of `v22_cfg` / `v23_cfg` outside allowlist.
4. Imports of old V model names from canonical module outside compatibility tests.
5. Direct old V model usages outside M5 implementation/compatibility tests.

It should not fail on alias definitions themselves.

## Required behavior check

In the same test or a companion test, verify aliases still exist:

```python
from src.apps.runner_model_factory import (
    create_conscious_dreamer,
    create_conscious_dreamer_config,
    create_conscious_dreamer_v23,
    create_v23_config,
)
from src.shared.config import make_v23_config_from_unified
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified
```

Verify the functions are still callable on a minimal runner-like config and return matching config types where practical.

Do not construct full model if this duplicates existing runtime smoke tests; checking config-level compatibility is enough here.

## What not to do

Do not delete:

```text
create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified
ConsciousDreamerV2/V21/V22/V23/V2_3
self.v22_cfg/self.v23_cfg compatibility assignments
```

Do not edit docs.

Do not change runtime behavior.

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_alias_retirement_audit.py \
  src/apps/runner_model_factory.py \
  src/shared/config.py \
  src/shared/conscious_dreamer_config.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py

pytest tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
pytest tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke
```

## Expected result

After this task:

1. Deprecated aliases are still defined.
2. Internal project usage of deprecated aliases is reduced or eliminated where safe.
3. A smoke test prevents new internal deprecated alias usage.
4. Remaining alias usage is explicit and allowlisted.
5. Runtime behavior does not change.
6. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Alias audit:
- aliases still defined:
  ...
- internal usages replaced:
  ...
- remaining allowlisted usages:
  ...
- unexpected usages:
  none / list

Notes:
- no aliases deleted
- no docs touched
- no runtime behavior changed
```
