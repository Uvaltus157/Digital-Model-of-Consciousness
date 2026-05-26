# Codex Task: M5 ConsciousDreamer Internal Cleanup Boundary

## Context

The M5 world model previously accumulated several historical implementation versions:

- `ConsciousDreamerV2`
- `ConsciousDreamerV21`
- `ConsciousDreamerV22`
- `ConsciousDreamerV23`
- `ConsciousDreamerV2_3`

The project now has a canonical public API:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

The canonical runner/shared path is now:

```text
runner_unified_init.py
  -> runner_model_factory.py
    -> src/shared/conscious_dreamer_config.py
      -> src/shared/model_dimensions.py
      -> canonical ConsciousDreamerConfig
```

The previous app/shared-level cleanup has passed:

```bash
pytest tests/smoke
# 81 passed
```

## Goal

Create a boundary test and cleanup map that guarantees historical versioned M5 names are used only where they are allowed.

Do **not** rename M5 files yet.

Do **not** delete `ConsciousDreamerV21`, `ConsciousDreamerV22`, or `ConsciousDreamerV23`.

This task is only about protecting the boundary between:

```text
external app/shared/runtime API
```

and

```text
internal M5 implementation layers
```

## Search targets

Scan the repository for these names:

```text
ConsciousDreamerV2
ConsciousDreamerV21
ConsciousDreamerV22
ConsciousDreamerV23
ConsciousDreamerV2_3
ConsciousDreamerV21Config
ConsciousDreamerV22Config
ConsciousDreamerV23Config
v22_cfg
v23_cfg
make_v22_config_from_world
make_v23_config_from_unified
create_v23_config
create_conscious_dreamer_v23
```

## Required new test

Add a smoke test:

```text
tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
```

The test should parse Python files with `ast`, not just grep raw text.

## Allowed usage

The versioned names are allowed only in these places:

```text
src/modules/m05_world_model_attention_workspace/models/
```

Reason: these are internal M5 implementation layers.

Also allow explicit compatibility tests, for example files matching:

```text
tests/smoke/*canonical*
tests/smoke/*compatibility*
tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
```

Also allow compatibility aliases in these files only:

```text
src/apps/runner_model_factory.py
src/shared/config.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py
```

These aliases are temporary and must remain backward compatible.

## Forbidden usage

The test must fail if versioned names appear in:

```text
src/apps/
src/shared/     except explicitly allowed compatibility aliases
src/platform/
src/modules/    except src/modules/m05_world_model_attention_workspace/models/
```

Examples of forbidden imports/calls outside M5 internals:

```python
from ...conscious_dreamer_object_imagery import ConsciousDreamerV23
from ...conscious_dreamer_inner_speech import ConsciousDreamerV22
make_v22_config_from_world(...)
make_v23_config_from_unified(...)
ConsciousDreamerV23(...)
```

## Required cleanup behavior

If the new boundary test finds real external usage, replace it with canonical API:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)
```

For runner config from `UnifiedV510Config`, use:

```python
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified
```

Do not do broad rewrites. Make the smallest safe replacement.

## What not to change

Do not touch documentation:

```text
docs/
docs/html/
README*
```

Do not rename these files yet:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_full.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_inner_speech.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_object_imagery.py
```

Do not remove compatibility aliases yet:

```text
create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified
self.v22_cfg
self.v23_cfg
```

Only restrict where they are allowed.

## Expected result

After this task:

1. `src/apps` uses only canonical M5 API.
2. `src/shared` uses only canonical M5 API, except explicit compatibility aliases.
3. Historical versioned names remain inside M5 implementation files only.
4. Boundary rule is enforced by a smoke test.
5. Existing compatibility tests still pass.

## Commands to run

Run:

```bash
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke/test_conscious_dreamer_canonical_api.py
pytest tests/smoke/test_conscious_dreamer_config_helper.py
pytest tests/smoke/test_model_dimensions_helpers.py
pytest tests/smoke/test_conscious_dreamer_config_decoupling.py
pytest tests/smoke/test_conscious_dreamer_config_compatibility.py
pytest tests/smoke/test_no_app_level_versioned_conscious_dreamer.py
pytest tests/smoke/test_legacy_unified_viewer_canonical_api.py
pytest tests/smoke
```

## If tests fail

If tests fail:

1. Do not do a large rewrite.
2. Show the traceback.
3. Fix only the smallest necessary location.
4. Preserve compatibility aliases.
5. Re-run the failing test and then all smoke tests.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Remaining versioned usages:
- allowed internal M5 usages
- compatibility aliases
- unexpected usages, if any
```
