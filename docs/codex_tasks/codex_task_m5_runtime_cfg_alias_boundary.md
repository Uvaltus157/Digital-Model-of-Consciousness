# Codex Task: M5 Runtime Config Alias Boundary

## Context

The M5 cleanup has separated the API into:

### Canonical public API

```python
ConsciousDreamer
ConsciousDreamerConfig
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

Previous cleanup tasks addressed:

- versioned model aliases;
- factory/config aliases;
- canonical/package exports;
- compatibility facades.

Now focus on the remaining runtime config attribute aliases:

```text
self.v22_cfg
self.v23_cfg
```

The canonical runtime attribute should be:

```text
self.model_cfg
```

## Goal

Make `self.model_cfg` the only normal runtime config attribute used by project code.

Keep `self.v22_cfg` and `self.v23_cfg` only as temporary compatibility assignments for older checkpoints, old diagnostics, and external compatibility.

Do **not** delete `self.v22_cfg` / `self.v23_cfg` yet.

This task should:

1. Find all reads/writes of `v22_cfg` and `v23_cfg`.
2. Replace normal internal reads with `model_cfg`.
3. Keep only narrow compatibility assignments.
4. Add a smoke test to prevent new normal usage of `v22_cfg` / `v23_cfg`.

## Important rules

Do **not** delete compatibility assignments yet.

Do **not** change checkpoint loading behavior.

Do **not** change runtime logic.

Do **not** change model architecture, model config values, losses, state dict keys, or training behavior.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** start MuJoCo/viewer in tests.

## Required search

Search for:

```text
v22_cfg
v23_cfg
self.v22_cfg
self.v23_cfg
```

Classify every occurrence:

```text
allowed compatibility assignment
allowed compatibility test
allowed legacy viewer bridge
should be replaced with self.model_cfg
unexpected usage
```

## Allowed compatibility assignments

These assignments may remain temporarily:

```python
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

Allowed only in runtime initialization/legacy compatibility code, for example:

```text
src/apps/runner_unified_init.py
src/apps/unified_conscious_viewer.py
```

If another file has a real backward-compatibility reason, keep the allowlist narrow and explain it in the test comment.

## Required cleanup

Replace normal internal reads like:

```python
self.v23_cfg
self.v22_cfg
```

with:

```python
self.model_cfg
```

unless the line is explicitly setting the compatibility alias:

```python
self.v23_cfg = self.model_cfg
self.v22_cfg = self.model_cfg
```

Do not remove those assignments in this task.

If any checkpoint-loading code expects `v23_cfg`, inspect carefully. Prefer leaving checkpoint compatibility untouched, but normal runtime logic should use `model_cfg`.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_runtime_cfg_alias_boundary.py
```

The test should parse Python files with `ast`.

It should fail if project code reads or writes `.v22_cfg` or `.v23_cfg` outside the allowlist.

### Suggested allowlist

Allow these files:

```text
src/apps/runner_unified_init.py
src/apps/unified_conscious_viewer.py
tests/smoke/test_conscious_dreamer_runtime_cfg_alias_boundary.py
tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
```

Also allow files matching `tests/smoke/*compat*` or `tests/smoke/*alias*`, if needed.

Keep allowlist narrow.

### AST detection

Detect attribute nodes:

```python
ast.Attribute(attr="v22_cfg")
ast.Attribute(attr="v23_cfg")
```

Allowed pattern:

```python
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

That is an `ast.Assign` where the target is `self.v22_cfg` / `self.v23_cfg`.

Everything else should be considered a usage and should be rejected unless the file is allowlisted.

Suggested helper:

```python
def _is_self_attr(node, attr):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )
```

For each file, collect offenders:

```text
path:line uses .v23_cfg
path:line uses .v22_cfg
```

### Behavior check

In the same test, verify that known compatibility assignments still exist in the expected files.

For example, parse `src/apps/runner_unified_init.py` and/or `src/apps/unified_conscious_viewer.py` and assert that at least one compatibility assignment exists:

```python
self.v23_cfg = self.model_cfg
```

If both files are expected to keep both `v22_cfg` and `v23_cfg`, assert that. If only one file has both, keep the test aligned with actual code and explain the expected boundary.

## Required runtime smoke compatibility check

Add a simple test that instantiates the lightweight legacy config object only if it does not start MuJoCo.

Do **not** instantiate `UnifiedSystemV57` if that starts MuJoCo.

Prefer static AST checks for assignment boundaries.

If there is an existing smoke-safe object that exposes `model_cfg`, use it. Otherwise avoid runtime construction.

## What not to do

Do not remove:

```text
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

Do not edit docs.

Do not run viewer/MuJoCo.

Do not change checkpoint logic.

Do not change model internals.

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_runtime_cfg_alias_boundary.py \
  src/apps/runner_unified_init.py \
  src/apps/unified_conscious_viewer.py

pytest tests/smoke/test_conscious_dreamer_runtime_cfg_alias_boundary.py
pytest tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
pytest tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py
pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
pytest tests/smoke
```

## Expected result

After this task:

1. Normal runtime code uses `self.model_cfg`.
2. `self.v22_cfg` / `self.v23_cfg` remain only as compatibility assignments.
3. A smoke test prevents new normal usage of old cfg aliases.
4. Runtime behavior does not change.
5. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Runtime cfg alias audit:
- normal usages replaced:
  ...
- compatibility assignments kept:
  ...
- unexpected usages:
  none / list

Notes:
- self.model_cfg is the normal runtime config attribute
- self.v22_cfg/self.v23_cfg preserved only as compatibility bridges
- no docs touched
- no runtime behavior changed
```
