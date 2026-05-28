# Codex Task: M5 Factory Aliases Compatibility Boundary

## Context

The M5 model class naming cleanup has moved old versioned model aliases into the compatibility facade:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_compat.py
```

Now focus on the remaining factory/config aliases:

```text
create_v23_config()
create_conscious_dreamer_v23()
make_v23_config_from_unified()
```

These names should remain available for backward compatibility, but normal project code and normal tests should use canonical names:

```text
create_conscious_dreamer_config()
create_conscious_dreamer()
make_conscious_dreamer_config_from_unified()
```

## Goal

Make factory alias usage explicit and compatibility-only.

Do **not** delete the aliases yet.

This task should:

1. Ensure the aliases still exist.
2. Ensure they are thin wrappers over the canonical functions.
3. Ensure normal project code does not call/import the old aliases.
4. Add tests that prevent new internal usage of old factory aliases.
5. Keep compatibility tests allowed.

## Important rules

Do **not** delete:

```text
create_v23_config()
create_conscious_dreamer_v23()
make_v23_config_from_unified()
```

Do **not** change model logic.

Do **not** change runner behavior.

Do **not** rename files.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** remove `self.v22_cfg` / `self.v23_cfg` compatibility assignments in this task.

Do **not** add runtime warnings unless existing tests already expect warnings. Prefer docstring comments over `warnings.warn`, because warnings may pollute smoke output.

## Canonical functions

Canonical factory functions:

```python
from src.apps.runner_model_factory import (
    create_conscious_dreamer_config,
    create_conscious_dreamer,
)
```

Canonical shared config function:

```python
from src.shared.conscious_dreamer_config import (
    make_conscious_dreamer_config_from_unified,
)
```

## Compatibility aliases

Compatibility aliases should remain in:

```text
src/apps/runner_model_factory.py
  create_v23_config()
  create_conscious_dreamer_v23()

src/shared/config.py
  make_v23_config_from_unified()
```

They should be implemented as thin wrappers:

```python
def create_v23_config(cfg, speech_vocab_size=None):
    return create_conscious_dreamer_config(cfg, speech_vocab_size=speech_vocab_size)
```

```python
def create_conscious_dreamer_v23(cfg, device, speech_vocab_size=None):
    return create_conscious_dreamer(cfg, device, speech_vocab_size=speech_vocab_size)
```

```python
def make_v23_config_from_unified(cfg):
    return make_conscious_dreamer_config_from_unified(cfg)
```

Add short docstrings if missing:

```python
"""Compatibility alias. New code should use create_conscious_dreamer_config()."""
```

## Required search

Search for these names:

```text
create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified
```

Classify every occurrence:

```text
alias definition
compatibility test
allowed backward-compat import check
should be replaced
unexpected usage
```

## Required cleanup

Replace normal internal usage:

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

Keep old aliases only for compatibility definitions and explicit compatibility tests.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py
```

The test should parse Python files with `ast`.

It should fail if normal project code imports or calls the old factory aliases outside the allowlist.

### Deprecated factory alias names

```python
DEPRECATED_FACTORY_ALIASES = {
    "create_v23_config",
    "create_conscious_dreamer_v23",
    "make_v23_config_from_unified",
}
```

### Suggested allowlist

Allow these files:

```text
src/apps/runner_model_factory.py
src/shared/config.py
tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py
tests/smoke/test_conscious_dreamer_config_compatibility.py
tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
```

Also allow files matching `tests/smoke/*compat*` or `tests/smoke/*alias*`, if the project already uses that pattern.

Keep the allowlist narrow.

### AST checks

The test should detect:

1. `ImportFrom` importing deprecated factory aliases.
2. `Call` nodes calling deprecated factory aliases.
3. Optional: attribute calls like `factory.create_v23_config(...)` if easy.

Suggested logic:

```python
def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
```

Then reject calls where `_call_name(node.func)` is in `DEPRECATED_FACTORY_ALIASES`.

Do not fail on function definitions inside the allowed files.

## Required behavior checks

In the same test, verify aliases still work and match canonical functions.

Create minimal runner-like config:

```python
from types import SimpleNamespace

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
```

Verify config aliases:

```python
from src.apps.runner_model_factory import (
    create_conscious_dreamer_config,
    create_v23_config,
)
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified
from src.shared.config import make_v23_config_from_unified

canonical = create_conscious_dreamer_config(cfg, speech_vocab_size=128)
legacy = create_v23_config(cfg, speech_vocab_size=128)

assert type(legacy) is type(canonical)
assert legacy.symbolic_report.text_vocab_size == canonical.symbolic_report.text_vocab_size == 128

canonical_shared = make_conscious_dreamer_config_from_unified(cfg)
legacy_shared = make_v23_config_from_unified(cfg)

assert type(legacy_shared) is type(canonical_shared)
assert legacy_shared.data.action_dim == canonical_shared.data.action_dim
assert legacy_shared.data.embodied_dim == canonical_shared.data.embodied_dim
assert legacy_shared.data.hand_motor_dim == canonical_shared.data.hand_motor_dim
assert legacy_shared.data.tactile_dim == canonical_shared.data.tactile_dim
assert legacy_shared.data.body_state_dim == canonical_shared.data.body_state_dim
```

Do **not** construct full model here if the runtime factory smoke test already covers model construction. Keep this test focused on alias boundary and config-level compatibility.

## Optional docstring check

If easy, assert the compatibility aliases have docstrings mentioning `Compatibility` or `alias`.

Example:

```python
assert "Compatibility" in (create_v23_config.__doc__ or "")
```

If adding this causes brittle tests, skip it.

## What not to do

Do not delete compatibility aliases.

Do not modify docs.

Do not change runner runtime behavior.

Do not remove `self.v23_cfg` or `self.v22_cfg`.

Do not start MuJoCo/viewer.

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py \
  src/apps/runner_model_factory.py \
  src/shared/config.py \
  src/shared/conscious_dreamer_config.py

pytest tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py
pytest tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
pytest tests/smoke/test_conscious_dreamer_config_compatibility.py
pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
pytest tests/smoke
```

## Expected result

After this task:

1. Factory aliases still exist.
2. Factory aliases are thin compatibility wrappers.
3. Normal internal usage of old factory aliases is gone.
4. A smoke test prevents new normal usage of old factory aliases.
5. Runtime behavior does not change.
6. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Factory alias audit:
- aliases still defined:
  create_v23_config
  create_conscious_dreamer_v23
  make_v23_config_from_unified

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
