# Codex Task: M5 Move Factory Aliases to Compatibility Modules

## Context

The M5 cleanup has already separated model-class APIs:

### Canonical model API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
)
```

### Model compatibility API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV21,
    ConsciousDreamerV22,
    ConsciousDreamerV23,
)
```

The remaining deprecated aliases are factory/config aliases:

```text
create_v23_config()
create_conscious_dreamer_v23()
make_v23_config_from_unified()
```

Canonical functions are:

```text
create_conscious_dreamer_config()
create_conscious_dreamer()
make_conscious_dreamer_config_from_unified()
```

The previous boundary test should already prevent normal project code from using deprecated factory aliases.

Now move these deprecated factory aliases out of the canonical modules and into explicit compatibility modules.

## Goal

Create explicit compatibility modules for factory aliases:

```text
src/apps/runner_model_factory_compat.py
src/shared/conscious_dreamer_config_compat.py
```

After this task, deprecated factory aliases should be imported from these compatibility modules only.

Canonical modules should become clean:

```text
src/apps/runner_model_factory.py
  -> canonical functions only

src/shared/conscious_dreamer_config.py
  -> canonical helper only

src/shared/config.py
  -> should not define make_v23_config_from_unified anymore
```

## Important rules

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** change runtime logic.

Do **not** change model architecture, state dict keys, checkpoint loading, losses, or training behavior.

Do **not** remove model compatibility facade:

```text
conscious_dreamer_compat.py
```

Do **not** remove runtime cfg compatibility assignments yet:

```text
self.v22_cfg = self.model_cfg
self.v23_cfg = self.model_cfg
```

Do **not** rename files.

## Required changes

### 1. Add `src/apps/runner_model_factory_compat.py`

Create a small compatibility module:

```python
from __future__ import annotations

"""Compatibility factory aliases for historical M5 ConsciousDreamer V23 names.

New code should import from `src.apps.runner_model_factory`:
    create_conscious_dreamer_config
    create_conscious_dreamer
"""

from typing import Any

from src.apps.runner_model_factory import (
    create_conscious_dreamer,
    create_conscious_dreamer_config,
)


def create_v23_config(cfg: Any, speech_vocab_size: int | None = None) -> Any:
    """Compatibility alias. New code should use create_conscious_dreamer_config()."""
    return create_conscious_dreamer_config(cfg, speech_vocab_size=speech_vocab_size)


def create_conscious_dreamer_v23(cfg: Any, device: Any, speech_vocab_size: int | None = None) -> Any:
    """Compatibility alias. New code should use create_conscious_dreamer()."""
    return create_conscious_dreamer(cfg, device, speech_vocab_size=speech_vocab_size)


__all__ = [
    "create_v23_config",
    "create_conscious_dreamer_v23",
]
```

### 2. Add `src/shared/conscious_dreamer_config_compat.py`

Create:

```python
from __future__ import annotations

"""Compatibility config alias for historical M5 V23 config factory.

New code should import:
    make_conscious_dreamer_config_from_unified
from:
    src.shared.conscious_dreamer_config
"""

from typing import Any

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamerConfig
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified


def make_v23_config_from_unified(cfg: Any) -> ConsciousDreamerConfig:
    """Compatibility alias. New code should use make_conscious_dreamer_config_from_unified()."""
    return make_conscious_dreamer_config_from_unified(cfg)


__all__ = ["make_v23_config_from_unified"]
```

### 3. Clean canonical modules

Remove these deprecated functions from:

```text
src/apps/runner_model_factory.py
```

Remove:

```text
create_v23_config()
create_conscious_dreamer_v23()
```

Keep:

```text
create_conscious_dreamer_config()
create_conscious_dreamer()
```

Remove this deprecated function from:

```text
src/shared/config.py
```

Remove:

```text
make_v23_config_from_unified()
```

Keep canonical helpers/imports intact.

If removing `make_v23_config_from_unified()` from `src/shared/config.py` breaks many old compatibility tests, update those tests to import it from:

```python
from src.shared.conscious_dreamer_config_compat import make_v23_config_from_unified
```

Do not re-add the alias to `src/shared/config.py`.

### 4. Update tests

Update compatibility tests so deprecated aliases are imported from new compatibility modules:

```python
from src.apps.runner_model_factory_compat import (
    create_v23_config,
    create_conscious_dreamer_v23,
)
```

and:

```python
from src.shared.conscious_dreamer_config_compat import (
    make_v23_config_from_unified,
)
```

Canonical tests should use canonical functions only.

Do not weaken assertions.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_factory_aliases_in_compat_modules.py
```

The test should verify:

### Compatibility modules work

```python
from types import SimpleNamespace

from src.apps.runner_model_factory import create_conscious_dreamer_config
from src.apps.runner_model_factory_compat import create_v23_config
from src.shared.conscious_dreamer_config import make_conscious_dreamer_config_from_unified
from src.shared.conscious_dreamer_config_compat import make_v23_config_from_unified

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

canonical = create_conscious_dreamer_config(cfg, speech_vocab_size=128)
legacy = create_v23_config(cfg, speech_vocab_size=128)

assert type(legacy) is type(canonical)
assert legacy.symbolic_report.text_vocab_size == canonical.symbolic_report.text_vocab_size == 128

canonical_shared = make_conscious_dreamer_config_from_unified(cfg)
legacy_shared = make_v23_config_from_unified(cfg)

assert type(legacy_shared) is type(canonical_shared)
assert legacy_shared.data.action_dim == canonical_shared.data.action_dim
```

### Canonical modules no longer expose deprecated aliases

```python
import src.apps.runner_model_factory as runner_factory
import src.shared.config as shared_config

assert not hasattr(runner_factory, "create_v23_config")
assert not hasattr(runner_factory, "create_conscious_dreamer_v23")
assert not hasattr(shared_config, "make_v23_config_from_unified")
```

### Deprecated factory imports boundary

Add AST checks that fail if these aliases are imported from canonical modules:

```text
src.apps.runner_model_factory.create_v23_config
src.apps.runner_model_factory.create_conscious_dreamer_v23
src.shared.config.make_v23_config_from_unified
```

Allowed imports are only from:

```text
src.apps.runner_model_factory_compat
src.shared.conscious_dreamer_config_compat
```

## Update or replace older boundary test

If `tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py` still expects aliases in canonical modules, update it to the new rule:

```text
deprecated factory aliases live in compatibility modules only
```

Do not keep contradictory tests.

## What not to do

Do not remove runtime cfg aliases.

Do not remove model V aliases from `conscious_dreamer_compat.py`.

Do not edit docs.

Do not start MuJoCo/viewer.

Do not construct full model in this test unless already needed by existing tests.

## Commands to run

Run:

```bash
python3 -m py_compile \
  src/apps/runner_model_factory.py \
  src/apps/runner_model_factory_compat.py \
  src/shared/config.py \
  src/shared/conscious_dreamer_config.py \
  src/shared/conscious_dreamer_config_compat.py \
  tests/smoke/test_conscious_dreamer_factory_aliases_in_compat_modules.py

pytest tests/smoke/test_conscious_dreamer_factory_aliases_in_compat_modules.py
pytest tests/smoke/test_conscious_dreamer_factory_aliases_compat_only.py
pytest tests/smoke/test_conscious_dreamer_alias_retirement_audit.py
pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_canonical_synthetic_step.py
pytest tests/smoke
```

## Expected result

After this task:

1. Deprecated factory aliases live in explicit compatibility modules.
2. Canonical factory/config modules no longer expose deprecated factory aliases.
3. Existing compatibility remains available through new compat modules.
4. Normal internal code uses canonical factory names only.
5. Runtime behavior does not change.
6. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Factory alias cleanup:
- moved to:
  src/apps/runner_model_factory_compat.py
  src/shared/conscious_dreamer_config_compat.py

- removed from canonical modules:
  src/apps/runner_model_factory.py
  src/shared/config.py

- internal imports updated:
  ...

- unexpected usages:
  none / list

Notes:
- no docs touched
- no runtime cfg aliases removed
- no runtime behavior changed
```
