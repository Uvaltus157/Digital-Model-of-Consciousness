# Codex Task: M5 Canonical Factory Runtime Smoke

## Context

M5 naming/API cleanup is now almost complete. The code should have:

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

### Compatibility API

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_compat import (
    ConsciousDreamerV2,
    ConsciousDreamerV21,
    ConsciousDreamerV22,
    ConsciousDreamerV23,
    ConsciousDreamerV2_3,
)
```

The previous task added a final API integrity test. The next step is to verify that the canonical factory path can actually instantiate the model and produce initial state without using any old V-name entrypoint.

## Goal

Add a lightweight runtime smoke test for the canonical M5 factory path.

This is not a full training/runtime test. It should only prove:

1. Canonical config creation works.
2. Canonical model construction works.
3. `initial_state()` works.
4. `runner_model_factory.create_conscious_dreamer_config()` works.
5. `runner_model_factory.create_conscious_dreamer()` works.
6. Compatibility factory aliases still point to the canonical path.

## Important rules

Do **not** change model logic.

Do **not** change training logic.

Do **not** change runner loop behavior.

Do **not** touch documentation:

```text
docs/
docs/html/
README*
```

Do **not** delete compatibility aliases:

```text
create_v23_config
create_conscious_dreamer_v23
make_v23_config_from_unified
```

Do **not** use GPU. Test must run on CPU.

Do **not** run full MuJoCo or viewer.

## Required new smoke test

Add:

```text
tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
```

## Test 1: direct canonical model construction

Create a small canonical config from explicit dimensions:

```python
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    ConsciousDreamerConfig,
    make_conscious_dreamer_config_from_world,
)

cfg = make_conscious_dreamer_config_from_world(
    image_height=32,
    image_width=48,
    body_state_dim=83,
    tactile_dim=42,
    hand_motor_dim=44,
    embodied_dim=15,
    action_dim=24,
    text_vocab_size=128,
)

assert isinstance(cfg, ConsciousDreamerConfig)

model = ConsciousDreamer(cfg)
state = model.initial_state(batch_size=1, device="cpu")
assert isinstance(state, dict)
assert state
```

If the model expects `torch.device`, use:

```python
import torch
device = torch.device("cpu")
state = model.initial_state(batch_size=1, device=device)
```

Do not call full `model.step()` yet unless it is already known to be cheap and deterministic.

## Test 2: runner model factory config path

Create a minimal runner-like config:

```python
from types import SimpleNamespace

runner_cfg = SimpleNamespace(
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

Then verify:

```python
from src.apps.runner_model_factory import (
    create_conscious_dreamer_config,
    create_conscious_dreamer,
    create_v23_config,
    create_conscious_dreamer_v23,
    create_torch_device,
    seed_torch,
)

device = create_torch_device(runner_cfg)
seed = seed_torch(runner_cfg)
assert str(device) == "cpu"
assert seed == 123

canonical_cfg = create_conscious_dreamer_config(runner_cfg, speech_vocab_size=128)
legacy_cfg = create_v23_config(runner_cfg, speech_vocab_size=128)

assert type(canonical_cfg) is type(legacy_cfg)
assert canonical_cfg.symbolic_report.text_vocab_size == 128
assert legacy_cfg.symbolic_report.text_vocab_size == 128

model = create_conscious_dreamer(runner_cfg, device, speech_vocab_size=128)
legacy_model = create_conscious_dreamer_v23(runner_cfg, device, speech_vocab_size=128)

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import ConsciousDreamer
assert isinstance(model, ConsciousDreamer)
assert isinstance(legacy_model, ConsciousDreamer)

state = model.initial_state(batch_size=1, device=device)
legacy_state = legacy_model.initial_state(batch_size=1, device=device)
assert isinstance(state, dict)
assert isinstance(legacy_state, dict)
```

If constructing two full models is too slow, replace the legacy model construction check with a monkeypatch or identity-style check, but prefer real construction if smoke runtime remains acceptable.

## Test 3: optimizer kwargs remain stable

Verify:

```python
from src.apps.runner_model_factory import optimizer_kwargs, model_factory_snapshot

assert optimizer_kwargs(runner_cfg) == {"lr": 1e-4, "weight_decay": 0.0}

snap = model_factory_snapshot(runner_cfg, speech_vocab_size=128)
assert snap.device == "cpu"
assert snap.seed == 123
assert snap.text_vocab_size == 128
assert snap.optimizer_lr == 1e-4
assert snap.optimizer_weight_decay == 0.0
```

## If test is too slow

If model construction is too slow for smoke tests:

1. Mark only the full construction test with a narrow `pytest.mark.slow` **only if the project already uses slow markers**.
2. Otherwise keep the test minimal and instantiate only one model.
3. Do not remove the canonical config/factory checks.

## What not to do

Do not add MuJoCo dependencies to this test.

Do not create OpenCV/Open3D windows.

Do not run full runner.

Do not touch docs.

Do not change model internals just to make the test easier.

## Commands to run

Run:

```bash
python3 -m py_compile \
  tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py \
  src/apps/runner_model_factory.py \
  src/modules/m05_world_model_attention_workspace/models/conscious_dreamer.py \
  src/shared/conscious_dreamer_config.py

pytest tests/smoke/test_conscious_dreamer_canonical_factory_runtime.py
pytest tests/smoke/test_conscious_dreamer_api_integrity.py
pytest tests/smoke/test_conscious_dreamer_canonical_exports_clean.py
pytest tests/smoke/test_conscious_dreamer_package_exports.py
pytest tests/smoke/test_conscious_dreamer_versioned_imports_use_compat.py
pytest tests/smoke/test_conscious_dreamer_compat_facade.py
pytest tests/smoke/test_versioned_conscious_dreamer_usage_boundaries.py
pytest tests/smoke
```

## Expected result

After this task:

1. Canonical M5 model can be constructed on CPU.
2. Canonical M5 initial state can be created.
3. Runner model factory uses canonical config/model path.
4. Compatibility factory aliases still work.
5. No runtime behavior changes.
6. Smoke tests pass.

## Final report format

Report back with:

```text
Changed files:
- ...

Tests:
- command -> result

Notes:
- canonical factory runtime smoke added
- CPU-only model construction validated
- compatibility factory aliases preserved
- no docs touched
- no runner/MuJoCo/viewer started
```
