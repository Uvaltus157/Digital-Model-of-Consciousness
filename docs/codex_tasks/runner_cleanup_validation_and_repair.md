# Codex task: validate and repair V5.10 runner cleanup

## Goal

Validate the new decomposed V5.10 runner architecture and fix all import, smoke-test and first-runtime-startup errors introduced by the runner cleanup.

The architectural intent is already decided. Do not revert the cleanup back to the old monolithic `src/apps/runner.py`. The task is to make the decomposed version work.

## Current architecture

The root launch path should remain:

```text
runner.py
→ src/apps/runner_entry.py
→ src/apps/runner_config.py
→ src/apps/runner_system_factory.py
→ src/apps/runner.py / UnifiedSystemV510 compatibility wrapper
→ src/apps/runner_unified_init.py
→ src/apps/runner_loop.py
```

`src/apps/runner.py` is now supposed to be a slim compatibility module that keeps the public import path `src.apps.runner.UnifiedSystemV510` stable. It should not become monolithic again.

The main runtime construction is now in:

```text
src/apps/runner_unified_init.py
```

The extracted helper modules include:

```text
src/apps/runner_config.py
src/apps/runner_system_factory.py
src/apps/runner_unified_init.py
src/apps/runner_loop.py
src/apps/runner_teachers.py
src/apps/runner_training_flags.py
src/apps/runner_hover_config.py
src/apps/runner_optimizer.py
src/apps/runner_services.py
src/apps/runner_runtime_state.py
src/apps/runner_startup_state.py
src/apps/runner_modes.py
src/apps/runner_components.py
src/apps/runner_model_factory.py
src/apps/runner_memory_factory.py
src/apps/runner_visualizer_factory.py
src/apps/runner_world_factories.py
src/apps/runner_dynamic_rig_config.py
src/apps/runner_patches.py
```

## Required validation commands

Run these from the repository root:

```bash
pytest tests/smoke
```

Then run the normal startup command:

```bash
python runner.py --config-path config --config-name runner
```

If the full runtime requires graphics/MuJoCo and the current environment is headless, still fix all errors that occur before display/runtime limitations. If a failure is purely environmental, document it clearly in the final report.

## Repair rules

1. Do not revert `src/apps/runner.py` to the old monolithic file.
2. Preserve the public import path:
   ```python
   from src.apps.runner import UnifiedSystemV510
   ```
3. Preserve the normal launch command:
   ```bash
   python runner.py --config-path config --config-name runner
   ```
4. Preserve the fallback behavior in root `runner.py` if present.
5. Do not change checkpoint format.
6. Do not change optimizer param group semantics unless a test/runtime error proves it is necessary.
7. Do not delete the extracted helper modules.
8. Keep helper modules lightweight where possible. Avoid importing MuJoCo/Open3D/PyQt at import time unless unavoidable.
9. Fix one traceback at a time, then rerun `pytest tests/smoke`.
10. Add or update smoke tests for every bug fixed.

## Likely issues to check

Check these areas first:

### 1. Import cycles

Especially between:

```text
src/apps/runner.py
src/apps/runner_entry.py
src/apps/runner_patches.py
src/apps/runner_unified_init.py
src/apps/runner_system_factory.py
```

If an import cycle appears, prefer moving imports inside functions instead of reintroducing monolithic imports.

### 2. Missing names in `runner_unified_init.py`

Verify all functions/classes used by `initialize_unified_system_v510()` are imported or defined. In particular, check:

```text
load_inner_speech_teacher_from_config
ModuleTrainingGate
ModuleDebugStatusServer
IPCControlServer
DynamicAgentRigController
DynamicAgentRigControlConfig
```

### 3. Duplicate service startup

`runner_unified_init.py` may still start `ModuleDebugStatusServer` and `IPCControlServer`, and `runner_system_factory.py` also calls `ensure_runner_services()`. Ensure this remains idempotent and does not bind the same port twice.

If duplicate startup occurs, make service creation only happen through `ensure_runner_services()`.

### 4. Startup state overwrite

`initialize_unified_system_v510()` sets startup/runtime fields and `runner_system_factory.py` applies `apply_runtime_state()` / `apply_startup_state()` afterward. Ensure the final state matches config and does not break checkpoint load or model initialization.

### 5. Model init order

Preserve this dependency order:

```text
cfg safety defaults
hover config patch
device + torch seed
speech teacher + vocab
v23 config with speech vocab size
ConsciousDreamerV23 model
base optimizer
leg control head
inner object system + optimizer param group
self core + optimizer param group
replay/quality/novelty/emotional drive
world + dynamic rig controller
visualizers
state tensors
checkpoint load
module training gate
optimizer rebuild
module debug status write
services/window flags
```

If the order is wrong, fix the extracted initializer instead of moving code back into `runner.py`.

### 6. Smoke tests that may need updates

Run and repair:

```text
tests/smoke/test_runner_config.py
tests/smoke/test_runner_entry.py
tests/smoke/test_runner_teachers.py
tests/smoke/test_runner_loop.py
tests/smoke/test_runner_training_flags.py
tests/smoke/test_runner_hover_config.py
tests/smoke/test_runner_patches.py
tests/smoke/test_runner_modes.py
tests/smoke/test_runner_startup_state.py
tests/smoke/test_runner_runtime_state.py
tests/smoke/test_runner_services.py
tests/smoke/test_runner_optimizer.py
tests/smoke/test_runner_components.py
tests/smoke/test_runner_world_factories.py
tests/smoke/test_runner_factories_more.py
tests/smoke/test_runner_system_factory.py
```

## Expected final state

After the task, these should pass:

```bash
pytest tests/smoke
```

And this should either start normally or fail only at a documented environment limitation:

```bash
python runner.py --config-path config --config-name runner
```

## Final report format

Return a concise report with:

```text
1. Commands run.
2. Errors found.
3. Files changed.
4. Tests added/updated.
5. Final test result.
6. Runtime startup result.
7. Any remaining environment-only limitation.
```
