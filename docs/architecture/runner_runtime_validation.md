# Runner runtime validation

This document is the next checkpoint after the successful smoke-test pass for the decomposed V5.10 runner.

## Current known status

```text
pytest tests/smoke: passed
```

The architecture-level runner decomposition is complete. The next step is runtime startup validation.

## Command to run

Run from repository root:

```bash
python runner.py --config-path config --config-name runner
```

## Expected launch chain

```text
runner.py
→ src/apps/runner_entry.py
→ build_runner_config(cfg_raw)
→ build_unified_system(cfg_obj, UnifiedSystem)
→ UnifiedSystem.__init__ via runner_unified_init.py
→ UnifiedSystem.run via runner_loop.py
```

## What counts as success

Success means one of the following:

```text
1. Runner starts and enters life loop.
2. Runner starts far enough to hit a known environment-only graphics/runtime limitation.
```

Environment-only examples:

```text
OpenGL / EGL / X display unavailable
MuJoCo viewer cannot open because server is headless
Open3D window cannot open because display is unavailable
PyQt cannot initialize because no display is available
```

These should be documented, not treated as architecture failures.

## What counts as code failure

These require repair:

```text
ImportError / ModuleNotFoundError
NameError
AttributeError caused by missing config field or renamed helper
TypeError caused by wrong factory kwargs
duplicate IPC/status server bind error
checkpoint load crash caused by changed init order
optimizer crash caused by changed param groups
missing runtime field that used to be initialized by old runner.py
```

## Repair procedure

1. Copy the first traceback.
2. Identify which extracted helper owns the failing responsibility.
3. Fix that helper, not the slim `runner.py`, unless the public compatibility wrapper itself is the source of the error.
4. Add/update a smoke test that would catch the failure next time.
5. Run:

```bash
pytest tests/smoke
```

6. Run again:

```bash
python runner.py --config-path config --config-name runner
```

7. Repeat until startup succeeds or reaches environment-only limitation.

## Helper ownership map

| Symptom | First file to inspect |
|---|---|
| config merge / missing config key | `src/apps/runner_config.py` |
| Hydra entrypoint issue | `src/apps/runner_entry.py` |
| construction order issue | `src/apps/runner_system_factory.py` |
| model/optimizer base init | `src/apps/runner_model_factory.py` |
| full system init field missing | `src/apps/runner_unified_init.py` |
| teacher/vocab issue | `src/apps/runner_teachers.py` |
| train/passive module flags | `src/apps/runner_training_flags.py` |
| hover/flight safety values | `src/apps/runner_hover_config.py` |
| optimizer rebuild | `src/apps/runner_optimizer.py` |
| startup windows/sensors | `src/apps/runner_startup_state.py` |
| runtime counters/log paths | `src/apps/runner_runtime_state.py` |
| IPC/status services | `src/apps/runner_services.py` |
| life loop / viewer sync / shutdown | `src/apps/runner_loop.py` |
| object/self/emotional components | `src/apps/runner_components.py` |
| replay/quality/novelty | `src/apps/runner_memory_factory.py` |
| visualizer construction | `src/apps/runner_visualizer_factory.py` |
| MuJoCo world | `src/apps/runner_world_factories.py` |
| dynamic rig kwargs | `src/apps/runner_dynamic_rig_config.py` |

## Final report template

```text
Command: pytest tests/smoke
Result: passed / failed

Command: python runner.py --config-path config --config-name runner
Result: started / code failure / environment limitation

If failure:
First traceback:
...

Fix applied:
...

Files changed:
...

Tests added/updated:
...
```
