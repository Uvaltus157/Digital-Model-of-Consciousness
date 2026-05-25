# Runner unload map

This document records the completed decomposition of the former heavy V5.10 runner.

The goal was to move stable responsibilities out of `src/apps/runner.py` into small, testable app-level helpers while preserving the public launch command and the public import path.

## Current status

```text
Status: smoke tests passed
Validation reported by user: pytest tests/smoke passed
Runtime startup validation: next step
```

The monolithic runner cleanup is complete at the architecture level. Further work should be runtime-traceback repair only, not a return to the old monolithic file.

## Current launch path

```text
runner.py
→ src/apps/runner_entry.py
→ src/apps/runner_config.py
→ src/apps/runner_system_factory.py
→ src/apps/runner.py / UnifiedSystemV510 compatibility wrapper
→ src/apps/runner_unified_init.py
→ src/apps/runner_loop.py
```

The root `runner.py` routes through the slim entrypoint by default.

Fallback, if still present in the root launcher:

```bash
CWMS_LEGACY_RUNNER=1 python runner.py --config-path config --config-name runner
```

## Current `src/apps/runner.py` role

`src/apps/runner.py` is now a slim compatibility wrapper.

It keeps this public import path stable:

```python
from src.apps.runner import UnifiedSystemV510
```

It no longer owns the old monolithic responsibilities. The class now binds its main hooks to extracted helpers:

```text
UnifiedSystemV510.__init__
→ src/apps/runner_unified_init.py

UnifiedSystemV510.run
→ src/apps/runner_loop.py

UnifiedSystemV510.resolve_module_training_flags_from_config
→ src/apps/runner_training_flags.py

UnifiedSystemV510._force_hover_flight_runtime_config
→ src/apps/runner_hover_config.py

UnifiedSystemV510.rebuild_optimizer_from_trainable_modules
→ src/apps/runner_optimizer.py
```

## Extracted responsibilities

| New file | Responsibility |
|---|---|
| `src/apps/runner_entry.py` | Hydra entrypoint only |
| `src/apps/runner_config.py` | Hydra/OmegaConf filtering, structured config merge, resolved config rendering |
| `src/apps/runner_system_factory.py` | Construction sequence: instantiate system, normalize runtime/startup state, ensure services, apply mode |
| `src/apps/runner_unified_init.py` | Extracted `UnifiedSystemV510.__init__` |
| `src/apps/runner_loop.py` | Outer life loop, train thread, MuJoCo viewer sync, shutdown/save/close sequence |
| `src/apps/runner_teachers.py` | Inner-speech teacher module resolution and loading |
| `src/apps/runner_training_flags.py` | Module train/passive flag resolution from `module_debug.module_modes` |
| `src/apps/runner_hover_config.py` | Flight-safe hover clamps for `dynamic_agent_rig` |
| `src/apps/runner_optimizer.py` | Optimizer rebuild from selected trainable modules |
| `src/apps/runner_services.py` | IPC/status service boundary and metadata hooks |
| `src/apps/runner_runtime_state.py` | Mutable runtime bookkeeping state |
| `src/apps/runner_startup_state.py` | Startup/window/sensor flags |
| `src/apps/runner_modes.py` | Startup mode handling for `mode=train` / `mode=training` |
| `src/apps/runner_components.py` | Object imagery, self core, emotional drive and object visualizer factories |
| `src/apps/runner_model_factory.py` | Device/seed/model/v23 config/base optimizer helpers |
| `src/apps/runner_memory_factory.py` | Replay/quality/novelty helpers |
| `src/apps/runner_visualizer_factory.py` | Inner-world and visualizer snapshot helpers |
| `src/apps/runner_world_factories.py` | MuJoCo world factory boundary |
| `src/apps/runner_dynamic_rig_config.py` | Dynamic agent rig config-to-kwargs helper |
| `src/apps/runner_patches.py` | Idempotent compatibility layer documenting/mirroring extracted hooks |
| `src/apps/runtime_wiring.py` | Future boundary for cross-module runtime wiring |
| `src/apps/system_factory.py` | Reserved scaffold / compatibility concept; active factory is `runner_system_factory.py` |
| `src/apps/bootstrap.py` | Future bootstrap/environment setup boundary |

## Current slim entrypoint shape

`src/apps/runner_entry.py` should stay close to this shape:

```python
print("Resolved config:\n" + render_resolved_runner_config(cfg_raw))
cfg_obj = build_runner_config(cfg_raw)

system = build_unified_system(cfg_obj, UnifiedSystemV510)
system.run()
```

It should not become a second heavy runner.

## Smoke-test validation

Smoke tests were added around the extracted layers.

Primary validation command:

```bash
pytest tests/smoke
```

Known result:

```text
pytest tests/smoke: passed
```

Important smoke-test areas:

```text
runner config normalization
runner entry routing
teacher loading resolver
life-loop importability
training flag resolver
hover config clamps
patch compatibility layer
mode handling
startup state
runtime state
service startup
optimizer rebuild
component factories
world/dynamic rig kwargs
system factory
module registry / module file map
M10/M13 scaffolds
```

## Runtime validation command

Next validation step:

```bash
python runner.py --config-path config --config-name runner
```

If this fails, repair should be by traceback and helper module, not by restoring a monolithic `runner.py`.

## Remaining likely runtime-only risk areas

These are not architecture blockers; they are runtime integration areas to verify with real launch logs:

```text
MuJoCo/OpenGL/display availability
Open3D/PyQt availability
checkpoint loading with the new init path
optimizer state compatibility after optimizer rebuild
duplicate IPC/status service startup
slot_4d JSON-RPC streamer startup/shutdown
camera/action/object visualizer window toggles
```

## Safety rules for future runner work

1. Do not restore the old monolithic `src/apps/runner.py`.
2. Preserve `python runner.py --config-path config --config-name runner` as the main launch command.
3. Preserve `from src.apps.runner import UnifiedSystemV510`.
4. Do not change checkpoint format during runner cleanup.
5. Do not change optimizer param group semantics without a specific traceback and compatibility note.
6. Keep helper modules lightweight where possible; avoid MuJoCo/Open3D/PyQt imports at module import time unless unavoidable.
7. Add or update smoke tests for every new helper or bug fix.
8. If runtime launch fails, fix one traceback at a time and rerun `pytest tests/smoke`.
9. Treat environment-only failures separately from code failures.
10. Keep `runner_entry.py` and `runner.py` thin.

## Completion summary

```text
Architecture-level runner decomposition: complete
Smoke tests: passed
Runtime startup: pending real launch validation
Next work type: traceback-driven runtime repair only
```
