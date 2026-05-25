# Runner unload map

This document records the staged decomposition of the former heavy V5.10 runner.

The goal is not to rewrite the runtime in one large risky change. The goal is to move stable responsibilities out of `src/apps/runner.py` into small, testable app-level helpers while preserving current behavior.

## Current launch path

```text
runner.py
→ src/apps/runner_entry.py
→ src/apps/runner_config.py
→ src/apps/runner_patches.py
→ src/apps/runner.py / UnifiedSystemV510
→ src/apps/runner_loop.py
```

The root `runner.py` now routes through the slim entrypoint by default.

Fallback:

```bash
CWMS_LEGACY_RUNNER=1 python runner.py --config-path config --config-name runner
```

This bypasses `runner_entry.py` and enters the legacy heavy runner directly.

## Extracted responsibilities

| New file | Extracted responsibility | Previous location | Current integration method |
|---|---|---|---|
| `src/apps/runner_config.py` | Hydra/OmegaConf filtering, structured config merge, resolved config rendering | `src/apps/runner.py main()` | Used directly by `runner_entry.py` |
| `src/apps/runner_teachers.py` | Inner-speech teacher module resolution and loading | `load_inner_speech_teacher_from_config()` in `src/apps/runner.py` | Patched through `runner_patches.py` |
| `src/apps/runner_loop.py` | Outer life loop, train thread, MuJoCo viewer sync, shutdown/save/close sequence | `UnifiedSystemV510.run()` | Patched through `runner_patches.py` |
| `src/apps/runner_training_flags.py` | Module train/passive flag resolution from `module_debug.module_modes` | `UnifiedSystemV510.resolve_module_training_flags_from_config()` | Patched through `runner_patches.py` |
| `src/apps/runner_hover_config.py` | Flight-safe hover clamps for `dynamic_agent_rig` | `UnifiedSystemV510._force_hover_flight_runtime_config()` | Patched through `runner_patches.py` |
| `src/apps/runner_modes.py` | Startup mode handling for `mode=train` / `mode=training` | `runner_entry.py` / old `runner.py main()` block | Used directly by `runner_entry.py` |
| `src/apps/runner_patches.py` | Centralized behavior-preserving monkey patches for extracted helpers | Inline patch code in `runner_entry.py` | Used directly by `runner_entry.py` |
| `src/apps/runtime_wiring.py` | Future boundary for cross-module runtime wiring | Currently scattered across runner and M1 runtime inheritance | Documentation/scaffold only |
| `src/apps/system_factory.py` | Future system construction boundary | Currently `UnifiedSystemV510(cfg_obj)` inside `runner_entry.py` | Reserved scaffold |
| `src/apps/bootstrap.py` | Future bootstrap/environment setup boundary | Root `runner.py` and entrypoint path setup | Reserved scaffold |

## Current slim entrypoint shape

`src/apps/runner_entry.py` should stay close to this shape:

```python
print("Resolved config:\n" + render_resolved_runner_config(cfg_raw))
cfg_obj = build_runner_config(cfg_raw)

system = UnifiedSystemV510(cfg_obj)
apply_runner_mode(system, cfg_obj)
system.run()
```

It should not become a second heavy runner.

## Current patch layer

`src/apps/runner_patches.py` currently replaces these heavy-runner hooks:

```text
src.apps.runner.load_inner_speech_teacher_from_config
UnifiedSystemV510.run
UnifiedSystemV510.resolve_module_training_flags_from_config
UnifiedSystemV510._force_hover_flight_runtime_config
```

This is an intentional transitional mechanism. It lets the default launch path use extracted helpers without performing a risky full rewrite of `src/apps/runner.py`.

## Smoke tests added

```text
tests/smoke/test_runner_config.py
tests/smoke/test_runner_entry.py
tests/smoke/test_runner_teachers.py
tests/smoke/test_runner_loop.py
tests/smoke/test_runner_training_flags.py
tests/smoke/test_runner_hover_config.py
tests/smoke/test_runner_patches.py
tests/smoke/test_runner_modes.py
```

Recommended validation:

```bash
pytest tests/smoke
```

## What still remains inside `src/apps/runner.py`

`src/apps/runner.py` still owns the heavy runtime class and construction sequence:

```text
UnifiedSystemV510
  model construction
  optimizer construction
  inner object system construction
  self core construction
  visualizer construction
  replay / quality / novelty construction
  emotional drive construction
  MuJoCo world construction
  dynamic agent rig controller construction
  IPC/status server construction
  startup window flags
  checkpoint load/save interaction
```

This is acceptable for now. The large class is still the source of truth for actual runtime state.

## Next safe extraction candidates

### 1. `runner_components.py`

Extract object construction into small helpers, but do not change behavior:

```text
create_world_model(cfg, device)
create_inner_object_system(cfg, device)
create_self_core(cfg, device)
create_emotional_drive(cfg)
create_visualizers(cfg)
create_world(cfg, device)
create_dynamic_agent_rig_controller(cfg, world)
```

Risk: medium. These helpers touch Torch, MuJoCo, visualizers and optimizer param groups.

### 2. `runner_startup_state.py`

Extract startup booleans and window flags:

```text
training_enabled
show_inner_world_window
show_camera_preview_window
show_action_outputs_window
show_module_debug_window
show_inner_object_window
show_event_code_visualizer_window
show_inner_object_open3d_window
sensor enable flags
manual action IPC flags
```

Risk: low/medium. The flags are many but mostly simple.

### 3. `runner_services.py`

Extract service startup:

```text
ModuleDebugStatusServer
IPCControlServer
slot_4d_jsonrpc_streamer
external control flags
sensor preview metadata
```

Risk: medium. Must preserve start/stop order.

### 4. `runner_optimizer.py`

Extract optimizer construction and param group addition:

```text
create_optimizer(model, cfg)
add_inner_object_params(optimizer, inner_object_system)
add_self_core_params(optimizer, self_core)
```

Risk: medium. Optimizer state/checkpoint compatibility must be watched.

### 5. Replace patching with direct imports

Once the extracted helpers are stable, edit `src/apps/runner.py` directly so it imports and uses them natively. Then remove corresponding monkey patches from `runner_patches.py`.

Risk: medium. Do this one hook at a time.

## Target future structure

The long-term target is:

```text
runner.py                       # root compatibility launcher
src/apps/runner_entry.py         # Hydra entrypoint only
src/apps/system_factory.py       # builds UnifiedSystem/runtime object
src/apps/runner_components.py    # heavy component creation helpers
src/apps/runner_services.py      # IPC/status/window service startup
src/apps/runner_loop.py          # outer life loop
src/apps/runtime_wiring.py       # cross-module wiring boundary
src/apps/runner_config.py        # config normalization
src/apps/runner_modes.py         # mode handling
```

And then `src/apps/runner.py` can either become:

```text
src/apps/unified_system_v510.py
```

or be kept as a compatibility import wrapper around the new runtime class location.

## Safety rules for future runner unload work

1. Keep `CWMS_LEGACY_RUNNER=1` fallback until a full runtime smoke run passes.
2. Do not move multiple heavy construction blocks in one step.
3. Add a smoke test for every extracted helper.
4. Avoid importing MuJoCo/Open3D/PyQt at import time in lightweight helper modules.
5. Preserve `python runner.py --config-path config --config-name runner` as the main launch command.
6. Do not change checkpoint format during runner unloading.
7. Do not change optimizer param groups without documenting checkpoint compatibility.
8. Keep behavior-preserving patches centralized in `runner_patches.py` until direct integration is safe.
