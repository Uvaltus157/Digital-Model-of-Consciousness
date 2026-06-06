# Code Generation Order DNA

This is the recommended order for regenerating implementation code.

## Phase 0 — package skeleton

```text
pyproject.toml
requirements.txt
runner.py
src/apps/
src/modules/
src/shared/
src/platform/
tests/
```

## Phase 1 — shared config and types

```text
src/shared/config.py
src/shared/types.py
src/shared/tensor_utils.py
src/shared/event_bus/
```

## Phase 2 — M8 status and IPC first

Build observability before model intelligence:

```text
module_status_runtime.py
module_debug_status_ipc.py
control_panel.py
status schemas
```

## Phase 3 — safety gates

```text
SleepGate
TrainGate
M3 ActionGuard
```

## Phase 4 — M1 object slot path

```text
ObjectSlotMemory
build_inner_object_vision_proposals
_memory_update_forced_slot
Inner Object 3D
M1 imit
```

## Phase 5 — M5 seed bus and boundary

```text
common get_m5_focus_seed
FocusFeedbackBoundary
M5 latent prototype imit
M5 Learning Quality monitor
```

## Phase 6 — unconscious contour

```text
M11
M13
M4
M2
Replay Quality Monitor
Sleep Replay Monitor
```

## Phase 7 — conscious contour

```text
M10
M9
M7
M15
M12
M14
conscious monitors
```

## Phase 8 — real training

```text
M1 training
M5 training
M2 replay training
M10/M7/M15/M12 training
```

## Phase 9 — Level 5 comparison

```text
real latents/content vs prototype/imit latents/content
disable imitator only after equivalence passes
```
