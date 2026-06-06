# Architecture Genome

Top-level DNA of the project.

## Runtime layers

```text
runner.py
↓
src/apps/runner.py
↓
UnifiedSystem = composition of runtime mixins
↓
src/apps/life_runtime.py
↓
M1–M15 modules
↓
M8 status/monitor/UI
```

## M1–M15 module map

| Module | Owns |
|---|---|
| M1 | Object imagery, object slots, inner object 3D |
| M2 | Event dream replay, unconscious replay planner |
| M3 | Self action causality, action output, motor guard |
| M4 | Long dynamic memory, identity token, object stability |
| M5 | World model, attention workspace, FocusFeedbackBoundary |
| M6 | Learning, sleep consolidation, train/sleep gates |
| M7 | Inner speech, thought stream |
| M8 | Debug visual control, IPC, monitors |
| M9 | Self core, body ownership, agency |
| M10 | Global conscious broadcast |
| M11 | Motivational homeostasis, stress/relief/curiosity |
| M12 | Metacognition, uncertainty, confidence |
| M13 | Autobiographical memory, episodes |
| M14 | Semantic grounding |
| M15 | Counterfactual imagination/planning |

## Two-contour architecture

The system has two interacting contours:

```text
1. Unconscious contour:
   fast affect/replay/object loop.

2. Conscious contour:
   selected content, broadcast, inner speech, planning, metacognition.
```

## Strict unconscious loop

```text
World / body
↓
M1 object imagery
↓
M5 world workspace
↓
M11 affect
↓
M13 episodic memory + M4 identity
↓
M2 unconscious replay planner
↓
M5 FocusFeedbackBoundary seed
↓
M3 action proposal / motor guard
```

## Strict conscious loop

```text
M1 object imagery + M5 workspace
↓
M10 global conscious broadcast
↓
M9 self core / agency / body ownership grounding
↓
M7 inner speech / subjective stream
↓
M15 counterfactual imagination / planning
↓
M12 metacognitive confidence/uncertainty
↓
M14 semantic grounding + M13/M4 memory context
↓
M3 controlled action proposal / ActionGuard
↓
M5 conscious focus seed through common seed bus
```

## Sleep/replay invariants

```text
M1 sensors weakened/off
M3 outward motor output blocked
M11 active
M13 active
M4 active
M2 active
M5 receives replay/prototype seed through the common seed bus
M10/M7/M15 may be reduced or used only for dream report/debug
```

## Common buses

```text
IPC Action Bus
Module Status Bus
M5 Seed Bus
Object Slot Bus
Conscious Broadcast Bus
Subjective Stream Bus
Sleep/Train Gate
Debug/Imit Bus
```

## Non-negotiable rules

```text
No hidden direct M1 raw sensors → M2.
M2 must feed M5 through the same seed bus used by M15.
M5 seeds must pass through FocusFeedbackBoundary.
M10 selects/broadcasts; it does not mutate memories directly.
M7 verbalizes; it does not control M3 directly.
M15 plans; it does not write focus_context directly.
Imitators must be explicit and labeled source=*_imit.
Imitators live under module/imit/.
Monitors are read-only.
Sleep must block outward M3 motor output.
```
