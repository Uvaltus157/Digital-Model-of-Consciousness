# Rebuild From DNA

How to reconstruct the project architecture from scratch.

## 1. Create skeleton

```text
config/
docs/architecture/
docs/architecture/dna/
src/apps/
src/modules/
src/platform/
src/shared/
tests/
```

## 2. Create M1–M15 module dirs

```text
src/modules/m01_object_imagery/
src/modules/m02_event_dream_replay/
src/modules/m03_self_action_causality/
src/modules/m04_long_dynamic_memory/
src/modules/m05_world_model_attention_workspace/
src/modules/m06_learning_sleep_consolidation/
src/modules/m07_inner_speech_thoughts/
src/modules/m08_debug_visual_control/
src/modules/m09_self_core/
src/modules/m10_global_conscious_broadcast/
src/modules/m11_motivational_homeostasis/
src/modules/m12_metacognition_monitor/
src/modules/m13_autobiographical_memory/
src/modules/m14_semantic_grounding/
src/modules/m15_counterfactual_imagination_planning/
```

Each module should eventually contain:

```text
README.md
runtime.py
state.py / memory.py / model.py as needed
debug.py / status.py
imit/
tests
```

## 3. Build buses

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

## 4. Build unconscious contour

```text
M1 → M5 → M11 → M13/M4 → M2 → M5 → M3 guard
```

## 5. Build conscious contour

```text
M1/M5 → M10 → M7 → M15 → M12/M9/M14 context → M3 / M5 seed bus
```

## 6. Build M8 observability first

Before training, every important path must have:

```text
status payload
M8 window/table
contract test
live-smoke path
```

## 7. Add imitators before training

```text
M1 object-slot latents
M5 latent prototypes
M11 emotion probes
M13 memory episode probes
M2 replay material probes
M10 conscious broadcast probes
M7 inner speech probes
M15 counterfactual planning probes
M12 confidence/uncertainty probes
```

## 8. Validate levels 1–3 before training

```text
Level 1: wiring/contracts
Level 2: learned-like imit latents
Level 3: downstream behavior
```

## 9. Replace imitators with real training

```text
imit output contract
↓
real trained output contract
↓
same downstream behavior
↓
compare with prototype
↓
disable imit for that module
```
