# UI DNA

This document defines M8 control panel and monitor behavior.

## M8 buttons

Alias: M8 button registry.

Required buttons:

```text
M8:
  Module Debug
  Run Module Lab
  Sleep Replay Monitor
  Replay Quality Monitor
  M5 Learning Quality
  M5 Latent Prototypes
  M1 Object Slot Imit
```

Future conscious buttons:

```text
Conscious Broadcast Monitor
Inner Speech Monitor
Counterfactual Planning Monitor
Metacognition Monitor
Self Core Monitor
```

## IPC action messages

```text
m1_object_slot_imit_inject
m1_object_slot_imit_clear
m5_latent_prototype_inject
m5_latent_prototype_clear
sleep_replay_mode_set
dream_probe_inject
module_lab_run
```

Future conscious actions:

```text
m10_broadcast_probe_inject
m7_inner_speech_probe_inject
m15_counterfactual_probe_inject
m12_metacognition_probe_inject
```

## status windows

Each window must have:

```text
open method
refresh method
raw JSON
status labels
button style
runner-connected enable/disable
```

## Inner Object 3D requirements

Alias: Inner Object 3D UI.

Must show:

```text
selected_slot
slot confidence
z_obj norm
decoded shape / point cloud / debug fallback primitive
source label
```

If decoder is untrained:

```text
debug_imit_fallback_shape = True
```

must be visible for fallback cube/tetra/morph.

## conscious monitors

```text
Conscious Broadcast Monitor
Self Core Monitor
Inner Speech Monitor
Counterfactual Planning Monitor
Metacognition Monitor
Semantic Grounding Monitor
```
