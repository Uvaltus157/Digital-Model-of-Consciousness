# Unconscious sleep/replay loop v1

This patch fixes the strictly unconscious loop:

```text
M11 affect / activation
        ↓
M2 event_dream_replay  ←  M13 long episodic memory
        ↑
        │
M4 long dynamic identity
        ↓
M5 FocusFeedbackBoundary
        ↓
M5 inner playback / world model
        ↓
M11 emotional evaluation
```

## What changes

### M2 no longer directly mutates focus_context by default

Before:

```text
M2 replay_context -> out["focus_context"] blend
```

After:

```text
M2 replay_context -> next_focus_context_seed
                 -> M5 FocusFeedbackBoundary
                 -> workspace_seed + preconscious_seed
```

This is the same M5 entry where M15 sends the conscious-loop seed.

### M4 is included

M2 now reads:

```text
out["long_dynamic_memory"]["dynamic_identity_context"]
out["long_dynamic_memory"]["identity_stability"]
out["long_dynamic_memory"]["identity_novelty"]
out["long_dynamic_memory"]["dynamic_memory_gate"]
```

and blends M4 context into `replay_context`.

### Runtime order

After M11 affect is computed:

```text
M13 retrieval
M2 event/dream replay
M2 stores next M5 seed
```

The next M5 call receives that seed through `model_step(..., model_stage="pre_observe")`.

## Install

```bash
unzip -o unconscious_sleep_loop_patch_v1.zip -d .
python scripts/apply_unconscious_sleep_loop_patch_v1.py
```
