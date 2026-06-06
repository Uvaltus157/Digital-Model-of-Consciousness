# Conscious loop patch v2

V2 is more correct than v1 because it adds a true M5 boundary receptor instead
of directly mixing the feedback vector into `workspace_seed`.

## Loop

```text
M5 focus_context
  -> M10 global broadcast
  -> M9 self binding
  -> M11 affect + optional M12 metacognition
  -> M15 post-self thought chain
  -> next_focus_context_seed
  -> M5 FocusFeedbackBoundary
  -> workspace_seed + preconscious_seed
```

## New M5 layer

```text
src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
```

This layer receives:

```python
focus_context_seed
focus_context_seed_gate
workspace_seed
```

and produces:

```python
workspace_seed'
preconscious_seed_delta
learned_gate
total_gate
```

## Why v2 is better

V1 did:

```text
enhanced_focus_context -> workspace_seed arithmetic mix
```

V2 does:

```text
enhanced_focus_context -> M5 learned receptor -> workspace_seed + preconscious_seed boundary
```

So the feedback becomes a real M5 submodule.

## Install

Recommended on a clean branch:

```bash
unzip -o conscious_loop_patch_v2.zip -d .
python scripts/apply_conscious_loop_patch_v2.py
```

If v1 was already applied, either revert v1 or inspect anchors manually.
