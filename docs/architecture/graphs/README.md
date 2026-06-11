# Architecture Graphs

This folder is the fixed source-of-truth location for editable Graphviz `.dot`
architecture diagrams.

## Fixed DOT format

All graphs must follow `DOT_FORMAT_GUIDE.md`:

1. clusters contain only nodes
2. all edges are declared at the top level
3. no `ltail` / `lhead`
4. no normal edges inside clusters
5. every NN node is marked `NN:` and has `fillcolor="#ffd6d6"`
6. long text goes into `description`, not into huge visible labels

## Current fixed graph set

| File | Purpose |
| --- | --- |
| `unconscious_contour_architecture.dot` | Architecture-level unconscious contour |
| `unconscious_contour_runtime_life_step.dot` | Runtime order of the unconscious loop inside `life_step()` |
| `module_m1_architecture.dot` | Target architectural M1 graph |
| `module_m1_code.dot` | Current implementation/code graph of M1 |
| `module_m2_event_dream_replay.dot` | M2 event dream replay and M5 seed path |
| `module_m4_long_dynamic_memory.dot` | M4 long dynamic identity/passport structure |
| `module_m5_seed_bus.dot` | M5 FocusFeedbackBoundary and seed priority bus |

## Naming note

`module_m1_object_imagery.dot` was renamed to `module_m1_code.dot` because it
describes the current code/implementation, while `module_m1_architecture.dot`
describes the target M1 architecture.
