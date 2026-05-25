# Current repository structure

This document describes the current post-reorganization layout of the project.

The repository now uses a clear separation between top-level launchers, application orchestration, consciousness modules, platform infrastructure, shared utilities, documentation, and generated runtime artifacts.

```text
Digital-Model-of-Consciousness/
├── README.md
├── runner.py                         # thin compatibility launcher
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── config/
│   └── runner.yaml                   # main Hydra configuration
│
├── docs/
│   ├── architecture/
│   │   ├── current_structure.md
│   │   ├── module_file_map.md
│   │   ├── module_file_map.json
│   │   └── module_migration_plan.md
│   └── images/
│
└── src/
    ├── apps/
    │   ├── runner.py                 # main V5.10 orchestration entrypoint
    │   ├── life_runtime.py
    │   └── unified_conscious_viewer.py
    │
    ├── modules/
    │   ├── m01_object_imagery/
    │   ├── m02_event_dream_replay/
    │   ├── m03_self_action_causality/
    │   ├── m04_long_dynamic_memory/
    │   ├── m05_world_model_attention_workspace/
    │   ├── m06_learning_sleep_consolidation/
    │   ├── m07_inner_speech_thoughts/
    │   ├── m08_debug_visual_control/
    │   ├── m09_self_core/
    │   ├── m10_global_conscious_broadcast/
    │   ├── m11_motivational_homeostasis/
    │   ├── m12_metacognition_monitor/
    │   ├── m13_autobiographical_memory/
    │   ├── m14_semantic_grounding/
    │   └── m15_counterfactual_imagination_planning/
    │
    ├── platform/
    │   ├── mujoco_world/
    │   ├── ipc/
    │   ├── gui/
    │   └── scene_builder/
    │
    └── shared/
        ├── config.py
        ├── checkpointing.py
        ├── console_colors.py
        └── event_bus/
```

## Layer rules

### `runner.py`

The root `runner.py` is a compatibility launcher. It should stay small and only normalize paths/environment before delegating to `src/apps/runner.py`.

### `src/apps/`

Application-level orchestration. This layer wires modules together, owns the main runtime class composition, and may import from many modules.

### `src/modules/`

Architecture-level consciousness modules `M1` through `M15`. A module should primarily own its own state, runtime, models, memory, debug helpers, and visualization that belongs to that module.

### `src/platform/`

Infrastructure that is not one consciousness module: MuJoCo world, scene building, low-level GUI support, IPC transport, rendering support, and hardware/simulator adapters.

### `src/shared/`

Common configuration dataclasses, utility functions, lightweight event bus, schemas, common types, and project-wide helpers.

### `config/`

Main Hydra configuration. Module-specific configs may live under `src/modules/mXX_.../config` and be added through Hydra search paths.

### generated / ignored directories

These directories should normally be local/runtime artifacts, not committed:

```text
checkpoints/
data/
runs/
logs/
artifacts/
outputs/
inner_world_frames/
open3d_exports/
```

## Current architectural caution

`src/modules/m01_object_imagery/runtime.py` currently coordinates many cross-module runtime mixins. This is acceptable during migration, but the target direction is to move cross-module wiring into `src/apps/runtime_wiring.py` or shared event/context interfaces so that M1 does not become the hidden central orchestrator.
