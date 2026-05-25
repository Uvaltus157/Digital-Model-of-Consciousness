# Module migration plan

The repository has already moved toward the `src/apps`, `src/modules`, `src/platform`, `src/shared` structure. This plan records the remaining safe steps.

## Stage 0 — current structure documentation

Status: in progress.

Deliverables:

- `docs/architecture/current_structure.md`
- `docs/architecture/module_file_map.md`
- `docs/architecture/module_file_map.json`

No code behavior changes.

## Stage 1 — complete inventory

Generate a full file inventory and classify every important source/config/doc file.

Rules:

- Do not move files.
- Do not rename public classes/functions.
- Mark ambiguous files as `medium` or `low` confidence.
- Keep `module_file_map.json` valid JSON.

## Stage 2 — package metadata and repository hygiene

Add or maintain:

- `.gitignore`
- `requirements.txt`
- `pyproject.toml`
- optional `.env.example`

Make sure large runtime outputs are ignored:

```text
checkpoints/
data/
runs/
logs/
artifacts/
outputs/
```

## Stage 3 — runner decomposition without behavior change

Current `src/apps/runner.py` is the main orchestration file. It should stay functional while gradually extracting:

```text
src/apps/bootstrap.py
src/apps/system_factory.py
src/apps/runtime_wiring.py
```

Rules:

- Keep root `runner.py` compatible.
- Preserve current Hydra CLI behavior.
- Do not change runtime semantics.
- Add smoke tests before major extraction.

## Stage 4 — M1 dependency cleanup

Current `M1_OBJECT_IMAGERY` runtime imports and inherits cross-module runtime mixins. Move this wiring upward into `src/apps/runtime_wiring.py` or a shared runtime context.

Target direction:

```text
M1 owns object imagery only.
M2/M3/M4/M12/M15 communicate through runtime wiring or shared event/context interfaces.
```

## Stage 5 — scaffolds for missing architecture modules

Add minimal scaffolds for modules that exist architecturally but are not yet implemented.

Recommended first scaffolds:

```text
src/modules/m10_global_conscious_broadcast/state.py
src/modules/m10_global_conscious_broadcast/runtime.py
src/modules/m10_global_conscious_broadcast/debug.py
src/modules/m10_global_conscious_broadcast/README.md

src/modules/m13_autobiographical_memory/state.py
src/modules/m13_autobiographical_memory/memory.py
src/modules/m13_autobiographical_memory/runtime.py
src/modules/m13_autobiographical_memory/debug.py
src/modules/m13_autobiographical_memory/README.md
```

## Stage 6 — split M15 scripted control from imagination/planning

Separate procedural scenario controllers from the conceptual M15 pipeline.

Suggested files:

```text
adaptive_scenario_controller.py      # scripted scenario controller
future_scenario_generator.py         # future scenario proposal
imagined_action_rollout.py           # imagined rollout
outcome_evaluator.py                 # risk/value estimate
candidate_plan.py                    # plan selected before action
```

## Stage 7 — module configs

Move module-specific YAML fragments into module `config/` folders and add Hydra search paths as needed.

Candidates:

```text
M1: object_image / object_image_open3d
M6: train / sleep_sensors
M9: self_core
M11: emotional_drive
M14: latent_semantic_map
```

## Stage 8 — compatibility wrappers if additional moves are needed

If files are moved after this restructure, leave wrappers at old paths.

Wrapper template:

```python
"""Compatibility wrapper.

This module moved during M1-M15 architecture migration.
"""

from new.module.path import *  # noqa: F401,F403
```

If the old module was executable and the new module exposes `main`, preserve:

```python
if __name__ == "__main__":
    main()
```

## Stage 9 — tests and smoke checks

Add checks for:

- import of `src.apps.runner`;
- import of module scaffolds;
- config parsing;
- no accidental import cycles in lightweight modules;
- JSON validity for `module_file_map.json`.

## Non-goals

- Do not rewrite the whole architecture in one PR.
- Do not delete legacy code without a replacement.
- Do not change runner behavior while documenting structure.
- Do not force all modules to be implemented before the runtime stabilizes.
