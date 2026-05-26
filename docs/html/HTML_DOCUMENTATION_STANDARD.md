# HTML documentation standard

This file defines the stable format for project HTML documentation.

The goal is to keep every new documentation page consistent, expandable and easy to compare across modules. Do not replace detailed pages with shorter versions. Every update should preserve existing useful sections and add new detail on top.

## Language policy

- Repository HTML documentation is written in English and stored in `docs/html`.
- Russian copies may be generated as downloadable archives for review.
- Do not commit Russian archive copies into the repository unless explicitly requested.

## Folder layout

```text
docs/html/
  index.html
  assets/
    style.css
  apps/
    runner_skeleton.html
  modules/
    m1_object_imagery.html
    m2_event_dream_replay.html
    ...
  HTML_DOCUMENTATION_STANDARD.md
```

## Required page structure

Every substantial HTML documentation page should follow this order:

```text
1. Header
2. Short purpose / role
3. Place in the global architecture
4. Strict top-down pipeline
5. Main data structures / runtime state
6. Source file map table
7. Runtime flow
8. Connections to other modules
9. Debug / validation checklist
10. Typical failure modes
11. Readiness / completion criteria
12. Footer with maintenance note
```

A page may add extra sections, but should not omit the required sections unless the page is only a small index or navigation page.

## Standard source file table

Use this table format for source maps:

| File | Link | What's inside | How it works |
|---|---|---|---|
| `file.py` | `open source` | Main classes, configs, functions, state objects. | Runtime role and data flow. |

Rules:

1. `File` contains the short filename or relative component path.
2. `Link` contains a local relative link to the source file.
3. `What's inside` describes classes, configs, functions and important state.
4. `How it works` explains the role in the runtime pipeline.
5. Do not use vague entries such as “helper file” without naming what the helper does.
6. If a file is part of the module runtime, model, visualizer, IPC, factory or debug path, include it.
7. If a file is legacy but still relevant, mark it as legacy/diagnostic and explain why it remains documented.

## Link style

Local links should work when browsing inside the repository checkout.

From module pages:

```html
<a href="../../../src/modules/m01_object_imagery/runtime.py">open source</a>
```

From app pages:

```html
<a href="../../../src/apps/runner_entry.py">open source</a>
```

From `docs/html/index.html`:

```html
<a href="../src/apps/runner.py">open source</a>
```

## Visual style

Use the shared stylesheet:

```html
<link rel="stylesheet" href="../assets/style.css" />
```

For top-level index pages:

```html
<link rel="stylesheet" href="assets/style.css" />
```

Use the existing visual vocabulary:

```text
badge red    = neural/model component
badge blue   = runtime/sensor/control bridge
badge green  = memory/state/factory integration
badge purple = visual/debug/IPC/inspection
badge warn   = pending validation or environment limitation
badge good   = completed/passing status
```

## Pipeline section standard

The pipeline should be written as a strict top-down chain. Prefer this structure:

```html
<section class="card">
  <h2>Strict top-down pipeline</h2>
  <div class="flow">
    <div class="row"><div class="row-title">0. Inputs</div><div class="row-body">...</div></div>
    <div class="arrow">↓</div>
    <div class="row"><div class="row-title">1. Stage</div><div class="row-body">...</div></div>
  </div>
</section>
```

Each row should describe one conceptual stage only. Do not merge unrelated pipelines into one row.

## Runtime flow standard

Every module page should include a plain ordered list explaining what happens during one life step:

```text
1. What input arrives.
2. What gates/validation happen.
3. What model/fusion/transformation runs.
4. What state is updated.
5. What outputs are produced.
6. What other modules receive those outputs.
7. What visual/debug surfaces are updated.
```

## Debug checklist standard

Use a table:

| Signal | Expected behavior |
|---|---|
| `active_slot_index` | Does not jump randomly between objects. |

The checklist must contain signals that can actually be observed in logs, debug windows, state dictionaries, IPC status or visualizers.

## Typical failure modes standard

Use short, actionable bullets:

```text
- Symptom: likely cause and first file to inspect.
```

Example:

```text
- Slot identity jumping: inspect sticky thresholds and proposal slot lock in `ObjectSlotMemory`.
```

## Readiness criteria standard

Every module page should end with a checklist of completion conditions:

```text
- Stable core state exists.
- Runtime gates work.
- Outputs are consumed by downstream modules.
- Debug surface exposes enough state to validate behavior.
- Smoke/runtime tests cover the important path.
```

## Module page naming

Use lowercase module IDs and descriptive names:

```text
m1_object_imagery.html
m2_event_dream_replay.html
m3_self_action_causality.html
m4_long_dynamic_memory.html
m5_world_model_workspace.html
m6_learning_sleep.html
m7_inner_speech.html
m8_debug_visual_control.html
m9_self_core.html
m10_conscious_broadcast.html
m11_motivation_homeostasis.html
m12_metacognition.html
m13_autobiographical_memory.html
m14_semantic_grounding.html
m15_imagination_planning.html
```

## Quality rules

1. Keep pages detailed.
2. Do not remove existing useful sections.
3. Prefer tables for source maps.
4. Prefer strict vertical flow for pipelines.
5. Add local links to source files whenever a file is mentioned as important.
6. Keep English repository docs and Russian downloadable copies structurally aligned.
7. If a page is updated in English, regenerate the Russian copy when requested.
8. Keep `docs/html/index.html` rich enough to explain the whole project, not just a list of links.
9. Avoid undocumented “magic” modules: if a runtime path exists, add it to the relevant table.
10. Treat this file as the source of truth for HTML documentation style.
