# Module file map

This is a draft post-restructure map for the current `src/`-based project layout.

The project is already partially migrated from technical folders into architecture-level modules:

```text
src/apps/
src/modules/m01_object_imagery/
src/modules/m02_event_dream_replay/
...
src/modules/m15_counterfactual_imagination_planning/
src/platform/
src/shared/
```

The JSON source of truth is:

```text
docs/architecture/module_file_map.json
```

## Current high-confidence anchors

| Path | Group | Role | Status | Note |
|---|---|---|---|---|
| `runner.py` | apps | entrypoint | keep | Root compatibility launcher. |
| `src/apps/runner.py` | apps | entrypoint | keep | Main V5.10 orchestration entrypoint. |
| `src/shared/config.py` | shared | config | keep | Shared dataclass configuration. |
| `src/modules/m01_object_imagery/runtime.py` | M1_OBJECT_IMAGERY | runtime | keep | Object imagery runtime, but currently has cross-module wiring. |
| `src/modules/m08_debug_visual_control/module_debug_status_ipc.py` | M8_DEBUG_VISUAL_CONTROL | debug | keep | Module Debug status IPC server. |
| `src/modules/m15_counterfactual_imagination_planning/config/adaptive_scenario_controller.yaml` | M15_COUNTERFACTUAL_IMAGINATION_PLANNING | config | keep | Scenario controller config; future split recommended. |

## Module intent

### M1_OBJECT_IMAGERY

Owns object imagery: sensors/fusion/object slots/latent object/reconstruction heads.

Current caution: `runtime.py` imports many cross-module runtime mixins. This should be gradually moved to an application wiring layer.

### M2_EVENT_DREAM_REPLAY

Owns event tokenization, event sentence memory, event replay, dream-event decoding.

### M3_SELF_ACTION_CAUSALITY

Owns action heads, manual override, expected/observed comparison, causal attribution.

### M4_LONG_DYNAMIC_MEMORY

Owns dynamic object identity, object passport, restore/reuse, long object memory.

### M5_WORLD_MODEL_ATTENTION_WORKSPACE

Owns Dreamer/world model, RSSM, novelty, attention, active workspace.

### M6_LEARNING_SLEEP_CONSOLIDATION

Owns training loop, sleep sensor gating, replay training, consolidation.

### M7_INNER_SPEECH_THOUGHTS

Owns inner speech, subjective stream, thought decoding.

### M8_DEBUG_VISUAL_CONTROL

Owns module debug window, debug status IPC, debug schemas, visual/control state.

### M9_SELF_CORE

Owns SelfCore, agency, body ownership, egocentric frame, temporal self.

### M10_GLOBAL_CONSCIOUS_BROADCAST

Owns conscious candidates, competition, content selection, broadcast packet.

Current status: scaffold recommended.

### M11_MOTIVATIONAL_HOMEOSTASIS

Owns curiosity/safety/mastery/fatigue drives and motivation state.

### M12_METACOGNITION_MONITOR

Owns confidence monitors, uncertainty, doubt, need_to_check.

### M13_AUTOBIOGRAPHICAL_MEMORY

Owns self episodes, personal timeline, discovery/mistake memory, emotional tags, autobiographical recall.

Current status: scaffold recommended.

### M14_SEMANTIC_GROUNDING

Owns word-to-object/action/sensation/self grounding and concept memory.

### M15_COUNTERFACTUAL_IMAGINATION_PLANNING

Owns future scenario generation, counterfactual model, imagined rollout, outcome evaluation, candidate plan.

Current caution: scripted/adaptive scenario control should be separated from true counterfactual imagination/planning.

## Next mapping work

1. Generate a complete file inventory from the Git tree.
2. Fill `module_file_map.json` with every source/config/doc file.
3. Mark ambiguous files as `medium` or `low` confidence.
4. Only after that, perform safe moves with compatibility wrappers.
