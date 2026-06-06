# Runtime Wiring DNA

This document defines how code should wire modules together.

## root launch

Alias: Root launch.

```text
runner.py
↓
src/apps/runner.py
↓
UnifiedSystem
↓
life step loop
```

## UnifiedSystem composition

A regenerated implementation should compose module mixins in `src/apps/runner.py`
or future `src/apps/system_factory.py`.

Expected conceptual composition:

```python
class UnifiedSystem(
    ObjectImageryRuntimeMixin,              # M1
    EventDreamReplayRuntimeMixin,           # M2
    ActionRuntimeMixin,                     # M3
    LongDynamicMemoryRuntimeMixin,          # M4
    WorldModelWorkspaceRuntimeMixin,        # M5
    LearningSleepRuntimeMixin,              # M6
    InnerSpeechRuntimeMixin,                # M7
    ModuleStatusRuntimeMixin,               # M8
    SelfCoreRuntimeMixin,                   # M9
    GlobalConsciousBroadcastRuntimeMixin,   # M10
    MotivationalHomeostasisRuntimeMixin,    # M11
    MetacognitionRuntimeMixin,              # M12
    AutobiographicalMemoryRuntimeMixin,     # M13
    SemanticGroundingRuntimeMixin,          # M14
    CounterfactualPlanningRuntimeMixin,     # M15
    # imitators
    M1ObjectSlotLatentImitRuntimeMixin,
    M5LatentPrototypeRuntimeMixin,
):
    ...
```

## awake step order

Alias: Awake step order.

```text
1. read sensors / simulator
2. M1 object imagery
3. M5 workspace update
4. M11 affect update
5. M4 identity update
6. M13 episode update
7. M2 unconscious replay candidate
8. M10 conscious candidate selection
9. M9 self grounding
10. M7 inner speech
11. M15 counterfactual plan
12. M12 confidence/uncertainty
13. common M5 seed bus
14. M3 action guard/output
15. M8 status publish
```

## sleep/replay step order

Alias: Sleep/replay step order.

```text
1. sleep mode ON
2. M1 sensors weakened/off
3. M3 previous motor states zeroed
4. M3 outward output blocked
5. M11 active
6. M13 active
7. M4 active
8. M2 replay material selected
9. M5 seed through FocusFeedbackBoundary
10. replay quality metrics update
11. optional dream report through M10/M7
12. M8 status publish
```

## Common M5 seed bus

Possible seed sources:

```text
debug/imit seed
M15 conscious seed
M2 unconscious replay seed
fallback none
```

The exact priority must be explicit and tested.

Recommended priority:

```text
1. explicit active debug/imit seed
2. active conscious M15 seed
3. active unconscious M2 replay seed
4. none
```

Every seed source must return:

```text
seed: tensor [B, focus_context_dim]
gate: tensor/scalar [B, 1] or compatible
source label
```

## Status publish

Each life step should update:

```text
self.latest_out
self.last_status
module_debug_status.json or IPC status payload
```

M8 reads status. M8 does not own model logic.
