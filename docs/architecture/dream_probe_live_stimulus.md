# Dream Probe / Replay Probe

Adds live diagnostic stimulus buttons for sleep/replay debugging.

This is not a permanent model feature. It is a runtime diagnostic probe:

```text
M8 Sleep Replay Monitor
    Probe curiosity
    Probe stress
    Probe replay seed
    Probe mixed
    Clear probe
```

IPC action:

```text
dream_probe_inject
```

Payload examples:

```json
{"kind": "curiosity", "intensity": 0.85, "duration": 80}
{"kind": "stress", "intensity": 0.85, "duration": 80}
{"kind": "replay_seed", "intensity": 0.75, "duration": 60}
{"kind": "mixed", "intensity": 0.75, "duration": 80}
{"kind": "clear"}
```

Rules:

```text
- no raw M1 → M2 path
- no direct out["focus_context"] mutation
- stress/curiosity probe affects M11 inputs before EmotionalDrive.compute(...)
- replay_seed probe uses the existing M2/M5 seed bus:
  _event_dream_next_focus_seed + _event_dream_next_focus_gate
```
