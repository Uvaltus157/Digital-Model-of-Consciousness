# Sleep Replay Monitor

Adds a dedicated M8 window for watching the live unconscious sleep/replay loop.

## Status payload

Runner status IPC exposes:

```text
sleep_replay_monitor
```

with sections:

```text
m1  sensor gates
m11 affect
m13 autobiographical retrieval
m4  dynamic identity
m2  event/dream replay
m5  FocusFeedbackBoundary seed
m3  sleep motor guard
```

## M8 UI

M8 tab receives:

```text
Sleep Replay Monitor
```

The window updates from `last_status["sleep_replay_monitor"]` every status poll.
