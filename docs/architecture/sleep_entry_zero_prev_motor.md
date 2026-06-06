# Sleep entry zero prev motor state

When entering full sleep/replay mode, the previous awake motor command can still
be stored in:

```text
prev_embodied_action
prev_hand_motor
```

`life_runtime.py` uses these values at the very start of the next step before
the current step's `sleep_motor_guard` has a chance to run.

This patch adds a transition guard inside `SleepSensorsMixin.apply_sleep_sensor_state(...)`:

```text
awake/partial_cut -> sleep
    zero prev_embodied_action
    zero prev_hand_motor
```

It only fires on a transition into full sleep:

```text
old_sleep == False
new_sleep == True
```

It does not fire on partial sensor cuts.
