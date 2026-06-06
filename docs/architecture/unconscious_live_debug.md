# Unconscious live debug v1

Adds runtime visibility for the unconscious sleep/replay loop.

## Architecture

```text
M1 -> M5 -> M11 -> M2 -> FocusFeedbackBoundary -> M5
      |             ^
      v             |
     M4 ------------+
                   M13 -> M2

M5 -> M3 -> body/world -> M1
```

## Live trace

The runtime can print:

```text
[unconscious_loop step=120]
sleep=1 state=sleep
m11: val=-0.12 ar=0.64 stress=0.48 panic=0.20 cur=0.72
m13: rel=0.61 eps=20
m4: token=obj_003 gate=0.77 stab=0.82 nov=0.14
m2: gate=1.00 should=1 pressure=0.69 sal=0.74 src=event
m5_seed: gate=0.08 norm=11.4 fb=0.03
m3_sleep_block=1
```

## Motor guard

In full sleep mode:

```text
video/contact/imu disabled -> is_full_sleep_mode() == True
```

M5 may still propose actions internally, but `SleepMotorGuardRuntimeMixin`
zeros external executable motor tensors:

```text
embodied_targets
hand_ctrl
leg_ctrl
```

The original proposed values are kept as:

```text
imagined_embodied_targets
imagined_hand_ctrl
imagined_leg_ctrl
imagined_action_ids
```

## Behavioral scenarios

Run:

```bash
python scripts/module_lab/scenario_unconscious_replay.py --json
```

It checks:

```text
calm_no_replay
curiosity_replay
bad_prediction_dream
object_identity_replay
```
