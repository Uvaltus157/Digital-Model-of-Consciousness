# Reference Traces DNA

Reference traces prove that regenerated code behaves like the intended organism.

## Trace format

Every trace should record:

```text
scenario_name
step range
active modules
IPC actions
status snapshots
tensor norms
verdict
```

## Required traces

### 1. Awake baseline

```text
sleep_mode = false
M1 sensors active
M3 not blocked
M8 status updates
```

Pass:

```text
no NaN/Inf
status payload complete
```

### 2. Sleep/replay guard

```text
sleep_mode = true
M1 sensors weakened/off
M3 motor blocked
M11/M13/M4/M2 active
```

Pass:

```text
M3 sleep_blocked = True
prev_embodied_action = 0
prev_hand_motor = 0
```

### 3. M1 cube/tetra slots

Action:

```text
M1 Object Slot Imit → Fill cube slot1 + tetra slot2
```

Pass:

```text
slot1 z_norm > 0
slot1 confidence > 0
slot2 z_norm > 0
slot2 confidence > 0
selected_slot = 2
Inner Object 3D nonempty or fallback visible
```

### 4. M5 latent prototype

Action:

```text
M5 Latent Prototypes → Inject cube
```

Pass:

```text
seed_norm > 0
seed_gate > 0
seed_response > 0
FocusFeedbackBoundary path used
```

### 5. Replay quality

Action:

```text
Sleep/replay mode ON
Dream/replay seed active
```

Pass:

```text
Replay Quality Monitor shows verdict
quality metrics not all zero
integration metrics meaningful
```

### 6. Conscious contour probe

Action:

```text
Inject fake M10 broadcast packet
Inject fake M7 inner speech
Inject fake M15 plan
```

Pass:

```text
M9 self grounding visible
M12 confidence visible
M3 guard respected
Subjective stream UI updates
```
