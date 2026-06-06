# Monitors and Windows DNA

M8 is the observability layer.

## Required windows

```text
Module Status Table
Sleep Replay Monitor
Replay Quality Monitor
M5 Learning Quality
M5 Latent Prototypes
M1 Object Slot Imit
Inner Object 3D
Conscious Broadcast Monitor
Inner Speech Monitor
Counterfactual Planning Monitor
Metacognition Monitor
Self Core Monitor
Cameras/Depth
Actions
Sensors
```

## Inner Object 3D

Shows internal object representation, not external world.

```text
z_obj_slots[selected_slot]
confidence_slots[selected_slot]
decoded object / point cloud / debug fallback primitive
```

Difference:

```text
Camera window:
    raw sensory input

World/Open3D/MuJoCo:
    external scene

Inner Object 3D:
    internal object slot memory
```

## Replay Quality

```text
dream_pressure_delta
relief_delta
coherence_delta
stress_delta
selected_episode_summary
selected_identity_token
quality_score
verdict
```

## M5 Learning Quality

```text
train_loss
prediction_error
reconstruction_error
latent_coherence
seed_response
identity/object proxies
```

## Conscious Broadcast Monitor

Shows:

```text
candidate list
selected conscious content
selected slot/episode
broadcast confidence
source modules
```

## Inner Speech Monitor

Shows:

```text
inner_speech_text
thought_tokens
subjective_stream_packet
```

## Counterfactual Planning Monitor

Shows:

```text
imagined_rollouts
candidate_plan
expected_outcome
risk/value
conscious_focus_seed
```

## Metacognition Monitor

Shows:

```text
confidence
uncertainty
doubt
need_to_check
```

## Rule

Every new runtime/probe/imit must expose:

```text
status payload
M8 visible path
contract test
live-smoke check
```
