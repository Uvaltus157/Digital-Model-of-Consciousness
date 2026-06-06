# Module Contracts DNA

Every module must have a stable contract:

```text
input
output
status payload
monitor visibility
tests
live-smoke path
```

## Standard contract checklist

```text
import works
runtime method exists
payload keys exist
tensor shapes match config
dtype/device safe
no NaN/Inf
source label correct
status publishes
M8 UI receives status
downstream reaction visible
```

## Key contracts

### M1 Object Imagery

```text
Input:
    sensor summaries OR m1_object_slot_imit proposals

Output:
    vision_proposals
    target_slots
    proposal_kinds
    target_names
    z_obj_slots
    confidence_slots
    selected slot

Rule:
    imit z_obj latents must not be forced through raw sensory fusion.
    M1 imit z_obj latents must not be forced through raw sensory fusion.
```

### M2 Event Dream Replay

```text
Input:
    M11 affect
    M13 episodes
    M4 identity/context
    M5 current workspace

Output:
    replay_gate
    dream_pressure
    event_salience
    next_focus_context_seed
    next_focus_context_seed_gate

Rule:
    M2 must not write focus_context directly.
```

### M3 Action Guard

```text
Input:
    action proposal
    sleep mode / motor guard

Output:
    action or blocked action

Rule:
    in sleep/replay mode outward motor output must remain blocked.
    M3 blocks outward motor output in sleep/replay.
```

### M5 World Model

```text
Input:
    focus_context_seed
    focus_context_seed_gate
    observation/workspace state

Output:
    focus_context
    workspace payload
    seed_response
    prediction/reconstruction metrics

Rule:
    seeds pass through FocusFeedbackBoundary.
    M5 seeds pass through FocusFeedbackBoundary.
```

### M7 Inner Speech

```text
Input:
    M10 broadcast_packet
    M14 grounded concepts
    M13 memory context
    M9 self-state

Output:
    inner_speech_text
    thought_tokens
    subjective_stream_packet

Rule:
    M7 verbalizes selected content; it does not drive M3 directly.
    M7 verbalizes, does not drive M3 directly.
```

### M10 Global Conscious Broadcast

```text
Input:
    M5 workspace/focus_context
    M1 object slots
    M11 affect salience
    M13 episodes
    M4 identity/context
    M12 confidence
    M9 self-state

Output:
    conscious_content
    broadcast_packet
    selected_slot
    selected_episode
    broadcast_confidence

Rule:
    M10 selects and broadcasts; it does not mutate memory or motor output directly.
    M10 selects/broadcasts, does not mutate memory/motor directly.
```

### M12 Metacognition

```text
Input:
    M5 errors
    M10 broadcast confidence
    M15 plan uncertainty
    M11 affect pressure

Output:
    confidence
    uncertainty
    doubt
    need_to_check

Rule:
    M12 evaluates reliability; it does not replace selected content.
```

### M15 Counterfactual Planning

```text
Input:
    M10 broadcast
    M7 thought stream
    M5 world model
    M11 affect/value
    M12 uncertainty
    M13 memory
    M9 self/body constraints

Output:
    imagined_rollouts
    candidate_plan
    expected_outcome
    conscious_focus_seed

Rule:
    M15 uses the common M5 seed bus and FocusFeedbackBoundary.
    M15 uses common M5 seed bus and does not write focus_context directly.
```

### M9 Self Core

```text
Input:
    M5 focus_context
    body/self state
    ownership/agency signals

Output:
    self_state
    agency_grounding
    body_ownership_grounding

Rule:
    M9 grounds conscious content in I/body/agency.
```

### M8 Debug Visual Control

```text
Input:
    status payloads
    IPC responses

Output:
    windows, monitors, control buttons

Rule:
    read-only except explicit action messages.
```
