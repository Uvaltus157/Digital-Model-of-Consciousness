# Conscious Loop DNA

This document fixes the conscious contour of the architecture.

The project has two interacting contours:

```text
1. Unconscious contour:
   fast sensor/affect/replay loop, insect-like, no explicit subjective broadcast.

2. Conscious contour:
   selected content, global broadcast, inner speech, counterfactual imagination,
   deliberate planning, self-report, and controlled action proposal.
```

The conscious contour is not a replacement for the unconscious contour.
It sits above it and uses the same lower buses and memories.

---

## 1. Conscious contour overview

```text
M1 Object Imagery
↓
M5 World Model / Attention Workspace
↓
M10 Global Conscious Broadcast
↓
M9 Self Core / Agency / Body Ownership grounding
↓
M7 Inner Speech / Thought Stream
↓
M15 Counterfactual Imagination / Planning
↓
M12 Metacognitive confidence / uncertainty check
↓
M3 Action Proposal / Action Guard
↓
M8 monitors / subjective stream UI
```

Context providers / side inputs:

```text
M14 Semantic Grounding
M13 Autobiographical Memory
M4 Long Dynamic Memory
M11 Motivational Homeostasis
```

M9 and M12 are not merely optional side context:

```text
M9 grounds conscious content in “I / body / agency”.
M12 checks confidence, uncertainty, doubt, and need_to_check before action.
```

Full conscious loop:

```text
M1 + M5
↓
M10 selects conscious content
↓
M9 grounds selected content in self/body/agency
↓
M7 converts grounded content into inner speech / thought stream
↓
M15 generates imagined futures and candidate plans
↓
M12 evaluates uncertainty/confidence/doubt
↓
M14 grounds symbols/concepts
↓
M13 provides autobiographical context
↓
M3 receives controlled action proposal through ActionGuard
↓
M5 receives conscious seed through the same FocusFeedbackBoundary path
```

---

## 2. Conscious modules

### M10 — Global Conscious Broadcast

Owns:

```text
candidate collection
competition/selection
global broadcast packet
selected conscious content
```

Inputs:

```text
M5 workspace/focus_context
M1 object slots
M11 affect salience
M13 episodes
M4 identity/context
M12 uncertainty/confidence
M9 self-state
```

Outputs:

```text
conscious_content
broadcast_packet
selected_slot
selected_episode
broadcast_salience
broadcast_confidence
```

Contract:

```text
M10 selects and broadcasts; it does not directly mutate memories or motor output.
```

---

### M7 — Inner Speech / Thought Stream

Owns:

```text
inner speech
subjective text stream
thought decoding
reportable content
```

Inputs:

```text
M10 broadcast_packet
M14 grounded concepts
M13 autobiographical context
M9 self-state
```

Outputs:

```text
inner_speech_text
thought_tokens
subjective_stream_packet
```

Contract:

```text
M7 verbalizes/structures selected content; it does not select motor actions directly.
```

---

### M15 — Counterfactual Imagination / Planning

Owns:

```text
future scenario generation
counterfactual rollout
outcome evaluation
candidate plan
```

Inputs:

```text
M10 broadcast_packet
M7 thought stream
M5 world model
M11 affect/value
M12 confidence/uncertainty
M13 memories
M9 self/body constraints
```

Outputs:

```text
imagined_rollouts
candidate_plan
expected_outcome
conscious_focus_seed
```

Contract:

```text
M15 can propose a conscious focus seed,
but it must use the common M5 seed bus and FocusFeedbackBoundary.
```

---

### M12 — Metacognition

Owns:

```text
confidence
uncertainty
doubt
need_to_check
error awareness
```

Inputs:

```text
M5 prediction/reconstruction error
M10 broadcast confidence
M15 plan uncertainty
M11 affect pressure
```

Outputs:

```text
confidence
uncertainty
doubt
need_to_check
metacognitive_status
```

Contract:

```text
M12 evaluates reliability; it does not replace the selected content.
```

---

### M9 — Self Core

Owns:

```text
agency
body ownership
self-location
egocentric frame
temporal self-state
```

Inputs:

```text
body state
M3 action/efference
M1/M5 perception
M11 affect
```

Outputs:

```text
self_state
agency_score
body_ownership
egocentric_context
```

Contract:

```text
M9 grounds conscious content in self/body context.
```

---

### M14 — Semantic Grounding

Owns:

```text
word ↔ object/action/sensation grounding
concept memory
semantic labels
```

Inputs:

```text
M1 object slots
M7 words/thoughts
M5 latent context
M13 memory summaries
```

Outputs:

```text
grounded_concepts
semantic_labels
word_object_links
```

Contract:

```text
M14 grounds symbols; it does not decide replay or action.
```

---

## 3. Conscious vs unconscious seed rule

Both M2 and M15 can provide seeds to M5, but both must use the same boundary:

```text
M2 unconscious replay seed
        \
         → common M5 seed bus → FocusFeedbackBoundary → M5 workspace
        /
M15 conscious plan/imagery seed
```

Never:

```text
M15 directly mutates out["focus_context"]
M2 directly mutates out["focus_context"]
```

Priority policy must be explicit:

```text
debug/imit seed, if intentionally active
conscious seed, if deliberate focus is active
unconscious replay seed, during sleep/replay or dream pressure
fallback: no seed
```

If the project uses a different priority order, it must be documented in the M5 seed bus docs and tests.

---

## 4. Conscious awake flow

```text
M1 sees object / M5 builds world state
↓
M10 chooses reportable content
↓
M7 verbalizes: “I see / I remember / I plan ...”
↓
M15 imagines possible actions
↓
M12 estimates confidence/risk
↓
M3 receives action candidate
↓
M3 executes only if not blocked by guard/sleep/safety
```

---

## 5. Conscious sleep relationship

During sleep/replay:

```text
unconscious contour remains primary
M3 motor output remains blocked
M10/M7/M15 may be reduced, offline, or used only for dream report/debug
M2/M11/M13/M4/M5 replay loop remains active
```

Possible dream-report mode:

```text
M2/M13 replay material
↓
M5 dream workspace
↓
M10 selects dream content
↓
M7 produces dream report text
↓
M3 remains blocked
```

Contract:

```text
dream report is allowed
dream motor output is blocked
```

---

## 6. Conscious monitoring

M8 should eventually show:

```text
Conscious Broadcast Monitor:
    candidates
    selected content
    broadcast confidence
    source modules

Inner Speech Monitor:
    thought tokens
    inner speech text
    subjective stream

Counterfactual Planning Monitor:
    imagined rollouts
    candidate plan
    predicted outcome
    risk/value

Metacognition Monitor:
    confidence
    uncertainty
    doubt
    need_to_check

Self Core Monitor:
    agency
    ownership
    egocentric frame
```

---

## 7. Conscious loop validation levels

### Level 1

```text
M10/M7/M15/M12/M9/M14 status payloads exist.
IPC/monitors can see them.
Contracts pass.
```

### Level 2

```text
imit conscious content:
    fake broadcast packet
    fake inner speech
    fake imagined rollout
    fake confidence/risk
```

### Level 3

```text
downstream behaves as if conscious modules were trained:
    selected content appears in subjective stream
    M15 plan produces valid action candidate
    M12 confidence changes behavior/monitor
```

### Level 4

```text
replace conscious imitators with trained M10/M7/M15/M12/M9/M14 modules.
```

### Level 5

```text
compare trained conscious latents/content with prototype/imit conscious payloads.
```

---

## 8. Conscious contour pass criteria

PASS if:

```text
M10 selects a single broadcast content
M7 can verbalize it
M15 can create a candidate plan
M12 can report confidence/uncertainty
M9 grounds content in self/body
M14 grounds words/concepts
M15 uses common M5 seed bus
M3 guard still controls outward action
M8 monitors show the whole chain
```

FAIL if:

```text
conscious seed bypasses FocusFeedbackBoundary
M7 directly controls M3
M10 mutates memories directly
M15 writes M5 focus_context directly
M3 acts during sleep
M12 uncertainty is ignored by monitors/status
conscious and unconscious sources are not labeled
```
