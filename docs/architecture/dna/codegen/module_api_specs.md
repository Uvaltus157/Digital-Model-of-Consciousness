# Module API Specs DNA

This document defines public runtime APIs that a regenerated implementation must expose.

The exact internal model may change, but public contracts should remain stable.

---

# Common runtime conventions

Every module runtime mixin should follow this pattern:

```python
class MxxRuntimeMixin:
    def build_mxx_status(self) -> dict: ...
    def reset_mxx_state(self) -> None: ...
```

If the module has an imitator:

```python
class MxxSomethingImitRuntimeMixin:
    def request_mxx_imit(self, payload: dict) -> dict: ...
    def mxx_imit_status(self) -> dict: ...
```

Status payloads must be JSON-safe except optional tensors hidden from IPC.

---

# M1 — Object Imagery

## Required files

```text
src/modules/m01_object_imagery/runtime.py
src/modules/m01_object_imagery/imit/
src/modules/m01_object_imagery/README.md
```

## Required methods

```python
class ObjectImageryRuntimeMixin:
    def build_inner_object_vision_proposals(self, obs: dict):
        ...

    def _run_progressive_inner_object_system(self, obs: dict, out: dict) -> dict:
        ...

    def _memory_update_forced_slot(
        self,
        state: dict,
        z_update,
        vision_strength,
        touch_strength,
        force_slot_index: int,
    ) -> dict:
        ...
```

## Required outputs

```text
z_obj_slots: [B, num_slots, object_latent_dim]
confidence_slots: [B, num_slots, 1]
active_slot_index: int or tensor
selected_slot: int
inner_object_debug: dict
```

## M1 imit API

```python
class M1ObjectSlotLatentImitRuntimeMixin:
    def make_m1_object_slot_latent(self, kind: str, *, alpha: float = 0.5, device=None, dim=None):
        ...

    def request_m1_object_slot_latents(self, payload: dict | None = None) -> dict:
        ...

    def get_m1_imit_inner_object_proposals(self, scene=None):
        ...

    def m1_object_slot_imit_status(self) -> dict:
        ...
```

## M1 imit contract

```text
cube → slot 1
tetrahedron → slot 2
morph → slot 3
proposal_kind = m1_imit_dynamic_object
target_slots preserved
direct z_update bypass used
```

---

# M2 — Event Dream Replay

## Required methods

```python
class EventDreamReplayRuntimeMixin:
    def get_event_dream_focus_seed(self, stage: str = "model_step"):
        ...

    def get_m5_focus_seed(self, stage: str = "model_step"):
        ...

    def build_event_dream_replay_status(self) -> dict:
        ...
```

## Required outputs

```text
next_focus_context_seed
next_focus_context_seed_gate
replay_gate
dream_pressure
selected_episode_id
selected_identity_token
event_salience
```

## Contract

```text
M2 must never mutate M5 focus_context directly.
M2 must feed M5 only through common M5 seed bus.
```

---

# M3 — Self Action Causality

## Required methods

```python
class ActionRuntimeMixin:
    def apply_ipc_action(self, message: dict) -> None:
        ...

    def apply_action_guard(self, action: dict, *, sleep_mode: bool) -> dict:
        ...

    def build_action_status(self) -> dict:
        ...
```

## Required action guard outputs

```text
motor_blocked
sleep_blocked
prev_embodied_action_zeroed
prev_hand_motor_zeroed
outward_action
blocked_reason
```

## Contract

```text
Sleep/replay mode blocks outward motor output.
Debug/imit probes must not bypass ActionGuard.
```

---

# M4 — Long Dynamic Memory

## Required outputs

```text
identity_token
identity_stability
identity_novelty
object_passport
long_memory_match
```

## Contract

```text
M4 provides object identity/context to M2/M10/M13.
It does not select replay by itself.
```

---

# M5 — World Model / Attention Workspace

## Required methods

```python
class WorldModelWorkspaceRuntimeMixin:
    def get_m5_focus_seed(self, stage: str = "model_step"):
        ...

    def apply_focus_feedback_boundary(self, workspace, seed, gate):
        ...

    def build_m5_learning_quality_status(self) -> dict:
        ...
```

## Required outputs

```text
focus_context
workspace_norm
seed_response
prediction_error
reconstruction_error
latent_coherence
```

## M5 imit API

```python
class M5LatentPrototypeRuntimeMixin:
    def make_m5_latent_prototype(self, kind: str = "cube", *, device=None, dim=None):
        ...

    def request_m5_latent_prototype(self, payload: dict | None = None) -> dict:
        ...

    def get_m5_latent_prototype_focus_seed(self, stage: str = "model_step"):
        ...

    def m5_latent_prototype_status(self) -> dict:
        ...
```

## Contract

```text
All M5 seeds pass through FocusFeedbackBoundary.
No direct out["focus_context"] mutation by imit/probe.
```

---

# M6 — Learning / Sleep Consolidation

## Required outputs

```text
training_enabled
cfg_train_enabled
train_steps
last_train_loss
last_train_reason
last_train_error
sleep_mode
sensor_mask
```

## Contract

```text
Training must obey sleep/train gates.
Sleep replay can run without outward motor action.
```

---

# M7 — Inner Speech / Thought Stream

## Required methods

```python
class InnerSpeechRuntimeMixin:
    def build_inner_speech(self, broadcast_packet: dict) -> dict:
        ...

    def build_subjective_stream_status(self) -> dict:
        ...
```

## Required outputs

```text
inner_speech_text
thought_tokens
subjective_stream_packet
```

## Contract

```text
M7 verbalizes selected conscious content.
M7 must not directly control M3.
```

---

# M8 — Debug Visual Control

## Required files

```text
src/modules/m08_debug_visual_control/control_panel.py
src/modules/m08_debug_visual_control/module_status_runtime.py
src/modules/m08_debug_visual_control/module_debug_status_ipc.py
```

## Required behavior

```text
read status payloads
send explicit IPC actions
open monitors/windows
never mutate model state except explicit action messages
```

---

# M9 — Self Core

## Required outputs

```text
self_state
agency_score
body_ownership
egocentric_context
temporal_self_state
```

## Contract

```text
M9 grounds conscious content in I/body/agency before inner speech.
```

---

# M10 — Global Conscious Broadcast

## Required methods

```python
class GlobalConsciousBroadcastRuntimeMixin:
    def collect_conscious_candidates(self, out: dict) -> list[dict]:
        ...

    def select_conscious_content(self, candidates: list[dict]) -> dict:
        ...

    def build_broadcast_packet(self, selected: dict) -> dict:
        ...

    def build_m10_status(self) -> dict:
        ...
```

## Required outputs

```text
conscious_content
broadcast_packet
selected_slot
selected_episode
broadcast_confidence
broadcast_salience
```

## Contract

```text
M10 selects and broadcasts.
M10 must not mutate memory or motor output directly.
```

---

# M11 — Motivational Homeostasis

## Required outputs

```text
stress
panic
curiosity
relief
fatigue
mastery
valence
coherence
dream_pressure
```

## Contract

```text
M11 remains active in sleep/replay.
M11 influences M2 and M10 through affect/salience.
```

---

# M12 — Metacognition

## Required outputs

```text
confidence
uncertainty
doubt
need_to_check
metacognitive_verdict
```

## Contract

```text
M12 checks reliability before deliberate action/planning.
```

---

# M13 — Autobiographical Memory

## Required outputs

```text
selected_episode
episode_summary
episode_embedding
episode_emotional_tags
episode_identity_link
```

## Contract

```text
M13 provides episode material to M2 and context to M10/M15.
```

---

# M14 — Semantic Grounding

## Required outputs

```text
grounded_concepts
semantic_labels
word_object_links
action_word_links
```

## Contract

```text
M14 grounds words/concepts; it does not select action.
```

---

# M15 — Counterfactual Imagination / Planning

## Required outputs

```text
imagined_rollouts
candidate_plan
expected_outcome
risk_value_score
conscious_focus_seed
conscious_focus_seed_gate
```

## Contract

```text
M15 can propose M5 focus seed.
It must use common M5 seed bus and FocusFeedbackBoundary.
It must not write focus_context directly.
```
