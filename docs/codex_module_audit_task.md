# Codex task: DMoC module/runtime audit

## Goal

Проверить, что текущий код DMoC соответствует принятому алгоритму M1–M15, что все новые модули подключены в runtime, видны в stats/checkpoint/tests, и что из активного runtime убраны прямые versioned-имена вроде `UnifiedSystemV510` / `UnifiedSystemV57`, кроме явно разрешённых legacy alias.

Основной алгоритм должен быть таким:

```text
M1 inner_object
→ M4 long dynamic identity
→ M11 affect_latents
→ M4 identity-context blend
→ M13 autobiographical retrieval
→ M2 event/dream replay
→ M15 thought-chain search
→ M10 global conscious broadcast
→ M9 self-binding
→ M7 inner speech
→ M12 metacognition
→ M14 semantic action grounding
→ M13 write episode
```

Главное правило: исправлять только реальные несоответствия. Не менять архитектурную идею и не возвращать старые V2/V21/V22/V23 naming paths.

---

## 1. Проверить активный runtime assembly

Проверить файлы:

```text
src/apps/runner.py
src/apps/runner_entry.py
src/apps/runner_system_factory.py
src/apps/unified_runtime_base.py
src/apps/unified_conscious_viewer.py
```

Требования:

1. Основной публичный runtime-класс должен называться:

```python
UnifiedSystem
```

2. `UnifiedSystemV510` допускается только как legacy alias в `src/apps/runner.py`:

```python
UnifiedSystemV510 = UnifiedSystem
```

3. Активный `runner.py` не должен напрямую наследоваться от `UnifiedSystemV57`. Он должен наследоваться от versionless alias:

```python
from src.apps.unified_runtime_base import UnifiedRuntimeBase
```

и далее:

```python
class UnifiedSystem(..., UnifiedRuntimeBase):
    ...
```

4. `UnifiedSystemV57` допускается только внутри legacy-слоя:

```text
src/apps/unified_conscious_viewer.py
src/apps/unified_runtime_base.py
```

5. Новый код должен импортировать `UnifiedSystem`, а не `UnifiedSystemV510`.

6. Проверить, что `runner_entry.py` использует:

```python
from src.apps.runner import UnifiedSystem
system = build_unified_system(cfg_obj, UnifiedSystem)
```

---

## 2. Проверить mixin-подключения в UnifiedSystem

В `src/apps/runner.py` класс `UnifiedSystem` должен включать runtime mixins:

```text
M1  ObjectImageryRuntimeMixin
M2  EventDreamReplayRuntimeMixin
M3  ActionRuntimeMixin / ActionOutputsMixin
M4  DynamicObjectPassportRuntimeMixin + LongDynamicMemoryRuntimeMixin
M6  SleepSensorsMixin + TrainingRuntimeMixin
M7  InnerSpeechRuntimeMixin
M8  ModuleStatusRuntimeMixin
M9  SelfCoreRuntimeMixin
M10 GlobalBroadcastRuntimeMixin
M12 MetacognitionRuntimeMixin
M13 AutobiographicalMemoryRuntimeMixin
M14 SemanticActionRuntimeMixin
M15 ThoughtChainRuntimeMixin
```

Также должны быть сохранены platform/runtime mixins:

```text
CameraPreviewMixin
ExternalControlMixin
IPCRuntimeMixin
CheckpointingMixin
LegBirdRuntimeMixin
InnerVisualRuntimeMixin
TetraDynamicSlotDiagnosticMixin
LifeRuntimeMixin
UnifiedRuntimeBase
```

Если какого-то mixin нет — добавить.

---

## 3. Проверить порядок вызовов в life_runtime.py

Файл:

```text
src/apps/life_runtime.py
```

Должно быть:

```python
out["inner_object"] = self.compute_inner_object_image(obs, out)
self._compute_long_dynamic_memory(obs, out)
out["self_core"] = self.compute_self_core(obs, out)
self._apply_conscious_action_guard(obs, out)
...
emotion = self.emotional_drive.compute(out, obs)
...
self._write_autobiographical_episode(obs, out)
```

Проверить, что:

- M4 вызывается после M1 и до `compute_self_core`.
- M14 вызывается после `compute_self_core`, потому что M12 создаётся внутри self-core branch после M7.
- M13 write выполняется после M14/emotion, чтобы episode видел conscious action и affect.
- `emotional_drive.compute(...)` использует cache/reuse, а не создаёт конфликтующий второй affect packet.

---

## 4. Проверить порядок pre-self ветки в self_core_runtime.py

Файл:

```text
src/modules/m09_self_core/self_core_runtime.py
```

В `compute_self_core(...)` порядок должен быть:

```python
self.ensure_self_core_ready()
self._ensure_affect_packet_for_self_core(obs, out)
self._run_pre_self_long_dynamic_memory(obs, out)
self._run_pre_self_autobiographical_retrieval(obs, out)
self._run_pre_self_event_dream_replay(obs, out)
self._run_pre_self_thought_chain(obs, out)
self._run_pre_self_global_broadcast(obs, out)
# then M9 self_core(...)
# then M7 inner speech
# then M12 metacognition
```

Проверить, что:

- M11 affect создаётся до M4/M13/M2/M15/M10/M9.
- M4 blend идёт до M13/M2/M15.
- M13 retrieval идёт до M2/M15.
- M2 replay идёт до M15.
- M15 идёт до M10.
- M10 идёт до M9.
- M7 идёт после M9.
- M12 идёт после M7.

---

## 5. Проверить contracts out[...]

Каждый модуль должен создавать ожидаемые ключи.

### M1

```python
out["inner_object"]
```

### M2

```python
out["event_dream_replay"] = {
    "replay_context",
    "replay_gate",
    "event_salience",
    "dream_pressure",
    "should_replay",
    "selected_event_sentence",
    "selected_event_kind",
    "selected_event_slot_token",
    "selected_episode_summary",
    "replay_source",
}
```

### M4

```python
out["long_dynamic_memory"] = {
    "dynamic_identity_context",
    "dynamic_memory_gate",
    "should_bind_identity",
    "identity_token",
    "identity_stability",
    "identity_similarity",
    "identity_novelty",
    "identity_dynamic_score",
    "passport_count",
    "passport_slot",
    "passport_created",
    "selected_sentence",
    "episode_summary",
    "identity_source",
    "replay_z",
}
```

### M5

```python
out["focus_context"]
out["workspace_out"]
out["values"]
out["focus"]
```

### M7

```python
out["inner_speech"]
```

Runtime fallback may still tolerate `symbolic_report`, but new code should prefer `inner_speech` / `decoded_report`.

### M9

```python
out["self_core"]
```

### M10

```python
out["broadcast"]
```

### M12

```python
out["metacognition"]
```

### M13

```python
out["autobiographical_memory"]
```

### M14

```python
out["conscious_action"]
out["semantic_action"]
```

### M15

```python
out["thought_chain"]
```

---

## 6. Проверить config defaults

Проверить:

```text
src/shared/config.py
config/*.yaml
```

В config должны быть безопасные default-секции для:

```text
event_dream_replay
long_dynamic_memory
autobiographical_memory
semantic_action
global_broadcast
metacognition
thought_chain
self_core
```

Особенно проверить поля:

### event_dream_replay

```text
enabled
replay_context_dim
event_code_dim
replay_threshold
focus_blend
blend_replay_into_focus
use_m13_context
use_event_memory
max_recent_events_scan
print_every_steps
```

### long_dynamic_memory

```text
enabled
context_dim
focus_blend
blend_into_focus
stability_threshold
novelty_threshold
use_passport_manager
use_event_memory
print_every_steps
```

### autobiographical_memory

```text
enabled
memory_dim
max_episodes
retrieval_topk
write_every_steps
blend_retrieved_into_focus
focus_blend
min_relevance_for_blend
print_every_steps
```

### semantic_action

```text
enabled
hold_threshold
verify_threshold
high_doubt_threshold
min_action_scale
soft_hold_scale
explore_threshold
positive_delta_threshold
emergency_threshold
print_every_steps
```

Если runtime использует `getattr(self.cfg, "...")`, но секции нет — добавить dataclass/config default.

---

## 7. Проверить stats

Файлы:

```text
src/apps/life_stats_runtime.py
src/apps/life_stats_builders.py
```

Требования:

1. `life_stats_runtime.py` должен быть коротким runtime-shell.
2. Большой stats dictionary должен быть вынесен в `life_stats_builders.py`.
3. В `latest_stats` должны быть поля M2/M4/M10/M12/M13/M14/M15.

### M2 stats

```text
m2_replay_gate
m2_should_replay
m2_event_salience
m2_dream_pressure
m2_blended_into_focus
m2_replay_source
m2_selected_event_kind
m2_selected_event_sentence
```

### M4 stats

```text
m4_identity_token
m4_dynamic_memory_gate
m4_identity_stability
m4_identity_similarity
m4_identity_novelty
m4_identity_dynamic_score
m4_should_bind_identity
m4_passport_count
m4_passport_slot
m4_blended_into_focus
m4_selected_sentence
m4_episode_summary
```

### M10 stats

```text
m10_selected_source
m10_priority
m10_gate
```

### M12 stats

```text
m12_confidence
m12_doubt
m12_verify
m12_hold
```

### M13 stats

```text
m13_episode_count
m13_retrieval_relevance
m13_blended_into_focus
m13_last_summary
```

### M14 stats

```text
m14_semantic_intent
m14_target_source
m14_goal_text
m14_grounding_confidence
m14_expected_outcome
m14_action_scale
m14_reason
```

### M15 stats

```text
m15_best_chain_score
m15_predicted_affect_delta
```

Also verify side-effect traces include:

```python
maybe_print_long_dynamic_memory_trace(...)
maybe_print_event_dream_replay_trace(...)
maybe_print_thought_chain_trace(...)
maybe_print_global_broadcast_trace(...)
maybe_print_metacognition_trace(...)
maybe_print_autobiographical_memory_trace(...)
maybe_print_semantic_action_trace(...)
```

---

## 8. Проверить checkpoint

Файл:

```text
src/shared/checkpointing.py
```

Checkpoint должен сохранять и загружать:

```text
model
optimizer
global_step
train_steps
quality
inner_object_system
inner_object_state
inner_object_slot_snapshots
event_latent_memory
dynamic_object_passports
autobiographical_memory
self_core
self_core_state
leg_control_head
runtime_state
prev_embodied_action
prev_hand_motor
prev_leg_motor
```

Проверить:

- `autobiographical_memory.state_dict()` / `load_state_dict(...)` работает.
- `dynamic_object_passports.state_dict()` / `load_state_dict(...)` работает.
- M13 episodes не теряются между запусками.
- M4 passports не теряются между запусками.
- checkpoint save remains atomic via temp file + `os.replace`.

---

## 9. Проверить tests

Обязательно запустить:

```bash
python -m pytest tests/test_conscious_contracts.py -q
python -m pytest tests/test_m02_event_dream_replay.py -q
python -m pytest tests/test_m04_long_dynamic_memory.py -q
```

Если эти проходят, запустить:

```bash
python -m pytest tests -q
```

Если тесты падают из-за:

```text
missing config section
missing out[...] key
shape mismatch
import error
legacy UnifiedSystemV510 / UnifiedSystemV57 import
symbolic_report-only path
checkpoint state mismatch
```

исправить минимально.

---

## 10. Добавить недостающие tests

Если нет отдельного теста на runtime class naming, добавить тест:

```python
def test_unified_system_is_primary_runtime_class():
    from src.apps.runner import UnifiedSystem, UnifiedSystemV510
    assert UnifiedSystemV510 is UnifiedSystem
```

Если можно импортировать без MuJoCo side effects, добавить проверку методов:

```python
def test_unified_system_has_current_runtime_methods():
    from src.apps.runner import UnifiedSystem
    required = [
        "compute_long_dynamic_memory",
        "compute_event_dream_replay",
        "compute_autobiographical_retrieval",
        "compute_thought_chain",
        "compute_global_broadcast",
        "compute_inner_speech",
        "compute_metacognition",
        "compute_conscious_action",
        "write_autobiographical_episode",
    ]
    for name in required:
        assert hasattr(UnifiedSystem, name), name
```

Если import too heavy, use source-level AST/text test instead.

Also add or update tests proving M4 + M13 + M2 + M15 can coexist with compatible focus_context dimensions.

---

## 11. Не делать

Do not:

```text
- не переименовывать архитектурные модули M1–M15;
- не возвращать старые версии V2/V21/V22/V23;
- не делать M10 motivation;
- не делать M11 broadcast;
- не удалять manual override;
- не заменять low-level action heads на M14 полностью;
- не ломать fallback compatibility с symbolic_report, пока визуализаторы ещё могут читать старые поля;
- не удалять legacy aliases, если это ломает старые imports;
- не переписывать всё крупными изменениями, если достаточно минимальной правки.
```

---

## 12. Ожидаемый итоговый отчёт Codex

В конце работы выдать отчёт:

```text
1. Найденные несоответствия
2. Исправленные файлы
3. Добавленные/изменённые tests
4. Команды pytest, которые были запущены
5. Что прошло
6. Что не прошло и почему
7. Остаточные legacy aliases, если они оставлены намеренно
```

Критерий готовности:

```text
Активная runtime-сборка должна использовать UnifiedSystem без версионного имени.
Runtime-порядок должен соответствовать:
M1 → M4 → M11 → M4 blend → M13 → M2 → M15 → M10 → M9 → M7 → M12 → M14 → M13 write.
Все новые M2/M4/M13/M14 модули должны быть видны в stats/checkpoint/tests.
```
