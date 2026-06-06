# Codex task: проверить бессознательный контур / функциональную схему насекомого

## Контекст

В проект DMoC уже встроен бессознательный replay/sleep-контур, который должен соответствовать функциональной схеме насекомого:

```text
M1 → M5 → M11 → M2 → M5 → M11
          ↑
          │
        M4/M13 → M2
```

Более точно:

```text
M1 sensors
    ↓
M5 world model / inner playback
    ↓
M11 affect / valence / arousal / stress / curiosity
    ↓
M2 event_dream_replay selector
    ↑               ↑
    │               │
M13 episodic memory M4 object/agent identity
    ↓
M2 replay_context
    ↓
M5 FocusFeedbackBoundary
    ↓
M5 workspace_seed + preconscious_seed
    ↓
M11 оценивает результат проигрывания
```

Нужно проверить **только бессознательный контур**, без анализа сознательного M9/M10/M15-контура, кроме одной проверки: M2 должен использовать тот же вход M5, куда M15 кладёт conscious seed.

## Важные архитектурные правила

### Правильно

```text
M1 → M5
M5 → M11
M11 → M2
M13 → M2
M4 → M2
M2 replay_context → M5 FocusFeedbackBoundary
M5 → M11
```

### Неправильно

```text
M1 → M2 напрямую
M2 напрямую заменяет out["focus_context"] как основной путь
M11 напрямую строит сцену в M5
M2 идёт в отдельный dream-вход M5, отличный от FocusFeedbackBoundary
```

M2 должен класть `replay_context` туда же, куда M15 кладёт `enhanced_focus_context` в сознательном контуре:

```text
focus_context_seed
focus_context_seed_gate
→ M5 FocusFeedbackBoundary
→ workspace_seed + preconscious_seed
```

---

# Что нужно проверить

Ничего не применять заново. Никаких веток создавать не нужно. Проверить текущий код и, если есть реальные ошибки, внести минимальные правки.

## 1. Проверить M5-вход FocusFeedbackBoundary

Файл:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
```

Проверь, что `step(...)` принимает:

```python
focus_context_seed=None
focus_context_seed_gate=None
```

Проверь, что seed входит в M5 через:

```python
focus_feedback = self.focus_feedback_boundary(
    workspace_seed=attn["workspace_seed"],
    focus_context_seed=focus_context_seed,
    focus_context_seed_gate=focus_context_seed_gate,
)
```

Проверь, что результат влияет на:

```python
attn["workspace_seed"] = focus_feedback["workspace_seed"]
```

и позже на:

```python
preconscious_seed = self.focus_feedback_boundary.apply_preconscious_seed(...)
```

То есть M5 должен иметь общий вход для внутренних seed-сигналов.

## 2. Проверить сам FocusFeedbackBoundary

Файл:

```text
src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
```

Проверь:

```python
class FocusFeedbackBoundary(nn.Module)
```

Он должен принимать:

```python
workspace_seed
focus_context_seed
focus_context_seed_gate
```

и возвращать:

```python
workspace_seed
preconscious_delta
external_gate
learned_gate
total_gate
seed_norm
active
```

Метод:

```python
apply_preconscious_seed(...)
```

должен подмешивать `preconscious_delta` через `total_gate`.

## 3. Проверить M2 EventDreamReplay

Файлы:

```text
src/modules/m02_event_dream_replay/event_dream_replay.py
src/modules/m02_event_dream_replay/event_dream_runtime.py
```

Проверь, что M2 принимает/использует:

```text
out["affect"] от M11
out["autobiographical_memory"] от M13
out["long_dynamic_memory"] от M4
out["focus_context"] от M5
event_latent_memory, если есть
```

Проверь, что в `EventDreamReplayConfig` есть параметры:

```python
blend_replay_into_focus: bool = False
use_m13_context: bool = True
use_m4_context: bool = True
m4_context_weight: float = 0.20
seed_to_m5_boundary: bool = True
seed_gate_gain: float = 1.0
apply_stage: str = "pre_observe"
seed_only_in_sleep: bool = True
```

Проверь, что `replay_context` собирается примерно так:

```python
replay_context =
    focus_weight * focus_context
  + event_weight * event_vec
  + m13_weight * m13_context
  + m4_weight * m4_context
```

M4 context должен быть:

```python
out["long_dynamic_memory"]["dynamic_identity_context"]
```

и M2 должен учитывать также:

```python
identity_stability
identity_novelty
dynamic_memory_gate
identity_token
selected_sentence
```

## 4. Проверить, что M2 не мутирует focus_context как основной путь

В `event_dream_runtime.py` проверь, что прямое:

```python
out["focus_context"] = focus + ...
```

либо отсутствует, либо защищено legacy-условием:

```python
bool(cfg.blend_replay_into_focus)
and not bool(getattr(cfg, "seed_to_m5_boundary", True))
```

То есть по умолчанию должно быть:

```text
M2 replay_context → next_focus_context_seed → M5 FocusFeedbackBoundary
```

а не:

```text
M2 replay_context → out["focus_context"]
```

## 5. Проверить seed bus M2 → M5

Файл:

```text
src/modules/m02_event_dream_replay/event_dream_runtime.py
```

Проверь наличие методов:

```python
get_event_dream_focus_seed(...)
get_m5_focus_seed(...)
_store_event_dream_m5_seed(...)
```

`_store_event_dream_m5_seed(...)` должен брать:

```python
packet["replay_context"]
packet["replay_gate"] или packet["should_replay"]
packet["dream_pressure"]
```

и сохранять:

```python
self._event_dream_next_focus_seed
self._event_dream_next_focus_gate
packet["next_focus_context_seed"]
packet["next_focus_context_seed_gate"]
packet["target_m5_boundary"] = "FocusFeedbackBoundary(workspace_seed + preconscious_seed)"
packet["seed_source"] = "m02_event_dream_replay"
```

`get_m5_focus_seed(...)` должен использовать приоритет:

```text
1. M2 dream/replay seed в sleep mode
2. M15 conscious loop seed как fallback
```

То есть M2 и M15 должны входить в M5 через один общий seed-интерфейс.

## 6. Проверить model_step

Файл:

```text
src/apps/unified_conscious_viewer.py
```

Проверь, что `model_step(...)` принимает:

```python
model_stage: str = "main"
focus_context_seed=None
focus_context_seed_gate=None
```

Проверь, что seed запрашивается через общий метод:

```python
self.get_m5_focus_seed(stage=model_stage)
```

а не только через:

```python
self.get_conscious_loop_focus_seed(...)
```

Потом seed должен передаваться в:

```python
self.model.step(
    ...,
    focus_context_seed=focus_context_seed,
    focus_context_seed_gate=focus_context_seed_gate,
)
```

## 7. Проверить порядок в life_runtime

Файл:

```text
src/apps/life_runtime.py
```

Проверь, что M1 не идёт в M2 напрямую. Сенсоры должны идти только через observe → M5:

```python
obs0 = self.world.observe(...)
out0 = self.model_step(obs0, self.state, model_stage="pre_observe")
...
obs = self.world.observe(...)
out = self.model_step(obs, self.state, model_stage="main")
```

Проверь, что после M11 emotion:

```python
emotion = self.emotional_drive.compute(out, obs)
out["emotion"] = emotion
if isinstance(emotion.get("affect"), dict):
    out["affect"] = emotion["affect"]
```

выполняется бессознательный блок:

```python
self.compute_autobiographical_retrieval(obs, out)  # M13
self.compute_event_dream_replay(obs, out)          # M2
```

И желательно trace:

```python
self.maybe_print_event_dream_replay_trace(out)
self.maybe_print_autobiographical_memory_trace(out)
```

Важно: `compute_event_dream_replay(...)` должен вызываться **после** `out["affect"]`, потому что M11 управляет M2 через affect.

## 8. Проверить M4

Файл:

```text
src/modules/m04_long_dynamic_memory/long_dynamic_memory_runtime.py
```

Проверь, что M4 создаёт:

```python
out["long_dynamic_memory"]
```

с ключами:

```python
dynamic_identity_context
dynamic_memory_gate
identity_stability
identity_novelty
identity_token
selected_sentence
```

M2 должен читать этот пакет и использовать его при сборке `replay_context`.

## 9. Проверить M13

Файл:

```text
src/modules/m13_autobiographical_memory/autobiographical_memory_runtime.py
```

Проверь, что retrieval создаёт:

```python
out["autobiographical_memory"]
```

с ключами:

```python
retrieved_context
retrieval_relevance
summary
```

И что M2 читает эти данные.

## 10. Проверить M11

Файл:

```text
src/modules/m11_motivational_homeostasis/emotional_drive_bivalent.py
```

Проверь, что M11 строит:

```python
out["emotion"]["affect"]
```

и что там есть:

```python
affect_latents
valence
arousal
stress_latent
panic_latent
curiosity_latent
comfort_latent
relief_latent
expected_affect_delta
intrinsic_reward
```

M2 должен использовать affect как эмоциональное давление, а не напрямую получать M1.

---

# Запрещённые паттерны

Найти и убедиться, что их нет или они отключены по умолчанию:

```python
# M1 напрямую в M2
compute_event_dream_replay(obs, ...)
# где M2 читает raw obs как основной источник сцены

# M2 напрямую заменяет M5 focus как основной путь
out["focus_context"] = replay_context

# прямой bypass вокруг FocusFeedbackBoundary
self.model.step(..., dream_context=...)
self.model.step(..., replay_context=...)
```

Допустимо только legacy, если оно выключено по умолчанию:

```python
blend_replay_into_focus=False
seed_to_m5_boundary=True
```

---

# Compile checks

Запусти:

```bash
python -m py_compile src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
python -m py_compile src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_replay.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_runtime.py
python -m py_compile src/modules/m04_long_dynamic_memory/long_dynamic_memory_runtime.py
python -m py_compile src/modules/m13_autobiographical_memory/autobiographical_memory_runtime.py
python -m py_compile src/modules/m11_motivational_homeostasis/emotional_drive_bivalent.py
python -m py_compile src/apps/unified_conscious_viewer.py
python -m py_compile src/apps/life_runtime.py
python -m py_compile src/shared/config.py
```

---

# Import checks

Запусти:

```bash
python - <<'PY'
from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig
from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin
from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary
from src.apps.runner import UnifiedSystem
print("imports ok")
PY
```

---

# Smoke test M2 replay seed

Запусти минимальный тест M2 без MuJoCo:

```bash
python - <<'PY'
import torch
from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig

cfg = EventDreamReplayConfig(
    replay_context_dim=256,
    event_code_dim=8,
    blend_replay_into_focus=False,
    use_m13_context=True,
    use_m4_context=True,
    seed_to_m5_boundary=True,
)
m2 = EventDreamReplay(cfg)

out = {
    "focus_context": torch.randn(1, 256),
    "affect": {
        "panic_latent": torch.tensor([[0.3]]),
        "stress_latent": torch.tensor([[0.2]]),
        "curiosity_latent": torch.tensor([[0.7]]),
        "expected_affect_delta": torch.tensor([[0.1]]),
    },
    "autobiographical_memory": {
        "retrieved_context": torch.randn(1, 256),
        "retrieval_relevance": torch.tensor([[0.5]]),
        "summary": "test episode",
    },
    "long_dynamic_memory": {
        "dynamic_identity_context": torch.randn(1, 256),
        "dynamic_memory_gate": torch.tensor([[0.8]]),
        "identity_stability": torch.tensor([[0.6]]),
        "identity_novelty": torch.tensor([[0.2]]),
        "identity_token": "obj:test",
        "selected_sentence": "same object identity",
    },
}

packet = m2.compute(out=out, event_memory=None, dream_mode=True)
assert packet["replay_context"].shape == (1, 256)
assert packet["replay_gate"].shape == (1, 1)
assert "selected_identity_token" in packet
assert "dynamic_memory_gate" in packet
print("M2 replay smoke ok")
PY
```

---

# Smoke test FocusFeedbackBoundary

```bash
python - <<'PY'
import torch
from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary

b = 1
boundary = FocusFeedbackBoundary(
    focus_context_dim=256,
    workspace_seed_dim=256,
    thought_dim=192,
)

workspace_seed = torch.randn(b, 256)
focus_seed = torch.randn(b, 256)
gate = torch.tensor([[0.1]])

packet = boundary(
    workspace_seed=workspace_seed,
    focus_context_seed=focus_seed,
    focus_context_seed_gate=gate,
)

assert packet["workspace_seed"].shape == (b, 256)
assert packet["preconscious_delta"].shape == (b, 192)
assert packet["total_gate"].shape == (b, 1)

pre = torch.randn(b, 192)
pre2 = boundary.apply_preconscious_seed(pre, packet)
assert pre2.shape == (b, 192)

print("FocusFeedbackBoundary smoke ok")
PY
```

---

# Что исправлять

Исправлять только реальные ошибки:

- compile/import errors;
- M2 не видит M4 context;
- M2 всё ещё напрямую мутирует `focus_context` по умолчанию;
- `model_step(...)` всё ещё спрашивает только conscious seed;
- нет `get_m5_focus_seed(...)`;
- `life_runtime.py` вызывает M2 до `out["affect"]`;
- отсутствуют stage labels `pre_observe` / `main`;
- shape mismatch в M2/M4/M13 replay context;
- runtime падает, если M4 или M13 ещё пустые.

Не переписывать архитектуру заново.

---

# Ожидаемый итог

В отчёте напиши:

1. Какие файлы проверены.
2. Прошли ли compile/import checks.
3. Прошёл ли smoke test M2.
4. Прошёл ли smoke test FocusFeedbackBoundary.
5. Подтверждение, что `M1 → M2` напрямую отсутствует.
6. Подтверждение, что M2 кладёт `replay_context` в тот же M5 seed-вход, что и M15.
7. Подтверждение, что M4 и M13 входят в M2.
8. Подтверждение итоговой схемы:

```text
M1 → M5 → M11 → M2 → M5
          ↑      ↑
          M4     M13
```

9. Какие минимальные правки внесены, если были.
10. Какие риски остались.
