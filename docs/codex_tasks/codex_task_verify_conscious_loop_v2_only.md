# Codex task: проверить уже встроенную петлю сознания v2 по сути

## Контекст

В проект DMoC уже встроена петля сознания v2. Ничего применять, распаковывать, создавать ветки или накатывать patch не нужно.

Нужно только проверить, что реализация по смыслу и по коду соответствует архитектуре:

```text
M5 → M10 → M9 → M11/M12 → M15(post-self) → M5
```

Главный принцип:

```text
raw self_state НЕ должен напрямую попадать в M5
```

Правильный путь:

```text
M9 self_bound_context
+ subjective_affect_state
+ affect_latents
+ focus_context
→ M15 post-self
→ enhanced_focus_context
→ gated next_focus_context_seed
→ M5 FocusFeedbackBoundary
→ workspace_seed + preconscious_seed
```

## Главная цель проверки

Проверить, что петля сознания действительно замкнута правильно:

1. M5 создаёт `focus_context`.
2. M10 делает broadcast / global access.
3. M9 связывает фокус с self-моделью и создаёт `self_bound_context`.
4. M11 даёт актуальный affect.
5. M12, если включён, даёт confidence/doubt gate.
6. M15 запускается в post-self режиме.
7. M15 создаёт `enhanced_focus_context`.
8. Runtime сохраняет его как `next_focus_context_seed`.
9. Следующий вызов M5 получает этот seed через `FocusFeedbackBoundary`.
10. `FocusFeedbackBoundary` влияет на `workspace_seed` и `preconscious_seed`.

## Ничего не делать

Не нужно:

```bash
git checkout -b ...
unzip ...
python scripts/apply_...
git apply ...
```

Не нужно переписывать архитектуру.

Не нужно менять имена модулей.

Не нужно добавлять новый patch.

Задача — проверить уже существующий код и, если есть явная ошибка, предложить минимальную правку.

---

# Проверки по файлам

## 1. Проверить наличие M5 FocusFeedbackBoundary

Файл:

```text
src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
```

Проверь, что там есть класс:

```python
class FocusFeedbackBoundary(nn.Module):
```

Он должен принимать:

```python
workspace_seed
focus_context_seed
focus_context_seed_gate
```

И возвращать пакет с ключами примерно:

```python
workspace_seed
preconscious_delta
external_gate
learned_gate
total_gate
seed_norm
active
```

Также должен быть метод:

```python
apply_preconscious_seed(...)
```

Он должен аккуратно подмешивать `preconscious_delta` в `preconscious_seed` через gate.

Проверь, что размерности безопасно pad/trim-ятся и нет прямой зависимости от `self_state`.

---

## 2. Проверить интеграцию FocusFeedbackBoundary в M5

Файл:

```text
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
```

Проверь:

### Импорт

```python
from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary
```

### Инициализация

В `__init__` должен быть создан слой:

```python
self.focus_feedback_boundary = FocusFeedbackBoundary(...)
```

### Сигнатура step

`step(...)` должен принимать:

```python
focus_context_seed=None
focus_context_seed_gate=None
```

### Вход после AttentionController

После:

```python
attn = self.attention(...)
```

должно быть что-то логически эквивалентное:

```python
focus_feedback = self.focus_feedback_boundary(
    workspace_seed=attn["workspace_seed"],
    focus_context_seed=focus_context_seed,
    focus_context_seed_gate=focus_context_seed_gate,
)
attn["workspace_seed"] = focus_feedback["workspace_seed"]
```

### Влияние на preconscious_seed

После `Workspace(...)` и до thought/reflection loop должно быть:

```python
preconscious_seed = self.focus_feedback_boundary.apply_preconscious_seed(
    preconscious_seed,
    focus_feedback,
)
```

или эквивалентная логика.

### Debug output

В `out` желательно должен быть пакет:

```python
out["focus_feedback"]
```

или return key:

```python
"focus_feedback": {...}
```

В нём должны быть видны:

```python
active
external_gate
learned_gate
total_gate
seed_norm
```

---

## 3. Проверить runtime seed-передачу в M5

Файл:

```text
src/apps/unified_conscious_viewer.py
```

Проверь `model_step(...)`.

Он должен принимать:

```python
model_stage: str = "main"
focus_context_seed=None
focus_context_seed_gate=None
```

Если seed явно не передан, он должен получать его из runtime-loop:

```python
self.get_conscious_loop_focus_seed(stage=model_stage)
```

Затем должен передавать в `self.model.step(...)`:

```python
focus_context_seed=focus_context_seed
focus_context_seed_gate=focus_context_seed_gate
```

Проверь, что все существующие вызовы `model_step(...)` не сломаны, потому что новые параметры имеют defaults.

---

## 4. Проверить stage labels в life_runtime

Файл:

```text
src/apps/life_runtime.py
```

Проверь, что два вызова M5 различаются по stage:

```python
out0 = self.model_step(obs0, self.state, model_stage="pre_observe")
out = self.model_step(obs, self.state, model_stage="main")
```

Это нужно, чтобы config `conscious_loop.apply_stage` мог решать, куда подавать feedback:

```text
both | pre_observe | main
```

---

## 5. Проверить runtime-модуль петли

Файл:

```text
src/modules/m15_counterfactual_imagination_planning/conscious_loop_runtime.py
```

Проверь, что есть:

```python
class ConsciousLoopRuntimeMixin:
```

Обязательные методы:

```python
ensure_conscious_loop_ready(...)
get_conscious_loop_focus_seed(...)
compute_conscious_loop_feedback(...)
maybe_print_conscious_loop_trace(...)
```

## get_conscious_loop_focus_seed

Должен возвращать:

```python
focus_context_seed
focus_context_seed_gate
```

и учитывать:

```python
cfg.conscious_loop.enabled
cfg.conscious_loop.apply_stage
```

## compute_conscious_loop_feedback

Должен запускаться после M9 и после появления affect.

Он должен:

1. Проверить наличие `out["focus_context"]`.
2. Проверить наличие `out["self_core"]["self_bound_context"]`.
3. Вызвать M15 в post-self режиме:

```python
self.compute_thought_chain({}, out, pre_self_binding=False)
```

4. Взять:

```python
post_chain["enhanced_focus_context"]
```

5. Посчитать gate.
6. Сохранить:

```python
self._conscious_loop_next_focus_seed
self._conscious_loop_next_focus_gate
out["conscious_loop"]
out["next_focus_context_seed"]
out["next_focus_context_seed_gate"]
```

---

## 6. Проверить, что feedback считается после affect

Файл:

```text
src/apps/life_runtime.py
```

Проверь порядок.

Желательно:

```python
out["self_core"] = self.compute_self_core(obs, out)

...

emotion = self.emotional_drive.compute(out, obs)
out["emotion"] = emotion
if isinstance(emotion.get("affect"), dict):
    out["affect"] = emotion["affect"]

# потом, после affect:
self.compute_metacognition(obs, out)  # если есть
self.compute_conscious_loop_feedback(obs, out)
```

Почему это важно:

M15 controller принимает:

```python
focus_context
affect_latents
self_bound_context
subjective_affect_state
```

Если `affect_latents` ещё нет, post-self chain будет неполным.

---

## 7. Проверить M15 post-self режим

Файл:

```text
src/modules/m15_counterfactual_imagination_planning/thought_chain_runtime.py
```

Проверь, что `compute_thought_chain(..., pre_self_binding=False)` реально передаёт в controller:

```python
self_bound_context
subjective_affect_state
```

А не зануляет их.

В pre-self режиме можно занулять:

```python
self_bound_context=None
subjective_affect_state=None
```

Но в post-self режиме они должны использоваться.

---

## 8. Проверить M15 controller

Файл:

```text
src/modules/m15_counterfactual_imagination_planning/thought_chain_controller.py
```

Проверь, что `ThoughtChainController.forward(...)` принимает:

```python
focus_context
affect_latents
self_bound_context
subjective_affect_state
```

И что `enhanced_focus_context` создаётся примерно так:

```python
focus_delta = ...
focus_gate = ...
enhanced_focus_context = focus_context + focus_gate * focus_delta
```

Это и есть seed для следующего M5.

---

## 9. Проверить config

Файл:

```text
src/shared/config.py
```

Должен быть config:

```python
@dataclass
class ConsciousLoopRuntimeConfig:
    enabled: bool = True
    feedback_gain: float = 0.22
    min_gate: float = 0.00
    max_gate: float = 0.22
    require_self_binding: bool = True
    use_metacognition_gate: bool = True
    apply_stage: str = "both"
    print_every_steps: int = 30
```

И в `UnifiedConfig`:

```python
conscious_loop: ConsciousLoopRuntimeConfig = field(default_factory=ConsciousLoopRuntimeConfig)
```

---

## 10. Проверить runner mixin

Файл:

```text
src/apps/runner.py
```

Должен быть импорт:

```python
from src.modules.m15_counterfactual_imagination_planning.conscious_loop_runtime import ConsciousLoopRuntimeMixin
```

И в `UnifiedSystem` должно быть:

```python
ThoughtChainRuntimeMixin,
ConsciousLoopRuntimeMixin,
GlobalBroadcastRuntimeMixin,
```

Порядок важен: `ConsciousLoopRuntimeMixin` должен иметь доступ к `compute_thought_chain(...)`.

---

# Запрещённые паттерны

Найди и убедись, что их нет:

```python
focus_context_seed = out["self_core"]["self_state"]
```

```python
out["focus_context"] = out["self_core"]["self_state"]
```

```python
self_state → M5
```

```python
raw_self_state
```

```python
focus_context_seed = self_state
```

Если найдёшь прямой путь `self_state → M5`, это ошибка.

---

# Compile checks

Запусти:

```bash
python -m py_compile src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
python -m py_compile src/modules/m15_counterfactual_imagination_planning/conscious_loop_runtime.py
python -m py_compile src/apps/life_runtime.py
python -m py_compile src/apps/runner.py
python -m py_compile src/apps/unified_conscious_viewer.py
python -m py_compile src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
python -m py_compile src/modules/m15_counterfactual_imagination_planning/thought_chain_runtime.py
python -m py_compile src/modules/m15_counterfactual_imagination_planning/thought_chain_controller.py
python -m py_compile src/shared/config.py
```

---

# Import checks

Запусти:

```bash
python - <<'PY'
from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary
from src.modules.m15_counterfactual_imagination_planning.conscious_loop_runtime import ConsciousLoopRuntimeMixin
from src.apps.runner import UnifiedSystem
print("imports ok")
PY
```

---

# Smoke test FocusFeedbackBoundary

Запусти:

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
assert packet["learned_gate"].shape == (b, 1)

pre = torch.randn(b, 192)
pre2 = boundary.apply_preconscious_seed(pre, packet)
assert pre2.shape == (b, 192)

print("focus feedback boundary smoke ok")
PY
```

---

# Optional runtime smoke check

Если можно создать модель без запуска MuJoCo GUI, проверь один M5 step с seed:

```python
out = system.model_step(
    obs,
    system.state,
    model_stage="main",
    focus_context_seed=torch.randn(1, 256, device=system.device),
    focus_context_seed_gate=torch.tensor([[0.1]], device=system.device),
)
assert "focus_feedback" in out
```

Если это невозможно из-за MuJoCo/GUI, пропусти и напиши почему.

---

# Что исправлять

Исправляй только реальные ошибки:

- import error;
- syntax error;
- shape mismatch;
- неправильный порядок вызова feedback до affect;
- отсутствие `ConsciousLoopRuntimeConfig`;
- отсутствие `focus_context_seed` в `model_step`;
- отсутствие передачи seed в M5;
- прямой raw `self_state → M5`;
- неправильный post-self M15 вызов;
- падение smoke test.

Не переписывай архитектуру заново.

---

# Ожидаемый отчёт

В конце дай отчёт:

1. Какие файлы проверены.
2. Какие проверки прошли.
3. Какие ошибки найдены.
4. Какие минимальные правки внесены.
5. Подтверждение, что прямого `raw self_state → M5` нет.
6. Подтверждение, что петля работает как:

```text
M5 focus_context
→ M10 broadcast
→ M9 self_bound_context
→ M11/M12 gate
→ M15 post-self enhanced_focus_context
→ M5 FocusFeedbackBoundary
```

7. Команды, которыми повторить проверку.
