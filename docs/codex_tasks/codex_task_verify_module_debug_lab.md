# Codex task: проверить лабораторию отладки модулей DMoC

## Цель

Проверить, что добавленная лаборатория модулей корректно тестирует отдельные модули DMoC и весь бессознательный контур.

Нужно проверить именно модульную отладку:

```text
fixtures → module → output contract → behavioral checks → report
```

Главная архитектура, которую должны подтверждать тесты:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13

M5 → M3 → тело/мир → M1
```

## Важные правила архитектуры

Проверить, что тесты и лаборатория закрепляют эти invariants:

```text
1. M1 не идёт напрямую в M2.
2. M1 / сенсоры идут в M5.
3. M5 строит focus_context / workspace / inner playback.
4. M11 получает результат M5 и создаёт affect.
5. M2 получает affect от M11.
6. M13 идёт в M2 через retrieved_context.
7. M4 идёт в M2 через dynamic_identity_context.
8. M4 получает вход не напрямую от M1, а из inner_object / z_obj, созданного после M5.
9. M2 не мутирует out["focus_context"] напрямую по умолчанию.
10. M2 кладёт replay_context в тот же M5 seed-вход, куда M15 кладёт conscious seed:
    focus_context_seed + focus_context_seed_gate → FocusFeedbackBoundary.
11. M3 управляется от M5, а не от M2/M11/M13 напрямую.
```

---

# Что уже должно быть добавлено patch-ем

Проверить наличие файлов:

```text
scripts/module_lab/module_fixture_factory.py
scripts/module_lab/run_module_lab.py

tests/module_contracts/test_m02_event_dream_replay_contract.py
tests/module_contracts/test_m02_runtime_seed_bus_contract.py
tests/module_contracts/test_m04_long_dynamic_memory_contract.py
tests/module_contracts/test_m05_focus_feedback_boundary_contract.py
tests/module_contracts/test_m11_emotional_drive_contract.py
tests/module_contracts/test_m13_autobiographical_memory_contract.py
tests/module_contracts/test_unconscious_loop_contract.py

docs/architecture/module_debug_lab.md
```

Если каких-то файлов нет — найти причину и добавить их минимально, не переписывая проект.

---

# 1. Проверить fixture factory

Файл:

```text
scripts/module_lab/module_fixture_factory.py
```

Проверить, что он создаёт искусственные входы без MuJoCo:

```python
make_fake_obs()
make_fake_m5_out()
make_fake_affect()
make_fake_m13_memory()
make_fake_m4_identity()
make_fake_inner_object()
make_fake_event_memory()
make_unconscious_loop_out()
FakePassportManager
```

Проверить helper-и:

```python
assert_tensor(...)
assert_gate(...)
finite_tensor(...)
scalar(...)
randn(...)
```

## Ожидание

Фабрика должна давать стабильные shapes:

```text
focus_context: [1, 256]
workspace_out: [1, 256]
object_repr: [1, 128]
thought_candidate: [1, 192]
retrieved_context: [1, 256]
dynamic_identity_context: [1, 256]
```

Не должно быть NaN/Inf.

---

# 2. Проверить M11 contract test

Файл:

```text
tests/module_contracts/test_m11_emotional_drive_contract.py
```

Проверить, что тест создаёт fake M5 output и прогоняет:

```python
EmotionalDrive.compute(out, obs)
```

Проверить, что обязательные выходы есть:

```text
emotion["affect"]
affect_latents
valence
arousal
pain_latent
stress_latent
fear_latent
panic_latent
comfort_latent
relief_latent
curiosity_latent
discovery_latent
coherence_latent
expected_affect_delta
intrinsic_reward
```

## Ожидание

M11 должен быть проверен как оценщик результата M5:

```text
M5 → M11
```

M11 не должен получать M1 напрямую как основной источник сцены.

---

# 3. Проверить M13 contract test

Файл:

```text
tests/module_contracts/test_m13_autobiographical_memory_contract.py
```

Проверить сценарии:

```text
1. Память пустая.
2. Запись эпизода.
3. Retrieval после записи.
```

Обязательные выходы:

```text
retrieved_context
retrieval_relevance
summary
retrieved_episode_count
```

## Ожидание

M13 должен подтверждать связь:

```text
M13 → M2
```

M13 не должен идти напрямую в M5 / FocusFeedbackBoundary в бессознательном контуре.

---

# 4. Проверить M4 contract test

Файл:

```text
tests/module_contracts/test_m04_long_dynamic_memory_contract.py
```

Проверить, что M4 получает:

```text
inner_object / z_obj
focus_context из out
FakePassportManager
```

и возвращает:

```text
dynamic_identity_context
dynamic_memory_gate
identity_token
identity_stability
identity_novelty
passport_slot
selected_sentence
```

## Ожидание

Контракт должен закреплять:

```text
M1 → M5 → inner_object / z_obj → M4 → M2
```

А не:

```text
M1 → M4 напрямую
```

---

# 5. Проверить M2 EventDreamReplay contract test

Файл:

```text
tests/module_contracts/test_m02_event_dream_replay_contract.py
```

Проверить, что M2 получает:

```text
out["focus_context"] от M5
out["affect"] от M11
out["autobiographical_memory"]["retrieved_context"] от M13
out["long_dynamic_memory"]["dynamic_identity_context"] от M4
event_memory optional
dream_mode=True
```

Проверить обязательные выходы:

```text
replay_context
replay_gate
event_salience
dream_pressure
should_replay
replay_source
selected_identity_token
dynamic_memory_gate
selected_episode_summary
```

Проверить behavioral check:

```text
panic/stress выше → dream_pressure не должен уменьшаться
```

## Ожидание

M2 должен быть selector-ом replay, а не сенсорным модулем.

Правильная связь:

```text
M11 → M2
M13 → M2
M4 → M2
M5 focus_context → M2
M2 replay_context → FocusFeedbackBoundary → M5
```

---

# 6. Проверить M2 runtime seed bus test

Файл:

```text
tests/module_contracts/test_m02_runtime_seed_bus_contract.py
```

Проверить, что dummy runtime без полного runner-а может вызвать:

```python
compute_event_dream_replay(...)
get_event_dream_focus_seed(...)
get_m5_focus_seed(...)
_store_event_dream_m5_seed(...)
```

Проверить:

```text
M2 replay_context сохраняется как seed
seed shape == [1, 256]
gate существует
apply_stage="pre_observe" работает
stage="main" по умолчанию не возвращает seed
```

## Ожидание

M2 должен класть seed в общий M5 вход:

```text
focus_context_seed + focus_context_seed_gate
```

И не создавать отдельный dream-вход в M5.

---

# 7. Проверить M5 FocusFeedbackBoundary contract test

Файл:

```text
tests/module_contracts/test_m05_focus_feedback_boundary_contract.py
```

Проверить:

```python
FocusFeedbackBoundary(
    workspace_seed=...,
    focus_context_seed=...,
    focus_context_seed_gate=...
)
```

Обязательные выходы:

```text
active
workspace_seed
external_gate
learned_gate
total_gate
workspace_delta
preconscious_delta
seed_norm
```

Проверить:

```text
total_gate в диапазоне 0..0.35
workspace_seed shape == [1, 256]
preconscious_delta shape == [1, 192]
apply_preconscious_seed(...) сохраняет shape
при seed=None total_gate=0 и runaway нет
```

## Ожидание

Это общий receptor M5 для:

```text
M2 replay seed
M15 conscious seed
```

---

# 8. Проверить integration contract бессознательного контура

Файл:

```text
tests/module_contracts/test_unconscious_loop_contract.py
```

Проверить, что тест строит цепочку:

```text
fake M5 out
↓
M11 emotion / affect
↓
M13 retrieve
↓
M4 from inner_object / z_obj
↓
M2 replay
↓
M5 FocusFeedbackBoundary
```

Проверить, что в этом тесте M2 не получает raw M1:

```python
assert "left" not in out
assert "right" not in out
assert "depth" not in out
```

## Ожидание

Тест должен подтверждать итоговую схему:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13

M5 → M3
```

---

# 9. Проверить command-line лабораторию

Файл:

```text
scripts/module_lab/run_module_lab.py
```

Проверить запуск:

```bash
python scripts/module_lab/run_module_lab.py --module all
python scripts/module_lab/run_module_lab.py --module m11
python scripts/module_lab/run_module_lab.py --module m13
python scripts/module_lab/run_module_lab.py --module m4
python scripts/module_lab/run_module_lab.py --module m02
python scripts/module_lab/run_module_lab.py --module m05
python scripts/module_lab/run_module_lab.py --module loop
```

Проверить JSON-режим:

```bash
python scripts/module_lab/run_module_lab.py --module loop --json
```

## Ожидание

Каждый запуск должен возвращать короткий отчёт:

```text
module
status
ключевые scalar-метрики
```

Например:

```text
M2:
    replay_gate
    dream_pressure
    event_salience
    should_replay
    source
    identity

M4:
    identity_token
    dynamic_memory_gate
    identity_stability
    identity_novelty

M11:
    valence
    arousal
    stress
    panic
    curiosity
```

---

# 10. Запустить compile checks

Запустить:

```bash
python -m py_compile scripts/module_lab/module_fixture_factory.py
python -m py_compile scripts/module_lab/run_module_lab.py

python -m py_compile tests/module_contracts/test_m02_event_dream_replay_contract.py
python -m py_compile tests/module_contracts/test_m02_runtime_seed_bus_contract.py
python -m py_compile tests/module_contracts/test_m04_long_dynamic_memory_contract.py
python -m py_compile tests/module_contracts/test_m05_focus_feedback_boundary_contract.py
python -m py_compile tests/module_contracts/test_m11_emotional_drive_contract.py
python -m py_compile tests/module_contracts/test_m13_autobiographical_memory_contract.py
python -m py_compile tests/module_contracts/test_unconscious_loop_contract.py
```

---

# 11. Запустить pytest

Запустить:

```bash
pytest tests/module_contracts
```

Если pytest нет, проверить хотя бы импорт и прямой запуск лаборатории.

---

# 12. Запустить direct smoke commands

```bash
python scripts/module_lab/run_module_lab.py --module all
python scripts/module_lab/run_module_lab.py --module loop --json
```

---

# 13. Что исправлять

Исправлять только реальные ошибки:

```text
compile errors
import errors
shape mismatch
NaN / Inf
неверные ключи в contracts
M2 не видит M4 context
M2 не видит M13 context
M2 не создаёт replay_context
M2 runtime seed bus не возвращает seed
FocusFeedbackBoundary не принимает seed
M4 не создаёт dynamic_identity_context
M13 retrieval не создаёт retrieved_context
M11 affect не содержит нужные поля
```

Не переписывать архитектуру заново.

---

# 14. Что НЕ делать

Не делать:

```text
не подключать MuJoCo
не запускать полный life loop
не трогать визуализаторы
не менять M9/M10/M15 сознательный контур
не создавать новую архитектуру
не делать прямой M1 → M2
не добавлять отдельный dream_context вход в M5
не делать M13 → M5 напрямую
не делать M4 → M5 напрямую как основной replay-путь
```

---

# 15. Ожидаемый отчёт Codex

В конце написать отчёт:

```text
1. Какие файлы проверены.
2. Какие новые tests/module_contracts найдены.
3. Прошли ли py_compile checks.
4. Прошёл ли pytest tests/module_contracts.
5. Прошёл ли python scripts/module_lab/run_module_lab.py --module all.
6. Подтверждение: M1 не идёт в M2 напрямую.
7. Подтверждение: M13 идёт в M2.
8. Подтверждение: M4 идёт в M2 и получает вход из inner_object / z_obj.
9. Подтверждение: M2 кладёт replay_context в FocusFeedbackBoundary seed-вход M5.
10. Подтверждение: M3 управляется от M5.
11. Какие минимальные правки внесены.
12. Какие риски остались.
```

Итоговая строка должна быть:

```text
Module debug lab confirms the unconscious loop contract:
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5,
with M4/M13 feeding M2 and M5 controlling M3.
```
