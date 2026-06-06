# Codex task: проверить behavioral scenarios бессознательного sleep/replay контура

## Цель

Проверить не просто contract tests, а именно **поведенческие сценарии** бессознательного контура:

```text
fixtures → scenario → M2/M4/M11/M13/M5 behavior → expected outcome
```

Нужно убедиться, что сценарии подтверждают смысловую работу схемы:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13
```

Главный файл:

```text
scripts/module_lab/scenario_unconscious_replay.py
```

Главный pytest:

```text
tests/module_contracts/test_unconscious_behavioral_scenarios.py
```

---

## 1. Проверить наличие сценариев

В файле:

```text
scripts/module_lab/scenario_unconscious_replay.py
```

Должны быть сценарии:

```text
calm_no_replay
curiosity_replay
bad_prediction_dream
object_identity_replay
```

Проверить, что `run_all()` запускает все эти сценарии и возвращает:

```python
{
    "status": "ok" или "fail",
    "scenarios": [...]
}
```

---

## 2. Проверить calm_no_replay

Смысл:

```text
спокойствие
низкий panic
низкий stress
низкий curiosity
нет сильного M13 context
нет сильного M4 identity gate
dream_mode=False
```

Ожидание:

```text
dream_pressure низкий
replay не должен активно включаться
```

Минимальный критерий:

```text
dream_pressure < 0.50
```

Проверить, что сценарий не проходит случайно из-за отсутствия M2, а реально вызывает:

```python
EventDreamReplay.compute(...)
```

---

## 3. Проверить curiosity_replay

Смысл:

```text
высокое curiosity
есть M13 retrieved_context
есть M4 dynamic_identity_context
dream_mode=True
```

Ожидание:

```text
M2 должен сильнее активировать replay
dream_pressure повышается
event_salience повышается
identity доступна
```

Минимальные критерии:

```text
dream_pressure >= 0.35
event_salience >= 0.25
identity не ломает пакет
```

Проверить, что M2 реально получает:

```text
out["affect"]["curiosity_latent"]
out["autobiographical_memory"]["retrieved_context"]
out["long_dynamic_memory"]["dynamic_identity_context"]
```

---

## 4. Проверить bad_prediction_dream

Смысл:

```text
высокий panic
высокий stress
dream_mode=True
```

Ожидание:

```text
dream_pressure должен стать высоким
```

Минимальный критерий:

```text
dream_pressure >= 0.50
```

Проверить, что сценарий сравнивается с calm scenario:

```text
bad_prediction_dream.dream_pressure >= calm_no_replay.dream_pressure
```

---

## 5. Проверить object_identity_replay

Смысл:

```text
M4 получает inner_object / z_obj
FakePassportManager даёт stable identity
M4 возвращает dynamic_identity_context
```

Ожидание:

```text
identity_token существует
dynamic_memory_gate > 0
identity_stability числовой
identity_novelty числовой
```

Проверить, что M4 не получает raw M1 sensors напрямую, а вход идёт через:

```text
inner_object / z_obj
```

---

## 6. Запустить direct scenario script

Из корня проекта:

```bash
python scripts/module_lab/scenario_unconscious_replay.py --json
```

Ожидаемый результат:

```json
{
  "status": "ok",
  "scenarios": [
    {
      "name": "calm_no_replay",
      "pass": true
    },
    {
      "name": "curiosity_replay",
      "pass": true
    },
    {
      "name": "bad_prediction_dream",
      "pass": true
    },
    {
      "name": "object_identity_replay",
      "pass": true
    }
  ]
}
```

Если `status="fail"` — найти конкретный scenario и исправить минимально.

---

## 7. Запустить pytest

```bash
pytest tests/module_contracts/test_unconscious_behavioral_scenarios.py
```

Ожидание:

```text
all tests passed
```

Если падает — исправить причину, не ослабляя смысл сценариев.

---

## 8. Дополнительная проверка вместе с module lab

Запустить:

```bash
python scripts/module_lab/run_module_lab.py --module all
python scripts/module_lab/run_module_lab.py --module loop
python scripts/module_lab/scenario_unconscious_replay.py --json
```

Потом:

```bash
pytest tests/module_contracts/test_unconscious_behavioral_scenarios.py
pytest tests/module_contracts/test_unconscious_loop_contract.py
pytest tests/module_contracts/test_m02_event_dream_replay_contract.py
```

---

## 9. Что можно исправлять

Исправлять только реальные проблемы:

```text
import errors
shape mismatch
NaN / Inf
неверные ключи в fake out
M2 не видит affect
M2 не видит M13 context
M2 не видит M4 context
M4 fake passport не подходит текущему API
сценарий слишком хрупкий к текущим thresholds
```

Если threshold слишком строгий из-за изменения формулы M2, можно поправить threshold, но нельзя превращать сценарий в бессмысленный smoke-test.

---

## 10. Что нельзя делать

Не делать:

```text
не переписывать архитектуру
не подключать MuJoCo
не запускать полный live runner
не делать M1 → M2 напрямую
не делать M13 → M5 напрямую
не делать M4 → M5 напрямую
не включать прямую mutation out["focus_context"] как основной путь
не удалять проверку calm vs bad_prediction
не удалять проверку M4 identity
```

---

## 11. Ожидаемый отчёт Codex

В конце написать отчёт:

```text
1. Проверен файл scripts/module_lab/scenario_unconscious_replay.py.
2. Проверен файл tests/module_contracts/test_unconscious_behavioral_scenarios.py.
3. Результат python scripts/module_lab/scenario_unconscious_replay.py --json.
4. Результат pytest tests/module_contracts/test_unconscious_behavioral_scenarios.py.
5. Подтверждение:
   ✅ calm_no_replay проходит
   ✅ curiosity_replay проходит
   ✅ bad_prediction_dream проходит
   ✅ object_identity_replay проходит
   ✅ bad_prediction_dream pressure >= calm pressure
   ✅ M4 identity доступна для M2
   ✅ M13 context доступен для M2
   ✅ M11 affect влияет на M2 dream_pressure
6. Какие минимальные исправления внесены.
7. Остались ли риски.
```

Финальная строка отчёта должна быть:

```text
Behavioral scenarios confirmed:
calm state suppresses replay, curiosity/M13/M4 activate replay, stress/panic increases dream pressure, and M4 identity is available to M2.
```
