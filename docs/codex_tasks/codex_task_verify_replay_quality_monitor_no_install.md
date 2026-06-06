# Codex task: проверить Replay Quality Monitor в текущем репозитории, без установки patch

## Главное правило

Ничего не устанавливать и не применять.

Не запускать:

```bash
unzip ...
python scripts/apply_...
```

Проверять только текущее состояние репозитория. Если чего-то нет — зафиксировать как проблему. Исправлять только минимальные реальные ошибки, без переписывания архитектуры.

---

## Цель проверки

Убедиться, что **Replay Quality Monitor** уже встроен и корректно показывает качество sleep/replay:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13

M5 → M3, но во сне M3 наружу заблокирован
```

Монитор должен отвечать не просто “replay активен?”, а:

```text
выбран ли эпизод M13
выбран ли identity M4
доходит ли M2 seed до M5
падает ли dream_pressure
растёт ли relief/coherence
есть ли признаки интеграции опыта
```

---

## 1. Проверить наличие файлов

Проверить, что есть:

```text
src/modules/m08_debug_visual_control/replay_quality_monitor_status.py
tests/module_contracts/test_replay_quality_monitor_status_contract.py
docs/architecture/replay_quality_monitor.md
```

Если файла нет — report FAIL, не применять patch.

---

## 2. Compile checks

Запустить:

```bash
python -m py_compile src/modules/m08_debug_visual_control/replay_quality_monitor_status.py
python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
```

Ожидание: без ошибок.

---

## 3. Pytest checks

Запустить:

```bash
pytest tests/module_contracts/test_replay_quality_monitor_status_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
pytest tests/module_contracts/test_dream_probe_runtime_contract.py
pytest tests/module_contracts/test_m02_runtime_seed_bus_contract.py
pytest tests/module_contracts/test_unconscious_behavioral_scenarios.py
```

Если есть время:

```bash
pytest tests/module_contracts
```

---

## 4. Проверить module_status_runtime.py

Файл:

```text
src/modules/m08_debug_visual_control/module_status_runtime.py
```

Должен быть импорт:

```python
from src.modules.m08_debug_visual_control.replay_quality_monitor_status import build_replay_quality_monitor_status
```

В status payload должен быть ключ:

```python
"replay_quality_monitor": build_replay_quality_monitor_status(self),
```

Важно: он не должен удалить или заменить:

```text
sleep_replay_monitor
last_module_lab_result
input_sensors_enabled
sleep_sensor_mask
```

---

## 5. Проверить replay_quality_monitor_status.py

Функция:

```python
build_replay_quality_monitor_status(system)
```

должна читать:

```python
out = getattr(system, "latest_out", {}) or {}
```

и извлекать:

```text
out["affect"]
out["emotion"]
out["autobiographical_memory"]
out["long_dynamic_memory"]
out["event_dream_replay"]
out["dream_probe"]
```

Обязательные top-level keys:

```text
global_step
full_sleep
sensor_state
quality_score
quality_ema
verdict
selected_episode_summary
selected_identity_token
selected_identity_sentence
replay_source
m2
affect
m13
m4
m5
dream_probe
history
samples
```

Nested keys:

```text
m2:
    replay_gate
    should_replay
    dream_pressure
    event_salience
    dream_pressure_delta
    dream_pressure_trend

affect:
    stress
    stress_delta
    panic
    panic_delta
    relief
    relief_delta
    curiosity
    valence
    valence_delta
    coherence
    coherence_delta
    expected_affect_delta

m13:
    relevance
    episodes
    summary

m4:
    identity_token
    dynamic_memory_gate
    identity_stability
    identity_novelty
    selected_sentence

m5:
    seed_gate
    seed_norm
```

---

## 6. Проверить delta logic

`replay_quality_monitor_status.py` должен хранить состояние в:

```python
system._replay_quality_monitor_state
```

И repeated status poll на том же `global_step` не должен затирать delta обратно в 0.

Проверить логику:

```python
if last_step == step:
    delta = state["last_delta"]
else:
    delta = current - prev
    state["prev"] = current
    state["last_delta"] = delta
    state["last_step"] = step
```

Это критично, потому что status poll может идти чаще, чем life step.

---

## 7. Проверить quality_score и verdict

Проверить:

```text
0.0 <= quality_score <= 1.0
```

`quality_score` должен учитывать:

```text
replay_gate
event_salience
M13 retrieval_relevance
M4 dynamic_memory_gate
M5 seed_gate
M5 seed_norm
pressure_improvement
relief_gain
stress_improvement
coherence_gain
```

Допустимые verdict:

```text
idle
weak
replaying
integrating
```

Логика:

```text
idle         replay почти нет
weak         слабый сигнал
replaying    replay/seed заметны
integrating  pressure падает или relief/coherence растут
```

---

## 8. Проверить control_panel.py

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

Проверить:

```python
"btn_replay_quality_monitor"
self.btn_replay_quality_monitor = QtWidgets.QPushButton("Replay Quality Monitor")
self.btn_replay_quality_monitor.clicked.connect(self.open_replay_quality_monitor_window)
self.refresh_replay_quality_monitor_window()
```

`btn_replay_quality_monitor` должен быть:

```text
во вкладке M8
в runner-dependent controls
стилизован в refresh_ui
```

---

## 9. Проверить окно Replay Quality Monitor

Метод:

```python
open_replay_quality_monitor_window(...)
```

Окно должно показывать:

```text
Replay selection:
    verdict
    quality_score
    quality_ema
    replay_source
    selected_identity_token
    selected_episode_summary

M2 replay dynamics:
    replay_gate
    should_replay
    event_salience
    dream_pressure
    Δ dream_pressure
    pressure trend

Affect integration:
    stress
    Δ stress
    relief
    Δ relief
    coherence
    Δ coherence
    expected_affect_delta

Memory / identity support:
    M13 relevance
    M13 episodes
    M4 gate
    M4 stability
    M4 novelty
    M5 seed_gate
    M5 seed_norm
```

Raw JSON должен показывать:

```python
last_status["replay_quality_monitor"]
```

---

## 10. Live-smoke

Запустить runner фактической командой проекта, например:

```bash
python -m src.apps.runner
```

Запустить control panel:

```bash
python src/modules/m08_debug_visual_control/control_panel.py
```

Проверить:

```text
STATUS IPC: receiving
M8 tab содержит Replay Quality Monitor
окно открывается
header показывает step / verdict / quality / ema
raw JSON обновляется
```

---

## 11. Live-smoke: sleep без probe

Включить:

```text
Сон / replay mode
```

Ожидается:

```text
sensor_state = sleep
full_sleep = True
```

В Replay Quality Monitor допустимо:

```text
verdict = idle или weak
quality_score около 0
selected_episode_summary может быть пустой
selected_identity_token может быть пустой
```

Это нормально, если replay тихий.

---

## 12. Live-smoke: Probe stress

В Sleep Replay Monitor нажать:

```text
Probe stress
```

В Replay Quality Monitor смотреть:

```text
dream_pressure
Δ dream_pressure
stress
Δ stress
relief
Δ relief
quality_score
verdict
```

Ожидаемо:

```text
stress_delta или panic_delta меняется
dream_pressure меняется
quality_score может вырасти
```

Если всё 0 — проблема скорее в Dream Probe path, не в Replay Quality Monitor.

---

## 13. Live-smoke: Probe replay seed

Нажать:

```text
Probe replay seed
```

Ожидаемо:

```text
M5 seed_gate > 0
M5 seed_norm > 0
quality_score > 0
verdict = replaying или weak/replaying
```

Если `M5 seed_gate = 0` и `M5 seed_norm = 0`, проверить:

```text
DreamProbeRuntimeMixin._inject_probe_replay_seed
_event_dream_next_focus_seed
_event_dream_next_focus_gate
get_m5_focus_seed(...)
```

---

## 14. Live-smoke: признаки integrating

После stress/replay probe дать несколько шагов.

Искать:

```text
dream_pressure_delta < 0
relief_delta > 0
coherence_delta > 0
stress_delta < 0
```

Если один из этих признаков есть, verdict должен стать:

```text
integrating
```

Если replay идёт, но интеграции пока нет, verdict может быть:

```text
replaying
```

Это не ошибка.

---

## 15. Проверить read-only архитектуру

Replay Quality Monitor не должен:

```text
мутировать out
создавать replay seed
подмешивать dream probe
менять M11/M2/M5
отключать M3 guard
делать M1 raw sensors → M2
делать M13 → M5 напрямую
делать M4 → M5 напрямую
мутировать out["focus_context"]
```

Он должен быть только diagnostic/status layer.

---

## PASS criteria

PASS, если:

```text
✅ файл replay_quality_monitor_status.py есть
✅ status payload содержит replay_quality_monitor
✅ M8 UI содержит Replay Quality Monitor
✅ окно открывается
✅ raw JSON обновляется
✅ quality_score в диапазоне 0..1
✅ verdict корректный
✅ repeated status poll не стирает delta
✅ Probe stress виден в pressure/stress/quality
✅ Probe replay seed виден в M5 seed_gate/seed_norm
✅ старые sleep/replay tests проходят
✅ monitor read-only
```

---

## FAIL criteria

FAIL, если:

```text
нет replay_quality_monitor_status.py
нет status key replay_quality_monitor
кнопка M8 отсутствует
окно не открывается
raw JSON пустой при работающем status IPC
quality_score NaN/Inf/out of range
delta всегда затирается в 0
Probe replay seed не отражается в M5 seed fields
Replay Quality Monitor мутирует модель или out
сломался Sleep Replay Monitor / Dream Probe / M2 seed bus
```

---

## Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. Files inspected.
3. Compile results.
4. Pytest results.
5. Status payload result.
6. M8 UI result.
7. Live-smoke:
   - sleep without probe
   - Probe stress
   - Probe replay seed
   - integration signs
8. Any minimal fixes made.
9. Remaining risks.
```

Финальная строка:

```text
Replay Quality Monitor verified:
status payload, M8 UI, live probe response, replay seed visibility, and read-only architecture are confirmed.
```
