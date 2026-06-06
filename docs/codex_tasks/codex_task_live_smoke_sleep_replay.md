# Codex task: live-smoke проверка sleep/replay в реальном запуске DMoC

## Цель

Провести **ручную live-smoke проверку** в реальном запуске DMoC, а не только unit/contract tests.

Нужно подтвердить, что при включении **Сон / replay mode** через M8-пульт реально работает полный бессознательный sleep/replay-контур:

```text
M1 sensors cut
↓
M5 world model / inner playback
↓
M11 affect
↓
M13 autobiographical retrieval + M4 dynamic identity
↓
M2 event/dream replay selector
↓
next_focus_context_seed + next_focus_context_seed_gate
↓
M5 FocusFeedbackBoundary
↓
M5 inner playback на следующем шаге

M5 → M3 action proposal
но во сне:
M3/body external motor output blocked
```

Итоговая архитектура:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13

M5 → M3 → body/world → M1
```

Во сне должно быть:

```text
video/contact/imu OFF
M1 external sensors zeroed
M3 external motor output zeroed
M5 internal action imagination allowed
M11/M13/M4/M2 active
M2 replay seed enters M5 through common seed bus
```

---

# 0. Важно

Это **live-smoke**, не переписывание архитектуры.

Не надо заново проектировать модули.  
Нужно запустить, включить режим сна, посмотреть логи/окна, проверить фактические runtime-состояния, минимально исправить реальные ошибки.

---

# 1. Перед запуском выполнить быстрые проверки

Запустить:

```bash
python -m py_compile src/apps/runner.py
python -m py_compile src/apps/life_runtime.py
python -m py_compile src/apps/unified_conscious_viewer.py

python -m py_compile src/modules/m06_learning_sleep_consolidation/sleep_sensors.py
python -m py_compile src/modules/m03_self_action_causality/sleep_motor_guard.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_runtime.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_replay.py
python -m py_compile src/modules/m02_event_dream_replay/unconscious_loop_trace.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
python -m py_compile src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py
python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py
```

Потом:

```bash
pytest tests/module_contracts/test_sleep_entry_zero_prev_motor_contract.py
pytest tests/module_contracts/test_sleep_motor_guard_contract.py
pytest tests/module_contracts/test_m02_runtime_seed_bus_contract.py
pytest tests/module_contracts/test_unconscious_loop_trace_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
```

Если что-то падает — исправить минимально и снова запустить.

---

# 2. Запустить runner

Из корня проекта:

```bash
python -m src.apps.runner
```

Если проект запускается иначе в этом репозитории, использовать фактическую команду запуска runner-а, но обязательно сохранить в отчёте точную команду.

Ожидается:

```text
runner стартует
IPC server поднят
module status IPC доступен
ошибок import/compile нет
```

---

# 3. Запустить PyQt control panel

Открыть пульт:

```bash
python src/modules/m08_debug_visual_control/control_panel.py
```

или фактическую команду проекта для `pyqt_control_panel_ipc.py`, если она используется.

Проверить:

```text
STATUS IPC: receiving
step увеличивается
кнопки M8 доступны
```

---

# 4. Проверить M8 вкладку

Во вкладке M8 должны быть доступны:

```text
Module Lab
Sleep Replay Monitor
```

Также должна быть кнопка:

```text
Сон / replay mode
```

Если `Сон / replay mode` расположен не внутри M8, а в верхней sensor row — это допустимо, но в отчёте указать фактическое место.

---

# 5. Открыть Sleep Replay Monitor

Нажать:

```text
Sleep Replay Monitor
```

Ожидается отдельное окно:

```text
M8 Sleep Replay Monitor
```

В нём должны быть секции:

```text
M1 sensors
M11 affect
M13 autobiographical retrieval
M4 dynamic identity
M2 event/dream replay
M5 seed + M3 guard
raw JSON area
```

До включения сна допустимо:

```text
state=awake
full_sleep=0
m3 sleep_blocked=OFF
m2 replay_gate может быть 0
m5 seed_gate может быть 0
```

---

# 6. Включить Сон / replay mode

В пульте нажать:

```text
Сон / replay mode
```

Ожидается, что UI переключает sensor gates:

```text
Video OFF
Tactile/Contact OFF
IMU OFF
```

В console runner должен появиться лог:

```text
[ipc][sleep_sensors] video=OFF contact=OFF imu=OFF state=sleep
```

Также должен появиться лог входа в сон:

```text
[sleep_replay] zeroed previous motor tail on sleep entry: ...
```

Если второй лог не появился, проверить, действительно ли был переход:

```text
awake/partial_cut → sleep
```

Если runner уже был в sleep до клика, reset может не сработать повторно — это нормально. Для проверки можно сначала выключить сон, потом снова включить.

---

# 7. Проверить M1 sensors cut

В Sleep Replay Monitor должно быть:

```text
M1 sensors:
    state = sleep
    video_on = OFF / False
    contact_on = OFF / False
    imu_on = OFF / False
```

В status/raw JSON должно быть:

```json
"full_sleep": true,
"sensor_state": "sleep",
"input_sensors_enabled": {
  "video": false,
  "contact": false,
  "imu": false
},
"sleep_sensor_mask": {
  "video": true,
  "contact": true,
  "imu": true
}
```

Если есть доступ к debug status payload, проверить:

```text
sleep_replay_monitor.m1.video_on == false
sleep_replay_monitor.m1.contact_on == false
sleep_replay_monitor.m1.imu_on == false
```

---

# 8. Проверить M3 motor block

В Sleep Replay Monitor должно быть:

```text
M3:
    sleep_blocked = ON / True
    blocked_norm >= 0
    blocked_keys contains:
        embodied_targets
        hand_ctrl
        leg_ctrl
```

В console trace должно быть:

```text
m3_sleep_block=1
```

Если `m3_sleep_block=0` при `state=sleep`, это ошибка.

Также проверить, что во сне external motor не исполняется:

```text
latest_embodied should be near zero
latest_hand_ctrl should be near zero
actuator ctrl samples should not receive fresh nonzero sleep action
```

Допустимо, что MuJoCo физика ещё имеет инерцию/остаточное положение, но новые motor commands должны быть заблокированы.

---

# 9. Проверить M11 active

В Sleep Replay Monitor секция M11 должна обновляться:

```text
valence
arousal
stress
panic
curiosity
```

Значения могут быть малыми, но должны быть числовыми, не NaN/Inf.

В console trace должно быть:

```text
m11: val=... ar=... stress=... panic=... cur=...
```

Если M11 пустой или все поля отсутствуют — проверить порядок в `life_runtime.py`.

---

# 10. Проверить M13 active

В Sleep Replay Monitor секция M13 должна показывать:

```text
relevance
episodes
summary
```

Ожидаемо:

```text
episodes >= 0
relevance числовой
summary может быть пустой в начале, но не должен ломать monitor
```

В console trace должно быть:

```text
m13: rel=... eps=...
```

Если M13 не заполняется, проверить:

```python
self.compute_autobiographical_retrieval(obs, out)
```

должен вызываться после M11 и до M2.

---

# 11. Проверить M4 active

В Sleep Replay Monitor секция M4 должна показывать:

```text
token
gate
stability
novelty
sentence
```

Ожидаемо:

```text
gate числовой
stability числовой
novelty числовой
token может быть пустой в начале, но при объектной памяти должен появиться
```

В console trace должно быть:

```text
m4: token=... gate=... stab=... nov=...
```

Если M4 пустой, проверить:

```python
out["inner_object"] = self.compute_inner_object_image(obs, out)
self._compute_long_dynamic_memory(obs, out)
```

и что M4 кладёт:

```python
out["long_dynamic_memory"]
```

---

# 12. Проверить M2 active

В Sleep Replay Monitor секция M2 должна показывать:

```text
replay_gate
should_replay
dream_pressure
event_salience
source
identity
```

В console trace должно быть:

```text
m2: gate=... should=... pressure=... sal=... src=...
```

Ожидаемо:

```text
dream_pressure числовой
event_salience числовой
replay_gate числовой
source не обязан быть всегда непустым, но при replay должен указывать источник
```

Если M2 пустой, проверить:

```python
self.compute_event_dream_replay(obs, out)
```

должен вызываться после M13/M4/M11.

---

# 13. Проверить M2 → M5 replay seed

В Sleep Replay Monitor секция M5 должна показывать:

```text
seed_gate
seed_norm
feedback_gate
```

Ожидаемо во сне:

```text
seed_gate >= 0
seed_norm >= 0
```

Если M2 активен и should_replay=1, seed_gate должен стать заметным.

В console trace должно быть:

```text
m5_seed: gate=... norm=... fb=...
```

Проверить в коде при необходимости:

```python
M2:
    packet["next_focus_context_seed"]
    packet["next_focus_context_seed_gate"]
    packet["target_m5_boundary"]

model_step:
    get_m5_focus_seed(stage=model_stage)
    self.model.step(... focus_context_seed=..., focus_context_seed_gate=...)
```

Если M2 gate есть, но M5 seed всегда 0 — проверить `apply_stage`, `seed_only_in_sleep`, `is_full_sleep_mode`.

---

# 14. Проверить unconscious_loop_trace

В console должны идти строки вида:

```text
[unconscious_loop step=N] sleep=1 state=sleep |
m11: val=... ar=... stress=... panic=... cur=... |
m13: rel=... eps=... |
m4: token=... gate=... stab=... nov=... |
m2: gate=... should=... pressure=... sal=... src=... |
m5_seed: gate=... norm=... fb=... |
m3_sleep_block=1
```

Если trace не печатается, проверить config:

```text
event_dream_replay.unconscious_trace_enabled
event_dream_replay.unconscious_trace_every_steps
```

и вызов:

```python
self.maybe_print_unconscious_loop_trace(out, obs)
```

---

# 15. Проверить Module Lab из M8

Открыть M8:

```text
Module Lab
```

Нажать по очереди:

```text
Run M2 test
Run M4 test
Run M11 test
Run M13 test
Run M5Boundary test
Run unconscious loop test
Run behavioral scenarios
Run all
```

Ожидается:

```text
last_module_lab_result обновляется
ok = true
kind = module_lab / behavioral_scenarios
```

Если окно не обновляет результат, проверить:

```text
module_lab_run IPC action
last_module_lab_result in status IPC
refresh_module_lab_window()
```

---

# 16. Выключить Сон / replay mode

Нажать:

```text
Сон / replay mode
```

Ожидается:

```text
video=ON contact=ON imu=ON state=awake
```

Monitor должен показать:

```text
full_sleep=0
M1 video_on/contact_on/imu_on = True
M3 sleep_blocked = False или перестаёт быть active
```

Console:

```text
[ipc][sleep_sensors] video=ON contact=ON imu=ON state=awake
```

---

# 17. Проверить повторное включение сна

Снова включить:

```text
Сон / replay mode
```

Ожидается повторный переход:

```text
awake → sleep
```

и снова:

```text
[sleep_replay] zeroed previous motor tail on sleep entry
m3_sleep_block=1
```

Это подтверждает, что reset previous motor tail срабатывает на каждом новом входе в full sleep.

---

# 18. Что считать PASS

Live-smoke считается пройденным, если подтверждено:

```text
✅ runner стартует
✅ PyQt control panel подключается к status IPC
✅ M8 Sleep Replay Monitor открывается
✅ Сон / replay mode включает video/contact/imu OFF
✅ status shows full_sleep=True and sensor_state=sleep
✅ M1 sensor gates show all external sensors off
✅ previous motor tail reset log appears on sleep entry
✅ M3 sleep_motor_guard blocks external motor output
✅ unconscious_loop_trace prints sleep=1 and m3_sleep_block=1
✅ M11 values update
✅ M13 values update
✅ M4 values update
✅ M2 values update
✅ M5 seed values visible
✅ Module Lab window can run all listed tests
✅ turning sleep OFF returns sensor_state=awake
✅ turning sleep ON again resets previous motor tail again
```

---

# 19. Что считать FAIL

Live-smoke failed, если:

```text
runner не стартует
control panel не подключается к status IPC
Sleep Replay Monitor не открывается
Сон / replay mode меняет только UI, но не runner state
full_sleep не становится True
M1 sensors не режутся
prev_embodied_action / prev_hand_motor не зануляются при входе в сон
M3 sleep_blocked остаётся False в sleep
unconscious_loop_trace не появляется
M11/M13/M4/M2 пустые или не обновляются
M2 работает, но M5 seed всегда не получает replay seed
Module Lab IPC не работает
сон выключается, но sensor state остаётся sleep
```

---

# 20. Минимальные исправления, если что-то падает

Исправлять только по факту:

```text
import error
compile error
button not connected
status key missing
IPC action not handled
status payload not updated
wrong field name
wrong method name
guard called after motor execution
trace not called
monitor reads wrong status path
```

Не переписывать архитектуру.

---

# 21. Финальный отчёт Codex

В конце дать отчёт:

```text
1. Commit/hash checked.
2. Commands run.
3. Compile results.
4. Pytest smoke results.
5. Runner launch command.
6. Control panel launch command.
7. Sleep ON evidence:
   - console lines
   - status fields
   - monitor fields
8. Sleep OFF evidence.
9. M1/M11/M13/M4/M2/M5/M3 observations.
10. Module Lab observations.
11. Any fixes made.
12. Remaining risks.
```

Финальная строка должна быть:

```text
Live-smoke sleep/replay confirmed:
M1 sensors cut, M11/M13/M4/M2 active, M2 seed reaches M5, and M3 external motor output is blocked in full sleep.
```
