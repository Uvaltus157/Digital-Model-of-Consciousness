# Codex task: regression check после исправления Dream Probe / M11 deltas

## Цель

После исправления проблемы “Dream Probe даёт нули” проверить, что вся система осталась архитектурно правильной и live-debug теперь реально показывает изменения.

Нужно проверить не только сам Dream Probe, но и отсутствие регрессий в sleep/replay контуре:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13

M5 → M3, но во сне M3 наружу заблокирован
```

---

# 1. Проверить, что Dream Probe теперь виден в monitor

В live-режиме:

```text
Сон / replay mode ON
Sleep Replay Monitor открыт
```

Нажать:

```text
Probe curiosity
Probe stress
Probe replay seed
Probe mixed
```

Ожидается:

```text
dream_probe.active = True
dream_probe.kind = curiosity / stress / replay_seed / mixed
dream_probe.remaining уменьшается
dream_probe.pulse > 0
```

Для `Probe curiosity`:

```text
Δ curiosity != 0
change_score > 0
```

Для `Probe stress`:

```text
Δ stress != 0 OR Δ panic != 0
change_score > 0
M2 dream_pressure меняется
```

Для `Probe replay seed`:

```text
M5 seed_gate > 0
M5 seed_norm > 0
```

---

# 2. Проверить, что delta не затирается repeated status poll

Проверить файл:

```text
src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py
```

Если status вызывается несколько раз на одном `global_step`, monitor не должен сразу перезаписывать ненулевые дельты в ноль.

Проверить, что логика примерно такая:

```python
step = int(payload.get("global_step", 0))
last_step = int(state.get("last_step", -1))

if step == last_step:
    payload["m11_delta"] = state.get("last_delta", zero_delta)
    payload["m11_range"] = state.get("last_range", ...)
    payload["m11_activity"] = state.get("last_activity", ...)
    return
```

И что при новом step дельта пересчитывается.

---

# 3. Проверить, что M11 cache invalidation работает только для probe

В `DreamProbeRuntimeMixin.apply_dream_probe_to_out(...)` проверить, что после изменения `out` cache M11 сбрасывается:

```python
out.pop("emotion", None)
out.pop("affect", None)
```

или:

```python
out["emotion"]["_emotion_cache_reusable"] = False
```

Но не отключать cache глобально в `EmotionalDrive.compute(...)`.

Запрещено:

```text
удалить весь M11 cache
пересчитывать M11 дважды каждый обычный шаг
ломать conscious loop cache path
```

---

# 4. Проверить, что Dream Probe не нарушает архитектуру

Проверить:

```text
Probe curiosity / stress:
    меняет только pre-M11 inputs:
        out["values"]
        object_imagery confidence
        reflection confidence
        self_core confidence

Probe replay_seed:
    использует seed-bus:
        _event_dream_next_focus_seed
        _event_dream_next_focus_gate

НЕ ДЕЛАТЬ:
    raw M1 → M2
    out["focus_context"] = ...
    dream_context отдельным входом в M5
    M13 напрямую в M5
    M4 напрямую в M5
```

---

# 5. Проверить sleep/replay не сломан

В live:

```text
Сон / replay mode ON
```

Ожидается:

```text
M1:
    video_on = False
    contact_on = False
    imu_on = False

M3:
    sleep_blocked = True
    blocked_keys contains embodied_targets / hand_ctrl / leg_ctrl

M11:
    valence/arousal/stress/panic/curiosity numeric

M13:
    relevance/episodes numeric

M4:
    gate/stability/novelty numeric

M2:
    replay_gate/dream_pressure/event_salience numeric

M5:
    seed_gate/seed_norm/feedback_gate numeric
```

---

# 6. Проверить Clear probe

Нажать:

```text
Clear probe
```

Ожидается:

```text
dream_probe.active = False
dream_probe.remaining = 0
probe перестаёт подмешиваться
через несколько шагов deltas снова затухают
```

Проверить console:

```text
[dream_probe] cleared
```

---

# 7. Проверить, что sleep OFF возвращает normal mode

Нажать:

```text
Сон / replay mode OFF
```

Ожидается:

```text
video/contact/imu = ON
full_sleep = False
sensor_state = awake
M3 sleep_blocked = False или guard reason = awake_or_partial_sensor_cut
```

---

# 8. Проверить Module Lab после fixes

Запустить из M8 Module Lab:

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
ok = true
last_module_lab_result обновляется
```

---

# 9. Запустить compile checks

```bash
python -m py_compile src/modules/m02_event_dream_replay/dream_probe_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
python -m py_compile src/modules/m03_self_action_causality/action_runtime.py
python -m py_compile src/apps/life_runtime.py
python -m py_compile src/apps/runner.py
```

---

# 10. Запустить pytest

```bash
pytest tests/module_contracts/test_dream_probe_runtime_contract.py
pytest tests/module_contracts/test_m11_affect_visibility_monitor_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
pytest tests/module_contracts/test_sleep_motor_guard_contract.py
pytest tests/module_contracts/test_m02_runtime_seed_bus_contract.py
pytest tests/module_contracts/test_unconscious_behavioral_scenarios.py
pytest tests/module_contracts/test_unconscious_loop_contract.py
```

Потом полный набор:

```bash
pytest tests/module_contracts
```

---

# 11. Что считать PASS

PASS, если:

```text
✅ Dream Probe IPC доходит до runner
✅ request_dream_probe создаёт active state
✅ apply_dream_probe_to_out вызывается до M11
✅ M11 cache invalidated только при probe
✅ Δ curiosity/stress/panic видны
✅ same-step status poll не затирает delta в 0
✅ replay_seed probe даёт M5 seed_gate/seed_norm
✅ M3 остаётся blocked во сне
✅ M1 sensors остаются OFF во сне
✅ Clear probe выключает стимул
✅ sleep OFF возвращает awake
✅ module lab и behavioral scenarios проходят
```

---

# 12. Что считать FAIL

FAIL, если:

```text
Dream Probe опять даёт нули
M11 delta видна только в raw JSON, но не в label
delta появляется на мгновение и снова 0 на том же step
Probe stress не влияет на stress/panic
Probe replay seed не влияет на M5 seed_gate/seed_norm
sleep_motor_guard перестаёт блокировать M3
M2 снова мутирует focus_context напрямую
Module Lab перестаёт запускаться
```

---

# 13. Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. Что именно было исправлено после нулей.
3. Compile results.
4. Pytest results.
5. Live observations:
   - Probe curiosity
   - Probe stress
   - Probe replay seed
   - Clear probe
   - Sleep OFF/ON
6. Подтверждение отсутствия архитектурных регрессий.
7. Остались ли риски.
```

Финальная строка:

```text
Dream Probe regression check passed:
live probes produce visible M11/M2/M5 changes, same-step polling does not erase deltas, and sleep/replay architecture remains intact.
```
