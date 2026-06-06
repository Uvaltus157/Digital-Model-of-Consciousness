# Codex task: диагностировать почему Dream Probe даёт нули в Sleep Replay Monitor

## Симптом

В live-режиме после добавления Dream Probe / Replay Probe и нажатия кнопок:

```text
Probe curiosity
Probe stress
Probe replay seed
Probe mixed
```

в Sleep Replay Monitor всё равно видны нули:

```text
Δ stress = 0
Δ panic = 0
Δ curiosity = 0
change_score = 0
M5 seed_gate / seed_norm может быть 0
dream_probe может не отображаться или active=false
```

Нужно найти, где рвётся цепочка.

---

# Целевая цепочка

Кнопка в M8 должна пройти полный путь:

```text
M8 Sleep Replay Monitor button
↓
make_action_message("dream_probe_inject", kind=..., intensity=..., duration=...)
↓
IPC
↓
ActionRuntimeMixin.apply_ipc_action(...)
↓
DreamProbeRuntimeMixin.request_dream_probe(...)
↓
self._dream_probe_state active
↓
life_runtime.py
    apply_dream_probe_to_out(out, obs)
    BEFORE self.emotional_drive.compute(out, obs)
↓
M11 EmotionalDrive sees changed out["values"] / confidence fields
↓
out["emotion"], out["affect"] change
↓
M2 computes dream_pressure/replay_gate
↓
latest_out contains dream_probe + affect + event_dream_replay
↓
module_status_runtime builds sleep_replay_monitor
↓
control_panel Sleep Replay Monitor refreshes labels
```

---

# 1. Verify files exist

Проверить наличие файлов:

```text
src/modules/m02_event_dream_replay/dream_probe_runtime.py
tests/module_contracts/test_dream_probe_runtime_contract.py
docs/architecture/dream_probe_live_stimulus.md
```

Проверить, что файл компилируется:

```bash
python -m py_compile src/modules/m02_event_dream_replay/dream_probe_runtime.py
```

---

# 2. Verify runner mixin is installed

Файл:

```text
src/apps/runner.py
```

Проверить импорт:

```python
from src.modules.m02_event_dream_replay.dream_probe_runtime import DreamProbeRuntimeMixin
```

Проверить наследование:

```python
DreamProbeRuntimeMixin
```

в `class UnifiedSystem(...)`.

Если миксин не подключён, `hasattr(self, "request_dream_probe")` будет False, и IPC-команда не будет работать.

---

# 3. Verify IPC action handler

Файл:

```text
src/modules/m03_self_action_causality/action_runtime.py
```

Проверить, что в `apply_ipc_action(...)` есть обработчик:

```python
elif action in ("dream_probe_inject", "dream_probe_clear"):
    if hasattr(self, "request_dream_probe"):
        if action == "dream_probe_clear":
            payload = {"kind": "clear", **dict(payload or {})}
        self.request_dream_probe(payload)
    else:
        print("[dream_probe] ignored: DreamProbeRuntimeMixin is not installed")
```

Если нет — добавить.

Также добавить диагностический print при любом получении action:

```python
print(f"[dream_probe][ipc] action={action} payload={payload}")
```

но только для `dream_probe_inject/dream_probe_clear`.

Ожидаемый console log после клика:

```text
[dream_probe][ipc] action=dream_probe_inject payload={'kind': 'curiosity', ...}
[dream_probe] requested kind=curiosity intensity=... duration=...
```

Если первого лога нет — кнопка/IPC не отправляет action.

Если первый есть, но второго нет — `request_dream_probe` не подключён.

---

# 4. Verify M8 buttons send the correct action

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

Проверить наличие метода:

```python
request_sleep_replay_probe(...)
```

Он должен отправлять:

```python
self.send(make_action_message(
    "dream_probe_inject",
    kind=payload_kind,
    intensity=float(intensity),
    duration=int(duration),
    source="m8_sleep_replay_monitor",
))
```

Проверить кнопки в `open_sleep_replay_monitor_window()`:

```python
btn_probe_curiosity.clicked.connect(lambda: self.request_sleep_replay_probe("curiosity", 0.85, 80))
btn_probe_stress.clicked.connect(lambda: self.request_sleep_replay_probe("stress", 0.85, 80))
btn_probe_replay.clicked.connect(lambda: self.request_sleep_replay_probe("replay_seed", 0.75, 60))
btn_probe_mixed.clicked.connect(lambda: self.request_sleep_replay_probe("mixed", 0.75, 80))
btn_probe_clear.clicked.connect(lambda: self.request_sleep_replay_probe("clear", 0.0, 1))
```

Проверить, что кнопки реально находятся в окне Sleep Replay Monitor.

Если кнопки есть, но при клике в console нет `[dream_probe][ipc]`, значит проблема в UI signal или `self.send(...)`.

---

# 5. Verify request_dream_probe state

Файл:

```text
src/modules/m02_event_dream_replay/dream_probe_runtime.py
```

Проверить `request_dream_probe(...)`.

После вызова должно быть:

```python
self._dream_probe_state = {
    "active": True,
    "kind": kind,
    "remaining": duration,
    "duration": duration,
    "intensity": intensity,
    ...
}
```

Добавить временный/постоянный debug print:

```python
print(f"[dream_probe][state] {self._dream_probe_state}")
```

Ожидаемый log:

```text
[dream_probe] requested kind=curiosity intensity=0.850 duration=80 source=m8_sleep_replay_monitor
[dream_probe][state] {'active': True, 'kind': 'curiosity', 'remaining': 80, ...}
```

Если state не active — исправить.

---

# 6. Verify apply_dream_probe_to_out is called in life_runtime

Файл:

```text
src/apps/life_runtime.py
```

Проверить, что перед M11 есть:

```python
if hasattr(self, "apply_dream_probe_to_out"):
    try:
        out = self.apply_dream_probe_to_out(out, obs)
    except Exception as e:
        ...
emotion = self.emotional_drive.compute(out, obs)
```

Критично:

```text
apply_dream_probe_to_out(...)
ДОЛЖЕН быть ДО emotional_drive.compute(...)
```

Если он стоит после M11 — M11 не увидит probe в текущем шаге.

Добавить debug print внутри `apply_dream_probe_to_out(...)`:

```python
print(
    f"[dream_probe][apply] step={getattr(self,'global_step',0)} "
    f"kind={kind} pulse={pulse:.4f} remaining={state['remaining']}"
)
```

Ожидаемый log после клика:

```text
[dream_probe][apply] step=123 kind=curiosity pulse=0.8500 remaining=79
[dream_probe][apply] step=124 kind=curiosity pulse=0.8394 remaining=78
...
```

Если `[dream_probe][apply]` не появляется, значит:
- mixin не подключён;
- life_runtime anchor не вставлен;
- method name mismatch;
- `apply_dream_probe_to_out` не callable.

---

# 7. Verify out is actually changed before M11

В `apply_dream_probe_to_out(...)` проверить эффекты:

## Curiosity probe

Должно сделать:

```python
out["values"]["curiosity"] >= pulse
out["dream_probe"]["kind"] == "curiosity"
```

Добавить print:

```python
print(
    f"[dream_probe][values] curiosity={_scalar(values.get('curiosity')):.4f} "
    f"coherence={_scalar(values.get('coherence')):.4f}"
)
```

## Stress probe

Должно снизить:

```python
out["values"]["coherence"]
out["object_imagery"]["object_confidence"]
out["preconscious_reflection_out"]["model_confidence"]
out["self_core"]["self_confidence"]
```

Добавить print:

```python
print(
    f"[dream_probe][stress_inputs] coherence={...} "
    f"object_conf={...} model_conf={...} self_conf={...}"
)
```

Если эти значения изменяются, но M11 delta всё равно 0 — проблема в M11/cache/status/monitor.

---

# 8. Check M11 cache issue

Файл:

```text
src/modules/m11_motivational_homeostasis/emotional_drive_bivalent.py
```

В `EmotionalDrive.compute(...)` в начале есть cache path:

```python
cached = out.get("emotion")
if isinstance(cached, dict) and bool(cached.get("_emotion_cache_reusable", False)):
    return cached
```

Проверить, нет ли в `out["emotion"]` уже cached emotion **до** sleep/replay M11 compute.

Если `out["emotion"]` существует до `apply_dream_probe_to_out(...)`, M11 может вернуть старый cached affect, и probe не будет виден.

Варианты минимального исправления:

### Вариант A: invalidate cache inside dream probe

В `apply_dream_probe_to_out(...)`, после изменения out:

```python
if isinstance(out.get("emotion"), dict):
    out.pop("emotion", None)
if isinstance(out.get("affect"), dict):
    out.pop("affect", None)
```

или мягче:

```python
if isinstance(out.get("emotion"), dict):
    out["emotion"]["_emotion_cache_reusable"] = False
```

### Вариант B: before life_runtime M11 compute

После `apply_dream_probe_to_out(...)`:

```python
if isinstance(out.get("dream_probe"), dict):
    out.pop("emotion", None)
    out.pop("affect", None)
```

Предпочтительно A, потому что probe сам отвечает за invalidation.

После фикса ожидается:

```text
Probe curiosity -> M11 curiosity_latent changes
Probe stress -> M11 stress_latent / panic_latent changes
```

---

# 9. Check whether latest_out receives updated out

Файл:

```text
src/apps/life_runtime.py
```

Сейчас может быть:

```python
self.latest_out = out
...
emotion = self.emotional_drive.compute(out, obs)
out["emotion"] = emotion
...
compute_event_dream_replay(...)
```

Если `latest_out` — ссылка на тот же dict, это нормально.  
Но если где-то later code replaces `out` with a new dict, status может читать старое.

Проверить live:

```python
print(
    "[dream_probe][latest_out] "
    f"has_probe={isinstance(getattr(self,'latest_out',{}).get('dream_probe'), dict)} "
    f"has_affect={isinstance(getattr(self,'latest_out',{}).get('affect'), dict)}"
)
```

Если `out["dream_probe"]` есть, но `latest_out["dream_probe"]` нет, то нужно после M2/trace обновлять:

```python
self.latest_out = out
```

лучше сразу после:

```python
self.maybe_print_unconscious_loop_trace(out, obs)
```

или после M2 block.

---

# 10. Verify sleep_replay_monitor_status includes dream_probe and fresh affect

Файл:

```text
src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py
```

Проверить:

```python
out = getattr(system, "latest_out", {}) or {}
```

Проверить, что payload содержит:

```python
"dream_probe": dict(out.get("dream_probe", getattr(system, "_dream_probe_state", {}) or {}) or {})
```

Проверить, что M11 берётся из свежего:

```python
affect = out.get("affect")
emotion = out.get("emotion")
```

Проверить, что `m11_delta` считается между последовательными status calls.

Если status вызывается несколько раз на один и тот же `global_step`, delta может становиться 0 сразу после первого poll.

Нужно добавить защиту: обновлять delta только если `global_step` изменился.

В `_attach_m11_visibility(...)` добавить:

```python
step = int(payload.get("global_step", 0))
last_step = int(state.get("last_step", -1))
if step == last_step:
    # do not overwrite prev/delta with same frame
    # keep last_delta / last_activity
    payload["m11_delta"] = state.get("last_delta", zero_delta)
    payload["m11_range"] = state.get("last_range", ...)
    payload["m11_activity"] = state.get("last_activity", ...)
    return
state["last_step"] = step
```

И сохранять:

```python
state["last_delta"] = payload["m11_delta"]
state["last_range"] = payload["m11_range"]
state["last_activity"] = payload["m11_activity"]
```

Это важно: если status poll вызывается чаще, чем life step, первый вызов считает delta, второй тут же переписывает delta нулём на том же кадре. Тогда UI почти всегда видит 0.

Это очень вероятная причина “опять нули”.

---

# 11. Verify monitor UI reads new fields

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

Проверить, что Sleep Replay Monitor содержит rows:

```text
Δ stress       -> m11_delta.stress
Δ panic        -> m11_delta.panic
Δ curiosity    -> m11_delta.curiosity
trend          -> m11_activity.trend
change_score   -> m11_activity.change_score
```

Проверить, что Dream probe box содержит:

```text
active      -> dream_probe.active
kind        -> dream_probe.kind
remaining   -> dream_probe.remaining
pulse       -> dream_probe.pulse
intensity   -> dream_probe.intensity
```

Если status raw JSON показывает изменения, но labels нет — проблема в UI mapping.

---

# 12. Add direct debug script if needed

Если live тяжело проверять, добавить скрипт:

```text
scripts/module_lab/debug_dream_probe_path.py
```

Он должен:

```python
1. создать dummy system with DreamProbeRuntimeMixin
2. request_dream_probe(kind="curiosity")
3. apply_dream_probe_to_out(out)
4. call EmotionalDrive.compute(out, obs)
5. print curiosity_latent before/after
6. request_dream_probe(kind="stress")
7. call EmotionalDrive.compute(...)
8. print stress_latent/panic_latent before/after
9. request replay_seed
10. verify _event_dream_next_focus_seed exists
```

Запуск:

```bash
python scripts/module_lab/debug_dream_probe_path.py
```

Ожидаемо:

```text
curiosity_before < curiosity_after
stress_after or panic_after changes for stress probe
replay_seed_gate > 0
```

---

# 13. Required tests

Запустить:

```bash
python -m py_compile src/modules/m02_event_dream_replay/dream_probe_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
python -m py_compile src/modules/m03_self_action_causality/action_runtime.py
python -m py_compile src/apps/life_runtime.py
python -m py_compile src/apps/runner.py
```

Запустить:

```bash
pytest tests/module_contracts/test_dream_probe_runtime_contract.py
pytest tests/module_contracts/test_m11_affect_visibility_monitor_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
```

Если добавлен debug script:

```bash
python scripts/module_lab/debug_dream_probe_path.py
```

---

# 14. Live check после фикса

Запустить runner и M8 control panel.

Включить:

```text
Сон / replay mode
```

Открыть:

```text
Sleep Replay Monitor
```

Нажать:

```text
Probe curiosity
```

Ожидается:

```text
console:
[dream_probe][ipc] action=dream_probe_inject ...
[dream_probe] requested kind=curiosity ...
[dream_probe][apply] step=... kind=curiosity pulse=...
[dream_probe][values] curiosity=...

monitor:
dream_probe.active = True
dream_probe.kind = curiosity
dream_probe.remaining decreases
dream_probe.pulse > 0
Δ curiosity != 0
change_score > 0
```

Нажать:

```text
Probe stress
```

Ожидается:

```text
monitor:
dream_probe.kind = stress
Δ stress != 0 OR Δ panic != 0
change_score > 0
M2 dream_pressure changes
```

Нажать:

```text
Probe replay seed
```

Ожидается:

```text
monitor:
dream_probe.kind = replay_seed
M5 seed_gate > 0
M5 seed_norm > 0
```

---

# 15. What to fix

Исправлять только реальные причины:

```text
button not sending IPC
action_runtime missing handler
DreamProbeRuntimeMixin not inherited
apply_dream_probe_to_out not called before M11
M11 emotion cache not invalidated after probe
latest_out not updated after probe/M11/M2
status delta overwritten by repeated polls on same global_step
monitor labels mapped to wrong keys
replay_seed not written into _event_dream_next_focus_seed
```

Не переписывать архитектуру.

---

# 16. What not to do

Не делать:

```text
не делать M1 → M2 напрямую
не мутировать out["focus_context"] напрямую
не отключать sleep_motor_guard
не отключать M11 cache глобально, только invalidate при probe
не делать permanent emotion amplification в M11
не превращать probe в training fixture
не подключать MuJoCo fake sensors
```

---

# 17. Expected final report

В конце написать:

```text
1. Что было причиной нулей:
   - IPC не доходил?
   - apply не вызывался?
   - M11 cache?
   - latest_out stale?
   - status delta overwritten by same-step polling?
   - UI labels wrong?
2. Какие файлы проверены.
3. Какие минимальные фиксы внесены.
4. Compile results.
5. Pytest results.
6. Debug script result, если добавлен.
7. Live check:
   Probe curiosity -> observed values
   Probe stress -> observed values
   Probe replay seed -> observed values
8. Остались ли риски.
```

Финальная строка должна быть:

```text
Dream Probe live path confirmed:
M8 IPC reaches runner, probe modifies pre-M11 inputs, M11 deltas become visible, replay seed reaches M5, and monitor no longer overwrites nonzero deltas with same-step zero polls.
```
