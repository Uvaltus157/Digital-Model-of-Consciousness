# Codex task: проверить M5 Learning Quality Baseline и кнопку M8

## Цель

Проверить текущий репозиторий после добавления режима:

```text
M5 Learning Quality Baseline
```

и разобраться с кнопкой/стилем в M8, включая warning:

```text
[warn] style M5 learning quality: anchor not found
```

Важно: проверить не только backend status, но и UI-путь:

```text
module_status_runtime
↓
status IPC
↓
control_panel.py
↓
M8 button
↓
M5 Learning Quality window
↓
live refresh
```

---

## 0. Главное правило

Не переписывать архитектуру.

M5 Learning Quality Baseline должен быть **read-only diagnostic monitor**.

Он не должен:

```text
мутировать M5 output
создавать replay seed
подмешивать Dream Probe
обучать модель
менять out["focus_context"]
обходить FocusFeedbackBoundary
отключать M3 sleep guard
делать M1 raw sensors → M2
```

Он только читает runtime/status и показывает:

```text
обучается ли M5
есть ли loss trend
есть ли seed response
есть ли latent/object/identity proxy
```

---

## 1. Проверить наличие файлов

Проверить, что есть:

```text
src/modules/m08_debug_visual_control/m5_learning_quality_status.py
tests/module_contracts/test_m5_learning_quality_status_contract.py
docs/architecture/m5_learning_quality_baseline.md
```

Если файла нет — это FAIL.

---

## 2. Compile checks

Запустить:

```bash
python -m py_compile src/modules/m08_debug_visual_control/m5_learning_quality_status.py
python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
```

Ожидание:

```text
без ошибок
```

---

## 3. Pytest checks

Запустить:

```bash
pytest tests/module_contracts/test_m5_learning_quality_status_contract.py
pytest tests/module_contracts/test_replay_quality_monitor_status_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
pytest tests/module_contracts/test_dream_probe_runtime_contract.py
pytest tests/module_contracts/test_m02_runtime_seed_bus_contract.py
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

Проверить импорт:

```python
from src.modules.m08_debug_visual_control.m5_learning_quality_status import build_m5_learning_quality_status
```

Проверить status payload:

```python
"m5_learning_quality": build_m5_learning_quality_status(self),
```

Важно: новый ключ не должен удалить старые:

```text
sleep_replay_monitor
replay_quality_monitor
last_module_lab_result
input_sensors_enabled
sleep_sensor_mask
```

Если `replay_quality_monitor` ещё не установлен в текущей ветке — это не должно ломать M5 Learning Quality, но report должен отметить фактическое состояние.

---

## 5. Проверить m5_learning_quality_status.py

Функция:

```python
build_m5_learning_quality_status(system)
```

Должна быть read-only и читать:

```python
system.latest_out
system.last_train_loss
system.last_train_reason
system.last_train_error
system.train_steps
system.training_enabled
system.cfg.train.enabled
system.latest_object_decoder_stats
system.latest_long_dynamic_memory_stats
```

Проверить обязательные top-level keys:

```text
global_step
train_steps
training_enabled
cfg_train_enabled
full_sleep
verdict
learning_quality
learning_quality_ema
last_train_reason
last_train_error
m5_loss
m5_latent
m5_seed_response
object_identity_proxy
baseline
current
delta
history
samples
note
```

Проверить nested keys:

```text
m5_loss:
    train_loss
    train_loss_delta
    train_loss_from_baseline
    train_loss_trend
    prediction_error
    prediction_error_delta
    prediction_error_from_baseline
    prediction_error_trend
    reconstruction_error
    reconstruction_error_delta
    reconstruction_error_from_baseline
    reconstruction_error_trend

m5_latent:
    latent_coherence
    latent_coherence_delta
    latent_coherence_from_baseline
    focus_norm
    workspace_norm
    obs_embed_norm

m5_seed_response:
    seed_gate
    seed_norm
    feedback_gate
    seed_response
    seed_norm_delta

object_identity_proxy:
    object_recon
    object_recon_delta
    identity_stability
    identity_stability_delta
    identity_novelty
    identity_novelty_delta
```

---

## 6. Проверить delta/baseline logic

Проверить, что состояние хранится в:

```python
system._m5_learning_quality_state
```

И repeated status poll на том же кадре не затирает delta.

Ожидаемая логика:

```python
if last_step == step and last_train_steps == train_steps:
    delta = state["last_delta"]
else:
    delta = current - prev
    state["prev"] = current
    state["last_delta"] = delta
    state["last_step"] = step
    state["last_train_steps"] = train_steps
```

Проверить, что baseline фиксируется один раз:

```python
if not state["baseline"]:
    state["baseline"] = current
```

---

## 7. Проверить verdict logic

Допустимые verdict:

```text
untrained_or_no_data
idle
seed_reactive_untrained
tracking
improving
training_error
```

Ожидания:

```text
untrained_or_no_data:
    нет train_steps и нет loss

seed_reactive_untrained:
    M5 получает seed response, но обучения ещё нет

tracking:
    есть train_steps, но явного улучшения пока нет

improving:
    train_loss / prediction_error / reconstruction_error падает
    или latent_coherence растёт

training_error:
    last_train_error непустой
```

Проверить:

```text
0.0 <= learning_quality <= 1.0
0.0 <= learning_quality_ema <= 1.0
```

---

## 8. Проверить control_panel.py: M8 button

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

Проверить, что есть:

```python
self.btn_m5_learning_quality = QtWidgets.QPushButton("M5 Learning Quality")
```

Проверить, что кнопка есть в M8 tab:

```python
"btn_m5_learning_quality"
```

в `MODULE_TAB_BUTTONS["m8"]`.

Проверить signal:

```python
self.btn_m5_learning_quality.clicked.connect(self.open_m5_learning_quality_window)
```

Проверить runner-dependent controls:

```python
self.btn_m5_learning_quality
```

Проверить refresh call:

```python
self.refresh_m5_learning_quality_window()
```

в `refresh_ui()`.

---

## 9. Разобраться с warning style anchor

Был warning:

```text
[warn] style M5 learning quality: anchor not found
```

Нужно проверить, решён ли он.

В `refresh_ui()` должна быть строка стилизации:

```python
self._style_plain_status_button(
    self.btn_m5_learning_quality,
    bool((self.last_status or {}).get("m5_learning_quality", {})),
    "M5 Learning Quality",
)
```

Однострочный вариант тоже допустим:

```python
self._style_plain_status_button(self.btn_m5_learning_quality, bool((self.last_status or {}).get("m5_learning_quality", {})), "M5 Learning Quality")
```

Если строки нет — добавить её рядом с:

```text
Replay Quality Monitor style
Sleep Replay Monitor style
Module Lab style
Module Debug style
```

Пример допустимого места:

```python
self._style_button(self.btn_module_debug, s.module_debug, "Module debug")
self._style_plain_status_button(self.btn_module_lab, False, "Module Lab")
self._style_plain_status_button(self.btn_sleep_replay_monitor, self._sleep_mode_active(), "Sleep Replay Monitor")
self._style_plain_status_button(
    self.btn_m5_learning_quality,
    bool((self.last_status or {}).get("m5_learning_quality", {})),
    "M5 Learning Quality",
)
```

Если `btn_replay_quality_monitor` есть, лучше ставить после него.

---

## 10. Проверить окно M5 Learning Quality

Должен быть метод:

```python
open_m5_learning_quality_window(...)
```

Окно должно показывать:

```text
Training / baseline:
    verdict
    learning_quality
    learning_quality_ema
    training_enabled
    cfg_train_enabled
    last_train_reason
    last_train_error

M5 losses:
    train_loss
    Δ train_loss
    train trend
    prediction_error
    Δ prediction
    reconstruction_error
    Δ reconstruction

M5 latent / seed response:
    latent_coherence
    Δ coherence
    focus_norm
    workspace_norm
    obs_embed_norm
    seed_gate
    seed_norm
    feedback_gate
    seed_response

Object / identity proxy:
    object_recon
    Δ object_recon
    identity_stability
    Δ stability
    identity_novelty
    Δ novelty
```

Raw JSON должен показывать:

```python
last_status["m5_learning_quality"]
```

---

## 11. Live-smoke

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
M8 tab содержит M5 Learning Quality
кнопка кликается
окно открывается
header показывает step / train_steps / verdict / q / ema
raw JSON обновляется
```

---

## 12. Live-smoke без обучения

Если M5 ещё не обучается, допустимо:

```text
verdict = untrained_or_no_data
или seed_reactive_untrained
или idle
train_loss = 0
prediction_error = 0
reconstruction_error = 0
```

Это не ошибка.

Главное, чтобы monitor честно показывал:

```text
метрик нет / обучения нет
```

и не утверждал, что M5 понимает мир.

---

## 13. Live-smoke с Dream Probe / replay seed

Если есть Sleep Replay Monitor и Dream Probe:

```text
Сон / replay mode ON
Probe replay seed
```

В M5 Learning Quality смотреть:

```text
seed_gate
seed_norm
feedback_gate
seed_response
```

Ожидаемо:

```text
seed_gate > 0
seed_norm > 0
seed_response > 0
```

Если seed_response остаётся 0, проверить:

```text
event_dream_replay.next_focus_context_seed
event_dream_replay.replay_context
event_dream_replay.next_focus_context_seed_gate
event_dream_replay.replay_gate
focus_feedback.total_gate
attention.focus_feedback_gate
```

---

## 14. Live-smoke с обучением

Если training loop доступен:

```text
включить Online training
дать несколько train steps
```

Ожидаемо:

```text
train_steps растёт
train_loss обновляется
last_train_reason обновляется
```

Если модель обучается и loss падает:

```text
train_loss_delta < 0
prediction_error_delta < 0, если metric доступен
reconstruction_error_delta < 0, если metric доступен
latent_coherence_delta > 0, если metric доступен
verdict = improving или tracking
```

Если train_steps растёт, но loss не меняется:

```text
verdict может быть tracking
```

Это допустимо.

---

## 15. Проверить read-only архитектуру

M5 Learning Quality не должен:

```text
вызывать optimizer.step()
запускать train_once()
менять self.model
менять out
создавать replay seed
обнулять моторы
менять sensors
изменять out["focus_context"]
```

Он должен только читать и хранить собственный status state:

```python
system._m5_learning_quality_state
```

---

## 16. PASS criteria

PASS, если:

```text
✅ m5_learning_quality_status.py есть
✅ test_m5_learning_quality_status_contract.py проходит
✅ module_status_runtime отдаёт m5_learning_quality
✅ control_panel имеет кнопку M5 Learning Quality в M8
✅ warning style anchor устранён или не влияет
✅ кнопка стилизуется в refresh_ui
✅ окно открывается
✅ raw JSON обновляется
✅ learning_quality в диапазоне 0..1
✅ verdict корректный
✅ repeated status poll не стирает delta
✅ seed_response виден после replay seed probe
✅ monitor read-only
```

---

## 17. FAIL criteria

FAIL, если:

```text
нет m5_learning_quality_status.py
нет status key m5_learning_quality
кнопка M8 отсутствует
окно не открывается
raw JSON пустой при работающем status IPC
style warning остался и кнопка не refresh-ится
learning_quality NaN/Inf/out of range
delta всегда затирается в 0
seed_response не виден при replay seed
monitor мутирует модель или out
сломался Sleep Replay Monitor / Replay Quality Monitor / Dream Probe
```

---

## 18. Что можно исправлять

Исправлять минимально:

```text
missing import
missing status key
wrong control_panel anchor
button not in M8
button not in runner-dependent list
missing style line
missing refresh call
wrong label path
compile error
test shape/key mismatch
```

Не переписывать архитектуру.

---

## 19. Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. Files inspected.
3. Compile results.
4. Pytest results.
5. Status payload result.
6. M8 UI/button result.
7. Was style anchor warning fixed?
8. Live-smoke:
   - without training
   - with replay seed probe
   - with training if available
9. Any minimal fixes made.
10. Remaining risks.
```

Финальная строка:

```text
M5 Learning Quality Baseline verified:
status payload, M8 button/window, style refresh, seed response, learning metrics, and read-only architecture are confirmed.
```
