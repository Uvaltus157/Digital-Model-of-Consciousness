# Codex task: проверить sleep/replay live-контур и модульную отладку DMoC

## Цель

Проверить текущий репозиторий DMoC после последних изменений и убедиться, что режим **сон / replay live** работает строго по архитектуре:

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

Итоговая схема:

```text
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5
      │           ↑
      ↓           │
     M4 ──────────┤
                 M13

M5 → M3 → body/world → M1
```

Во сне:

```text
video/contact/imu OFF
M1 external sensors zeroed
M3 external motor output zeroed
M5 internal action imagination allowed
M11/M13/M4/M2 active
M2 replay seed enters M5 through common seed bus
```

---

# Важные invariants

Проверить и подтвердить:

```text
1. M1 не идёт напрямую в M2.
2. M1 сенсоры во сне зануляются до входа в M5.
3. M3 моторные выходы во сне блокируются до исполнения во внешнем мире.
4. M5 может во сне продолжать генерировать imagined actions.
5. M11 остаётся активным и создаёт affect.
6. M13 retrieval вызывается после M11 affect и до M2.
7. M4 вызывается из inner_object / z_obj и даёт identity context для M2.
8. M2 вызывается после M11/M13/M4.
9. M2 использует affect + M13 + M4 + focus_context.
10. M2 не мутирует out["focus_context"] напрямую по умолчанию.
11. M2 сохраняет replay_context как next_focus_context_seed.
12. M5 получает seed через общий вход:
    focus_context_seed + focus_context_seed_gate.
13. get_m5_focus_seed(...) даёт приоритет:
    1) M2 dream/replay seed in sleep mode
    2) M15 conscious seed fallback
14. При входе в full sleep сразу зануляются:
    prev_embodied_action
    prev_hand_motor
15. M8 пульт умеет:
    - открыть Module Lab window;
    - запустить проверки M2/M4/M11/M13/M5Boundary/loop/scenarios;
    - включить/выключить Sleep / replay mode.
```

---

# 1. Проверить sleep sensors / M1 cut

Файл:

```text
src/modules/m06_learning_sleep_consolidation/sleep_sensors.py
```

Проверить:

```python
is_full_sleep_mode()
```

должен возвращать True только если:

```text
video_sensor_enabled == False
contact_sensor_enabled == False
imu_sensor_enabled == False
```

Проверить:

```python
gate_observation_for_sleep(obs)
```

должен занулять:

```text
video OFF:
    left
    right
    depth
    object_state

contact OFF:
    tactile
    contact
    contacts
    contact_sensors

imu OFF:
    pose
    body_state
    vestibular
    imu
    gyro
    accel
    balance_reward
```

Проверить, что после gate добавляются debug masks:

```text
obs["input_sensors_enabled"]
obs["sleep_sensor_mask"]
obs["sensor_gate_applied"] = True
```

Проверить, что `life_runtime.py` вызывает:

```python
obs0 = self.gate_observation_for_sleep(obs0)
obs = self.gate_observation_for_sleep(obs)
```

до `model_step(...)`.

---

# 2. Проверить zero previous motor tail on sleep entry

Файл:

```text
src/modules/m06_learning_sleep_consolidation/sleep_sensors.py
```

Проверить наличие helper:

```python
_zero_prev_motor_state_on_sleep_entry(...)
```

Он должен занулять:

```text
self.prev_embodied_action
self.prev_hand_motor
```

только при переходе:

```text
old_sleep == False
new_sleep == True
```

То есть:

```text
awake / partial_cut → full sleep:
    zero prev_embodied_action
    zero prev_hand_motor

partial_cut:
    не занулять

already sleep → sleep:
    не делать лишний reset
```

Проверить, что в `apply_sleep_sensor_state(...)` есть:

```python
old = self.input_sensors_enabled_dict_no_startup_apply()
old_sleep = ...

...
new = self.input_sensors_enabled_dict_no_startup_apply()
new_sleep = ...

if changed and (not old_sleep) and new_sleep:
    self._zero_prev_motor_state_on_sleep_entry()
```

Проверить тест:

```text
tests/module_contracts/test_sleep_entry_zero_prev_motor_contract.py
```

Он должен проходить.

---

# 3. Проверить sleep_motor_guard / M3 block

Файл:

```text
src/modules/m03_self_action_causality/sleep_motor_guard.py
```

Проверить, что при `sleep_mode=True` зануляются:

```text
out["embodied_targets"]
out["hand_ctrl"]
out["leg_ctrl"]
```

и сохраняются отладочные копии:

```text
out["imagined_embodied_targets"]
out["imagined_hand_ctrl"]
out["imagined_leg_ctrl"]
out["imagined_action_ids"]
```

Проверить, что:

```text
out["sleep_motor_guard"]["blocked"] == True
out["sleep_motor_guard"]["reason"] == "full_sleep_mode_blocks_external_motor_execution"
```

Проверить, что при `sleep_mode=False` моторные выходы не меняются.

Проверить тест:

```text
tests/module_contracts/test_sleep_motor_guard_contract.py
```

---

# 4. Проверить подключение SleepMotorGuardRuntimeMixin

Файл:

```text
src/apps/runner.py
```

Проверить импорты:

```python
from src.modules.m03_self_action_causality.sleep_motor_guard import SleepMotorGuardRuntimeMixin
```

Проверить, что `UnifiedSystem` наследует:

```python
SleepMotorGuardRuntimeMixin
```

---

# 5. Проверить применение sleep_motor_guard в life_runtime

Файл:

```text
src/apps/life_runtime.py
```

Проверить `pre_observe`:

```python
out0["leg_ctrl"] = self.compute_leg_control(out0)
out0 = self.apply_manual_leg_action_override(out0)

if hasattr(self, "apply_sleep_motor_guard"):
    out0 = self.apply_sleep_motor_guard(out0, stage="pre_observe")

self.apply_bird_leg_controls(out0["leg_ctrl"])
```

Важно: guard должен стоять **до**:

```python
self.apply_bird_leg_controls(...)
self.apply_dynamic_agent_rig_control(out0["embodied_targets"]...)
world.observe(... hand_controls=out0["hand_ctrl"] ...)
```

Проверить `main`:

```python
out["leg_ctrl"] = self.compute_leg_control(out)
out = self.apply_manual_leg_action_override(out)

if hasattr(self, "apply_sleep_motor_guard"):
    out = self.apply_sleep_motor_guard(out, stage="main")
```

Важно: guard должен стоять **до**:

```python
self.prev_embodied_action = out["embodied_targets"].detach()
self.prev_hand_motor = out["hand_ctrl"].detach()
replay.add(...)
```

---

# 6. Проверить M11 active

Файл:

```text
src/apps/life_runtime.py
```

Проверить, что после main M5 output вызывается:

```python
emotion = self.emotional_drive.compute(out, obs)
out["emotion"] = emotion
if isinstance(emotion.get("affect"), dict):
    out["affect"] = emotion["affect"]
```

И что это происходит **до**:

```python
compute_autobiographical_retrieval(...)
compute_event_dream_replay(...)
```

Проверить контракт:

```text
tests/module_contracts/test_m11_emotional_drive_contract.py
```

---

# 7. Проверить M13 retrieval active

Файлы:

```text
src/apps/life_runtime.py
src/modules/m13_autobiographical_memory/autobiographical_memory_runtime.py
```

Проверить, что после M11 affect вызывается:

```python
self.compute_autobiographical_retrieval(obs, out)
```

и что runtime кладёт:

```python
out["autobiographical_memory"] = memory
```

Пакет должен содержать:

```text
retrieved_context
retrieval_relevance
summary
retrieved_episode_count / episode_count
```

Проверить, что M13 retrieval идёт **в M2**, а не напрямую в M5 как основной sleep/replay path.

Legacy blend M13→focus_context допустим только если явно включён config-ом, но для бессознательного sleep/replay M13 должен участвовать через M2.

Проверить контракт:

```text
tests/module_contracts/test_m13_autobiographical_memory_contract.py
```

---

# 8. Проверить M4 active

Файлы:

```text
src/apps/life_runtime.py
src/modules/m04_long_dynamic_memory/long_dynamic_memory_runtime.py
```

Проверить порядок:

```python
out["inner_object"] = self.compute_inner_object_image(obs, out)
self._compute_long_dynamic_memory(obs, out)
```

В M4 runtime проверить:

```python
obj = out.get("inner_object")
```

и результат:

```python
out["long_dynamic_memory"] = packet
```

Пакет должен содержать:

```text
dynamic_identity_context
dynamic_memory_gate
identity_token
identity_stability
identity_novelty
passport_slot
selected_sentence
```

Проверить, что M4 получает вход не напрямую из M1 raw sensors, а из:

```text
M5 / inner_object / z_obj
```

Проверить контракт:

```text
tests/module_contracts/test_m04_long_dynamic_memory_contract.py
```

---

# 9. Проверить M2 active and correct inputs

Файлы:

```text
src/apps/life_runtime.py
src/modules/m02_event_dream_replay/event_dream_replay.py
src/modules/m02_event_dream_replay/event_dream_runtime.py
```

Проверить, что после M11/M13/M4 вызывается:

```python
self.compute_event_dream_replay(obs, out)
```

В `EventDreamReplay.compute(...)` проверить, что читаются:

```python
affect = out.get("affect")
memory13 = out.get("autobiographical_memory")
memory4 = out.get("long_dynamic_memory")
focus_context = out.get("focus_context")
```

Проверить, что replay_context собирается из:

```text
focus_context
event_vec
m13_context = memory13["retrieved_context"]
m4_context = memory4["dynamic_identity_context"]
```

Проверить, что salience/dream_pressure учитывает:

```text
panic_latent
stress_latent
curiosity_latent
retrieval_relevance
identity_stability
identity_novelty
dynamic_memory_gate
event_delta/contact/action
```

Проверить обязательные выходы:

```text
replay_context
replay_gate
event_salience
dream_pressure
should_replay
selected_episode_summary
selected_identity_token
selected_identity_sentence
identity_stability
identity_novelty
dynamic_memory_gate
replay_source
```

Проверить контракты:

```text
tests/module_contracts/test_m02_event_dream_replay_contract.py
tests/module_contracts/test_m02_runtime_seed_bus_contract.py
```

---

# 10. Проверить M2 → M5 seed bus

Файл:

```text
src/modules/m02_event_dream_replay/event_dream_runtime.py
```

Проверить config defaults:

```python
blend_replay_into_focus = False
use_m13_context = True
use_m4_context = True
seed_to_m5_boundary = True
apply_stage = "pre_observe"
seed_only_in_sleep = True
```

Проверить методы:

```python
_store_event_dream_m5_seed(...)
get_event_dream_focus_seed(...)
get_m5_focus_seed(...)
```

`_store_event_dream_m5_seed(packet)` должен сохранять:

```text
self._event_dream_next_focus_seed
self._event_dream_next_focus_gate
packet["next_focus_context_seed"]
packet["next_focus_context_seed_gate"]
packet["target_m5_boundary"] = "FocusFeedbackBoundary(workspace_seed + preconscious_seed)"
packet["seed_source"] = "m02_event_dream_replay"
```

`get_event_dream_focus_seed(stage)` должен возвращать seed только если:

```text
event_dream_replay.enabled == True
seed_only_in_sleep == False OR is_full_sleep_mode() == True
stage allowed by apply_stage
seed tensor exists
```

`get_m5_focus_seed(stage)` должен иметь приоритет:

```text
1. M2 dream/replay seed
2. M15 conscious seed fallback
```

---

# 11. Проверить, что M2 не мутирует focus_context по умолчанию

Файл:

```text
src/modules/m02_event_dream_replay/event_dream_runtime.py
```

Запрещённый путь по умолчанию:

```python
out["focus_context"] = focus + ...
```

Допустимо только legacy-условие:

```python
bool(cfg.blend_replay_into_focus)
and not bool(getattr(cfg, "seed_to_m5_boundary", True))
```

То есть по умолчанию должно быть:

```text
M2 replay_context
↓
next_focus_context_seed
↓
get_m5_focus_seed(...)
↓
model.step(... focus_context_seed=...)
↓
FocusFeedbackBoundary
```

а не:

```text
M2 replay_context
↓
out["focus_context"] прямой mutation
```

---

# 12. Проверить model_step / M5 receiving replay seed

Файл:

```text
src/apps/unified_conscious_viewer.py
```

Проверить:

```python
def model_step(..., model_stage: str = "main", focus_context_seed=None, focus_context_seed_gate=None)
```

и внутри:

```python
if focus_context_seed is None:
    if hasattr(self, "get_m5_focus_seed"):
        focus_context_seed, focus_context_seed_gate = self.get_m5_focus_seed(stage=model_stage)
    elif hasattr(self, "get_conscious_loop_focus_seed"):
        focus_context_seed, focus_context_seed_gate = self.get_conscious_loop_focus_seed(stage=model_stage)
```

Проверить, что seed передаётся в:

```python
self.model.step(
    ...
    focus_context_seed=focus_context_seed,
    focus_context_seed_gate=focus_context_seed_gate,
)
```

---

# 13. Проверить FocusFeedbackBoundary

Файлы:

```text
src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
```

Проверить, что M5 принимает:

```text
focus_context_seed
focus_context_seed_gate
```

через:

```python
FocusFeedbackBoundary
```

и результат влияет на:

```text
workspace_seed
preconscious_seed / preconscious_delta
```

Проверить контракт:

```text
tests/module_contracts/test_m05_focus_feedback_boundary_contract.py
```

---

# 14. Проверить live trace

Файл:

```text
src/modules/m02_event_dream_replay/unconscious_loop_trace.py
```

Проверить, что trace читает:

```text
out["affect"]
out["autobiographical_memory"]
out["long_dynamic_memory"]
out["event_dream_replay"]
out["focus_feedback"] / attention focus feedback gate
out["sleep_motor_guard"]
```

и печатает строку вида:

```text
[unconscious_loop step=N]
sleep=1 state=sleep |
m11: ...
m13: ...
m4: ...
m2: ...
m5_seed: ...
m3_sleep_block=1
```

Проверить подключение в `runner.py`:

```python
UnconsciousLoopTraceRuntimeMixin
```

Проверить вызов в `life_runtime.py`:

```python
self.maybe_print_unconscious_loop_trace(out, obs)
```

Проверить контракт:

```text
tests/module_contracts/test_unconscious_loop_trace_contract.py
```

---

# 15. Проверить Module Lab / imit layout

Проверить наличие per-module imit каталогов:

```text
src/modules/m01_object_imagery/imit/
src/modules/m02_event_dream_replay/imit/
src/modules/m03_self_action_causality/imit/
src/modules/m04_long_dynamic_memory/imit/
src/modules/m05_world_model_attention_workspace/imit/
src/modules/m11_motivational_homeostasis/imit/
src/modules/m13_autobiographical_memory/imit/
```

Проверить:

```text
scripts/module_lab/module_imit_registry.py
scripts/module_lab/run_module_imit_registry.py
scripts/module_lab/run_module_lab.py
scripts/module_lab/scenario_unconscious_replay.py
```

Запустить:

```bash
python scripts/module_lab/run_module_imit_registry.py --module all
python scripts/module_lab/run_module_lab.py --module all
python scripts/module_lab/scenario_unconscious_replay.py --json
```

Проверить тесты:

```text
tests/module_contracts/test_per_module_imit_layout_contract.py
tests/module_contracts/test_unconscious_behavioral_scenarios.py
tests/module_contracts/test_unconscious_loop_contract.py
```

---

# 16. Проверить M8 Module Lab window

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

Проверить, что во вкладке M8 есть кнопка:

```text
Module Lab
```

Она должна открывать окно с кнопками:

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

Проверить IPC mapping:

```text
Run M2 test                 -> module_lab_run {module: "m02"}
Run M4 test                 -> module_lab_run {module: "m4"}
Run M11 test                -> module_lab_run {module: "m11"}
Run M13 test                -> module_lab_run {module: "m13"}
Run M5Boundary test         -> module_lab_run {module: "m05"}
Run unconscious loop test   -> module_lab_run {module: "loop"}
Run behavioral scenarios    -> module_lab_run {module: "scenarios"}
Run all                     -> module_lab_run {module: "all"}
```

Проверить runner handler:

```text
src/modules/m08_debug_visual_control/module_lab_runtime.py
src/modules/m03_self_action_causality/action_runtime.py
```

Должен быть IPC action:

```text
module_lab_run
```

и результат должен сохраняться:

```text
self.last_module_lab_result
```

Проверить status IPC:

```text
src/modules/m08_debug_visual_control/module_status_runtime.py
```

должен отдавать:

```text
last_module_lab_result
```

Проверить тест:

```text
tests/module_contracts/test_m8_module_lab_runtime_contract.py
```

---

# 17. Проверить M8 Sleep / replay mode button

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

Проверить, что во вкладке M8 есть кнопка:

```text
Сон / replay mode
```

Она должна переключать:

```text
OFF -> ON:
    video_sensor_enabled = False
    contact_sensor_enabled = False
    imu_sensor_enabled = False
    sleep_video_cut = True
    sleep_contact_cut = True
    sleep_imu_cut = True

ON -> OFF:
    video_sensor_enabled = True
    contact_sensor_enabled = True
    imu_sensor_enabled = True
    sleep_video_cut = False
    sleep_contact_cut = False
    sleep_imu_cut = False
```

Проверить, что кнопка отправляет set_state IPC, а не только меняет локальный UI.

Проверить, что runner принимает это через:

```text
apply_sleep_sensor_state(...)
```

И что при включении full sleep срабатывает zero previous motor tail.

---

# 18. Запрещённые паттерны

Найти и убедиться, что они отсутствуют или выключены по умолчанию:

```python
# M1 raw sensors directly into M2
compute_event_dream_replay(raw_obs, ...)
# где M2 строит scene/replay напрямую из left/right/depth/tactile

# M2 direct focus mutation by default
out["focus_context"] = focus + replay_context

# separate dream_context input bypassing FocusFeedbackBoundary
self.model.step(..., dream_context=...)
self.model.step(..., replay_context=...)

# M13 directly feeding M5 as main sleep/replay path
M13 -> FocusFeedbackBoundary

# M4 directly feeding M5 as main sleep/replay path
M4 -> FocusFeedbackBoundary

# M3 body execution while full sleep mode is active
apply_dynamic_agent_rig_control(nonzero out["embodied_targets"]) in sleep
apply_bird_leg_controls(nonzero out["leg_ctrl"]) in sleep
world.observe(... hand_controls=nonzero out["hand_ctrl"]) in sleep
```

Legacy paths are acceptable only if disabled by default and guarded by config.

---

# 19. Compile checks

Run:

```bash
python -m py_compile src/modules/m06_learning_sleep_consolidation/sleep_sensors.py
python -m py_compile src/modules/m03_self_action_causality/sleep_motor_guard.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_replay.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_runtime.py
python -m py_compile src/modules/m02_event_dream_replay/unconscious_loop_trace.py
python -m py_compile src/modules/m04_long_dynamic_memory/long_dynamic_memory_runtime.py
python -m py_compile src/modules/m13_autobiographical_memory/autobiographical_memory_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
python -m py_compile src/modules/m08_debug_visual_control/module_lab_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py
python -m py_compile src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py
python -m py_compile src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py
python -m py_compile src/apps/life_runtime.py
python -m py_compile src/apps/unified_conscious_viewer.py
python -m py_compile src/apps/runner.py
```

---

# 20. Pytest checks

Run:

```bash
pytest tests/module_contracts/test_sleep_entry_zero_prev_motor_contract.py
pytest tests/module_contracts/test_sleep_motor_guard_contract.py
pytest tests/module_contracts/test_m02_event_dream_replay_contract.py
pytest tests/module_contracts/test_m02_runtime_seed_bus_contract.py
pytest tests/module_contracts/test_m04_long_dynamic_memory_contract.py
pytest tests/module_contracts/test_m05_focus_feedback_boundary_contract.py
pytest tests/module_contracts/test_m11_emotional_drive_contract.py
pytest tests/module_contracts/test_m13_autobiographical_memory_contract.py
pytest tests/module_contracts/test_unconscious_loop_contract.py
pytest tests/module_contracts/test_unconscious_loop_trace_contract.py
pytest tests/module_contracts/test_unconscious_behavioral_scenarios.py
pytest tests/module_contracts/test_per_module_imit_layout_contract.py
pytest tests/module_contracts/test_m8_module_lab_runtime_contract.py
```

Then run all module contracts:

```bash
pytest tests/module_contracts
```

---

# 21. Direct smoke commands

Run:

```bash
python scripts/module_lab/run_module_imit_registry.py --module all
python scripts/module_lab/run_module_lab.py --module all
python scripts/module_lab/run_module_lab.py --module loop
python scripts/module_lab/scenario_unconscious_replay.py --json
```

If the app can be started locally, also do a short live check:

```bash
python -m src.apps.runner
```

Then from PyQt M8:

```text
1. Click Sleep / replay mode ON.
2. Confirm console:
   [ipc][sleep_sensors] video=OFF contact=OFF imu=OFF state=sleep
   [sleep_replay] zeroed previous motor tail on sleep entry: ...
3. Confirm trace:
   [unconscious_loop step=N] sleep=1 state=sleep ...
4. Confirm m3_sleep_block=1.
5. Click Module Lab.
6. Run each:
   M2, M4, M11, M13, M5Boundary, unconscious loop, behavioral scenarios.
7. Confirm last_module_lab_result updates in window/status.
```

---

# 22. What to fix

Fix only real problems:

```text
compile errors
import errors
missing mixin imports
M8 button not visible
M8 window not opening
module_lab_run IPC not handled
last_module_lab_result not exposed in status
sleep sensors not cutting M1 inputs
prev motor tail not zeroed on sleep entry
sleep_motor_guard not applied before motor execution
M11 affect computed after M2
M13 retrieval missing before M2
M4 identity missing before M2
M2 not called in life_runtime
M2 direct focus mutation enabled by default
M2 seed bus missing
model_step still uses only conscious seed
FocusFeedbackBoundary not receiving focus_context_seed
shape mismatches
NaN/Inf in module tests
```

Do not rewrite architecture.

---

# 23. Expected final report

At the end, report:

```text
1. Current commit/hash checked.
2. Files inspected.
3. Compile check result.
4. Pytest result.
5. Direct smoke command result.
6. M8 UI checks result.
7. Confirmation:
   ✅ M1 sensors cut in sleep
   ✅ prev motor tail zeroed on sleep entry
   ✅ M3 external motor blocked
   ✅ M11 active
   ✅ M13 retrieval active
   ✅ M4 identity active
   ✅ M2 replay active
   ✅ M2 uses M11/M13/M4/focus_context
   ✅ M2 does not mutate focus_context by default
   ✅ M2 stores next_focus_context_seed
   ✅ M5 receives replay seed through FocusFeedbackBoundary
   ✅ M8 Module Lab works
   ✅ M8 Sleep / replay mode works
8. Any minimal fixes made.
9. Remaining risks.
```

Final line should be exactly:

```text
Sleep/replay live contract confirmed:
M1 → M5 → M11 → M2 → FocusFeedbackBoundary → M5,
with M4/M13 feeding M2 and M3 externally blocked during full sleep.
```
