# Codex task: проверить отладку 2-го уровня — имитация внутренних латентов M5

## Цель

Проверить шаг **“Отладка 2-го уровня: имитация внутренних learned latents M5”**.

Мы проверяем не обучение M5, а диагностический режим:

```text
как будто M5 уже выучила латенты cube / tetrahedron / morph,
и мы подаём эти устойчивые object-latents через штатный M5 seed/boundary path.
```

Ключевая идея:

```text
M8 button
↓
IPC action
↓
M5 imit runtime
↓
simulated learned object latent
↓
common M5 seed bus
↓
FocusFeedbackBoundary
↓
M5 workspace/focus_context
↓
status + monitors
↓
downstream reaction: M11 / M4 / M2 / Replay Quality / M5 Learning Quality
```

Важно:

```text
Это НЕ обучение M5.
Это НЕ настоящие веса M5.
Это имитационная подкладка learned latent prototypes для проверки проводки 2-го уровня.
```

---

# 0. Архитектурное правило

Все имитаторы должны лежать в каталоге `imit/`.

Для M5 правильный путь:

```text
src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py
```

Неправильный путь:

```text
src/modules/m05_world_model_attention_workspace/m5_latent_prototype_runtime.py
```

Если root-level файл существует — это FAIL, его надо удалить.

---

# 1. Проверить файлы

Проверить наличие:

```text
src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py
src/modules/m08_debug_visual_control/m5_latent_prototype_status.py
tests/module_contracts/test_m5_latent_prototype_simulator_contract.py
docs/architecture/m5_latent_prototype_simulator.md
```

Проверить отсутствие:

```text
src/modules/m05_world_model_attention_workspace/m5_latent_prototype_runtime.py
```

---

# 2. Проверить compile

Запустить:

```bash
python -m py_compile src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/m5_latent_prototype_status.py
python -m py_compile src/apps/runner.py
python -m py_compile src/modules/m02_event_dream_replay/event_dream_runtime.py
python -m py_compile src/modules/m03_self_action_causality/action_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
```

Ожидается:

```text
без ошибок
```

---

# 3. Проверить pytest

Запустить:

```bash
pytest tests/module_contracts/test_m5_latent_prototype_simulator_contract.py
```

Дополнительно регрессии:

```bash
pytest tests/module_contracts/test_m5_learning_quality_status_contract.py
pytest tests/module_contracts/test_replay_quality_monitor_status_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
pytest tests/module_contracts/test_m02_runtime_seed_bus_contract.py
pytest tests/module_contracts/test_dream_probe_runtime_contract.py
```

Если есть время:

```bash
pytest tests/module_contracts
```

---

# 4. Проверить runner.py

Файл:

```text
src/apps/runner.py
```

Должен быть import именно из `imit`:

```python
from src.modules.m05_world_model_attention_workspace.imit.m5_latent_prototype_runtime import M5LatentPrototypeRuntimeMixin
```

В `class UnifiedSystem(...)` должен быть mixin:

```python
M5LatentPrototypeRuntimeMixin
```

Ожидание:

```text
UnifiedSystem имеет методы:
    request_m5_latent_prototype(...)
    get_m5_latent_prototype_focus_seed(...)
    m5_latent_prototype_status(...)
```

---

# 5. Проверить M5 imit runtime

Файл:

```text
src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py
```

Должен содержать:

```python
class M5LatentPrototypeRuntimeMixin:
```

Проверить методы:

```python
make_m5_latent_prototype(...)
request_m5_latent_prototype(...)
get_m5_latent_prototype_focus_seed(...)
m5_latent_prototype_status(...)
```

Проверить, что это именно имитация, не обучение:

```text
не вызывает optimizer.step()
не вызывает train_once()
не меняет веса self.model
не пишет checkpoint
не мутирует real M5 weights
```

---

# 6. Проверить геометрические prototype descriptors

Должны быть как минимум:

```text
cube:
    faces = 6
    edges = 12
    vertices = 8
    square_faces = 6
    symmetry_order = 24

tetrahedron:
    faces = 4
    edges = 6
    vertices = 4
    triangular_faces = 4
    symmetry_order = 12
```

Проверить, что:

```python
make_m5_latent_prototype("cube")
make_m5_latent_prototype("tetrahedron")
```

возвращают:

```text
latent shape = (1, 256) или cfg.self_core.focus_context_dim
descriptor dict
```

Проверить, что cube и tetrahedron не одинаковые:

```python
cosine_similarity(cube, tetra) < 0.98
```

---

# 7. Проверить точки подкладки имитационных данных

## Точка 1 — IPC payload

Из M8 должно уходить:

```python
make_action_message(
    "m5_latent_prototype_inject",
    kind="cube" / "tetrahedron" / "morph",
    gate=...,
    duration=...,
    alpha=...,
    source="m8_m5_latent_prototype_window",
)
```

Для clear:

```python
kind="clear"
```

или action:

```text
m5_latent_prototype_clear
```

## Точка 2 — request_m5_latent_prototype(payload)

В `ActionRuntimeMixin.apply_ipc_action(...)` должен быть handler:

```python
elif action in ("m5_latent_prototype_inject", "m5_latent_prototype_clear"):
    ...
    self.request_m5_latent_prototype(payload)
```

Ожидаемый console log:

```text
[m5_latent_prototype][ipc] action=m5_latent_prototype_inject payload=...
[m5_latent_prototype][imit] kind=cube gate=... duration=...
```

## Точка 3 — создание simulated learned object latent

`request_m5_latent_prototype(...)` должен создать:

```python
self._m5_latent_prototype_seed
self._m5_latent_prototype_gate
self._m5_latent_prototype_state
```

Проверить state:

```text
active = True
kind = cube / tetrahedron / morph
remaining > 0
gate > 0
seed_norm > 0
cube_similarity
tetra_similarity
layout = imit
target_m5_boundary = FocusFeedbackBoundary(...)
```

## Точка 4 — common M5 seed bus

В `event_dream_runtime.py` метод:

```python
get_m5_focus_seed(stage)
```

должен проверять prototype seed до обычных источников:

```python
if hasattr(self, "get_m5_latent_prototype_focus_seed"):
    seed, gate = self.get_m5_latent_prototype_focus_seed(stage=stage)
    if torch.is_tensor(seed):
        return seed, gate

seed, gate = self.get_event_dream_focus_seed(stage=stage)
...
conscious seed
...
```

То есть приоритет:

```text
1. M5 latent prototype seed
2. M2 dream/replay seed
3. M15 conscious seed
```

## Точка 5 — M5 / FocusFeedbackBoundary

Проверить, что seed дальше идёт штатно как:

```text
focus_context_seed
focus_context_seed_gate
```

и не мутирует напрямую:

```text
out["focus_context"] = ...
```

Запрещено:

```text
прямая подмена focus_context
обход FocusFeedbackBoundary
прямое вмешательство в M5 weights
```

## Точка 6 — status publication

В `module_status_runtime.py` должен быть:

```python
from src.modules.m08_debug_visual_control.m5_latent_prototype_status import build_m5_latent_prototype_status
```

и в payload:

```python
"m5_latent_prototype": build_m5_latent_prototype_status(self),
```

---

# 8. Проверить M8 control_panel.py

Файл:

```text
src/modules/m08_debug_visual_control/control_panel.py
```

В M8 должна быть кнопка:

```text
M5 Latent Prototypes
```

Проверить:

```python
self.btn_m5_latent_prototype = QtWidgets.QPushButton("M5 Latent Prototypes")
```

Проверить, что кнопка добавлена в M8 tab:

```python
"btn_m5_latent_prototype"
```

Проверить signal:

```python
self.btn_m5_latent_prototype.clicked.connect(self.open_m5_latent_prototype_window)
```

Проверить refresh:

```python
self.refresh_m5_latent_prototype_window()
```

Проверить style:

```python
self._style_plain_status_button(
    self.btn_m5_latent_prototype,
    bool((self.last_status or {}).get("m5_latent_prototype", {}).get("active", False)),
    "M5 Latent Prototypes",
)
```

---

# 9. Проверить окно M5 Latent Prototypes

Должен быть метод:

```python
open_m5_latent_prototype_window(...)
```

Окно должно иметь кнопки:

```text
Inject cube latent
Inject tetrahedron latent
Inject cube↔tetra morph
Clear
```

Окно должно показывать:

```text
active
kind
gate
seed_norm
remaining
cube_similarity
tetra_similarity
layout
target_m5_boundary
source
note
```

Raw JSON должен показывать:

```python
last_status["m5_latent_prototype"]
```

---

# 10. Live-smoke проверка

Запустить runner:

```bash
python -m src.apps.runner
```

Запустить control panel:

```bash
python src/modules/m08_debug_visual_control/control_panel.py
```

Открыть M8:

```text
M5 Latent Prototypes
```

Нажать:

```text
Inject cube latent
```

Ожидается:

```text
console:
[m5_latent_prototype][ipc] action=m5_latent_prototype_inject ...
[m5_latent_prototype][imit] kind=cube gate=...

window/status:
active = True
kind = cube
gate > 0
seed_norm > 0
cube_similarity > tetra_similarity
remaining decreases
layout = imit
```

Нажать:

```text
Inject tetrahedron latent
```

Ожидается:

```text
kind = tetrahedron
tetra_similarity > cube_similarity
seed_norm > 0
```

Нажать:

```text
Inject cube↔tetra morph
```

Ожидается:

```text
kind = morph
cube_similarity and tetra_similarity both meaningful
descriptor contains morph_alpha
```

Нажать:

```text
Clear
```

Ожидается:

```text
active = False
remaining = 0
```

---

# 11. Проверить downstream monitors

После `Inject cube latent` / `Inject tetrahedron latent` смотреть:

## M5 Learning Quality

Ожидается:

```text
seed_gate > 0
seed_norm > 0
seed_response > 0
```

## Replay Quality Monitor

Может показать:

```text
quality_score меняется
verdict может стать weak / replaying
```

Но это не обязано быть `integrating`, потому что это не настоящее обучение.

## Sleep Replay Monitor

Смотреть:

```text
M5 seed fields
M11 affect response
M4 identity fields
M2 replay response
```

Ожидание:

```text
есть реакция проводки
нет NaN/Inf
```

---

# 12. Проверить, что это не ломает сон/replay

Включить:

```text
Сон / replay mode
```

Проверить:

```text
M1 sensors OFF
M3 sleep_blocked = True
M5 latent prototype seed всё ещё идёт через FocusFeedbackBoundary
не создаёт моторный выход наружу
```

Если prototype active во сне:

```text
M3 должен оставаться blocked
```

---

# 13. Проверить read-only / non-training безопасность

M5 latent prototype simulator не должен:

```text
вызывать optimizer.step()
вызывать train_once()
менять параметры self.model
сохранять checkpoint
подменять out["focus_context"] напрямую
обходить FocusFeedbackBoundary
создавать фейковое утверждение, что M5 обучена
```

Допустимо:

```text
хранить self._m5_latent_prototype_seed
хранить self._m5_latent_prototype_gate
хранить self._m5_latent_prototype_state
писать status
давать seed/gate через get_m5_latent_prototype_focus_seed()
```

---

# 14. Что считать PASS

PASS, если:

```text
✅ runtime лежит в m05/.../imit/
✅ root-level runtime отсутствует
✅ runner импортирует mixin из imit
✅ IPC action работает
✅ cube/tetra/morph prototype создаются
✅ seed/gate идут через get_m5_focus_seed()
✅ FocusFeedbackBoundary путь не обойдён
✅ status payload m5_latent_prototype есть
✅ M8 кнопка/окно есть
✅ raw JSON обновляется
✅ cube_similarity/tetra_similarity различают cube/tetra
✅ M5 Learning Quality видит seed_response
✅ сон/replay не сломан
✅ M3 остаётся blocked во сне
✅ pytest и compile проходят
```

---

# 15. Что считать FAIL

FAIL, если:

```text
runtime лежит не в imit/
root-level runtime остался
runner импортирует из неправильного пути
нет m5_latent_prototype status key
M8 кнопка отсутствует
кнопка не отправляет IPC
seed не доходит в get_m5_focus_seed
prototype напрямую мутирует out["focus_context"]
prototype меняет веса M5
prototype ломает M2/M15 seed bus
M3 не blocked во сне
NaN/Inf в status
cube и tetra latents одинаковые
```

---

# 16. Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. Files inspected.
3. Compile results.
4. Pytest results.
5. Imit layout result.
6. Runner/mixin result.
7. IPC result.
8. M5 seed bus result.
9. M8 UI/window result.
10. Live-smoke:
    - cube latent
    - tetrahedron latent
    - morph latent
    - clear
11. Downstream monitor observations:
    - M5 Learning Quality
    - Replay Quality
    - Sleep Replay
12. Safety/read-only confirmation.
13. Any minimal fixes made.
14. Remaining risks.
```

Финальная строка:

```text
M5 latent prototype 2nd-level debug verified:
imit layout, IPC payload injection, simulated cube/tetra/morph latents, common M5 seed bus, FocusFeedbackBoundary path, status publication, M8 UI, and downstream monitor response are confirmed.
```
