# Codex task: проверить M1 Object Slot Latent Imitator и отображение в Inner Object 3D

## Цель

Проверить новый шаг отладки:

```text
симитировать в M1 внутренние object-latents куба и тетраэдра,
заполнить ими object-slots,
автоматически выбрать слот,
и убедиться, что Inner Object 3D показывает выбранный слот.
```

Это **не обучение M1/M5** и не подмена настоящего зрения.

Это отладка второго уровня:

```text
как будто M1 уже сформировал object latent,
и мы проверяем:
    slot memory
    selected slot
    inner object 3D viewer
    downstream object-slot path
```

---

# 0. Архитектурное правило

Все имитаторы должны лежать в каталоге `imit/`.

Правильный путь:

```text
src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py
```

Неправильно:

```text
src/modules/m01_object_imagery/m1_object_slot_latent_runtime.py
```

Если root-level файл появился — удалить или отметить FAIL.

---

# 1. Проверить файлы

Проверить наличие:

```text
src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py
src/modules/m08_debug_visual_control/m1_object_slot_imit_status.py
tests/module_contracts/test_m1_object_slot_latent_imit_contract.py
docs/architecture/m1_object_slot_latent_imit.md
```

Проверить наличие:

```text
src/modules/m01_object_imagery/imit/__init__.py
```

---

# 2. Compile checks

Запустить:

```bash
python -m py_compile src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/m1_object_slot_imit_status.py
python -m py_compile src/apps/runner.py
python -m py_compile src/modules/m01_object_imagery/runtime.py
python -m py_compile src/modules/m03_self_action_causality/action_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py
python -m py_compile src/modules/m08_debug_visual_control/control_panel.py
```

Ожидается:

```text
без ошибок
```

---

# 3. Pytest

Запустить:

```bash
pytest tests/module_contracts/test_m1_object_slot_latent_imit_contract.py
```

Дополнительно регрессии:

```bash
pytest tests/module_contracts/test_m5_latent_prototype_simulator_contract.py
pytest tests/module_contracts/test_m5_learning_quality_status_contract.py
pytest tests/module_contracts/test_replay_quality_monitor_status_contract.py
pytest tests/module_contracts/test_sleep_replay_monitor_status_contract.py
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

Должен быть import:

```python
from src.modules.m01_object_imagery.imit.m1_object_slot_latent_runtime import M1ObjectSlotLatentImitRuntimeMixin
```

В `UnifiedSystem(...)` должен быть mixin:

```python
M1ObjectSlotLatentImitRuntimeMixin
```

Проверить, что `UnifiedSystem` имеет методы:

```python
request_m1_object_slot_latents(...)
get_m1_imit_inner_object_proposals(...)
m1_object_slot_imit_status(...)
```

---

# 5. Проверить M1 imit runtime

Файл:

```text
src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py
```

Должен содержать:

```python
class M1ObjectSlotLatentImitRuntimeMixin:
```

Проверить методы:

```python
make_m1_object_slot_latent(...)
request_m1_object_slot_latents(...)
get_m1_imit_inner_object_proposals(...)
m1_object_slot_imit_status(...)
```

Проверить, что runtime создаёт latents:

```text
cube
tetrahedron
morph
```

Проверить descriptors:

```text
cube:
    faces=6
    edges=12
    vertices=8
    square_faces=6

tetrahedron:
    faces=4
    edges=6
    vertices=4
    triangular_faces=4
```

Проверить shape:

```python
make_m1_object_slot_latent("cube")
```

возвращает:

```text
latent shape = (1, cfg.object_image.latent_dim)
обычно (1, 128)
```

---

# 6. Проверить главную точку подкладки данных

Главная точка подкладки:

```python
ObjectImageryRuntimeMixin.build_inner_object_vision_proposals(...)
```

Внутри после создания `scene` и после начального debug-state должен быть hook:

```python
if hasattr(self, "get_m1_imit_inner_object_proposals"):
    m1_imit = self.get_m1_imit_inner_object_proposals(scene)
    if isinstance(m1_imit, dict) and torch.is_tensor(m1_imit.get("proposals")):
        self._inner_object_proposal_target_slots = list(m1_imit.get("target_slots", []))
        self._inner_object_proposal_kinds = list(m1_imit.get("proposal_kinds", []))
        self._inner_object_proposal_target_names = list(m1_imit.get("target_names", []))
        self._inner_object_dynamic_debug = {
            "dynamic_ready": True,
            "dynamic_source": "m1_object_slot_imit",
            "slot_update_allowed": True,
            ...
        }
        return m1_imit["proposals"]
```

Это должно быть **до обычной sensory/dynamic логики**, чтобы imit мог заполнить слоты даже без настоящего зрения/обучения.

---

# 7. Проверить allowed_kinds

В `_run_progressive_inner_object_system(...)` есть фильтр:

```python
allowed_kinds = {"dynamic_object", "dynamic_event", "dream_replay"}
```

Он должен включать:

```python
"m1_imit_dynamic_object"
```

Иначе proposals от M1 imit будут отброшены.

Ожидаемо:

```python
allowed_kinds = {
    "dynamic_object",
    "dynamic_event",
    "dream_replay",
    "m1_imit_dynamic_object",
}
```

---

# 8. Проверить запись в конкретные slots

Путь должен быть:

```text
get_m1_imit_inner_object_proposals(...)
↓
proposals: [1, P, latent_dim]
target_slots: [1, 2]
proposal_kinds: ["m1_imit_dynamic_object", ...]
target_names: ["cube", "tetrahedron"]
↓
_run_progressive_inner_object_system(...)
↓
_memory_update_forced_slot(... force_slot_index=target_slot)
```

Критично:

```text
cube должен попасть в slot 1
tetrahedron должен попасть в slot 2
morph должен попасть в slot 3
```

По умолчанию кнопка должна делать:

```text
Fill cube slot1 + tetra slot2
selected_slot = 2
```

---

# 9. Проверить выбор слота для Inner Object 3D

В `request_m1_object_slot_latents(...)` при `auto_select_slot=True` должно вызываться:

```python
self._m1_select_inner_object_slot(selected_slot)
```

Проверить, что метод делает:

```python
self.inner_object_viz.requested_dream_slot_index = slot
self._ipc_inner_object_dream_slot_index = slot
self._m1_object_slot_imit_selected_slot = slot
```

Ожидаемо после Fill cube+tetra:

```text
selected_slot = 2
inner_object_viz.requested_dream_slot_index = 2
_ipc_inner_object_dream_slot_index = 2
```

То есть Inner Object 3D должен показывать slot 2 — тетраэдр.

---

# 10. Проверить action_runtime.py

Файл:

```text
src/modules/m03_self_action_causality/action_runtime.py
```

Должен быть IPC handler:

```python
elif action in ("m1_object_slot_imit_inject", "m1_object_slot_imit_clear"):
    ...
    self.request_m1_object_slot_latents(payload)
```

Ожидаемый console log:

```text
[m1_object_slot_imit][ipc] action=m1_object_slot_imit_inject payload=...
[m1_object_slot_imit] kind=... items=... selected_slot=...
```

---

# 11. Проверить status payload

Файл:

```text
src/modules/m08_debug_visual_control/module_status_runtime.py
```

Должен быть import:

```python
from src.modules.m08_debug_visual_control.m1_object_slot_imit_status import build_m1_object_slot_imit_status
```

В status payload должен быть ключ:

```python
"m1_object_slot_imit": build_m1_object_slot_imit_status(self),
```

Проверить, что старые ключи не удалены:

```text
sleep_replay_monitor
replay_quality_monitor
m5_learning_quality
m5_latent_prototype
last_module_lab_result
```

Если часть этих ключей отсутствует в ветке — отметить фактическое состояние, но M1 imit не должен ломать существующие.

---

# 12. Проверить M8 control_panel.py

В M8 должна быть кнопка:

```text
M1 Object Slot Imit
```

Проверить:

```python
self.btn_m1_object_slot_imit = QtWidgets.QPushButton("M1 Object Slot Imit")
```

Проверить, что кнопка есть в M8 tab:

```python
"btn_m1_object_slot_imit"
```

Проверить signal:

```python
self.btn_m1_object_slot_imit.clicked.connect(self.open_m1_object_slot_imit_window)
```

Проверить style:

```python
self._style_plain_status_button(
    self.btn_m1_object_slot_imit,
    bool((self.last_status or {}).get("m1_object_slot_imit", {}).get("active", False)),
    "M1 Object Slot Imit",
)
```

Проверить refresh:

```python
self.refresh_m1_object_slot_imit_window()
```

---

# 13. Проверить окно M1 Object Slot Imit

Окно должно иметь кнопки:

```text
Fill cube slot1 + tetra slot2
Cube → slot1
Tetra → slot2
Morph → slot3
Clear
```

Окно должно показывать:

```text
active
kind
remaining
selected_slot
items
last_slots
last_names
layout
target
source
note
```

Raw JSON должен показывать:

```python
last_status["m1_object_slot_imit"]
```

---

# 14. Live-smoke: Fill cube + tetra

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
M1 Object Slot Imit
```

Нажать:

```text
Fill cube slot1 + tetra slot2
```

Ожидается:

```text
console:
[m1_object_slot_imit][ipc] action=m1_object_slot_imit_inject ...
[m1_object_slot_imit] kind=... items=[("cube", 1), ("tetrahedron", 2)] selected_slot=2

status:
active = True
selected_slot = 2
last_slots = [1, 2]
last_names = ["cube", "tetrahedron"]
```

---

# 15. Live-smoke: Inner Object 3D

После Fill cube+tetra открыть/обновить:

```text
Inner Object 3D
```

Ожидаемо:

```text
выбран slot 2
slot 2 имеет confidence > 0
slot 2 имеет z_obj norm > 0
viewer показывает selected slot
```

Если Inner Object 3D показывает пусто:

Проверить:

```text
inner_object_viz.requested_dream_slot_index
_ipc_inner_object_dream_slot_index
active_slot_index
z_obj_slots[2]
confidence_slots[2]
```

Если slot заполнен, но окно не переключилось — проблема в viewer selection path.

Если slot не заполнен — проблема в proposal hook / allowed_kinds / force_slot_index path.

---

# 16. Live-smoke: Cube slot1

Нажать:

```text
Cube → slot1
```

Ожидается:

```text
selected_slot = 1
last_slots = [1]
last_names = ["cube"]
inner_object_viz.requested_dream_slot_index = 1
```

---

# 17. Live-smoke: Tetra slot2

Нажать:

```text
Tetra → slot2
```

Ожидается:

```text
selected_slot = 2
last_slots = [2]
last_names = ["tetrahedron"]
inner_object_viz.requested_dream_slot_index = 2
```

---

# 18. Live-smoke: Morph slot3

Нажать:

```text
Morph → slot3
```

Ожидается:

```text
selected_slot = 3
last_slots = [3]
last_names = ["morph"]
inner_object_viz.requested_dream_slot_index = 3
```

---

# 19. Clear

Нажать:

```text
Clear
```

Ожидается:

```text
active = False
remaining = 0
```

Важно: Clear не обязан очищать уже записанные slots. Он только выключает imit-подачу.

---

# 20. Проверить read-only / non-training безопасность

M1 Object Slot Imit не должен:

```text
вызывать optimizer.step()
вызывать train_once()
менять веса M1/M5
сохранять checkpoint
выдавать себя за настоящее обучение
ломать sensor gates
обходить ObjectSlotMemory
писать все slots сразу без target_slots
```

Допустимо:

```text
создать simulated z_obj latent
подать proposals в build_inner_object_vision_proposals
задать _inner_object_proposal_target_slots
задать requested_dream_slot_index
публиковать status
```

---

# 21. Проверить сон/replay guard

Если включён:

```text
Сон / replay mode
```

то:

```text
M3 motor output должен оставаться blocked
M1 sensors могут быть OFF
M1 imit может писать diagnostic slots только как explicit debug action
```

Если это нежелательно во сне, отметить как design question, но не ломать guard.

---

# 22. PASS criteria

PASS, если:

```text
✅ runtime лежит в m01/.../imit/
✅ runner импортирует M1ObjectSlotLatentImitRuntimeMixin
✅ IPC action работает
✅ build_inner_object_vision_proposals получает imit proposals
✅ allowed_kinds включает m1_imit_dynamic_object
✅ cube пишет slot 1
✅ tetrahedron пишет slot 2
✅ selected_slot выставляется
✅ inner_object_viz.requested_dream_slot_index выставляется
✅ status payload m1_object_slot_imit есть
✅ M8 кнопка/окно есть
✅ raw JSON обновляется
✅ Inner Object 3D показывает выбранный slot или slot реально заполнен
✅ compile и pytest проходят
✅ monitor/read-only архитектура не ломает обучение и M3 guard
```

---

# 23. FAIL criteria

FAIL, если:

```text
runtime не в imit/
нет M1ObjectSlotLatentImitRuntimeMixin в runner
нет IPC handler
proposals отбрасываются allowed_kinds
slots не заполняются
slot заполняется не тот
selected_slot не выставляется
Inner Object 3D не переключается на выбранный slot
status key отсутствует
M8 кнопка отсутствует
imit меняет веса или запускает обучение
M3 guard ломается во сне
NaN/Inf в z_obj/confidence
```

---

# 24. Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. Files inspected.
3. Compile results.
4. Pytest results.
5. Imit layout result.
6. Runner/mixin result.
7. IPC result.
8. build_inner_object_vision_proposals hook result.
9. allowed_kinds result.
10. Slot write result:
    - cube slot1
    - tetra slot2
    - morph slot3
11. Inner Object 3D selected slot result.
12. Status/M8 UI result.
13. Sleep/replay guard result.
14. Any minimal fixes made.
15. Remaining risks.
```

Финальная строка:

```text
M1 object-slot latent imit verified:
simulated cube/tetra/morph latents enter M1 inner-object proposals, fill target slots, select the requested slot, update status/M8 UI, and are visible through the Inner Object 3D selected-slot path.
```
