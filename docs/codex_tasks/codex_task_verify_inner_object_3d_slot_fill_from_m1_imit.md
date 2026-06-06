# Codex task: проверить заполнение object-slot и появление 3D формы в Inner Object 3D

## Цель

Проверить, что после M1 Object Slot Latent Imitator:

```text
M8 → M1 Object Slot Imit → Fill cube slot1 + tetra slot2
```

происходит не только отправка IPC/status, а реально:

```text
cube latent пишет slot 1
tetrahedron latent пишет slot 2
selected_slot = 2
Inner Object 3D переключается на slot 2
в окне Inner Object 3D появляется непустая 3D форма / point cloud / gaussian / decoded object representation
```

Если форма не появляется, найти точную точку разрыва:

```text
1. IPC не дошёл
2. M1 imit proposals не создались
3. proposals отфильтрованы allowed_kinds
4. direct z_update bypass не сработал
5. _memory_update_forced_slot не записал slot
6. confidence_slots остались 0
7. selected_slot не передался в viewer
8. slot заполнен, но decoder / 3D recon не строит форму
9. viewer смотрит не на тот slot
```

---

# 0. Важное правило

Это проверка имитатора, не обучение.

M1 Object Slot Imit должен:

```text
создать simulated z_obj latent
подать его в object slot proposal path
записать target slot через force_slot_index
выбрать slot для Inner Object 3D
```

Он не должен:

```text
обучать M1/M5
менять веса
обходить ObjectSlotMemory полностью
ломать M3 sleep guard
выдавать себя за реальное зрение
```

---

# 1. Проверить, что direct-z fix установлен

Файл:

```text
src/modules/m01_object_imagery/runtime.py
```

В `_run_progressive_inner_object_system(...)` должен быть специальный путь:

```python
source = str(proposal_kinds[pi]) if pi < len(proposal_kinds) else "dynamic_object"

if source == "m1_imit_dynamic_object":
    # proposal is already z_obj-like latent
    z_update = v_i
    ...
    fused = {
        "z_update": z_update,
        "vision_strength": vision_strength,
        "touch_strength": touch_strength,
        ...
    }
else:
    fused = self.inner_object_system.fusion(v_i, tactile, body, hand, leg)

slot = self._memory_update_forced_slot(
    state,
    fused["z_update"],
    fused["vision_strength"],
    fused["touch_strength"],
    target_slot,
)
```

Если такого блока нет — это причина пустых slots. Внести минимальный fix.

---

# 2. Проверить allowed_kinds

В том же методе должен быть kind:

```python
allowed_kinds = {
    "dynamic_object",
    "dynamic_event",
    "dream_replay",
    "m1_imit_dynamic_object",
}
```

Если `m1_imit_dynamic_object` отсутствует, proposals будут отброшены, и slots останутся пустыми.

---

# 3. Проверить M1 imit runtime

Файл:

```text
src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py
```

Проверить методы:

```python
request_m1_object_slot_latents(...)
get_m1_imit_inner_object_proposals(...)
make_m1_object_slot_latent(...)
```

Проверить, что при `Fill cube slot1 + tetra slot2` state содержит:

```text
active = True
items = [
    {"kind": "cube", "slot": 1, ...},
    {"kind": "tetrahedron", "slot": 2, ...}
]
selected_slot = 2
auto_select_slot = True
layout = imit
```

Проверить, что `get_m1_imit_inner_object_proposals(scene)` возвращает:

```text
proposals.shape = [1, 2, latent_dim]
target_slots = [1, 2]
proposal_kinds = ["m1_imit_dynamic_object", "m1_imit_dynamic_object"]
target_names = ["cube", "tetrahedron"]
details contains norm > 0
```

---

# 4. Проверить hook в build_inner_object_vision_proposals

Файл:

```text
src/modules/m01_object_imagery/runtime.py
```

Метод:

```python
build_inner_object_vision_proposals(obs)
```

Должен иметь hook **до обычной sensory/dynamic логики**:

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
            "m1_imit_active": True,
            ...
        }
        return m1_imit["proposals"]
```

Если hook стоит после `if not video_sensor_enabled: return ...`, то во сне или при отключённом видео он может не сработать.  
Для отладки имитатор должен иметь возможность писать slots как explicit debug action даже при выключенном видео.

Если нужно, перенести hook выше video gate, но аккуратно.

---

# 5. Добавить временные debug prints для live проверки

На время проверки добавить prints или убедиться, что они уже есть.

## В request_m1_object_slot_latents

```python
print(
    "[m1_object_slot_imit][request] "
    f"kind={kind} items={items} selected_slot={selected_slot} duration={duration}"
)
```

## В get_m1_imit_inner_object_proposals

```python
print(
    "[m1_object_slot_imit][proposal] "
    f"slots={slots} names={names} shape={tuple(proposals.shape)} "
    f"norms={[round(d['norm'], 4) for d in details]}"
)
```

## В _run_progressive_inner_object_system before write

```python
print(
    "[inner_object][slot_write_attempt] "
    f"pi={pi} source={source} target_slot={target_slot} "
    f"v_shape={tuple(v_i.shape)} v_norm={float(v_i.norm().item())}"
)
```

## After _memory_update_forced_slot

```python
z_slots = slot.get("z_obj_slots")
c_slots = slot.get("confidence_slots")
if torch.is_tensor(z_slots) and torch.is_tensor(c_slots):
    print(
        "[inner_object][slot_write_done] "
        f"target_slot={target_slot} "
        f"z_norm={float(z_slots[:, target_slot, :].norm().item())} "
        f"conf={float(c_slots[:, target_slot, :].mean().item())}"
    )
```

Эти prints можно оставить как diagnostic, если они не шумят слишком сильно, или gated через debug flag.

---

# 6. Проверить запись slot memory после live action

После нажатия:

```text
M8 → M1 Object Slot Imit → Fill cube slot1 + tetra slot2
```

в runner console должны появиться:

```text
[m1_object_slot_imit][ipc] action=m1_object_slot_imit_inject ...
[m1_object_slot_imit][request] kind=...
[m1_object_slot_imit][proposal] slots=[1, 2] names=["cube", "tetrahedron"] shape=(1, 2, 128)
[inner_object][slot_write_attempt] source=m1_imit_dynamic_object target_slot=1 ...
[inner_object][slot_write_done] target_slot=1 z_norm>0 conf>0
[inner_object][slot_write_attempt] source=m1_imit_dynamic_object target_slot=2 ...
[inner_object][slot_write_done] target_slot=2 z_norm>0 conf>0
```

Если proposal log есть, но slot_write_attempt нет:

```text
проблема в _run_progressive_inner_object_system path
```

Если slot_write_attempt есть, но slot_write_done conf=0:

```text
проблема в _memory_update_forced_slot / ObjectSlotMemory strength path
```

Если slot_write_done есть, но viewer пустой:

```text
проблема в selected slot / viewer / 3D decoder/reconstructor
```

---

# 7. Проверить actual tensors в latest_out

Найти, где `inner_object` output кладётся в `latest_out`.

Проверить после live action:

```python
out = system.latest_out
inner = out.get("inner_object") or out.get("object_imagery") or out
```

Нужно найти реальные keys:

```text
z_obj_slots
confidence_slots
active_slot_index
z_obj
confidence
semantic_updated_slot
semantic_proposal_count
```

Проверить значения:

```python
z_obj_slots[:, 1, :].norm() > 0
confidence_slots[:, 1, :].mean() > 0

z_obj_slots[:, 2, :].norm() > 0
confidence_slots[:, 2, :].mean() > 0
```

Если `z_obj_slots/confidence_slots` не попадают в `latest_out`, проверить internal state:

```python
system._inner_object_state
system.inner_object_system.memory state
```

---

# 8. Проверить selected slot path

После Fill cube+tetra должно быть:

```text
selected_slot = 2
```

Проверить:

```python
system._m1_object_slot_imit_selected_slot == 2
system._ipc_inner_object_dream_slot_index == 2
system.inner_object_viz.requested_dream_slot_index == 2
```

Если `inner_object_viz` создаётся после action, проверить, что при открытии окна оно подхватывает:

```python
_ipc_inner_object_dream_slot_index
```

или последнюю `requested_dream_slot_index`.

Если viewer создаётся поздно и не знает selected slot — нужно при открытии Inner Object 3D установить выбранный slot из `_ipc_inner_object_dream_slot_index`.

---

# 9. Проверить Inner Object 3D viewer

Найти файл viewer, например:

```text
inner_object_visualizer*.py
inner_object_3d*.py
src/modules/m01_object_imagery/...
src/modules/m08_debug_visual_control/...
```

Проверить, что viewer:

```text
читает selected slot index
читает z_obj_slots / confidence_slots
берёт именно selected slot
не всегда показывает slot 0
не скрывает slot при confidence > 0
строит 3D форму из выбранного z_obj
```

Особенно проверить:

```python
requested_dream_slot_index
active_slot_index
selected_slot
slot_id
```

Если slot 2 заполнен, но viewer показывает slot 0 — исправить selection mapping.

---

# 10. Проверить появление 3D формы

В Inner Object 3D после `Fill cube slot1 + tetra slot2` должно быть хотя бы одно из:

```text
decoded mesh/points not empty
point_cloud point_count > 0
gaussian count > 0
3D primitive/proxy visible
z_obj norm visible
slot confidence visible
```

Если настоящая 3D-реконструкция требует trained decoder и поэтому геометрия пустая, нужно добавить честный fallback для imit/debug:

```text
если source == m1_object_slot_imit
и selected slot z_obj/confidence > 0
и learned decoder не дал форму,
показать fallback primitive:
    cube slot → cube proxy
    tetra slot → tetrahedron proxy
    morph slot → blended/simple proxy
```

Важно: fallback должен быть явно помечен:

```text
debug_imit_fallback_shape = True
```

и не выдавать себя за настоящую реконструкцию.

---

# 11. Проверить M8 status

Status payload должен содержать:

```python
"m1_object_slot_imit": build_m1_object_slot_imit_status(self)
```

В M8 окно должно показывать:

```text
active = True
kind = cube_tetra
selected_slot = 2
items = cube slot1, tetra slot2
last_slots = [1, 2]
last_names = ["cube", "tetrahedron"]
remaining decreasing
```

Если status active, но slots пустые — проблема после status, в proposal/write path.

Если slots заполнены, но status inactive — status path stale, но slot write может работать.

---

# 12. Test / script для проверки без UI

Добавить или временно сделать script:

```text
scripts/module_lab/debug_m1_object_slot_imit_live.py
```

Он должен:

```python
1. создать/получить system
2. request_m1_object_slot_latents({
       "kind": "cube_tetra",
       "cube_slot": 1,
       "tetra_slot": 2,
       "selected_slot": 2,
       "auto_select_slot": True,
   })
3. выполнить несколько life/runtime steps
4. найти z_obj_slots/confidence_slots
5. assert slot1 norm > 0
6. assert slot2 norm > 0
7. assert slot2 confidence > 0
8. assert selected_slot == 2
9. print PASS with exact numbers
```

Если full runner сложно создать, сделать contract-level unit test на:

```text
get_m1_imit_inner_object_proposals
allowed_kinds
direct-z branch around _memory_update_forced_slot
selected slot setter
```

---

# 13. Проверить sleep/replay mode

В режиме сна:

```text
M1 sensors can be OFF
M3 motor output blocked
M1 imit is explicit debug injection
```

Проверить:

```text
M1 Object Slot Imit не разблокирует M3
M3 sleep_blocked остаётся True
slot write допускается только потому, что это explicit debug action
```

---

# 14. PASS criteria

PASS, если после live action:

```text
✅ IPC дошёл
✅ m1_object_slot_imit active=True
✅ proposals shape = [1, 2, latent_dim]
✅ target_slots = [1, 2]
✅ allowed_kinds не отбрасывает m1_imit_dynamic_object
✅ direct-z bypass сработал
✅ slot 1 z_norm > 0
✅ slot 1 confidence > 0
✅ slot 2 z_norm > 0
✅ slot 2 confidence > 0
✅ selected_slot = 2
✅ inner_object_viz.requested_dream_slot_index = 2
✅ Inner Object 3D показывает slot 2
✅ 3D форма/proxy/point cloud не пустая
✅ clear выключает только imit, не ломая память
```

---

# 15. FAIL criteria

FAIL, если:

```text
IPC/status active, но proposals не создаются
proposals создаются, но target_slots пустые
m1_imit_dynamic_object отбрасывается allowed_kinds
proposal идёт через fusion вместо direct z_update
_memory_update_forced_slot не пишет confidence
z_obj_slots[1/2] norm = 0
confidence_slots[1/2] = 0
selected_slot не равен 2
viewer показывает slot 0 вместо slot 2
viewer получает slot 2, но скрывает его
3D форма полностью пустая без fallback/debug explanation
M3 sleep guard ломается
```

---

# 16. Что можно исправлять

Разрешены минимальные фиксы:

```text
перенести M1 imit hook выше video gate
добавить m1_imit_dynamic_object в allowed_kinds
добавить direct-z bypass
добавить debug prints
исправить selected_slot propagation
исправить viewer slot selection
добавить debug fallback primitive для cube/tetra/morph, если decoder не обучен
добавить status keys для z_norm/confidence выбранных slots
```

Не делать:

```text
не обучать M1/M5 в этом шаге
не менять архитектуру object memory
не удалять ObjectSlotMemory
не писать все slots без target slot
не отключать M3 guard
```

---

# 17. Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. Files inspected.
3. Compile results.
4. Pytest results.
5. Live IPC result.
6. Proposal result:
   shape, target_slots, target_names
7. Slot write result:
   slot1 z_norm/confidence
   slot2 z_norm/confidence
8. Selected slot result:
   _ipc_inner_object_dream_slot_index
   inner_object_viz.requested_dream_slot_index
9. Inner Object 3D result:
   selected slot displayed?
   3D form/point cloud/proxy count?
10. If 3D form is missing:
   exact reason
   fix applied or recommended
11. Sleep/M3 guard result.
12. Any minimal fixes made.
13. Remaining risks.
```

Финальная строка:

```text
M1 Object Slot Imit live slot fill verified:
cube and tetrahedron latents write nonzero z_obj/confidence into slots 1 and 2, slot 2 is selected, and Inner Object 3D displays the selected slot with a nonempty 3D form or explicit debug fallback.
```
