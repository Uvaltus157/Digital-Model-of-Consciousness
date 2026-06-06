# Codex task: проверить Architecture DNA / Code DNA и README после реорганизации

## Цель

Проверить, что новая документационная структура проекта зафиксирована правильно:

```text
docs/architecture/
    структура / миграция

docs/architecture/dna/
    Architecture DNA

docs/architecture/dna/codegen/
    Code DNA / implementation-level reconstruction specs
```

Нужно убедиться, что по документации можно восстановить архитектурный контекст проекта и начать заново писать совместимый код.

---

# 1. Проверить структуру каталогов

Должны существовать:

```text
docs/architecture/README.md

docs/architecture/current_structure.md
docs/architecture/module_file_map.md
docs/architecture/module_file_map.json
docs/architecture/module_migration_plan.md

docs/architecture/dna/README.md
docs/architecture/dna/architecture_genome.md
docs/architecture/dna/unconscious_loop_dna.md
docs/architecture/dna/conscious_loop_dna.md
docs/architecture/dna/module_contracts_dna.md
docs/architecture/dna/validation_levels_1_5_dna.md
docs/architecture/dna/imitators_dna.md
docs/architecture/dna/monitors_and_windows_dna.md
docs/architecture/dna/rebuild_from_dna.md
docs/architecture/dna/architecture_dna_manifest.json

docs/architecture/dna/codegen/README.md
docs/architecture/dna/codegen/module_api_specs.md
docs/architecture/dna/codegen/runtime_wiring_dna.md
docs/architecture/dna/codegen/file_inventory_dna_template.json
docs/architecture/dna/codegen/config_dna.md
docs/architecture/dna/codegen/ui_dna.md
docs/architecture/dna/codegen/training_dna.md
docs/architecture/dna/codegen/reference_traces_dna.md
docs/architecture/dna/codegen/code_generation_order.md
docs/architecture/dna/codegen/codegen_manifest.json
```

---

# 2. Проверить, что старые root-level DNA файлы не остались

Эти файлы не должны лежать прямо в `docs/architecture/`:

```text
docs/architecture/architecture_genome.md
docs/architecture/module_contracts_dna.md
docs/architecture/unconscious_loop_dna.md
docs/architecture/conscious_loop_dna.md
docs/architecture/validation_levels_1_5_dna.md
docs/architecture/imitators_dna.md
docs/architecture/monitors_and_windows_dna.md
docs/architecture/rebuild_from_dna.md
docs/architecture/architecture_dna_manifest.json
```

Если есть — перенести в:

```text
docs/architecture/dna/
```

или удалить дубликат, если правильная версия уже есть в `dna/`.

---

# 3. Проверить README.md

Root `README.md` должен упоминать новую структуру:

```text
docs/architecture/dna/
docs/architecture/dna/codegen/
```

Должны быть ссылки/пути:

```text
docs/architecture/dna/architecture_genome.md
docs/architecture/dna/unconscious_loop_dna.md
docs/architecture/dna/conscious_loop_dna.md
docs/architecture/dna/validation_levels_1_5_dna.md
docs/architecture/dna/rebuild_from_dna.md

docs/architecture/dna/codegen/module_api_specs.md
docs/architecture/dna/codegen/runtime_wiring_dna.md
docs/architecture/dna/codegen/config_dna.md
docs/architecture/dna/codegen/ui_dna.md
docs/architecture/dna/codegen/training_dna.md
docs/architecture/dna/codegen/reference_traces_dna.md
```

README должен явно показывать два контура:

```text
Unconscious contour:
    M1 → M5 → M11 → M13/M4 → M2 → M5 → M3 guard

Conscious contour:
    M1/M5 → M10 → M9 → M7 → M15 → M12/M14/M13/M4 → M3 / M5 seed bus
```

И уровни:

```text
Level 1: wiring and contracts
Level 2: simulated learned latents/content
Level 3: downstream behavior as if trained
Level 4: replace imitators with real training
Level 5: compare real latents/content with prototype/imit latents/content
```

---

# 4. Проверить Architecture DNA manifest

Файл:

```text
docs/architecture/dna/architecture_dna_manifest.json
```

Должен быть валидным JSON.

Проверить:

```text
root = docs/architecture/dna
required_docs содержит все DNA docs
sub_layers.codegen = docs/architecture/dna/codegen
contours.unconscious есть
contours.conscious есть
rules содержит:
    dna_docs_live_under_docs_architecture_dna
    code_dna_lives_under_docs_architecture_dna_codegen
    both_unconscious_and_conscious_contours_are_documented
    imitators_live_under_module_imit_dir
    M2_and_M15_use_common_M5_seed_bus
    M5_seeds_pass_through_FocusFeedbackBoundary
    M3_motor_output_blocked_in_sleep
```

---

# 5. Проверить Code DNA manifest

Файл:

```text
docs/architecture/dna/codegen/codegen_manifest.json
```

Должен быть валидным JSON.

Проверить:

```text
root = docs/architecture/dna/codegen
required_docs содержит все codegen docs
minimum_regeneration_targets содержит:
    M1 object slot path
    M5 seed bus and boundary
    M3 sleep/action guard
    M8 status and IPC
    unconscious contour
    conscious contour
    imitator to training transition
```

---

# 6. Проверить сознательный контур и M9

Файл:

```text
docs/architecture/dna/conscious_loop_dna.md
```

В overview M9 должен быть прямо в основной цепочке, а не только как context provider.

Правильный порядок:

```text
M1 Object Imagery
↓
M5 World Model / Attention Workspace
↓
M10 Global Conscious Broadcast
↓
M9 Self Core / Agency / Body Ownership grounding
↓
M7 Inner Speech / Thought Stream
↓
M15 Counterfactual Imagination / Planning
↓
M12 Metacognitive confidence / uncertainty check
↓
M3 Action Proposal / Action Guard
↓
M8 monitors / subjective stream UI
```

Проверить:

```text
M10 идёт до M9
M9 идёт до M7
M7 идёт до M15
M12 идёт до M3
```

Проверить, что документ объясняет:

```text
M9 grounds conscious content in I/body/agency
M12 checks confidence/uncertainty/doubt before action
M15 must use common M5 seed bus and FocusFeedbackBoundary
M3 remains blocked in sleep/dream motor output
```

---

# 7. Проверить architecture_genome.md

Файл:

```text
docs/architecture/dna/architecture_genome.md
```

Должен содержать:

```text
M1–M15 module map
Two-contour architecture
Strict unconscious loop
Strict conscious loop
Sleep/replay invariants
Common buses
Non-negotiable rules
```

Проверить, что в strict conscious loop M9 явно стоит перед M7:

```text
M10 global conscious broadcast
↓
M9 self core / agency / body ownership grounding
↓
M7 inner speech / subjective stream
```

---

# 8. Проверить module_contracts_dna.md

Файл:

```text
docs/architecture/dna/module_contracts_dna.md
```

Должны быть contracts минимум для:

```text
M1
M2
M3
M5
M7
M8
M9
M10
M12
M15
```

Обязательно проверить правила:

```text
M1 imit z_obj latents must not be forced through raw sensory fusion
M2 must not write focus_context directly
M3 blocks outward motor output in sleep/replay
M5 seeds pass through FocusFeedbackBoundary
M7 verbalizes, does not drive M3 directly
M9 grounds conscious content in I/body/agency
M10 selects/broadcasts, does not mutate memory/motor directly
M15 uses common M5 seed bus and does not write focus_context directly
```

---

# 9. Проверить validation_levels_1_5_dna.md

Файл:

```text
docs/architecture/dna/validation_levels_1_5_dna.md
```

Должны быть уровни:

```text
Level 1 — wiring and contracts
Level 2 — simulated learned latents/content
Level 3 — downstream behavior as if trained
Level 4 — replace imit with real training
Level 5 — compare real latents vs prototype latents/content
```

Проверить, что уровни относятся и к бессознательному, и к сознательному контуру.

Level 2 должен включать:

```text
M1 imit
M5 imit
M11 probe
M13 imit
M2 imit
M10 conscious broadcast probe
M7 inner speech probe
M15 counterfactual planning probe
M12 confidence/uncertainty probe
```

---

# 10. Проверить Code DNA: module_api_specs.md

Файл:

```text
docs/architecture/dna/codegen/module_api_specs.md
```

Должны быть публичные API/specs для M1–M15.

Минимально проверить наличие:

```text
ObjectImageryRuntimeMixin
M1ObjectSlotLatentImitRuntimeMixin
EventDreamReplayRuntimeMixin
ActionRuntimeMixin
WorldModelWorkspaceRuntimeMixin
M5LatentPrototypeRuntimeMixin
InnerSpeechRuntimeMixin
SelfCoreRuntimeMixin
GlobalConsciousBroadcastRuntimeMixin
MotivationalHomeostasisRuntimeMixin
MetacognitionRuntimeMixin
AutobiographicalMemoryRuntimeMixin
SemanticGroundingRuntimeMixin
CounterfactualPlanningRuntimeMixin
```

Проверить, что M1/M5/M10/M15 имеют explicit contracts.

---

# 11. Проверить Code DNA: runtime_wiring_dna.md

Файл:

```text
docs/architecture/dna/codegen/runtime_wiring_dna.md
```

Должны быть:

```text
root launch
UnifiedSystem composition
awake step order
sleep/replay step order
common M5 seed bus
status publish
```

Проверить awake step order содержит оба контура:

```text
M1
M5
M11
M4
M13
M2
M10
M9
M7
M15
M12
M3
M8
```

Проверить M5 seed bus priority явно описан:

```text
debug/imit seed
M15 conscious seed
M2 unconscious replay seed
none
```

---

# 12. Проверить Code DNA: file_inventory_dna_template.json

Файл:

```text
docs/architecture/dna/codegen/file_inventory_dna_template.json
```

Должен быть валидным JSON.

Проверить поля:

```text
required_fields_per_file:
    path
    module
    role
    public_classes
    public_methods
    inputs
    outputs
    status_keys
    tests
    notes
```

Проверить, что есть хотя бы starter entries для:

```text
src/apps/runner.py
src/modules/m01_object_imagery/runtime.py
src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py
```

---

# 13. Проверить Code DNA: config/ui/training/reference traces

Проверить файлы:

```text
docs/architecture/dna/codegen/config_dna.md
docs/architecture/dna/codegen/ui_dna.md
docs/architecture/dna/codegen/training_dna.md
docs/architecture/dna/codegen/reference_traces_dna.md
docs/architecture/dna/codegen/code_generation_order.md
```

Ожидания:

## config_dna.md

```text
object_image.latent_dim
object_image.num_slots
self_core.focus_context_dim
sleep gates
sensor mask
imitators
debug_ui
```

## ui_dna.md

```text
M8 buttons
IPC action messages
status windows
Inner Object 3D requirements
conscious monitors
```

## training_dna.md

```text
imit output contract → trained output contract
M1 training target
M5 training target
M2 training target
M10/M7/M15/M12 conscious training target
guards must not be disabled
```

## reference_traces_dna.md

```text
Awake baseline
Sleep/replay guard
M1 cube/tetra slots
M5 latent prototype
Replay quality
Conscious contour probe
```

## code_generation_order.md

```text
package skeleton
shared config/types
M8 status/IPC
safety gates
M1 object slots
M5 seed bus
unconscious contour
conscious contour
real training
Level 5 comparison
```

---

# 14. Запустить docs tests

Запустить:

```bash
python -m py_compile tests/docs/test_architecture_dna_docs.py
python -m py_compile tests/docs/test_architecture_conscious_loop_dna.py
python -m py_compile tests/docs/test_conscious_loop_m9_position.py
python -m py_compile tests/docs/test_architecture_code_dna_docs.py
python -m py_compile tests/docs/test_readme_architecture_dna_structure.py
```

Затем:

```bash
pytest tests/docs
```

Если какие-то тесты отсутствуют в текущей ветке — отметить это и проверить существующие.

---

# 15. Дополнительно: проверить links / duplicate docs

Проверить:

```bash
find docs/architecture -maxdepth 3 -type f | sort
```

Убедиться:

```text
DNA docs только в docs/architecture/dna/
Code DNA docs только в docs/architecture/dna/codegen/
README.md указывает на актуальные paths
нет дубликатов старых root DNA docs
```

---

# 16. PASS criteria

PASS, если:

```text
✅ docs/architecture/dna/ существует
✅ docs/architecture/dna/codegen/ существует
✅ README.md обновлён новой структурой
✅ Architecture DNA manifest валиден
✅ Code DNA manifest валиден
✅ unconscious_loop_dna.md есть
✅ conscious_loop_dna.md есть
✅ M9 явно в conscious contour перед M7
✅ M12 явно перед action guard
✅ module contracts включают сознательный и бессознательный контуры
✅ validation levels 1–5 описывают imit → training → comparison
✅ codegen specs описывают API, wiring, config, UI, training, traces
✅ tests/docs проходят
✅ старых root-level DNA дубликатов нет
```

---

# 17. FAIL criteria

FAIL, если:

```text
DNA docs лежат прямо в docs/architecture/
нет docs/architecture/dna/codegen/
README не упоминает DNA/codegen
conscious_loop_dna.md отсутствует
M9 только context provider, но не в основной цепочке
M15 пишет focus_context напрямую по документации
M2/M15 common seed bus не описан
M3 sleep guard не описан
Code DNA не содержит API specs
file_inventory_dna_template.json невалидный
tests/docs падают
```

---

# 18. Что можно исправлять

Разрешены минимальные исправления:

```text
добавить недостающие docs
перенести DNA docs в docs/architecture/dna/
добавить/исправить README links
обновить manifest JSON
обновить tests/docs
добавить M9 в conscious contour
добавить codegen docs
исправить устаревшие пути
удалить root-level DNA дубликаты
```

Не делать:

```text
не переписывать runtime code
не менять module code
не менять architecture semantics без отдельного обсуждения
не удалять current_structure/module_file_map/module_migration_plan
```

---

# 19. Финальный отчёт Codex

В конце написать:

```text
1. Commit/hash checked.
2. README result.
3. docs/architecture structure result.
4. docs/architecture/dna result.
5. docs/architecture/dna/codegen result.
6. Manifest validation result.
7. Conscious contour / M9 result.
8. Unconscious contour result.
9. Module contracts result.
10. Validation levels 1–5 result.
11. Code DNA API/wiring/config/UI/training/traces result.
12. tests/docs result.
13. Any fixes made.
14. Remaining risks.
```

Финальная строка:

```text
Architecture DNA / Code DNA documentation verified:
README, dna folder, codegen layer, unconscious and conscious contours, M9 self-core grounding, manifests, module contracts, validation levels, and docs tests are consistent and sufficient as a reconstruction scaffold.
```
