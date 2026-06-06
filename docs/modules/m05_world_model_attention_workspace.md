# M5 — World Model / Attention Workspace

## Общее описание

M5 — это **центральный модуль “модель мира + внимание + рабочее пространство”**. В проекте он оформлен как canonical `ConsciousDreamer`: внешний код должен брать текущую модель именно из `conscious_dreamer.py`, через `ConsciousDreamer`, `ConsciousDreamerConfig`, `make_conscious_dreamer_config_from_world()`.

По смыслу M5 — это “центр сборки сознательного состояния”. Он принимает разные сигналы от тела и мира, собирает их в общую латентную модель, выбирает фокус внимания, строит внутреннее состояние, делает предсказания, формирует план действия и отдаёт наружу результат для других модулей.

### Что входит в M5

Внутри M5 сейчас есть несколько слоёв:

```text
ConsciousDreamerCore
  → ConsciousDreamerMemoryThought
    → ConsciousDreamerInnerSpeech
      → ConsciousDreamerObjectImagery
        → ConsciousDreamer
```

Canonical `ConsciousDreamer` сейчас указывает на самый полный слой `ConsciousDreamerObjectImagery`, то есть финальная M5-модель включает object imagery поверх внутренней речи, памяти, мыслительного цикла и базовой модели мира.

### Основная задача M5

Главная задача M5 — **собрать единое внутреннее состояние из разных потоков**:

```text
левое/правое зрение
глубина
поза
body_state
тактильные сигналы
моторика рук
действие тела
object_state
прошлое внутреннее состояние
```

В базовом ядре M5 прямо указано, что оно включает мультимодальные энкодеры, AttentionController, ConsciousPlanner, ImaginationCore, ReflectiveLoop, object representation, предсказание действий тела/рук и стабильный output contract.

### Как идёт поток внутри M5

Упрощённо шаг M5 выглядит так:

```text
1. Закодировать сенсоры
2. Выбрать фокус внимания
3. Слить всё в общий latent
4. Обновить внутреннее состояние мира
5. Сформировать workspace
6. Сформировать self/body state
7. Сформировать object representation
8. Сделать reflection / confidence
9. Вообразить возможное будущее
10. Выбрать действие
11. Предсказать RGB/depth/reward/continue
12. Вернуть общий output для runner и других модулей
```

В коде это видно в `step()`: M5 кодирует vision, pose, body, tactile, motor, object_state и action, потом прогоняет их через attention, fusion, RSSM, workspace, self_model, object_repr, reflective_loop, imagination, planner и decoder.

### Что такое attention/workspace в M5

AttentionController получает набор модальностей:

```text
vision
pose
body
tactile
hand_motor
object
action
```

и вычисляет:

```text
tokens
context
workspace_seed
modality_weights
attention matrix
focus logits
focus index
```

То есть он решает: **какая информация сейчас важнее**, на что направить фокус, и что отправить в рабочее пространство.

Workspace потом превращает текущее состояние RSSM + seed внимания в:

```text
workspace
thought
report
```

То есть в “рабочую сцену сознания”, из которой потом рождаются мысль, отчёт, план и действие.

### Что M5 хранит в состоянии

`initial_state()` создаёт минимальное внутреннее состояние:

```text
rssm
prev_action_ids
prev_embodied_action
prev_hand_motor
```

То есть M5 помнит не только “что видит сейчас”, но и предыдущее действие, предыдущее тело/руки и латентное состояние мира.

### Что M5 выдаёт наружу

После одного шага M5 возвращает большой словарь, где есть:

```text
state
obs_embed
rssm
attention
workspace_out
thoughts
report
selves
reflection_out
object_repr
values
focus
action_logits
action_ids
embodied_targets
hand_ctrl
imagined
decoder
```

То есть M5 одновременно отдаёт: внутреннюю модель мира, внимание, мысль, self-состояние, объектное представление, план действия, моторные команды, воображаемые будущие состояния и декодированные предсказания.

### Слой памяти и мысли внутри M5

Поверх базового ядра есть слой `ConsciousDreamerMemoryThought`. Он добавляет:

```text
ThoughtLoop
AutobiographicalMemory
```

и сохраняет тот же `step()` API со стабильными output-ключами.

Память там работает как онлайн-эпизодическая память: она умеет читать контекст по query и записывать episode без градиентов. На шаге M5 читает memory_context, прогоняет thought_loop, получает final_thought, потом использует это для reflection, imagination и planner.

### Слой внутренней речи

Следующий слой — `ConsciousDreamerInnerSpeech`. Он добавляет встроенный `InnerSpeechLoop` / symbolic report layer. Каждый шаг возвращает `out["symbolic_report"]`, где есть:

```text
report_latent
inner_speech_sequence
symbol_ids
phoneme_ids
text_token_ids
confidence
```

То есть M5 не только “думает” в латентном виде, но ещё пытается перевести внутреннее состояние в символический отчёт: условную внутреннюю речь.

### Слой object imagery

Финальный слой — `ConsciousDreamerObjectImagery`. Он добавляет `ObjectImageryDecoder`, который строит object imagery из:

```text
object_repr
workspace
thought
reflection
```

и кладёт результат в `out["object_imagery"]`.

То есть M5 завершает поток так: из общей модели мира и внимания появляется **внутренний образ объекта**.

### Простыми словами

M5 — это не просто “нейросеть”. Это центральный узел, который делает примерно следующее:

```text
Я вижу / чувствую / двигаюсь
→ понимаю, что сейчас важно
→ обновляю модель мира
→ понимаю, где я и что происходит
→ вспоминаю похожие эпизоды
→ формирую мысль
→ оцениваю уверенность и ценность
→ воображаю возможные последствия
→ выбираю действие
→ выдаю команды телу/рукам
→ строю внутренний образ объекта
→ формирую внутренний символический отчёт
```

Если совсем коротко: **M5 — это центральное “сознательное рабочее пространство” DMoC, где сенсоры, память, внимание, self-модель, воображение, речь и действие собираются в один текущий поток состояния.**
