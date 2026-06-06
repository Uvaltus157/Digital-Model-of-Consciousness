# Per-module `imit/` layout

Central `scripts/module_lab/` remains only a launcher/registry.

Input imitation belongs to each module:

```text
src/modules/m01_object_imagery/imit/sensor_inputs.py
src/modules/m02_event_dream_replay/imit/event_dream_replay_inputs.py
src/modules/m03_self_action_causality/imit/motor_guard_inputs.py
src/modules/m04_long_dynamic_memory/imit/long_dynamic_memory_inputs.py
src/modules/m05_world_model_attention_workspace/imit/focus_feedback_inputs.py
src/modules/m11_motivational_homeostasis/imit/emotional_drive_inputs.py
src/modules/m13_autobiographical_memory/imit/autobiographical_memory_inputs.py
```

Registry:

```text
scripts/module_lab/module_imit_registry.py
scripts/module_lab/run_module_imit_registry.py
```

Run:

```bash
python scripts/module_lab/run_module_imit_registry.py --module all
pytest tests/module_contracts/test_per_module_imit_layout_contract.py
```

Architectural rule:

```text
M1 -> M5 -> M11 -> M2 -> FocusFeedbackBoundary -> M5
      |             ^
      v             |
     M4 ------------+
                   M13 -> M2
```

M2 imitator must not receive raw sensors (`left/right/depth`) directly.
M4 imitator must receive `inner_object / z_obj`, not raw sensors.
