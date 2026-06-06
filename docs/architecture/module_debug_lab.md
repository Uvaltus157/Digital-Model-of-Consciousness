# DMoC module debug lab

This lab checks modules in isolation without MuJoCo.

## Target unconscious loop

```text
M1 -> M5 -> M11 -> M2 -> M5 FocusFeedbackBoundary -> M5
      |             ^
      v             |
     M4 ------------+
                   M13 -> M2

M5 -> M3 -> body/world -> M1
```

Rules:

- M1 does not feed M2 directly.
- M2 does not mutate `out["focus_context"]` by default.
- M2 sends `replay_context` to the same M5 seed input as M15:
  `focus_context_seed` + `focus_context_seed_gate`.
- M4 is derived from `inner_object / z_obj`, not raw sensors.
- M13 feeds M2 with `retrieved_context`.
- M11 feeds M2 with affect/emotional pressure.

## Run all labs

```bash
python scripts/module_lab/run_module_lab.py --module all
```

## Run pytest contracts

```bash
pytest tests/module_contracts
```

## Compile checks

```bash
python -m py_compile scripts/module_lab/module_fixture_factory.py
python -m py_compile scripts/module_lab/run_module_lab.py
python -m py_compile tests/module_contracts/test_m02_event_dream_replay_contract.py
python -m py_compile tests/module_contracts/test_m02_runtime_seed_bus_contract.py
python -m py_compile tests/module_contracts/test_m04_long_dynamic_memory_contract.py
python -m py_compile tests/module_contracts/test_m05_focus_feedback_boundary_contract.py
python -m py_compile tests/module_contracts/test_m11_emotional_drive_contract.py
python -m py_compile tests/module_contracts/test_m13_autobiographical_memory_contract.py
python -m py_compile tests/module_contracts/test_unconscious_loop_contract.py
```
