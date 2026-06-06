# M8 Module Lab button

Adds a button to the M8 tab in `pyqt_control_panel_ipc.py` / `control_panel.py`:

```text
Run Module Lab
```

The button sends IPC:

```json
{
  "type": "action",
  "action": "module_lab_run",
  "payload": {
    "module": "all",
    "source": "m8_control_panel"
  }
}
```

Runner stores:

```python
self.last_module_lab_result
```

and status IPC exposes:

```text
last_module_lab_result
```
