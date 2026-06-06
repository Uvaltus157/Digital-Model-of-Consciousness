# M8 Module Lab window fix

The previous M8 patch added an IPC command button, but it did not open a window.

This fix changes the button behavior:

```text
M8 tab -> Module Lab button -> opens QDialog
```

The dialog has:

```text
Run all
Run loop
Run M2
Run scenarios
QPlainTextEdit result area
```

The result area is updated from:

```text
last_status["last_module_lab_result"]
```

which comes from runner status IPC.
