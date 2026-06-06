# M8 Module Lab full button set

The Module Lab window must expose separate buttons:

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

IPC mapping:

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
