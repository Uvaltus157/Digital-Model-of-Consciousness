# Supervisor folder layout

Repository:

```text
Conscious-World-Model-System
```

Place this whole folder at:

```text
<repo_root>/supervisor
```

For your server that means:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System/supervisor
```

Run from the repository root:

```bash
python supervisor/run_codex_full_autonomous_supervisor.py
```

You may also run from inside the supervisor folder:

```bash
cd supervisor
python run_codex_full_autonomous_supervisor.py
```

The launcher automatically uses the repository root as its working directory.

The launcher also appends the full JSON task into the prompt, including:

```text
must_verify
forbidden
success_marker
diagnostic_log
```

Task JSON:

```text
supervisor/tasks/full_autonomous_task.json
```

Logs are written to:

```text
/home/user/_project/temp/logs
```

Main diagnostic log:

```text
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

Supervisor log:

```text
/home/user/_project/temp/logs/supervisor_run.log
```
