from __future__ import annotations

"""Control panel launcher that uses the registry-backed Module Debug window.

This keeps the large existing `control_panel.py` untouched, but changes the
runtime default by injecting:

    --module-debug-script pyqt_module_debug_ipc_status_registry.py

before delegating to `control_panel.main()`.

Use this launcher as the safe registry-backed control panel entrypoint while
`control_panel.py` is gradually refactored.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "runner.yaml").exists():
            sys.path.insert(0, str(parent))
            break

import sys

from src.modules.m08_debug_visual_control import control_panel


def main() -> None:
    if "--module-debug-script" not in sys.argv:
        sys.argv.extend([
            "--module-debug-script",
            "pyqt_module_debug_ipc_status_registry.py",
        ])
    control_panel.main()


if __name__ == "__main__":
    main()
