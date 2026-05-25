from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = ROOT / "src" / "apps" / "runner_entry.py"
LEGACY_TARGET = ROOT / "src" / "apps" / "runner.py"


def resolve_target() -> Path:
    """Return the runner target.

    By default the root launcher now enters through the slim Hydra entrypoint
    (`runner_entry.py`). Set `CWMS_LEGACY_RUNNER=1` to route directly to the
    legacy heavy runner module while the migration is being validated.
    """
    if os.environ.get("CWMS_LEGACY_RUNNER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return LEGACY_TARGET
    return DEFAULT_TARGET


def normalize_config_path_arg() -> None:
    normalized: list[str] = []
    skip_next = False
    for idx, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--config-path" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            path = Path(value).expanduser()
            normalized.append(arg)
            normalized.append(str(path if path.is_absolute() else ROOT / path))
            skip_next = True
        elif arg.startswith("--config-path="):
            value = arg.split("=", 1)[1]
            path = Path(value).expanduser()
            normalized.append(f"--config-path={path if path.is_absolute() else ROOT / path}")
        else:
            normalized.append(arg)
    sys.argv[:] = normalized


def main() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    normalize_config_path_arg()
    runpy.run_path(str(resolve_target()), run_name="__main__")


if __name__ == "__main__":
    main()
