from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "modules" / "m08_debug_visual_control" / "control_panel.py"


def main() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
