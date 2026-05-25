from __future__ import annotations

"""Application bootstrap helpers.

The root `runner.py` currently prepares `PROJECT_ROOT`, normalizes Hydra
`--config-path`, and delegates to `src/apps/runner.py`.

This module is a lightweight placeholder for future behavior-preserving
bootstrap extraction. It deliberately avoids importing the heavy runner.
"""

from pathlib import Path
import os
import sys


def ensure_project_root(project_root: str | Path) -> Path:
    """Set PROJECT_ROOT and ensure the repository root is on sys.path."""
    root = Path(project_root).resolve()
    os.environ.setdefault("PROJECT_ROOT", str(root))
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
