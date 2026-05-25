from __future__ import annotations

"""Application-level system factory placeholder.

This file is intentionally small for now. The current V5.10 runner still owns
runtime class composition directly in `src/apps/runner.py`.

Future refactor target:

    cfg = load_config()
    system = build_unified_system(cfg)
    system.run()

The extraction should be behavior-preserving and should not change Hydra CLI
semantics or the root `runner.py` compatibility launcher.
"""

from typing import Any


def build_unified_system(*args: Any, **kwargs: Any) -> Any:
    """Reserved factory hook for the future runner decomposition.

    The current runner is still the source of truth. This function is not used
    yet because moving construction here requires careful import-cycle and
    Hydra smoke checks.
    """
    raise NotImplementedError(
        "build_unified_system is a reserved refactor hook. "
        "Use src/apps/runner.py until the behavior-preserving extraction is completed."
    )
