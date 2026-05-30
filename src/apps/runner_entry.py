from __future__ import annotations

"""Slim Hydra entrypoint for the unified DMoC runner.

This module keeps the heavy runtime class in `src/apps/runner.py`, but uses
app-level helpers for config normalization, behavior-preserving runtime patching
and system construction.
"""

import os
from pathlib import Path

import hydra

import src.apps.runner as runner_runtime
from src.apps.runner import UnifiedSystem
from src.apps.runner_config import build_runner_config, render_resolved_runner_config
from src.apps.runner_patches import apply_runner_patches
from src.apps.runner_system_factory import build_unified_system

apply_runner_patches(runner_runtime, UnifiedSystem)

os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))


@hydra.main(version_base=None, config_path="../../config", config_name="runner")
def main(cfg_raw) -> None:
    print("Resolved config:\n" + render_resolved_runner_config(cfg_raw))
    cfg_obj = build_runner_config(cfg_raw)

    system = build_unified_system(cfg_obj, UnifiedSystem)
    system.run()


if __name__ == "__main__":
    main()
