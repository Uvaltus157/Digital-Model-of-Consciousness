from __future__ import annotations

"""Slim Hydra entrypoint for the V5.10 runner.

This module keeps the heavy runtime class in `src/apps/runner.py` for now, but
uses small app-level helpers for config normalization, behavior-preserving
runtime patching, runtime-state normalization, startup-state normalization,
service startup and mode handling.
"""

import os
from pathlib import Path

import hydra

import src.apps.runner as runner_runtime
from src.apps.runner import UnifiedSystemV510
from src.apps.runner_config import build_runner_config, render_resolved_runner_config
from src.apps.runner_modes import apply_runner_mode
from src.apps.runner_patches import apply_runner_patches
from src.apps.runner_runtime_state import apply_runtime_state
from src.apps.runner_services import ensure_runner_services
from src.apps.runner_startup_state import apply_startup_state

apply_runner_patches(runner_runtime, UnifiedSystemV510)

os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))


@hydra.main(version_base=None, config_path="../../config", config_name="runner")
def main(cfg_raw) -> None:
    print("Resolved config:\n" + render_resolved_runner_config(cfg_raw))
    cfg_obj = build_runner_config(cfg_raw)

    system = UnifiedSystemV510(cfg_obj)
    apply_runtime_state(system, cfg_obj)
    apply_startup_state(system, cfg_obj)
    ensure_runner_services(system, cfg_obj)
    apply_runner_mode(system, cfg_obj)
    system.run()


if __name__ == "__main__":
    main()
