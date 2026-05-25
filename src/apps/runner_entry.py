from __future__ import annotations

"""Slim Hydra entrypoint for the V5.10 runner.

This module keeps the heavy runtime class in `src/apps/runner.py` for now, but
uses small app-level helpers for config normalization and behavior-preserving
runtime patching.
"""

import os
from pathlib import Path

import hydra

import src.apps.runner as runner_runtime
from src.apps.runner import UnifiedSystemV510
from src.apps.runner_config import build_runner_config, render_resolved_runner_config
from src.apps.runner_patches import apply_runner_patches

apply_runner_patches(runner_runtime, UnifiedSystemV510)

os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))


@hydra.main(version_base=None, config_path="../../config", config_name="runner")
def main(cfg_raw) -> None:
    print("Resolved config:\n" + render_resolved_runner_config(cfg_raw))
    cfg_obj = build_runner_config(cfg_raw)

    system = UnifiedSystemV510(cfg_obj)

    mode = str(getattr(cfg_obj, "mode", "run")).lower().strip()
    if mode in ("train", "training"):
        # Training is a parallel thread, not a separate sequential loop.
        # The inherited run() owns life_step and starts train_loop in background.
        system.training_enabled = bool(getattr(cfg_obj.control_startup, "training", False))
        if system.training_enabled:
            try:
                system.cfg.train.enabled = True
            except Exception:
                pass
            print("[mode=train] parallel train thread enabled; starting normal life run")
        else:
            print("[mode=train] training stays OFF because control_startup.training=false")

    system.run()


if __name__ == "__main__":
    main()
