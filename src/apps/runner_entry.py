from __future__ import annotations

"""Slim Hydra entrypoint for the V5.10 runner.

This module keeps the heavy runtime class in `src/apps/runner.py` for now, but
moves startup config normalization, teacher loading, training flag resolution,
hover safety patching and the outer life loop into small app-level helpers.
"""

import os
from pathlib import Path

import hydra

import src.apps.runner as runner_runtime
from src.apps.runner import UnifiedSystemV510
from src.apps.runner_config import build_runner_config, render_resolved_runner_config
from src.apps.runner_hover_config import force_hover_flight_runtime_config_for_system
from src.apps.runner_loop import run_unified_life_loop
from src.apps.runner_teachers import load_inner_speech_teacher_from_config
from src.apps.runner_training_flags import resolve_module_training_flags_for_system

# Behavior-preserving patches for the heavy runtime module.
# They avoid a risky full edit of the large runner.py while migration is ongoing.
runner_runtime.load_inner_speech_teacher_from_config = load_inner_speech_teacher_from_config
UnifiedSystemV510.run = run_unified_life_loop
UnifiedSystemV510.resolve_module_training_flags_from_config = resolve_module_training_flags_for_system
UnifiedSystemV510._force_hover_flight_runtime_config = force_hover_flight_runtime_config_for_system

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
