from __future__ import annotations

"""Runner mode handling for the slim V5.10 entrypoint.

This module extracts the startup mode block from `runner_entry.py` so the Hydra
entrypoint stays focused on config loading, system construction and `run()`.
"""

from typing import Any

TRAIN_MODES = {"train", "training"}


def normalize_runner_mode(cfg: Any) -> str:
    """Return normalized runner mode string from a config object."""
    return str(getattr(cfg, "mode", "run")).lower().strip()


def apply_runner_mode(system: Any, cfg: Any) -> str:
    """Apply mode-specific startup changes to an already constructed system.

    Currently only `train` / `training` needs special treatment. Training is a
    parallel thread owned by `system.run()`, not a separate sequential loop.

    Returns the normalized mode for diagnostics/tests.
    """
    mode = normalize_runner_mode(cfg)
    if mode not in TRAIN_MODES:
        return mode

    system.training_enabled = bool(getattr(cfg.control_startup, "training", False))
    if system.training_enabled:
        try:
            system.cfg.train.enabled = True
        except Exception:
            pass
        print("[mode=train] parallel train thread enabled; starting normal life run")
    else:
        print("[mode=train] training stays OFF because control_startup.training=false")
    return mode
