from __future__ import annotations

"""Module training flag resolution for the V5.10 runner.

This is part of the gradual runner unload. The logic was originally embedded
inside `UnifiedSystem.resolve_module_training_flags_from_config()`.
"""

from typing import Any, Dict

DEFAULT_MODULE_TRAINING_FLAGS: Dict[str, bool] = {
    "world_model": True,
    "object_imagery": True,
    "core_model": True,
    "action_heads": True,
    "leg_control": True,
    "self_core": True,
    "inner_speech": True,
    "long_dynamic_memory": True,
}

MODULE_MODE_ALIASES: Dict[str, bool] = {
    "train": True,
    "training": True,
    "on": True,
    "enabled": True,
    "active": True,
    "true": True,
    "1": True,
    "passive": False,
    "frozen": False,
    "freeze": False,
    "off": False,
    "disabled": False,
    "false": False,
    "0": False,
}


def resolve_module_training_flags_from_config_object(cfg: Any) -> Dict[str, bool]:
    """Resolve startup module train/passive flags from a config object.

    YAML example:

    ```yaml
    module_debug:
      module_modes:
        world_model: train
        object_imagery: passive
    ```

    Values:
    - `train` means the module participates in optimizer/train step.
    - `passive` means the module is used in life/forward but parameters freeze.
    """
    md = getattr(cfg, "module_debug", None)
    modes = getattr(md, "module_modes", None) if md is not None else None

    if not isinstance(modes, dict) or len(modes) == 0:
        print("[module_debug] module_modes missing/empty; using all modules in train mode")
        return dict(DEFAULT_MODULE_TRAINING_FLAGS)

    flags = dict(DEFAULT_MODULE_TRAINING_FLAGS)
    for key, value in modes.items():
        key = str(key).strip()
        if key not in flags:
            print(f"[module_debug] unknown module_modes key ignored: {key!r}")
            continue
        if isinstance(value, bool):
            flags[key] = bool(value)
            continue
        mode = str(value).lower().strip()
        if mode not in MODULE_MODE_ALIASES:
            print(f"[module_debug] unknown mode for {key}: {value!r}; using default=train")
            continue
        flags[key] = bool(MODULE_MODE_ALIASES[mode])

    print("[module_debug] startup module modes -> " + ", ".join(
        f"{k}={'train' if v else 'passive'}" for k, v in flags.items()
    ))
    return flags


def resolve_module_training_flags_for_system(system: Any) -> Dict[str, bool]:
    """Method-compatible wrapper for `UnifiedSystem` instances."""
    return resolve_module_training_flags_from_config_object(system.cfg)
