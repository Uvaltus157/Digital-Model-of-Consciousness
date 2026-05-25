from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_training_flags import (
    DEFAULT_MODULE_TRAINING_FLAGS,
    resolve_module_training_flags_from_config_object,
)


def _cfg(module_modes=None):
    return SimpleNamespace(module_debug=SimpleNamespace(module_modes=module_modes))


def test_missing_module_modes_defaults_to_all_train() -> None:
    flags = resolve_module_training_flags_from_config_object(_cfg(None))
    assert flags == DEFAULT_MODULE_TRAINING_FLAGS
    assert all(flags.values())


def test_train_and_passive_aliases_are_resolved() -> None:
    flags = resolve_module_training_flags_from_config_object(
        _cfg(
            {
                "world_model": "train",
                "object_imagery": "passive",
                "self_core": "enabled",
                "inner_speech": "disabled",
                "long_dynamic_memory": False,
            }
        )
    )
    assert flags["world_model"] is True
    assert flags["object_imagery"] is False
    assert flags["self_core"] is True
    assert flags["inner_speech"] is False
    assert flags["long_dynamic_memory"] is False


def test_unknown_key_is_ignored_and_unknown_mode_keeps_default() -> None:
    flags = resolve_module_training_flags_from_config_object(
        _cfg(
            {
                "unknown_module": "passive",
                "action_heads": "strange-mode",
            }
        )
    )
    assert "unknown_module" not in flags
    assert flags["action_heads"] is True
