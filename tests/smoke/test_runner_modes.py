from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_modes import apply_runner_mode, normalize_runner_mode


def _cfg(mode="run", startup_training=False, train_enabled=False):
    return SimpleNamespace(
        mode=mode,
        control_startup=SimpleNamespace(training=startup_training),
        train=SimpleNamespace(enabled=train_enabled),
    )


def test_normalize_runner_mode() -> None:
    assert normalize_runner_mode(_cfg("TRAIN")) == "train"
    assert normalize_runner_mode(_cfg(" training ")) == "training"
    assert normalize_runner_mode(SimpleNamespace()) == "run"


def test_apply_runner_mode_run_does_not_change_training() -> None:
    cfg = _cfg("run", startup_training=True, train_enabled=False)
    system = SimpleNamespace(training_enabled=False, cfg=cfg)

    mode = apply_runner_mode(system, cfg)

    assert mode == "run"
    assert system.training_enabled is False
    assert system.cfg.train.enabled is False


def test_apply_runner_mode_train_enables_parallel_training_when_startup_flag_true() -> None:
    cfg = _cfg("train", startup_training=True, train_enabled=False)
    system = SimpleNamespace(training_enabled=False, cfg=cfg)

    mode = apply_runner_mode(system, cfg)

    assert mode == "train"
    assert system.training_enabled is True
    assert system.cfg.train.enabled is True


def test_apply_runner_mode_train_keeps_training_off_when_startup_flag_false() -> None:
    cfg = _cfg("training", startup_training=False, train_enabled=False)
    system = SimpleNamespace(training_enabled=True, cfg=cfg)

    mode = apply_runner_mode(system, cfg)

    assert mode == "training"
    assert system.training_enabled is False
    assert system.cfg.train.enabled is False
