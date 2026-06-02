from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from src.apps.runner_config import build_runner_config
from src.apps.runner_thread_affinity import apply_thread_affinity


def test_runner_config_keeps_thread_affinity_section() -> None:
    cfg = build_runner_config(
        OmegaConf.create(
            {
                "thread_affinity": {
                    "enabled": True,
                    "strict": True,
                    "cpus": {
                        "train": 1,
                        "mujoco_viewer": [2, 3],
                    },
                }
            }
        )
    )

    assert cfg.thread_affinity.enabled is True
    assert cfg.thread_affinity.strict is True
    assert cfg.thread_affinity.cpus["train"] == 1
    assert cfg.thread_affinity.cpus["mujoco_viewer"] == [2, 3]


def test_apply_thread_affinity_uses_thread_native_id(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr("src.apps.runner_thread_affinity.os.sched_getaffinity", lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr(
        "src.apps.runner_thread_affinity.os.sched_setaffinity",
        lambda pid, cpus: calls.append((pid, set(cpus))),
    )

    cfg = SimpleNamespace(
        thread_affinity=SimpleNamespace(
            enabled=True,
            strict=True,
            cpus={"train": [1, 2]},
        )
    )
    thread = SimpleNamespace(native_id=12345)

    assert apply_thread_affinity(cfg, "train", thread) is True
    assert calls == [(12345, {1, 2})]
