from __future__ import annotations

"""CPU affinity helpers for runner-owned background threads."""

import os
from threading import Thread
from typing import Any, Iterable


def _enabled(cfg: Any) -> bool:
    affinity_cfg = getattr(cfg, "thread_affinity", None)
    return bool(getattr(affinity_cfg, "enabled", False))


def _strict(cfg: Any) -> bool:
    affinity_cfg = getattr(cfg, "thread_affinity", None)
    return bool(getattr(affinity_cfg, "strict", False))


def _configured_cpu_ids(cfg: Any, key: str) -> set[int]:
    affinity_cfg = getattr(cfg, "thread_affinity", None)
    raw_map = getattr(affinity_cfg, "cpus", {}) or {}
    if key not in raw_map:
        return set()

    value = raw_map[key]
    if value is None:
        return set()
    if isinstance(value, int):
        return {value} if value >= 0 else set()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return {int(part) for part in parts if part}
    if isinstance(value, Iterable):
        return {int(item) for item in value if int(item) >= 0}
    raise TypeError(f"Unsupported CPU affinity value for {key!r}: {value!r}")


def apply_thread_affinity(cfg: Any, key: str, thread: Thread | None, *, label: str | None = None) -> bool:
    """Pin a started Python thread to configured CPUs when supported.

    Linux exposes Python thread native ids as schedulable task ids. On other
    platforms, or when the current process cpuset does not allow the requested
    CPU, this helper either warns and continues or raises when strict mode is
    enabled.
    """
    if not _enabled(cfg):
        return False
    name = label or key
    try:
        cpu_ids = _configured_cpu_ids(cfg, key)
        if not cpu_ids:
            return False
        if thread is None:
            raise RuntimeError("thread object is not available")
        native_id = getattr(thread, "native_id", None)
        if native_id is None:
            raise RuntimeError("thread native_id is not available yet")
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("os.sched_setaffinity is not available on this platform")

        allowed = set(os.sched_getaffinity(0))
        unavailable = sorted(cpu_ids - allowed)
        if unavailable:
            raise ValueError(
                f"requested CPUs {unavailable} are outside current process affinity {sorted(allowed)}"
            )

        os.sched_setaffinity(int(native_id), cpu_ids)
        print(f"[thread_affinity] {name} -> CPUs {sorted(cpu_ids)}")
        return True
    except Exception as exc:
        message = f"[thread_affinity] {name} skipped: {exc}"
        if _strict(cfg):
            raise RuntimeError(message) from exc
        print(message)
        return False
