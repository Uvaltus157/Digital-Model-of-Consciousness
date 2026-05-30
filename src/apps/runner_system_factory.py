from __future__ import annotations

"""System factory for the slim unified runner path.

This module centralizes construction and post-construction normalization of the
unified runtime. The heavy system class lives in `src/apps/runner.py`, but the
entrypoint no longer needs to know the full startup sequence.
"""

from dataclasses import dataclass
from typing import Any, Type

from src.apps.runner_modes import apply_runner_mode
from src.apps.runner_runtime_state import RuntimeStateSnapshot, apply_runtime_state
from src.apps.runner_services import RunnerServiceSnapshot, ensure_runner_services
from src.apps.runner_startup_state import StartupStateSnapshot, apply_startup_state


@dataclass(frozen=True)
class BuiltSystemContext:
    """Diagnostics for the normalized system construction path."""

    runtime_state: RuntimeStateSnapshot
    startup_state: StartupStateSnapshot
    services: RunnerServiceSnapshot
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_state": self.runtime_state.to_dict(),
            "startup_state": self.startup_state.to_dict(),
            "services": self.services.to_dict(),
            "mode": self.mode,
        }


def build_unified_system(
    cfg: Any,
    system_cls: Type[Any],
    *,
    return_context: bool = False,
) -> Any | tuple[Any, BuiltSystemContext]:
    """Build and normalize the unified runtime system.

    The order is important and mirrors the current slim entrypoint:

    1. construct runtime class;
    2. normalize runtime bookkeeping;
    3. normalize startup/window/sensor flags;
    4. ensure app-level services;
    5. apply mode-specific startup behavior.
    """
    system = system_cls(cfg)
    runtime_state = apply_runtime_state(system, cfg)
    startup_state = apply_startup_state(system, cfg)
    services = ensure_runner_services(system, cfg)
    mode = apply_runner_mode(system, cfg)

    context = BuiltSystemContext(
        runtime_state=runtime_state,
        startup_state=startup_state,
        services=services,
        mode=mode,
    )
    if return_context:
        return system, context
    return system


__all__ = ["BuiltSystemContext", "build_unified_system"]
