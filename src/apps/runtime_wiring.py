from __future__ import annotations

"""Runtime wiring boundary for cross-module integration.

Purpose
-------
The long-term architecture should keep each M1-M15 module focused on its own
semantic responsibility. Cross-module composition should happen here or in a
similar app-level wiring layer instead of inside one module runtime.

Current state
-------------
`src/apps/runner.py` still performs the actual V5.10 runtime composition.
This module documents the boundary and provides small neutral data structures
that future refactors can use without changing current behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RuntimeWiringContext:
    """Shared app-level context for future module wiring.

    This is intentionally generic and dependency-light. It should not import
    MuJoCo, Torch, Open3D, PyQt, or the heavy runner.
    """

    modules: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def register_module(self, name: str, module: Any) -> None:
        self.modules[str(name)] = module

    def register_service(self, name: str, service: Any) -> None:
        self.services[str(name)] = service

    def get_module(self, name: str, default: Any = None) -> Any:
        return self.modules.get(str(name), default)

    def get_service(self, name: str, default: Any = None) -> Any:
        return self.services.get(str(name), default)
