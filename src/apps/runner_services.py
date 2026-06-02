from __future__ import annotations

"""Runtime service helpers for the V5.10 runner.

This module extracts the service-startup boundary from the heavy runner:

- Module Debug status IPC server;
- main IPC control server;
- initial external-control flags;
- sensor-preview metadata initialization.

`UnifiedSystem.__init__` still starts these services today. The
slim entrypoint calls `ensure_runner_services()` after construction. The helper
is idempotent: it starts a server only if the corresponding attribute is absent
or `None`, then re-applies lightweight metadata/flag initialization hooks.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.apps.runner_thread_affinity import apply_thread_affinity


@dataclass(frozen=True)
class RunnerServiceSnapshot:
    module_status_enabled: bool
    module_status_host: str
    module_status_port: int
    module_status_running: bool
    ipc_control_enabled: bool
    ipc_control_host: str
    ipc_control_port: int
    ipc_control_running: bool
    external_flags_initialized: bool
    sensor_preview_metadata_initialized: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _is_running_server(server: Any) -> bool:
    if server is None:
        return False
    # Most current service objects do not expose a stable `is_running()` API.
    # Attribute existence is enough for this idempotent startup boundary.
    return True


def _call_optional(system: Any, method_name: str) -> bool:
    method = getattr(system, method_name, None)
    if not callable(method):
        return False
    method()
    return True


def ensure_module_status_server(
    system: Any,
    cfg: Any,
    server_factory: Optional[Callable[[str, int], Any]] = None,
) -> bool:
    """Ensure Module Debug status IPC server exists when enabled."""
    ipc_cfg = getattr(cfg, "module_status_ipc", None)
    if not bool(getattr(ipc_cfg, "enabled", False)):
        system.module_status_server = getattr(system, "module_status_server", None)
        return False

    if _is_running_server(getattr(system, "module_status_server", None)):
        return True

    if server_factory is None:
        from src.modules.m08_debug_visual_control.module_debug_status_ipc import ModuleDebugStatusServer

        server_factory = ModuleDebugStatusServer

    server = server_factory(str(ipc_cfg.host), int(ipc_cfg.port))
    server.start()
    system.module_status_server = server
    apply_thread_affinity(cfg, "module_status", getattr(server, "thread", None), label="module status IPC")
    return True


def ensure_ipc_control_server(
    system: Any,
    cfg: Any,
    server_factory: Optional[Callable[[str, int], Any]] = None,
) -> bool:
    """Ensure main IPC control server exists when enabled."""
    ipc_cfg = getattr(cfg, "ipc_control", None)
    if not bool(getattr(ipc_cfg, "enabled", False)):
        system.ipc_server = getattr(system, "ipc_server", None)
        return False

    if _is_running_server(getattr(system, "ipc_server", None)):
        return True

    if server_factory is None:
        from src.platform.ipc.ipc_control_bus import IPCControlServer

        server_factory = IPCControlServer

    server = server_factory(str(ipc_cfg.host), int(ipc_cfg.port))
    server.start()
    system.ipc_server = server
    apply_thread_affinity(cfg, "ipc_control", getattr(server, "thread", None), label="IPC control")
    return True


def ensure_runner_services(
    system: Any,
    cfg: Any,
    module_status_server_factory: Optional[Callable[[str, int], Any]] = None,
    ipc_server_factory: Optional[Callable[[str, int], Any]] = None,
) -> RunnerServiceSnapshot:
    """Ensure app-level runtime services are present and metadata hooks ran."""
    module_status_running = ensure_module_status_server(system, cfg, module_status_server_factory)
    ipc_running = ensure_ipc_control_server(system, cfg, ipc_server_factory)

    external_flags_initialized = _call_optional(system, "_write_initial_external_control_flags")
    sensor_preview_metadata_initialized = _call_optional(system, "_init_sensor_preview_metadata")

    module_status_cfg = getattr(cfg, "module_status_ipc", None)
    ipc_cfg = getattr(cfg, "ipc_control", None)

    return RunnerServiceSnapshot(
        module_status_enabled=bool(getattr(module_status_cfg, "enabled", False)),
        module_status_host=str(getattr(module_status_cfg, "host", "")),
        module_status_port=int(getattr(module_status_cfg, "port", 0) or 0),
        module_status_running=_is_running_server(getattr(system, "module_status_server", None)) or module_status_running,
        ipc_control_enabled=bool(getattr(ipc_cfg, "enabled", False)),
        ipc_control_host=str(getattr(ipc_cfg, "host", "")),
        ipc_control_port=int(getattr(ipc_cfg, "port", 0) or 0),
        ipc_control_running=_is_running_server(getattr(system, "ipc_server", None)) or ipc_running,
        external_flags_initialized=external_flags_initialized,
        sensor_preview_metadata_initialized=sensor_preview_metadata_initialized,
    )
