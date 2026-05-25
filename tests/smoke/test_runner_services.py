from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_services import ensure_ipc_control_server, ensure_module_status_server, ensure_runner_services


class DummyServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.started = False

    def start(self) -> None:
        self.started = True


def _cfg(module_status_enabled=True, ipc_enabled=True):
    return SimpleNamespace(
        module_status_ipc=SimpleNamespace(enabled=module_status_enabled, host="127.0.0.1", port=8766),
        ipc_control=SimpleNamespace(enabled=ipc_enabled, host="127.0.0.1", port=8765),
    )


def test_ensure_module_status_server_starts_when_enabled() -> None:
    system = SimpleNamespace(module_status_server=None)
    started = ensure_module_status_server(system, _cfg(module_status_enabled=True), DummyServer)

    assert started is True
    assert isinstance(system.module_status_server, DummyServer)
    assert system.module_status_server.started is True
    assert system.module_status_server.port == 8766


def test_ensure_module_status_server_skips_when_disabled() -> None:
    system = SimpleNamespace(module_status_server=None)
    started = ensure_module_status_server(system, _cfg(module_status_enabled=False), DummyServer)

    assert started is False
    assert system.module_status_server is None


def test_ensure_ipc_control_server_starts_when_enabled() -> None:
    system = SimpleNamespace(ipc_server=None)
    started = ensure_ipc_control_server(system, _cfg(ipc_enabled=True), DummyServer)

    assert started is True
    assert isinstance(system.ipc_server, DummyServer)
    assert system.ipc_server.started is True
    assert system.ipc_server.port == 8765


def test_ensure_runner_services_is_idempotent_and_calls_metadata_hooks() -> None:
    calls = []

    def _external():
        calls.append("external")

    def _sensor():
        calls.append("sensor")

    system = SimpleNamespace(
        module_status_server=None,
        ipc_server=None,
        _write_initial_external_control_flags=_external,
        _init_sensor_preview_metadata=_sensor,
    )

    snapshot1 = ensure_runner_services(system, _cfg(), DummyServer, DummyServer)
    snapshot2 = ensure_runner_services(system, _cfg(), DummyServer, DummyServer)

    assert snapshot1.module_status_running is True
    assert snapshot1.ipc_control_running is True
    assert snapshot2.module_status_running is True
    assert snapshot2.ipc_control_running is True
    assert system.module_status_server.started is True
    assert system.ipc_server.started is True
    assert calls == ["external", "sensor", "external", "sensor"]
