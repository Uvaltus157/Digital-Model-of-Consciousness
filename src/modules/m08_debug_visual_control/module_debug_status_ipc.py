
from __future__ import annotations

"""
module_debug_status_ipc.py

Dedicated IPC status server for Module Debug.

Why separate from the command IPC port:
- command IPC stays one-shot client -> runner
- status IPC is read-only for PyQt module-debug window
- no temp status file is needed

Protocol:
- TCP JSON-line on 127.0.0.1:8766
- client sends: {"type": "get_status"}
- server returns one JSON object line:
  {"type": "module_debug_status", "status": {...}}
"""

from dataclasses import dataclass
from threading import Thread, Event, Lock
from typing import Dict, Any, Optional
import json
import socket
import time


@dataclass
class ModuleDebugStatusIPCConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8766


class ModuleDebugStatusServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.host = str(host)
        self.port = int(port)
        self.stop_event = Event()
        self.thread: Optional[Thread] = None
        self.socket: Optional[socket.socket] = None
        self.bound_port: Optional[int] = None
        self._lock = Lock()
        self._status: Dict[str, Any] = {
            "updated_at": time.time(),
            "ready": False,
            "module_training": {},
            "trainable_counts": {},
        }

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        try:
            if self.socket is not None:
                self.socket.close()
        except Exception:
            pass

    def update_status(self, status: Dict[str, Any]):
        with self._lock:
            self._status = dict(status)
            self._status["updated_at"] = time.time()
            self._status["ready"] = True

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket = srv
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(0.25)
        self.bound_port = srv.getsockname()[1]
        print(f"[module_status_ipc] server listening on {self.host}:{self.bound_port}")

        while not self.stop_event.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = Thread(target=self._handle_client, args=(conn,), daemon=True)
            t.start()

    def _handle_client(self, conn: socket.socket):
        with conn:
            conn.settimeout(1.0)
            try:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                # Accept any request as get_status for robustness.
                payload = {
                    "type": "module_debug_status",
                    "status": self.get_status(),
                    "updated_at": time.time(),
                }
                conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            except Exception:
                pass


def request_module_debug_status(host: str = "127.0.0.1", port: int = 8766, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
    msg = {"type": "get_status", "updated_at": time.time()}
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            return None
        line = buf.split(b"\n", 1)[0]
        data = json.loads(line.decode("utf-8"))
        if data.get("type") == "module_debug_status":
            return data.get("status", {})
        return data
    except Exception as e:
        return None
