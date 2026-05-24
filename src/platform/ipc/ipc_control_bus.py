from __future__ import annotations

"""
ipc_control_bus.py

Local inter-process control bus for Conscious Viewer.

Transport:
    TCP on 127.0.0.1
Protocol:
    one JSON object per line

Why:
    PyQt5 process and OpenCV/MuJoCo process stay separate.
    No cv2 import in PyQt process.
    No shared JSON file polling.
"""

from dataclasses import dataclass, asdict
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Dict, Any
import json
import socket
import time


@dataclass
class IPCControlState:
    mujoco_next_run: bool = True
    inner_world: bool = True
    cameras: bool = True
    depth: bool = True
    training: bool = True
    close_aux_counter: int = 0
    stop: bool = False
    updated_at: float = 0.0
    connected: bool = False


class IPCControlServer:
    """
    Runs inside the main viewer process.
    Receives JSON-line commands from one or more control-panel clients.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = int(port)
        self.queue: Queue[Dict[str, Any]] = Queue()
        self.stop_event = Event()
        self.thread: Optional[Thread] = None
        self.socket: Optional[socket.socket] = None
        self.bound_port: Optional[int] = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        try:
            if self.socket is not None:
                self.socket.close()
        except Exception:
            pass

    def _run(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket = srv
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(0.25)
        self.bound_port = srv.getsockname()[1]
        print(f"[ipc] control server listening on {self.host}:{self.bound_port}")

        while not self.stop_event.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            t = Thread(target=self._handle_client, args=(conn, addr), daemon=True)
            t.start()

    def _handle_client(self, conn: socket.socket, addr) -> None:
        with conn:
            conn.settimeout(0.25)
            buf = b""
            while not self.stop_event.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                if not chunk:
                    break

                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                        if isinstance(msg, dict):
                            self.queue.put(msg)
                    except Exception as e:
                        print(f"[ipc] bad message from {addr}: {e}")

    def get_nowait(self) -> Optional[Dict[str, Any]]:
        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    def drain(self) -> list[Dict[str, Any]]:
        out = []
        while True:
            msg = self.get_nowait()
            if msg is None:
                break
            out.append(msg)
        return out


def send_ipc_message(host: str, port: int, msg: Dict[str, Any], timeout: float = 1.0) -> bool:
    """
    One-shot JSON-line sender. Used by standalone PyQt control app.
    """
    data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as sock:
            sock.sendall(data)
        return True
    except Exception as e:
        print(f"[ipc] send failed: {e}")
        return False


def make_set_state_message(**kwargs) -> Dict[str, Any]:
    return {
        "type": "set_state",
        "updated_at": time.time(),
        "state": kwargs,
    }


def make_toggle_message(key: str) -> Dict[str, Any]:
    return {
        "type": "toggle",
        "updated_at": time.time(),
        "key": key,
    }


def make_action_message(action: str, **payload) -> Dict[str, Any]:
    msg = {
        "type": "action",
        "updated_at": time.time(),
        "action": action,
    }
    if payload:
        msg["payload"] = payload
    return msg
