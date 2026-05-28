from __future__ import annotations

from dataclasses import dataclass
from threading import Thread, Event, Lock
from typing import Dict, Optional, Tuple
import time

import cv2
import numpy as np


@dataclass
class _FrameItem:
    frame: np.ndarray
    width: int
    height: int
    flags: int
    seq: int


class OpenCVGuiThread:
    """
    Dedicated OpenCV HighGUI thread.

    Rule:
        model/runtime thread must NOT call:
            cv2.namedWindow
            cv2.resizeWindow
            cv2.getWindowImageRect
            cv2.imshow
            cv2.waitKey
            cv2.pollKey
            cv2.destroyWindow

        visualizers only build numpy frames and submit them here.

    This avoids freezes when several visualizers are alive.
    """

    def __init__(self, fps: float = 12.0, idle_sleep_sec: float = 0.10):
        self.fps = float(fps)
        self.idle_sleep_sec = float(idle_sleep_sec)
        self.lock = Lock()
        self.stop_event = Event()
        self.stopped_event = Event()
        self.thread: Optional[Thread] = None
        self.frames: Dict[str, _FrameItem] = {}
        self.created: Dict[str, Tuple[int, int]] = {}
        self.displayed_seq: Dict[str, int] = {}
        self.to_close: set[str] = set()
        self.last_key: int = 255
        self.key_events: list[int] = []
        self.max_key_events: int = 32
        self.last_pump_time: float = 0.0
        self.seq_counter: int = 0

    def start(self) -> None:
        if self.stop_event.is_set():
            return
        if self.thread is not None and self.thread.is_alive():
            return
        self.stopped_event.clear()
        self.thread = Thread(target=self._run, name="OpenCVGuiThread", daemon=True)
        self.thread.start()

    def submit(self, window_name: str, frame: np.ndarray, width: int, height: int, flags: Optional[int] = None) -> None:
        if self.stop_event.is_set():
            return
        if frame is None:
            return
        self.start()
        if self.thread is None:
            return
        try:
            f = np.ascontiguousarray(frame.copy())
        except Exception:
            f = frame
        if flags is None:
            flags = cv2.WINDOW_NORMAL
            if hasattr(cv2, "WINDOW_FREERATIO"):
                flags |= cv2.WINDOW_FREERATIO
        with self.lock:
            self.seq_counter += 1
            self.frames[str(window_name)] = _FrameItem(f, int(width), int(height), int(flags), self.seq_counter)

    def close(self, window_name: str) -> None:
        if self.stop_event.is_set() and (self.thread is None or not self.thread.is_alive()):
            return
        self.start()
        if self.thread is None:
            return
        with self.lock:
            self.frames.pop(str(window_name), None)
            self.to_close.add(str(window_name))

    def shutdown(self, timeout: float = 3.0) -> None:
        self.stop_event.set()
        if self.thread is None:
            self.stopped_event.set()
            return
        with self.lock:
            self.frames.clear()
            self.to_close.update(self.created.keys())
        if self.thread.is_alive():
            self.thread.join(timeout=float(timeout))
        if self.thread.is_alive():
            # Give callers a clear diagnostic instead of letting Python finalizers
            # tear down Qt-backed HighGUI objects from the wrong thread silently.
            print("[opencv_gui_thread] shutdown timed out; HighGUI cleanup may continue in background")

    def key(self) -> int:
        with self.lock:
            return int(self.last_key)

    def consume_key(self) -> int:
        with self.lock:
            if not self.key_events:
                return 255
            return int(self.key_events.pop(0))

    def _snapshot(self):
        with self.lock:
            frames = dict(self.frames)
            to_close = set(self.to_close)
            self.to_close.clear()
        return frames, to_close

    def _ensure_window(self, name: str, item: _FrameItem) -> bool:
        size = (int(item.width), int(item.height))
        if name in self.created:
            old = self.created[name]
            # Avoid resize spam; only resize if size changed noticeably.
            if abs(old[0] - size[0]) < 8 and abs(old[1] - size[1]) < 8:
                return False

        try:
            cv2.namedWindow(name, int(item.flags))
        except Exception:
            pass
        try:
            if hasattr(cv2, "WND_PROP_ASPECT_RATIO") and hasattr(cv2, "WINDOW_FREERATIO"):
                cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        except Exception:
            pass
        try:
            cv2.resizeWindow(name, max(320, int(item.width)), max(240, int(item.height)))
        except Exception:
            pass

        self.created[name] = size
        return True

    def _run(self) -> None:
        delay = 1.0 / max(1.0, self.fps)
        try:
            while not self.stop_event.is_set():
                frames, to_close = self._snapshot()

                for name in to_close:
                    try:
                        cv2.destroyWindow(name)
                    except Exception:
                        pass
                    self.created.pop(name, None)
                    self.displayed_seq.pop(name, None)

                if not frames and not self.created:
                    with self.lock:
                        self.last_key = 255
                    time.sleep(max(0.02, self.idle_sleep_sec))
                    continue

                for name, item in frames.items():
                    try:
                        window_changed = self._ensure_window(name, item)
                        if window_changed or self.displayed_seq.get(name) != int(item.seq):
                            cv2.imshow(name, item.frame)
                            self.displayed_seq[name] = int(item.seq)
                    except Exception as e:
                        # Do not kill GUI thread because one frame failed.
                        print(f"[opencv_gui_thread] imshow failed for {name}: {e}")

                key = 255
                try:
                    key = cv2.waitKey(1) & 0xFF
                except Exception:
                    key = 255
                with self.lock:
                    self.last_key = int(key)
                    if int(key) != 255:
                        self.key_events.append(int(key))
                        if len(self.key_events) > self.max_key_events:
                            self.key_events = self.key_events[-self.max_key_events:]
                    self.last_pump_time = time.time()

                time.sleep(delay)
        finally:
            names = []
            with self.lock:
                names = list(self.created.keys()) + list(self.frames.keys()) + list(self.to_close)
                self.frames.clear()
                self.created.clear()
                self.displayed_seq.clear()
                self.to_close.clear()
                self.last_key = 255
                self.key_events.clear()
            for name in dict.fromkeys(names):
                try:
                    cv2.destroyWindow(name)
                except Exception:
                    pass
            for _ in range(5):
                try:
                    cv2.waitKey(1)
                except Exception:
                    break
                time.sleep(0.01)
            self.stopped_event.set()


_GLOBAL_GUI_THREAD = OpenCVGuiThread(fps=12.0)


def submit_cv2_frame(window_name: str, frame: np.ndarray, width: int, height: int, flags: Optional[int] = None) -> None:
    _GLOBAL_GUI_THREAD.submit(window_name, frame, width, height, flags)


def close_cv2_window(window_name: str) -> None:
    _GLOBAL_GUI_THREAD.close(window_name)


def get_cv2_last_key() -> int:
    return _GLOBAL_GUI_THREAD.key()


def consume_cv2_key() -> int:
    return _GLOBAL_GUI_THREAD.consume_key()


def shutdown_cv2_gui_thread(timeout: float = 1.5) -> None:
    _GLOBAL_GUI_THREAD.shutdown(timeout=timeout)
