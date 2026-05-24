from __future__ import annotations

from threading import Event, Lock, Thread
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np


class MujocoViewerThread:
    """
    Passive MuJoCo viewer running off the life loop.

    The viewer owns a private MjData instance. The model/life thread only copies
    lightweight state snapshots into this object, so viewer.sync() cannot block
    or race with mj_step() on the live world data.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        sync_fps: float = 8.0,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
    ) -> None:
        self.model = model
        self.sync_fps = float(sync_fps)
        self.show_left_ui = bool(show_left_ui)
        self.show_right_ui = bool(show_right_ui)

        self._lock = Lock()
        self._stop = Event()
        self._enabled = Event()
        self._thread: Optional[Thread] = None
        self._snapshot: Optional[dict[str, np.ndarray | float]] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, name="MujocoViewerThread", daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()

    def update_from(self, data: mujoco.MjData) -> None:
        snap: dict[str, np.ndarray | float] = {"time": float(data.time)}
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            value = getattr(data, name, None)
            if value is not None:
                snap[name] = np.asarray(value).copy()
        with self._lock:
            self._snapshot = snap

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._enabled.clear()
        if self._thread is not None:
            self._thread.join(timeout=float(timeout))

    def _take_snapshot(self) -> Optional[dict[str, np.ndarray | float]]:
        with self._lock:
            if self._snapshot is None:
                return None
            return dict(self._snapshot)

    def _apply_snapshot(self, data: mujoco.MjData, snapshot: dict[str, np.ndarray | float]) -> None:
        data.time = float(snapshot.get("time", data.time))
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            src = snapshot.get(name)
            dst = getattr(data, name, None)
            if src is None or dst is None:
                continue
            src_arr = np.asarray(src)
            if dst.shape == src_arr.shape:
                np.copyto(dst, src_arr)

    def _run(self) -> None:
        viewer = None
        local_data = mujoco.MjData(self.model)
        interval = 0.0 if self.sync_fps <= 0.0 else 1.0 / max(self.sync_fps, 1e-6)

        while not self._stop.is_set():
            if not self._enabled.is_set():
                if viewer is not None:
                    try:
                        viewer.close()
                    except Exception:
                        pass
                    viewer = None
                time.sleep(0.05)
                continue

            if viewer is None:
                try:
                    viewer = mujoco.viewer.launch_passive(
                        self.model,
                        local_data,
                        show_left_ui=self.show_left_ui,
                        show_right_ui=self.show_right_ui,
                    )
                except Exception as e:
                    print(f"[mujoco.viewer.thread] launch failed: {e}")
                    self._enabled.clear()
                    time.sleep(0.5)
                    continue

            if not viewer.is_running():
                try:
                    viewer.close()
                except Exception:
                    pass
                viewer = None
                self._enabled.clear()
                continue

            snapshot = self._take_snapshot()
            if snapshot is not None:
                try:
                    self._apply_snapshot(local_data, snapshot)
                    mujoco.mj_forward(self.model, local_data)
                    viewer.sync()
                except Exception as e:
                    print(f"[mujoco.viewer.thread] sync failed: {e}")
                    try:
                        viewer.close()
                    except Exception:
                        pass
                    viewer = None
                    self._enabled.clear()

            time.sleep(interval)

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
