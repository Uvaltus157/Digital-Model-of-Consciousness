from __future__ import annotations

import time

from src.platform.gui.opencv_gui_thread import get_cv2_last_key


class HighGUIEventPump:
    """
    Compatibility key pump for visualizers backed by OpenCVGuiThread.

    Why:
        HighGUI calls must stay in src/platform/gui/opencv_gui_thread.py. Older runtime
        code still calls pump_highgui_events(), so this wrapper now only exposes
        the last key observed by the dedicated GUI thread.

    Rule:
        visualizers submit frames; only OpenCVGuiThread calls HighGUI.
    """

    def __init__(self, min_interval_sec: float = 0.0):
        self.min_interval_sec = float(min_interval_sec)
        self.last_pump_time = 0.0
        self.last_key = 255
        self.pump_count = 0

    def pump(self) -> int:
        now = time.monotonic()
        if self.min_interval_sec > 0.0 and (now - self.last_pump_time) < self.min_interval_sec:
            return int(self.last_key)

        try:
            key = int(get_cv2_last_key())
        except Exception:
            key = 255

        self.last_key = int(key)
        self.last_pump_time = now
        self.pump_count += 1
        return int(key)


_GLOBAL_HIGHGUI_PUMP = HighGUIEventPump()


def pump_highgui_events() -> int:
    return _GLOBAL_HIGHGUI_PUMP.pump()


class HighGUIEventPumpRuntimeMixin:
    """
    Runtime mixin. Kept for older call sites that still ask for a per-frame key.
    """

    def update_highgui_event_pump(self) -> None:
        try:
            key = pump_highgui_events()
            self.last_highgui_key = int(key)
            if key in (27, ord("q")):
                # Do not shutdown whole app here; just expose the key.
                # Specific visualizers can close themselves on their own flags.
                pass
        except Exception as e:
            if not hasattr(self, "_highgui_event_pump_warned"):
                print(f"[highgui_event_pump] failed: {e}")
                self._highgui_event_pump_warned = True
