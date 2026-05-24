
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "runner.yaml").exists():
            sys.path.insert(0, str(parent))
            break

try:
    from src.shared.console_colors import install_colored_errors

    install_colored_errors()
except Exception:
    pass

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from src.platform.ipc.ipc_control_bus import make_set_state_message, send_ipc_message
from src.modules.m08_debug_visual_control.module_debug_status_ipc import request_module_debug_status


def project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "runner.yaml").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def child_process_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(root)
    existing = env.get("PYTHONPATH", "")
    paths = [str(root)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


MODULES: List[Tuple[str, str, str, str]] = [
    ("world_model", "World Model", "Perception", "#4F9DFF"),
    ("object_imagery", "Object Imagery", "Perception", "#66B5FF"),
    ("long_dynamic_memory", "Long Dynamic Memory", "Dynamic object", "#35E3C4"),
    ("core_model", "Core Model", "Integration", "#A18BFF"),
    ("action_heads", "Action Heads", "Action", "#4FD788"),
    ("leg_control", "Leg Control", "Action", "#56E2AE"),
    ("self_core", "SelfCore", "Self", "#C981FF"),
    ("inner_speech", "Inner Speech", "Language", "#FF7DD1"),
]

EDGES: List[Tuple[str, str, str]] = [
    ("world_model", "core_model", "latent state"),
    ("world_model", "object_imagery", "object slots"),
    ("long_dynamic_memory", "self_core", "dynamic object"),
    ("long_dynamic_memory", "object_imagery", "z_dynamic slot"),
    ("world_model", "long_dynamic_memory", "temporal context"),
    ("object_imagery", "long_dynamic_memory", "z_static stream"),
    ("object_imagery", "long_dynamic_memory", "z_static"),
    ("long_dynamic_memory", "object_imagery", "z_dynamic"),
    ("object_imagery", "self_core", "object percept"),
    ("core_model", "action_heads", "intent"),
    ("action_heads", "leg_control", "motor"),
    ("action_heads", "self_core", "agency"),
    ("core_model", "self_core", "workspace"),
    ("self_core", "inner_speech", "report"),
    ("world_model", "inner_speech", "semantics"),
    ("self_core", "action_heads", "self-guided"),
]

DEFAULT_FLAGS: Dict[str, bool] = {k: True for k, *_ in MODULES}
DEFAULT_SENSOR_FLAGS: Dict[str, bool] = {"video": True, "contact": True, "imu": True}

COLLECTIVE_PRESETS: List[Tuple[str, Dict[str, bool]]] = [
    ("Perception", {
        "world_model": True,
        "object_imagery": True,
        "core_model": False,
        "long_dynamic_memory": True,
        "action_heads": False,
        "leg_control": False,
        "self_core": False,
        "inner_speech": False,
    }),
    ("Dynamic Object Memory", {
        "world_model": True,
        "object_imagery": True,
        "long_dynamic_memory": True,
        "core_model": False,
        "action_heads": False,
        "leg_control": False,
        "self_core": False,
        "inner_speech": False,
    }),
    ("World + Core", {
        "world_model": True,
        "object_imagery": True,
        "core_model": True,
        "long_dynamic_memory": True,
        "action_heads": False,
        "leg_control": False,
        "self_core": False,
        "inner_speech": False,
    }),
    ("Action Stack", {
        "world_model": False,
        "object_imagery": False,
        "long_dynamic_memory": False,
        "core_model": True,
        "action_heads": True,
        "leg_control": True,
        "self_core": False,
        "inner_speech": False,
    }),
    ("Self Loop", {
        "world_model": True,
        "object_imagery": True,
        "long_dynamic_memory": True,
        "core_model": True,
        "action_heads": True,
        "leg_control": False,
        "self_core": True,
        "inner_speech": True,
    }),
    ("Language / Report", {
        "world_model": True,
        "object_imagery": True,
        "core_model": True,
        "long_dynamic_memory": True,
        "action_heads": False,
        "leg_control": False,
        "self_core": True,
        "inner_speech": True,
    }),
    ("All modules", {k: True for k, *_ in MODULES}),
    ("Freeze all", {k: False for k, *_ in MODULES}),
]


def local_sensor_state(flags: Dict[str, bool]) -> str:
    video = bool(flags.get("video", True))
    contact = bool(flags.get("contact", True))
    imu = bool(flags.get("imu", True))
    if video and contact and imu:
        return "awake"
    if not video and not contact and not imu:
        return "sleep"
    off = [k for k, v in {"video": video, "contact": contact, "imu": imu}.items() if not v]
    return "partial_cut:" + ",".join(off)


class StatusPill(QtWidgets.QLabel):
    def __init__(self, text="—"):
        super().__init__(text)
        self.setMinimumHeight(34)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.set_state("grey", text)

    def set_state(self, color: str, text: str):
        palette = {
            "green": ("#123F2B", "#73E9A0"),
            "grey": ("#242C39", "#748298"),
            "yellow": ("#4A3B19", "#FFD36D"),
            "purple": ("#372451", "#D2A8FF"),
            "red": ("#4A1F2E", "#FF8FA3"),
            "blue": ("#19355C", "#8DBDFF"),
        }
        bg, fg = palette.get(color, palette["grey"])
        self.setText(text)
        self.setStyleSheet(
            f"background:{bg}; color:{fg}; border:1px solid {fg}; "
            "border-radius:12px; padding:6px 12px; font-weight:900;"
        )


class StableScrollTextEdit(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self._last_text = None
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setStyleSheet("""
            QPlainTextEdit {
                background:#0F1722; color:#D7E4F9; border:1px solid #304059;
                border-radius:10px; padding:8px; font-family:Consolas, monospace;
                font-size:11px; selection-background-color:#37507A;
            }
            QScrollBar:vertical { background:#111A27; width:16px; margin:2px; border-radius:7px; border:1px solid #243145; }
            QScrollBar::handle:vertical { background:#6F84A3; min-height:34px; border-radius:7px; }
            QScrollBar::handle:vertical:hover { background:#9BB2D1; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:#111A27; }
            QScrollBar:horizontal { background:#111A27; height:14px; margin:2px; border-radius:7px; border:1px solid #243145; }
            QScrollBar::handle:horizontal { background:#6F84A3; min-width:34px; border-radius:7px; }
            QScrollBar::handle:horizontal:hover { background:#9BB2D1; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background:#111A27; }
        """)

    def set_stable_plain_text(self, text: str):
        if text == self._last_text:
            return
        self._last_text = text
        vb = self.verticalScrollBar()
        hb = self.horizontalScrollBar()
        vv = vb.value()
        hv = hb.value()
        self.setUpdatesEnabled(False)
        try:
            self.setPlainText(text)
        finally:
            self.setUpdatesEnabled(True)

        def restore():
            self.verticalScrollBar().setValue(min(vv, self.verticalScrollBar().maximum()))
            self.horizontalScrollBar().setValue(min(hv, self.horizontalScrollBar().maximum()))

        restore()


class ModuleCard(QtWidgets.QFrame):
    toggled = QtCore.pyqtSignal(str, bool)

    def __init__(self, key: str, title: str, group: str, color: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.color = color
        self.setObjectName("moduleCard")
        self.setCursor(QtCore.Qt.PointingHandCursor)

        self.checkbox = QtWidgets.QCheckBox(title)
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(lambda v: self.toggled.emit(self.key, bool(v)))

        self.group_label = QtWidgets.QLabel(group)
        self.count_label = QtWidgets.QLabel("trainable: —")
        self.metric_label = QtWidgets.QLabel("")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(5)
        lay.addWidget(self.checkbox)
        lay.addWidget(self.group_label)
        lay.addWidget(self.count_label)
        lay.addWidget(self.metric_label)
        lay.addStretch(1)
        self.set_from_status(True, None, True)

    def set_from_status(self, checked: bool, count=None, available=True):
        blocker = QtCore.QSignalBlocker(self.checkbox)
        self.checkbox.setChecked(bool(checked))
        self.checkbox.setEnabled(bool(available))
        del blocker

        if not available:
            count_text = "N/A"
        elif count is None:
            count_text = "trainable: —"
        else:
            count_text = f"trainable: {int(count):,}"
        self.count_label.setText(count_text)

        color = self.color if available else "#596171"
        bg = QtGui.QColor(color)
        bg.setAlpha(44 if checked and available else 22)
        self.setStyleSheet(f"""
            QFrame#moduleCard {{
                background:rgba({bg.red()},{bg.green()},{bg.blue()},{bg.alpha()});
                border:2px solid {color};
                border-radius:16px;
            }}
            QCheckBox {{
                color:{'white' if available else '#778092'};
                font-size:15px;
                font-weight:800;
                spacing:8px;
            }}
            QCheckBox::indicator {{
                width:18px; height:18px; border-radius:5px;
                border:1px solid #DDE7FF; background:rgba(255,255,255,20);
            }}
            QCheckBox::indicator:checked {{ background:{color}; border:1px solid {color}; }}
            QLabel {{ color:#AFC0D8; font-size:11px; border:none; }}
        """)

    def set_metric_text(self, text: str):
        try:
            self.metric_label.setText(str(text or ""))
        except Exception:
            pass

    def is_checked(self) -> bool:
        return bool(self.checkbox.isChecked())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.checkbox.isEnabled():
            self.checkbox.toggle()
            event.accept()
            return
        super().mousePressEvent(event)


class DiagramCanvas(QtWidgets.QWidget):
    moduleToggled = QtCore.pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1180, 650) #1240, 760)
        self.cards: Dict[str, ModuleCard] = {}
        self._card_training_active = {}
        self._training_blink_phase = False
        self._training_blink_timer = QtCore.QTimer(self)
        self._training_blink_timer.timeout.connect(self._tick_training_blink)
        self._training_blink_timer.start(450)
        self._ldm_learning_active = False
        self._ldm_blink_phase = False
        self._ldm_blink_timer = QtCore.QTimer(self)
        self._ldm_blink_timer.timeout.connect(self._tick_ldm_blink)
        self._ldm_blink_timer.start(450)
        self.positions = {
            "world_model": (0.12, 0.20),
            "object_imagery": (0.12, 0.62),
            "long_dynamic_memory": (0.30, 0.62),
            "core_model": (0.39, 0.20),
            "action_heads": (0.66, 0.20),
            "leg_control": (0.88, 0.20),
            "self_core": (0.48, 0.62),
            "inner_speech": (0.80, 0.62),
        }
        for key, title, group, color in MODULES:
            card = ModuleCard(key, title, group, color, self)
            card.toggled.connect(self.moduleToggled.emit)
            self.cards[key] = card

    def _tick_ldm_blink(self):
        self._ldm_blink_phase = not bool(self._ldm_blink_phase)
        card = self.cards.get("long_dynamic_memory")
        if card is None:
            return
        if bool(getattr(self, "_ldm_learning_active", False)):
            fg = "#E9FFE9" if self._ldm_blink_phase else "#39FF88"
            card.count_label.setStyleSheet(
                f"color:{fg}; font-size:12px; font-weight:900; border:none;"
            )

    def _tick_training_blink(self):
        self._training_blink_phase = not bool(getattr(self, "_training_blink_phase", False))
        active_map = getattr(self, "_card_training_active", {}) or {}
        for key in ("world_model", "object_imagery", "long_dynamic_memory"):
            card = self.cards.get(key)
            if card is None:
                continue
            active = bool(active_map.get(key, False))
            if active:
                fg = "#E9FFE9" if self._training_blink_phase else "#39FF88"
                card.count_label.setStyleSheet(
                    f"color:{fg}; font-size:12px; font-weight:900; border:none;"
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        cw = min(235, max(185, int(w * 0.165)))
        ch = min(172, max(132, int(h * 0.22)))
        for key, card in self.cards.items():
            rx, ry = self.positions[key]
            x = max(20, min(w - cw - 20, int(rx * w) - cw // 2))
            y = max(70, min(h - ch - 20, 70 + int(ry * (h - 120)) - ch // 2))
            card.setGeometry(x, y, cw, ch)

    def center(self, key: str) -> QtCore.QPointF:
        return QtCore.QPointF(self.cards[key].geometry().center())

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        p.fillRect(self.rect(), QtGui.QColor("#101722"))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 14), 1))
        for x in range(0, self.width(), 40):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 40):
            p.drawLine(0, y, self.width(), y)

        p.setPen(QtGui.QColor("#EAF2FF"))
        p.setFont(QtGui.QFont("Sans Serif", 18, QtGui.QFont.Bold))
        p.drawText(18, 32, "Module Debug Schematic")

        p.setFont(QtGui.QFont("Sans Serif", 10))
        p.setPen(QtGui.QColor("#9DB1CD"))
        p.drawText(18, 55, "Collective training presets are above; cards and edges show active trainable subsystems.")

        for a, b, label in EDGES:
            s0, e0 = self.center(a), self.center(b)
            on = (
                self.cards[a].is_checked()
                and self.cards[b].is_checked()
                and self.cards[a].checkbox.isEnabled()
                and self.cards[b].checkbox.isEnabled()
            )
            p.setPen(QtGui.QPen(QtGui.QColor("#73E9A0" if on else "#54657C"), 2.2 if on else 1.2))
            vec = e0 - s0
            length = (vec.x() ** 2 + vec.y() ** 2) ** 0.5 or 1.0
            ux, uy = vec.x() / length, vec.y() / length
            s = QtCore.QPointF(s0.x() + ux * 72, s0.y() + uy * 30)
            e = QtCore.QPointF(e0.x() - ux * 72, e0.y() - uy * 30)
            p.drawLine(s, e)

            ah = 10
            angle = QtCore.QLineF(s, e).angle()
            a1 = QtCore.QLineF.fromPolar(ah, angle + 150)
            a2 = QtCore.QLineF.fromPolar(ah, angle - 150)
            p.drawLine(e, e + QtCore.QPointF(a1.dx(), -a1.dy()))
            p.drawLine(e, e + QtCore.QPointF(a2.dx(), -a2.dy()))

            rect = QtCore.QRectF((s.x() + e.x()) / 2 - 47, (s.y() + e.y()) / 2 - 11, 94, 22)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(20, 28, 40, 225))
            p.drawRoundedRect(rect, 10, 10)
            p.setPen(QtGui.QColor("#B7C5DA"))
            p.setFont(QtGui.QFont("Sans Serif", 8))
            p.drawText(rect, QtCore.Qt.AlignCenter, label)
        p.end()

    def set_module_state(self, flags: Dict[str, bool], counts: Dict[str, int] | None = None, status: Dict | None = None):
        counts = counts or {}
        status = status or {}

        # Checkboxes are commands. Never disable a checkbox just because
        # trainable_count is absent/zero. Counts are diagnostics only.
        for key, card in self.cards.items():
            cnt = counts.get(key, None)
            if key == "long_dynamic_memory" and cnt is None:
                cnt = counts.get("long_dynamic_object_memory", None)
            card.set_from_status(bool(flags.get(key, False)), cnt, True)

        runner_module_training = status.get("module_training", {}) or {}
        effective_training = bool(status.get("effective_training", status.get("training", False)))
        last_train_reason = str(status.get("last_train_reason", ""))
        is_really_training = bool(effective_training and last_train_reason == "trained")
        global_loss = float(status.get("last_train_loss", 0.0) or 0.0)

        self._card_training_active = getattr(self, "_card_training_active", {}) or {}
        self._card_metrics_cache = getattr(self, "_card_metrics_cache", {}) or {}

        for key, card in self.cards.items():
            # Count aliases for modules whose UI key and internal key differ.
            if key == "long_dynamic_memory":
                ldm_status = status.get("long_dynamic_memory_status", {}) or {}
                ldm_learning = status.get("long_dynamic_memory_learning", {}) or {}
                merged = dict(ldm_learning)
                merged.update(ldm_status)
                trainable = int(
                    counts.get(
                        "long_dynamic_memory",
                        counts.get(
                            "long_dynamic_object_memory",
                            merged.get("trainable", merged.get("params", 0)),
                        ),
                    )
                    or 0
                )
            elif key == "object_imagery":
                trainable = int(counts.get("object_imagery", counts.get("inner_object_system", 0)) or 0)
                merged = {}
            elif key == "leg_control":
                trainable = int(counts.get("leg_control", counts.get("leg_control_head", 0)) or 0)
                merged = {}
            else:
                trainable = int(counts.get(key, 0) or 0)
                merged = {}

            # Prefer runner-confirmed flag. Fallback to local checkbox while waiting for ack.
            train_flag = bool(runner_module_training.get(key, flags.get(key, False)))
            active = bool(train_flag and trainable > 0 and is_really_training)
            self._card_training_active[key] = active

            cache = self._card_metrics_cache.get(
                key, {"loss": 0.0, "ema": 0.0, "reward": 0.0, "recon": 0.0}
            )

            # Metrics update only if this exact module is actively training.
            # If checkbox is OFF or global training is paused, freeze displayed EMA/loss.
            if active:
                if key == "long_dynamic_memory":
                    loss = float(merged.get("loss", cache.get("loss", 0.0)) or 0.0)
                    ema = float(merged.get("loss_ema", cache.get("ema", 0.0)) or 0.0)
                    reward = float(merged.get("reward_proxy", 0.0) or 0.0)
                    recon = float(merged.get("recon", cache.get("recon", 0.0)) or 0.0)
                else:
                    # Until per-module losses are separated, use global train loss
                    # only while this module is actually active.
                    loss = global_loss
                    prev_loss = float(cache.get("loss", loss))
                    reward = float(prev_loss) - float(loss)
                    prev_ema = cache.get("ema", None)
                    ema = float(loss) if prev_ema is None else 0.96 * float(prev_ema) + 0.04 * float(loss)
                    recon = 0.0

                cache = {
                    "loss": float(loss),
                    "ema": float(ema),
                    "reward": float(reward),
                    "recon": float(recon),
                }
                self._card_metrics_cache[key] = cache
            else:
                loss = float(cache.get("loss", 0.0) or 0.0)
                ema = float(cache.get("ema", 0.0) or 0.0)
                reward = 0.0
                recon = float(cache.get("recon", 0.0) or 0.0)

            state_text = "trained" if active else ("off" if not train_flag else last_train_reason[:18])
            sign = "+" if reward >= 0.0 else ""

            if hasattr(card, "metric_label"):
                if key == "long_dynamic_memory":
                    card.metric_label.setText(
                        f"loss   {loss:.5f}\n"
                        f"ema    {ema:.5f}\n"
                        f"reward {sign}{reward:.5f}\n"
                        f"recon  {recon:.5f}\n"
                        f"train  {state_text}"
                    )
                else:
                    card.metric_label.setText(
                        f"loss   {loss:.5f}\n"
                        f"ema    {ema:.5f}\n"
                        f"reward {sign}{reward:.5f}\n"
                        f"train  {state_text}"
                    )
                card.metric_label.setStyleSheet(
                    "color:#D5FFF0; font-size:10px; font-family:Consolas, monospace; "
                    "font-weight:800; border:none;"
                )

            if active:
                card.count_label.setStyleSheet("color:#39FF88; font-size:12px; font-weight:900; border:none;")
            elif train_flag and trainable > 0:
                card.count_label.setStyleSheet("color:#FFD36D; font-size:12px; font-weight:900; border:none;")
            else:
                card.count_label.setStyleSheet("color:#AFC0D8; font-size:11px; border:none;")

        self.update()

    def flags(self) -> Dict[str, bool]:
        return {k: c.is_checked() for k, c in self.cards.items()}


class ModulesTab(QtWidgets.QWidget):
    moduleToggled = QtCore.pyqtSignal(str, bool)
    presetRequested = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.preset_bar = QtWidgets.QFrame()
        self.preset_bar.setObjectName("presetBar")
        preset_lay = QtWidgets.QHBoxLayout(self.preset_bar)
        preset_lay.setContentsMargins(10, 8, 10, 8)
        preset_lay.setSpacing(8)

        title = QtWidgets.QLabel("Collective selective training:")
        title.setObjectName("barTitle")
        preset_lay.addWidget(title)

        for name, flags in COLLECTIVE_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _=False, f=flags: self.presetRequested.emit(dict(f)))
            preset_lay.addWidget(btn)
        preset_lay.addStretch(1)

        self.diagram = DiagramCanvas()
        self.diagram.moduleToggled.connect(self.moduleToggled.emit)

        root.addWidget(self.preset_bar, 0)
        root.addWidget(self.diagram, 1)

    def set_module_state(self, flags: Dict[str, bool], counts: Dict[str, int] | None = None, status: Dict | None = None):
        self.diagram.set_module_state(flags, counts, status)

    def flags(self) -> Dict[str, bool]:
        return self.diagram.flags()


class ActivityModesTab(QtWidgets.QWidget):
    sensorToggled = QtCore.pyqtSignal(str, bool)
    presetRequested = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        self.mode_label = QtWidgets.QLabel("MODE: unknown")
        root.addWidget(self.mode_label)

        box = QtWidgets.QGroupBox("Sensor gates")
        bl = QtWidgets.QVBoxLayout(box)
        self.sensor_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        for key, text in {
            "video": "Video / eyes",
            "contact": "Contact / touch",
            "imu": "IMU / vestibular body",
        }.items():
            cb = QtWidgets.QCheckBox(text)
            cb.setChecked(True)
            cb.toggled.connect(lambda v, k=key: self.sensorToggled.emit(k, bool(v)))
            self.sensor_checkboxes[key] = cb
            bl.addWidget(cb)
        root.addWidget(box)

        pbox = QtWidgets.QGroupBox("Activity presets")
        gl = QtWidgets.QGridLayout(pbox)
        presets = [
            ("Awake / active", {"video": True, "contact": True, "imu": True}),
            ("Sleep / dream", {"video": False, "contact": False, "imu": False}),
            ("Blind awake", {"video": False, "contact": True, "imu": True}),
            ("Body only", {"video": False, "contact": False, "imu": True}),
        ]
        for i, (title, flags) in enumerate(presets):
            btn = QtWidgets.QPushButton(title)
            btn.clicked.connect(lambda _=False, f=flags: self.presetRequested.emit(dict(f)))
            gl.addWidget(btn, i // 2, i % 2)
        root.addWidget(pbox)

        hint = QtWidgets.QLabel(
            "Full sleep = video OFF + contact OFF + IMU OFF.\n"
            "If any sensor returns, the system leaves dream mode.\n"
            "The top panel shows the live activity lamp and training button."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        root.addWidget(hint)
        root.addStretch(1)

    def set_sensor_flags(self, flags: Dict[str, bool]):
        for key, cb in self.sensor_checkboxes.items():
            blocker = QtCore.QSignalBlocker(cb)
            cb.setChecked(bool(flags.get(key, True)))
            del blocker

    def sensor_flags(self) -> Dict[str, bool]:
        return {k: bool(cb.isChecked()) for k, cb in self.sensor_checkboxes.items()}

    def set_mode(self, state: str, full_sleep: bool):
        if full_sleep:
            text, color = "SLEEP / DREAM MODE", "#6F45B8"
        elif str(state).startswith("partial_cut"):
            text, color = f"PARTIAL SENSOR CUT: {str(state).replace('partial_cut:', '')}", "#B8862B"
        elif state == "awake":
            text, color = "AWAKE / ACTIVE", "#1F7A45"
        else:
            text, color = f"MODE: {state}", "#31415C"
        self.mode_label.setText(text)
        self.mode_label.setStyleSheet(
            f"background:{color}; color:white; border:1px solid #DCE8F8; "
            "border-radius:12px; padding:14px; font-size:18px; font-weight:900;"
        )


class StatusBoardTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setSpacing(12)

        self.training = StableScrollTextEdit()
        self.sensors = StableScrollTextEdit()
        self.modules = StableScrollTextEdit()
        self.raw = StableScrollTextEdit()

        for i, (title, widget) in enumerate([
            ("Training / replay", self.training),
            ("Sensors / sleep", self.sensors),
            ("Modules", self.modules),
            ("Raw status JSON", self.raw),
        ]):
            box = QtWidgets.QGroupBox(title)
            lay = QtWidgets.QVBoxLayout(box)
            lay.addWidget(widget)
            grid.addWidget(box, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

    def update_status(self, data: Dict):
        self.training.set_stable_plain_text("\n".join([
            f"global_step: {data.get('global_step', 0)}",
            f"train_steps: {data.get('train_steps', 0)}",
            f"training_enabled(raw): {data.get('training_enabled', False)}",
            f"cfg_train_enabled: {data.get('cfg_train_enabled', False)}",
            f"effective_training: {data.get('effective_training', data.get('training', False))}",
            f"full_sleep: {data.get('full_sleep', False)}",
            f"replay_len: {data.get('replay_len', 0)}",
            f"replay_min_ready: {data.get('replay_min_ready', 0)}",
            f"replay_ready: {data.get('replay_ready', False)}",
            f"last_train_reason: {data.get('last_train_reason', '')}",
            f"last_train_loss: {data.get('last_train_loss', 0.0)}",
            f"last_train_error: {data.get('last_train_error', '')}",
            "",
            "object_decoder_stats:",
            json.dumps(data.get("object_decoder_stats", {}), indent=2, ensure_ascii=False),
        ]))
        self.sensors.set_stable_plain_text("\n".join([
            f"sensor_state: {data.get('sensor_state', 'unknown')}",
            f"full_sleep: {data.get('full_sleep', False)}",
            "",
            "input_sensors_enabled:",
            json.dumps(data.get("input_sensors_enabled", {}), indent=2, ensure_ascii=False),
            "",
            "sleep_sensor_mask:",
            json.dumps(data.get("sleep_sensor_mask", {}), indent=2, ensure_ascii=False),
        ]))
        self.modules.set_stable_plain_text(
            "module_training:\n"
            + json.dumps(data.get("module_training", {}), indent=2, ensure_ascii=False)
            + "\n\ntrainable_counts:\n"
            + json.dumps(data.get("trainable_counts", {}), indent=2, ensure_ascii=False)
            + "\n\nlong_dynamic_memory_status:\n"
            + json.dumps(data.get("long_dynamic_memory_status", {}), indent=2, ensure_ascii=False)
            + f"\n\nlast_module_training_seq: {data.get('last_module_training_seq', 0)}"
        )
        self.raw.set_stable_plain_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


class ModuleDebugPyQtWindow(QtWidgets.QMainWindow):
    def __init__(self, host="127.0.0.1", port=8765, status_host="127.0.0.1", status_port=8766):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.status_host = status_host
        self.status_port = int(status_port)

        self.flags = dict(DEFAULT_FLAGS)
        self.sensor_flags = dict(DEFAULT_SENSOR_FLAGS)
        self.training_enabled = True
        self.effective_training = True
        self.syncing_from_runner = False
        self.module_training_seq = 0
        self.pending_module_training_seq = 0
        self.last_status: Dict = {}
        self.last_status_ok = False

        self.setWindowTitle("Module Debug — tabs / top status / live IPC")
        self.resize(1420, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        root.addWidget(self._toolbar())

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        self.modules_tab = ModulesTab()
        self.modules_tab.moduleToggled.connect(self.on_module_toggled)
        self.modules_tab.presetRequested.connect(self.apply_module_preset)

        self.activity = ActivityModesTab()
        self.activity.sensorToggled.connect(self.on_sensor_toggled)
        self.activity.presetRequested.connect(self.set_sensor_preset)

        self.status_board = StatusBoardTab()

        self.tabs.addTab(self.modules_tab, "Modules / mnemonic graph")
        self.tabs.addTab(self.activity, "Activity modes")
        self.tabs.addTab(self.status_board, "Status board")

        self.status_bar = QtWidgets.QLabel("Ready")
        self.status_bar.setStyleSheet("color:#B7C5DA; font-size:12px;")
        root.addWidget(self.status_bar)

        self.apply_theme()
        self.sync_ui()

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.poll_runner_status)
        self.status_timer.start(500)

    def shutdown_timers(self):
        try:
            if self.status_timer.isActive():
                self.status_timer.stop()
        except Exception:
            pass

    def closeEvent(self, event):
        self.shutdown_timers()
        super().closeEvent(event)

    def _toolbar(self):
        bar = QtWidgets.QFrame()
        bar.setObjectName("topPanel")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        self.status_ipc_pill = StatusPill("STATUS IPC: —")
        lay.addWidget(self.status_ipc_pill)

        self.training_btn = QtWidgets.QPushButton("Training: —")
        self.training_btn.setCheckable(True)
        self.training_btn.clicked.connect(self.on_training_toggled)
        lay.addWidget(self.training_btn)

        self.activity_lamp = StatusPill("MODE: unknown")
        self.activity_lamp.setMinimumWidth(230)
        lay.addWidget(self.activity_lamp)

        send = QtWidgets.QPushButton("Send current state")
        send.clicked.connect(self.send_state)
        lay.addWidget(send)

        ping = QtWidgets.QPushButton("Ping")
        ping.clicked.connect(lambda: self.send_action("ping"))
        lay.addWidget(ping)

        lay.addStretch(1)
        return bar

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background:#0C121B; }
            QWidget { color:#DCE8F8; }
            QFrame#topPanel {
                background:#141D29;
                border:1px solid #243145;
                border-radius:14px;
            }
            QTabWidget::pane {
                border:1px solid #243145;
                border-radius:10px;
                background:#101722;
            }
            QTabBar::tab {
                background:#141D29;
                color:#B7C5DA;
                border:1px solid #243145;
                padding:9px 16px;
                border-top-left-radius:9px;
                border-top-right-radius:9px;
                margin-right:3px;
                font-weight:800;
            }
            QTabBar::tab:selected {
                background:#243449;
                color:#FFFFFF;
                border:1px solid #4F85FF;
            }
            QFrame, QGroupBox {
                background:#101722;
                border:1px solid #243145;
                border-radius:12px;
            }
            QFrame#presetBar {
                background:#141D29;
                border:1px solid #304059;
                border-radius:12px;
            }
            QLabel#barTitle {
                color:#EDF4FF;
                font-weight:900;
                border:none;
            }
            QGroupBox {
                margin-top:12px;
                padding:10px;
                font-weight:900;
            }
            QGroupBox::title {
                subcontrol-origin:margin;
                left:10px;
                padding:0 4px;
            }
            QPushButton {
                background:#1D2A3B;
                color:white;
                border:1px solid #37507A;
                border-radius:10px;
                padding:8px 12px;
                font-weight:800;
            }
            QPushButton:hover { background:#263B55; }
            QLineEdit, QSpinBox {
                background:#0F1722;
                color:white;
                border:1px solid #304059;
                border-radius:8px;
                padding:5px 8px;
            }
            QCheckBox {
                spacing:8px;
                font-weight:800;
            }
            QCheckBox::indicator {
                width:18px;
                height:18px;
                border-radius:5px;
                border:1px solid #DDE7FF;
                background:rgba(255,255,255,22);
            }
            QCheckBox::indicator:checked {
                background:#2B66F6;
                border:1px solid #4F85FF;
            }
            QLabel#hintLabel {
                color:#9DB1CD;
                font-size:13px;
                border:none;
            }
        """)

    def sync_ui(self):
        self.modules_tab.set_module_state(self.flags, self.last_status.get("trainable_counts", {}), self.last_status)
        self.activity.set_sensor_flags(self.sensor_flags)

        state = self.last_status.get("sensor_state", local_sensor_state(self.sensor_flags))
        full_sleep = bool(self.last_status.get("full_sleep", state == "sleep"))
        self.activity.set_mode(state, full_sleep)
        self.update_top_training_button()
        self.update_activity_lamp(state, full_sleep)

    def update_top_training_button(self):
        blocker = QtCore.QSignalBlocker(self.training_btn)
        self.training_btn.setChecked(bool(self.training_enabled))
        del blocker

        if self.training_enabled and self.effective_training:
            text = "Training: ON"
            color = "#1F7A45"
        elif self.training_enabled and not self.effective_training:
            text = "Training: ON / paused"
            color = "#85662A"
        else:
            text = "Training: OFF"
            color = "#7A2D4B"
        self.training_btn.setText(text)
        self.training_btn.setStyleSheet(
            f"background:{color}; color:white; border:1px solid #DCE8F8; "
            "border-radius:10px; padding:8px 12px; font-weight:900;"
        )

    def update_activity_lamp(self, state: str, full_sleep: bool):
        if full_sleep:
            self.activity_lamp.set_state("purple", "SLEEP / DREAM")
        elif str(state).startswith("partial_cut"):
            self.activity_lamp.set_state("yellow", f"PARTIAL: {str(state).replace('partial_cut:', '')}")
        elif state == "awake":
            self.activity_lamp.set_state("green", "AWAKE / ACTIVE")
        else:
            self.activity_lamp.set_state("grey", f"MODE: {state}")

    def on_module_toggled(self, key: str, checked: bool):
        if self.syncing_from_runner:
            return
        self.flags[key] = bool(checked)
        self.module_training_seq += 1
        self.pending_module_training_seq = self.module_training_seq
        self.send_state()

    def apply_module_preset(self, flags: Dict[str, bool]):
        if self.syncing_from_runner:
            return
        self.flags.update({k: bool(v) for k, v in flags.items() if k in self.flags})
        self.module_training_seq += 1
        self.pending_module_training_seq = self.module_training_seq
        self.sync_ui()
        self.send_state()

    def on_sensor_toggled(self, key: str, checked: bool):
        if self.syncing_from_runner:
            return
        self.sensor_flags[key] = bool(checked)
        self.sync_ui()
        self.send_state()

    def on_training_toggled(self, checked: bool):
        if self.syncing_from_runner:
            return
        self.training_enabled = bool(checked)
        self.update_top_training_button()
        self.send_state()

    def set_sensor_preset(self, flags: Dict[str, bool]):
        if self.syncing_from_runner:
            return
        self.sensor_flags.update({k: bool(v) for k, v in flags.items() if k in self.sensor_flags})
        self.sync_ui()
        self.send_state()

    def payload(self) -> Dict:
        mask = {k: not bool(v) for k, v in self.sensor_flags.items()}
        return {
            "module_training": dict(self.flags),
            "module_training_seq": int(self.module_training_seq),
            "training": bool(self.training_enabled),
            "input_sensors_enabled": dict(self.sensor_flags),
            "sleep_sensor_mask": mask,
            "video_sensor_enabled": bool(self.sensor_flags.get("video", True)),
            "contact_sensor_enabled": bool(self.sensor_flags.get("contact", True)),
            "imu_sensor_enabled": bool(self.sensor_flags.get("imu", True)),
            "sleep_video_cut": bool(mask.get("video", False)),
            "sleep_contact_cut": bool(mask.get("contact", False)),
            "sleep_imu_cut": bool(mask.get("imu", False)),
        }

    def send_state(self):
        ok = send_ipc_message(self.host, self.port, make_set_state_message(**self.payload()), timeout=0.8)
        self.status_bar.setText(
            f"{'sent' if ok else 'send failed'} | cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port} | {time.strftime('%H:%M:%S')}"
        )

    def send_action(self, action: str):
        ok = send_ipc_message(
            self.host,
            self.port,
            {"type": "action", "action": action, "updated_at": time.time()},
            timeout=0.8,
        )
        self.status_bar.setText(
            f"action {action}: {'sent' if ok else 'failed'} | cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port}"
        )

    def poll_runner_status(self):
        data = request_module_debug_status(self.status_host, self.status_port, timeout=0.35)
        if not data:
            self.last_status_ok = False
            self.status_ipc_pill.set_state("grey", "STATUS IPC: no signal")
            self.status_board.raw.set_stable_plain_text(
                f"Waiting for runner status IPC\n{self.status_host}:{self.status_port}\n\n"
                "Runner must have module_status_ipc.enabled=true"
            )
            return

        self.last_status_ok = True
        self.status_ipc_pill.set_state("green", "STATUS IPC: receiving")
        self.last_status = data

        runner_seq = int(data.get("last_module_training_seq", data.get("module_training_seq", 0)) or 0)

        self.syncing_from_runner = True
        try:
            flags = data.get("module_training", {})
            if isinstance(flags, dict) and runner_seq >= self.pending_module_training_seq:
                for k in self.flags:
                    if k in flags:
                        self.flags[k] = bool(flags[k])

            sensors = data.get("input_sensors_enabled")
            if not isinstance(sensors, dict) and isinstance(data.get("sleep_sensor_mask"), dict):
                sensors = {k: not bool(v) for k, v in data.get("sleep_sensor_mask").items()}
            if isinstance(sensors, dict):
                for k in self.sensor_flags:
                    if k in sensors:
                        self.sensor_flags[k] = bool(sensors[k])

            self.training_enabled = bool(data.get("training_enabled", self.training_enabled))
            self.effective_training = bool(data.get("effective_training", data.get("training", self.training_enabled)))

            self.sync_ui()
            self.status_board.update_status(data)
        finally:
            self.syncing_from_runner = False

        self.status_bar.setText(
            f"status ok | step={data.get('global_step', 0)} | "
            f"cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port} | "
            f"mode={data.get('sensor_state', 'unknown')} | {time.strftime('%H:%M:%S')}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--status-host", default="127.0.0.1")
    p.add_argument("--status-port", type=int, default=8766)
    return p.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = ModuleDebugPyQtWindow(args.host, args.port, args.status_host, args.status_port)
    app.aboutToQuit.connect(win.shutdown_timers)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
