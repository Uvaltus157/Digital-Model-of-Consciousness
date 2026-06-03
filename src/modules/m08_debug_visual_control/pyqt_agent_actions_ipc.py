
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
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from src.platform.ipc.ipc_control_bus import make_action_message, make_set_state_message, send_ipc_message
from src.modules.m08_debug_visual_control.module_debug_status_ipc import request_module_debug_status


BODY_CHANNELS = [
    ("vx", "Body vx", -2.0, 2.0),
    ("vy", "Body vy", -2.0, 2.0),
    ("vz", "Body vz", -2.0, 2.0),
    ("yaw", "Body yaw", -1.5, 1.5),
    ("pitch", "Body pitch", -1.5, 1.5),
    ("roll", "Body roll", -1.5, 1.5),
    ("head_yaw", "Head yaw", -1.0, 1.0),
    ("head_pitch", "Head pitch", -1.0, 1.0),
    ("head_roll", "Head roll", -1.0, 1.0),
]

ARM_CHANNELS = [
    ("L_sh_yaw", "L shoulder yaw", -1.0, 1.0),
    ("L_sh_pitch", "L shoulder pitch", -1.0, 1.0),
    ("L_elbow", "L elbow", -1.0, 1.0),
    ("R_sh_yaw", "R shoulder yaw", -1.0, 1.0),
    ("R_sh_pitch", "R shoulder pitch", -1.0, 1.0),
    ("R_elbow", "R elbow", -1.0, 1.0),
]

# IMPORTANT: this order matches realistic_hand_mjcf.both_hand_control_names():
# per hand: palm_roll, palm_pitch + 5 * (mcp_yaw, mcp, pip, dip)
# left 22 + right 22 = 44.
HAND_CHANNELS = [
    ("left_palm_roll", "L palm roll", 0.0, 1.0),
    ("left_palm_pitch", "L palm pitch", 0.0, 1.0),

    ("left_thumb_mcp_yaw", "L thumb yaw", 0.0, 1.0),
    ("left_thumb_mcp", "L thumb MCP", 0.0, 1.0),
    ("left_thumb_pip", "L thumb PIP", 0.0, 1.0),
    ("left_thumb_dip", "L thumb DIP", 0.0, 1.0),

    ("left_index_mcp_yaw", "L index yaw", 0.0, 1.0),
    ("left_index_mcp", "L index MCP", 0.0, 1.0),
    ("left_index_pip", "L index PIP", 0.0, 1.0),
    ("left_index_dip", "L index DIP", 0.0, 1.0),

    ("left_middle_mcp_yaw", "L middle yaw", 0.0, 1.0),
    ("left_middle_mcp", "L middle MCP", 0.0, 1.0),
    ("left_middle_pip", "L middle PIP", 0.0, 1.0),
    ("left_middle_dip", "L middle DIP", 0.0, 1.0),

    ("left_ring_mcp_yaw", "L ring yaw", 0.0, 1.0),
    ("left_ring_mcp", "L ring MCP", 0.0, 1.0),
    ("left_ring_pip", "L ring PIP", 0.0, 1.0),
    ("left_ring_dip", "L ring DIP", 0.0, 1.0),

    ("left_little_mcp_yaw", "L little yaw", 0.0, 1.0),
    ("left_little_mcp", "L little MCP", 0.0, 1.0),
    ("left_little_pip", "L little PIP", 0.0, 1.0),
    ("left_little_dip", "L little DIP", 0.0, 1.0),

    ("right_palm_roll", "R palm roll", 0.0, 1.0),
    ("right_palm_pitch", "R palm pitch", 0.0, 1.0),

    ("right_thumb_mcp_yaw", "R thumb yaw", 0.0, 1.0),
    ("right_thumb_mcp", "R thumb MCP", 0.0, 1.0),
    ("right_thumb_pip", "R thumb PIP", 0.0, 1.0),
    ("right_thumb_dip", "R thumb DIP", 0.0, 1.0),

    ("right_index_mcp_yaw", "R index yaw", 0.0, 1.0),
    ("right_index_mcp", "R index MCP", 0.0, 1.0),
    ("right_index_pip", "R index PIP", 0.0, 1.0),
    ("right_index_dip", "R index DIP", 0.0, 1.0),

    ("right_middle_mcp_yaw", "R middle yaw", 0.0, 1.0),
    ("right_middle_mcp", "R middle MCP", 0.0, 1.0),
    ("right_middle_pip", "R middle PIP", 0.0, 1.0),
    ("right_middle_dip", "R middle DIP", 0.0, 1.0),

    ("right_ring_mcp_yaw", "R ring yaw", 0.0, 1.0),
    ("right_ring_mcp", "R ring MCP", 0.0, 1.0),
    ("right_ring_pip", "R ring PIP", 0.0, 1.0),
    ("right_ring_dip", "R ring DIP", 0.0, 1.0),

    ("right_little_mcp_yaw", "R little yaw", 0.0, 1.0),
    ("right_little_mcp", "R little MCP", 0.0, 1.0),
    ("right_little_pip", "R little PIP", 0.0, 1.0),
    ("right_little_dip", "R little DIP", 0.0, 1.0),
]




LEG_CHANNELS = [
    ("L_hip_yaw", "L hip yaw", -1.0, 1.0),
    ("L_hip_pitch", "L hip pitch", -1.0, 1.0),
    ("L_knee", "L knee", -1.0, 1.0),
    ("L_ankle_pitch", "L ankle pitch", -1.0, 1.0),
    ("L_ankle_roll", "L ankle roll", -1.0, 1.0),
    ("L_toe_in", "L toe inner", -1.0, 1.0),
    ("L_toe_mid", "L toe mid", -1.0, 1.0),
    ("L_toe_out", "L toe outer", -1.0, 1.0),
    ("L_toe_rear", "L toe rear", -1.0, 1.0),
    ("R_hip_yaw", "R hip yaw", -1.0, 1.0),
    ("R_hip_pitch", "R hip pitch", -1.0, 1.0),
    ("R_knee", "R knee", -1.0, 1.0),
    ("R_ankle_pitch", "R ankle pitch", -1.0, 1.0),
    ("R_ankle_roll", "R ankle roll", -1.0, 1.0),
    ("R_toe_in", "R toe inner", -1.0, 1.0),
    ("R_toe_mid", "R toe mid", -1.0, 1.0),
    ("R_toe_out", "R toe outer", -1.0, 1.0),
    ("R_toe_rear", "R toe rear", -1.0, 1.0),
]


class StatusPill(QtWidgets.QLabel):
    def __init__(self, text="--"):
        super().__init__(text)
        self.setMinimumHeight(34)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.set_state("grey", text)

    def set_state(self, color: str, text: str):
        palette = {
            "green": ("#123F2B", "#73E9A0"),
            "grey": ("#242C39", "#748298"),
            "yellow": ("#4A3B19", "#FFD36D"),
            "red": ("#4A1F2E", "#FF8FA3"),
            "blue": ("#19355C", "#8DBDFF"),
        }
        bg, fg = palette.get(color, palette["grey"])
        self.setText(text)
        self.setStyleSheet(
            f"background:{bg}; color:{fg}; border:1px solid {fg}; "
            "border-radius:12px; padding:6px 12px; font-weight:900;"
        )


class ActionSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, key: str, label: str, min_value: float, max_value: float, parent=None):
        super().__init__(parent)
        self.key = key
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        self.label = QtWidgets.QLabel(label)
        self.label.setMinimumWidth(120)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(-1000, 1000)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(lambda _v: self._emit())

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(self.min_value, self.max_value)
        self.spin.setSingleStep(0.05)
        self.spin.setValue(0.0)
        self.spin.setFixedWidth(88)
        self.spin.valueChanged.connect(self._spin_changed)

        self.zero_btn = QtWidgets.QPushButton("0")
        self.zero_btn.setFixedWidth(34)
        self.zero_btn.clicked.connect(self.set_zero)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(8)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.zero_btn)

    def _emit(self):
        v = self.value()
        blocker = QtCore.QSignalBlocker(self.spin)
        self.spin.setValue(v)
        del blocker
        self.valueChanged.emit()

    def _spin_changed(self, v):
        if v >= 0:
            norm = v / max(1e-9, self.max_value)
        else:
            # Example: v=-2.0, min_value=-2.0 -> norm=-1.0
            norm = v / max(1e-9, abs(self.min_value))
        norm = max(-1.0, min(1.0, norm))
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(int(norm * 1000))
        del blocker
        self.valueChanged.emit()

    def value(self) -> float:
        norm = self.slider.value() / 1000.0
        if norm >= 0:
            return float(norm * self.max_value)
        # Negative slider side must map to a real negative value.
        # Example: norm=-1.0, min_value=-2.0 -> -2.0
        return float((-norm) * self.min_value)

    def set_value(self, v: float):
        v = max(self.min_value, min(self.max_value, float(v)))
        blocker = QtCore.QSignalBlocker(self.spin)
        self.spin.setValue(v)
        del blocker
        self._spin_changed(v)

    def set_zero(self):
        self.set_value(0.0)

    def set_enabled_soft(self, enabled: bool):
        self.slider.setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.zero_btn.setEnabled(enabled)


class AgentActionsWindow(QtWidgets.QMainWindow):
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, status_host: str = "127.0.0.1", status_port: int = 8766):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.status_host = status_host
        self.status_port = int(status_port)
        self.send_timer = QtCore.QTimer(self)
        self.send_timer.setSingleShot(True)
        self.send_timer.timeout.connect(self.send_actions)
        self._closing = False
        self.preset_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self.active_preset_key = ""
        self.syncing_from_runner = False
        self.local_edit_until = 0.0
        self.last_runner_status: Dict = {}
        self.last_status_ok = False
        self.rotate_tetra_checkbox: QtWidgets.QCheckBox | None = None
        self.rotate_cube_checkbox: QtWidgets.QCheckBox | None = None
        self.fly_cube_checkbox: QtWidgets.QCheckBox | None = None

        self.setWindowTitle("Manual Agent Actions — IPC neural output override")
        self.resize(980, 860)
        self.sliders: Dict[str, ActionSlider] = {}

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(self._toolbar())

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("tabs")
        tabs.addTab(self._group_panel("Body + Head", BODY_CHANNELS, "body"), "Body + Head")
        tabs.addTab(self._group_panel("Arms", ARM_CHANNELS, "arm"), "Arms")
        tabs.addTab(self._group_panel("Hands / Palms / Fingers", HAND_CHANNELS, "hand"), "Hands")
        tabs.addTab(self._group_panel("Bird Legs / Toes", LEG_CHANNELS, "leg"), "Legs")
        tabs.addTab(self._gesture_presets_panel(), "Gesture Presets")
        root.addWidget(tabs, 1)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setObjectName("status")
        root.addWidget(self.status)

        self.apply_dark_theme()
        self.refresh_enabled_state()

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.poll_runner_status)
        self.status_timer.start(500)

    def _toolbar(self):
        w = QtWidgets.QFrame()
        w.setObjectName("toolbar")
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        self.status_ipc_pill = StatusPill("STATUS IPC: --")
        lay.addWidget(self.status_ipc_pill)

        self.enable_override = QtWidgets.QCheckBox("Override neural outputs")
        self.enable_override.setChecked(False)
        self.enable_override.toggled.connect(self._override_toggled)
        lay.addWidget(self.enable_override)

        self.send_live = QtWidgets.QCheckBox("Send live")
        self.send_live.setChecked(True)
        lay.addWidget(self.send_live)

        self.send_btn = QtWidgets.QPushButton("Send Now")
        self.send_btn.setObjectName("accent")
        self.send_btn.clicked.connect(self.send_actions)
        lay.addWidget(self.send_btn)

        self.zero_body_btn = QtWidgets.QPushButton("Zero Body")
        self.zero_body_btn.clicked.connect(lambda: self.zero_group("body"))
        lay.addWidget(self.zero_body_btn)

        self.neutral_pose_btn = QtWidgets.QPushButton("Neutral Pose")
        self.neutral_pose_btn.clicked.connect(self.set_neutral_pose)
        lay.addWidget(self.neutral_pose_btn)

        self.level_agent_btn = QtWidgets.QPushButton("Level Agent")
        self.level_agent_btn.clicked.connect(self.level_agent)
        lay.addWidget(self.level_agent_btn)

        self.open_hands_btn = QtWidgets.QPushButton("Open Hands")
        self.open_hands_btn.clicked.connect(self.gesture_open_palms)
        lay.addWidget(self.open_hands_btn)

        self.curl_fingers_btn = QtWidgets.QPushButton("Curl Fingers")
        self.curl_fingers_btn.clicked.connect(self.gesture_strong_fist)
        lay.addWidget(self.curl_fingers_btn)

        self.zero_all_btn = QtWidgets.QPushButton("Zero All")
        self.zero_all_btn.setObjectName("danger")
        self.zero_all_btn.clicked.connect(self.zero_all)
        lay.addWidget(self.zero_all_btn)

        lay.addStretch(1)
        return w

    def _gesture_presets_panel(self):
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        root = QtWidgets.QVBoxLayout(frame)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Gesture Presets")
        title.setObjectName("panelTitle")
        root.addWidget(title)

        subtitle = QtWidgets.QLabel("One-click scenarios for fingers, palms, arms, and grasp testing.")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        gestures = [
            ("Neutral hands", "gesture_neutral_hands", self.gesture_neutral_hands),
            ("Open palms", "gesture_open_palms", self.gesture_open_palms),
            ("Splay fingers", "gesture_splay_fingers", self.gesture_splay_fingers),
            ("Soft curl", "gesture_soft_curl", self.gesture_soft_curl),
            ("Strong fist", "gesture_strong_fist", self.gesture_strong_fist),
            ("Pinch L", "gesture_pinch_left", lambda: self.gesture_pinch("left")),
            ("Pinch R", "gesture_pinch_right", lambda: self.gesture_pinch("right")),
            ("Pinch both", "gesture_pinch_both", self.gesture_pinch_both),
            ("Point L index", "gesture_point_index_left", lambda: self.gesture_point_index("left")),
            ("Point R index", "gesture_point_index_right", lambda: self.gesture_point_index("right")),
            ("Peace L", "gesture_peace_left", lambda: self.gesture_peace("left")),
            ("Peace R", "gesture_peace_right", lambda: self.gesture_peace("right")),
            ("Thumbs up both", "gesture_thumbs_up_both", self.gesture_thumbs_up_both),
            ("Cup palms", "gesture_cup_palms", self.gesture_cup_palms),
            ("Precision grasp", "gesture_precision_grasp", self.gesture_precision_grasp),
            ("Power grasp", "gesture_power_grasp", self.gesture_power_grasp),
            ("Reach forward", "gesture_reach_forward", self.gesture_reach_forward),
            ("Arms open", "gesture_arms_open", self.gesture_arms_open),
            ("Hands to center", "gesture_hands_to_center", self.gesture_hands_to_center),
            ("Touch object pose", "gesture_touch_object_pose", self.gesture_touch_object_pose),
        ]
        scenarios = [
            ("Имит action", "imit_action", self.gesture_imit_action),
            ("Fly to cube + palpate", "fly_to_cube_palpate", self.gesture_fly_to_cube_and_palpate),
            ("Grab small cube + rotate", "fly_to_small_cube_grasp_rotate", self.gesture_fly_to_small_cube_grasp_rotate),
        ]

        root.addWidget(self._preset_section("Gestures", gestures))
        root.addWidget(self._preset_section("Scenarios", scenarios))
        root.addWidget(self._floating_object_scenario_panel())

        note = QtWidgets.QLabel("Tip: keep Override neural outputs ON. Presets set sliders and send IPC.")
        note.setWordWrap(True)
        root.addWidget(note)
        root.addStretch(1)
        return frame

    def _floating_object_scenario_panel(self):
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        section_title = QtWidgets.QLabel("Floating Object Scenario")
        section_title.setObjectName("sectionTitle")
        lay.addWidget(section_title)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        btn = QtWidgets.QPushButton("Inspect floating objects")
        btn.setObjectName("presetButton")
        btn.setProperty("activePreset", False)
        btn.setMinimumHeight(42)
        btn.clicked.connect(self.gesture_fly_to_tetrahedron_inspect)
        self.preset_buttons["fly_to_tetrahedron_inspect"] = btn
        row.addWidget(btn, 1)

        self.rotate_tetra_checkbox = QtWidgets.QCheckBox("Rotate tetrahedron")
        self.rotate_cube_checkbox = QtWidgets.QCheckBox("Rotate cube")
        self.fly_cube_checkbox = QtWidgets.QCheckBox("Fly cube")
        self.rotate_tetra_checkbox.setChecked(False)
        self.rotate_cube_checkbox.setChecked(False)
        self.fly_cube_checkbox.setChecked(False)
        self.rotate_tetra_checkbox.stateChanged.connect(self._floating_object_options_changed)
        self.rotate_cube_checkbox.stateChanged.connect(self._floating_object_options_changed)
        self.fly_cube_checkbox.stateChanged.connect(self._floating_object_options_changed)
        row.addWidget(self.rotate_tetra_checkbox)
        row.addWidget(self.rotate_cube_checkbox)
        row.addWidget(self.fly_cube_checkbox)
        row.addStretch(1)

        lay.addLayout(row)
        return wrap

    def _floating_object_options_payload(self) -> Dict[str, bool]:
        return {
            "rotate_tetrahedron": bool(self.rotate_tetra_checkbox.isChecked()) if self.rotate_tetra_checkbox is not None else False,
            "rotate_cube": bool(self.rotate_cube_checkbox.isChecked()) if self.rotate_cube_checkbox is not None else False,
            "fly_cube": bool(self.fly_cube_checkbox.isChecked()) if self.fly_cube_checkbox is not None else False,
        }

    def _floating_object_scenario_active(self) -> bool:
        if self.active_preset_key == "fly_to_tetrahedron_inspect":
            return True
        scenario = self.last_runner_status.get("adaptive_scenario_status", {}) if isinstance(self.last_runner_status, dict) else {}
        return (
            isinstance(scenario, dict)
            and bool(self.last_runner_status.get("adaptive_scenario_active", False) or scenario.get("active", False))
            and str(scenario.get("scenario", "")) == "fly_to_tetrahedron_inspect"
        )

    def _floating_object_options_changed(self, *_args):
        if not self._floating_object_scenario_active():
            payload = self._floating_object_options_payload()
            self.status.setText(
                f"{time.strftime('%H:%M:%S')} floating options ready | "
                f"tetra={int(payload['rotate_tetrahedron'])} cube={int(payload['rotate_cube'])} "
                f"fly={int(payload['fly_cube'])}"
            )
            return
        self.send_action_command(
            "fly_to_tetrahedron_inspect",
            active_key="fly_to_tetrahedron_inspect",
            **self._floating_object_options_payload(),
        )

    def _preset_section(self, title: str, presets: List[Tuple[str, str, object]]):
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        section_title = QtWidgets.QLabel(title)
        section_title.setObjectName("sectionTitle")
        lay.addWidget(section_title)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        lay.addLayout(grid)

        for i, (label, key, fn) in enumerate(presets):
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName("presetButton")
            btn.setProperty("activePreset", False)
            btn.setMinimumHeight(42)
            btn.clicked.connect(fn)
            self.preset_buttons[str(key)] = btn
            grid.addWidget(btn, i // 4, i % 4)

        return wrap

    def _set_active_preset(self, key: str) -> None:
        self.active_preset_key = str(key or "")
        for preset_key, btn in self.preset_buttons.items():
            btn.setProperty("activePreset", bool(preset_key == self.active_preset_key))
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()

    def _clear_active_preset(self) -> None:
        if self.active_preset_key:
            self._set_active_preset("")

    def _set_override_checked_from_runner(self, checked: bool) -> None:
        blocker = QtCore.QSignalBlocker(self.enable_override)
        self.enable_override.setChecked(bool(checked))
        del blocker
        self.refresh_enabled_state()

    def _ensure_override_checked(self) -> None:
        if self.enable_override.isChecked():
            return
        blocker = QtCore.QSignalBlocker(self.enable_override)
        self.enable_override.setChecked(True)
        del blocker
        self.refresh_enabled_state()

    def _sync_group_from_runner(self, group: str, values) -> None:
        try:
            vals = [float(v) for v in values]
        except Exception:
            return
        if group == "leg":
            vals = self._runtime_leg_to_ui(vals)

        prefix = group + "."
        keys = [k for k in self.sliders if k.startswith(prefix)]
        for key, value in zip(keys, vals):
            slider = self.sliders.get(key)
            if slider is None:
                continue
            blocker = QtCore.QSignalBlocker(slider)
            slider.set_value(float(value))
            del blocker

    def _preset_key_from_runner_status(self, data: Dict) -> str:
        scenario = data.get("adaptive_scenario_status", {})
        if not isinstance(scenario, dict):
            scenario = {}
        if bool(data.get("adaptive_scenario_active", False) or scenario.get("active", False)):
            name = str(scenario.get("scenario", ""))
            if name in self.preset_buttons:
                return name

        gesture = data.get("adaptive_gesture_status", {})
        if isinstance(gesture, dict) and bool(gesture.get("active", False)):
            command = str(gesture.get("command", ""))
            if command in self.preset_buttons:
                return command
        return ""

    def _reset_runner_driven_ui(self) -> None:
        if time.time() < float(getattr(self, "local_edit_until", 0.0)):
            return
        self._clear_active_preset()
        self._set_override_checked_from_runner(False)

    def poll_runner_status(self):
        self.status_host = self.current_status_host()
        self.status_port = self.current_status_port()

        data = request_module_debug_status(self.status_host, self.status_port, timeout=0.35)
        status_age = float(time.time() - float(data.get("updated_at", 0.0))) if isinstance(data, dict) else 999.0
        if not data or not bool(data.get("ready", True)) or status_age > 2.5:
            self.last_status_ok = False
            self.status_ipc_pill.set_state("grey", "STATUS IPC: no signal")
            self._reset_runner_driven_ui()
            self.status.setText(
                f"status lost | cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port}"
            )
            return

        self.last_status_ok = True
        self.last_runner_status = data
        self.status_ipc_pill.set_state("green", "STATUS IPC: receiving")

        local_edit_active = time.time() < float(getattr(self, "local_edit_until", 0.0))
        if not local_edit_active:
            self.syncing_from_runner = True
            try:
                self._set_override_checked_from_runner(bool(data.get("manual_actions_enabled", False)))
                self._sync_group_from_runner("body", data.get("manual_body_action", []))
                self._sync_group_from_runner("arm", data.get("manual_arm_action", []))
                self._sync_group_from_runner("hand", data.get("manual_hand_action", []))
                self._sync_group_from_runner("leg", data.get("manual_leg_action", []))

                preset_key = self._preset_key_from_runner_status(data)
                self._set_active_preset(preset_key)
            finally:
                self.syncing_from_runner = False

        scenario = data.get("adaptive_scenario_status", {})
        phase = scenario.get("phase", "") if isinstance(scenario, dict) else ""
        self.status.setText(
            f"status ok | step={data.get('global_step', 0)} | "
            f"cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port} | "
            f"override={'ON' if data.get('manual_actions_enabled', False) else 'OFF'} | "
            f"mode={self.active_preset_key or 'manual/none'}"
            + (f" | phase={phase}" if phase else "")
        )

    def _set_slider_value(self, group: str, key: str, value: float):
        slider = self.sliders.get(f"{group}.{key}")
        if slider is not None:
            slider.set_value(float(value))

    def _set_hand(self, name: str, value: float):
        self._set_slider_value("hand", name, value)

    def _set_arm(self, name: str, value: float):
        self._set_slider_value("arm", name, value)

    def _set_body(self, name: str, value: float):
        self._set_slider_value("body", name, value)

    def _set_finger(self, side: str, finger: str, value: float):
        for joint in ("mcp", "pip", "dip"):
            self._set_hand(f"{side}_{finger}_{joint}", value)

    def _set_all_fingers(self, value: float):
        for side in ("left", "right"):
            for finger in ("thumb", "index", "middle", "ring", "little"):
                self._set_finger(side, finger, value)

    def _set_side_fingers(self, side: str, value: float):
        for finger in ("thumb", "index", "middle", "ring", "little"):
            self._set_finger(side, finger, value)

    def send_action_command(self, action: str, active_key: str | None = None, mark_active: bool = True, **payload):
        """
        Send only a high-level action/scenario command to the runner.

        UI stays as a control panel. Motion logic lives in the adaptive controller,
        not in this PyQt window.
        """
        try:
            if self.send_timer.isActive():
                self.send_timer.stop()
        except Exception:
            pass
        self.local_edit_until = time.time() + 1.2

        if not self.enable_override.isChecked():
            blocker = QtCore.QSignalBlocker(self.enable_override)
            self.enable_override.setChecked(True)
            del blocker
            self.refresh_enabled_state()

        ok = send_ipc_message(
            self.current_host(),
            self.current_port(),
            make_action_message(action, **payload),
        )
        if ok and mark_active:
            self._set_active_preset(str(active_key or action))
        if not ok:
            self._clear_active_preset()
            self._set_override_checked_from_runner(False)
        self.status.setText(
            f"{time.strftime('%H:%M:%S')} command={action} sent ok={ok} | "
            + ("" if ok else "command IPC unavailable; start runner first | ")
            + f"cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port}"
        )

    def stop_floating_object_scenario(self):
        try:
            if self.send_timer.isActive():
                self.send_timer.stop()
        except Exception:
            pass
        self.local_edit_until = time.time() + 1.2

        ok = send_ipc_message(
            self.current_host(),
            self.current_port(),
            make_action_message("stop_fly_to_tetrahedron_inspect"),
        )
        self._set_active_preset("")
        if ok:
            self._set_override_checked_from_runner(False)
        self.status.setText(
            f"{time.strftime('%H:%M:%S')} floating object inspect stopped ok={ok} | "
            + ("" if ok else "command IPC unavailable; start runner first | ")
            + f"cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port}"
        )

    def gesture_neutral_hands(self):
        self.send_action_command("gesture_neutral_hands")

    def gesture_open_palms(self):
        self.send_action_command("gesture_open_palms")

    def gesture_splay_fingers(self):
        self.send_action_command("gesture_splay_fingers")

    def gesture_soft_curl(self):
        self.send_action_command("gesture_soft_curl")

    def gesture_strong_fist(self):
        self.send_action_command("gesture_strong_fist")

    def gesture_pinch(self, side: str, send: bool = True):
        self.send_action_command(f"gesture_pinch_{side}")

    def gesture_pinch_both(self):
        self.send_action_command("gesture_pinch_left", mark_active=False)
        self.send_action_command("gesture_pinch_right", active_key="gesture_pinch_both")

    def gesture_point_index(self, side: str):
        self.send_action_command(f"gesture_point_index_{side}")

    def gesture_peace(self, side: str):
        self.send_action_command(f"gesture_peace_{side}")

    def gesture_thumbs_up_both(self):
        self.send_action_command("gesture_thumbs_up_both")

    def gesture_cup_palms(self):
        self.send_action_command("gesture_cup_palms")

    def gesture_precision_grasp(self):
        self.send_action_command("gesture_precision_grasp")

    def gesture_power_grasp(self):
        self.send_action_command("gesture_power_grasp")

    def gesture_reach_forward(self):
        self.send_action_command("gesture_reach_forward")

    def gesture_arms_open(self):
        self.send_action_command("gesture_arms_open")

    def gesture_hands_to_center(self):
        self.send_action_command("gesture_hands_to_center")

    def gesture_touch_object_pose(self):
        self.send_action_command("gesture_touch_object_pose")

    def gesture_imit_action(self):
        self.send_action_command(
            "imit_action",
            active_key="imit_action",
            mode="conscious",
            steps=1,
            blend_into_focus_context=True,
            focus_blend_weight=0.015,
        )

    def gesture_fly_to_cube_and_palpate(self):
        self.send_action_command("fly_to_cube_palpate")

    def gesture_fly_to_small_cube_grasp_rotate(self):
        self.send_action_command("fly_to_small_cube_grasp_rotate")

    def gesture_fly_to_tetrahedron_inspect(self):
        if self._floating_object_scenario_active():
            self.stop_floating_object_scenario()
            return
        self.send_action_command(
            "fly_to_tetrahedron_inspect",
            **self._floating_object_options_payload(),
        )



    def _group_panel(self, title: str, channels: List[Tuple[str, str, float, float]], group: str):
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        head = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("panelTitle")
        head.addWidget(title_label)
        head.addStretch(1)
        zero = QtWidgets.QPushButton("Zero")
        zero.clicked.connect(lambda: self.zero_group(group))
        head.addWidget(zero)
        lay.addLayout(head)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        inner = QtWidgets.QWidget()
        inner_lay = QtWidgets.QVBoxLayout(inner)
        inner_lay.setContentsMargins(2, 2, 2, 2)
        inner_lay.setSpacing(4)

        for key, label, mn, mx in channels:
            full_key = f"{group}.{key}"
            s = ActionSlider(full_key, label, mn, mx)
            s.valueChanged.connect(self._slider_changed)
            self.sliders[full_key] = s
            inner_lay.addWidget(s)

        inner_lay.addStretch(1)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        return frame

    def _override_toggled(self, checked: bool):
        if self.syncing_from_runner:
            return
        self.local_edit_until = time.time() + 1.2
        self.refresh_enabled_state()
        if checked:
            # When entering manual mode, send a clean neutral command immediately.
            # This prevents the previous neural output from being kept for one step.
            self.set_neutral_pose(send=False)
            self.send_actions()
        else:
            self.queue_send()

    def refresh_enabled_state(self):
        # Sliders must stay interactive even while override is OFF: the first
        # local edit auto-enables override in _slider_changed().
        en = True
        for s in self.sliders.values():
            s.set_enabled_soft(en)

    def _slider_changed(self):
        if self.syncing_from_runner:
            return
        self._ensure_override_checked()
        self.local_edit_until = time.time() + 1.2
        self._clear_active_preset()
        if self.send_live.isChecked():
            self.send_actions()

    def queue_send(self):
        if bool(getattr(self, "_closing", False)):
            return
        self.send_timer.start(80)

    def shutdown_timers(self):
        self._closing = True
        try:
            if self.send_timer.isActive():
                self.send_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "status_timer") and self.status_timer.isActive():
                self.status_timer.stop()
        except Exception:
            pass

    def current_host(self):
        return self.host

    def current_port(self):
        return int(self.port)

    def current_status_host(self):
        return self.status_host

    def current_status_port(self):
        return int(self.status_port)

    def values_for_group(self, group: str) -> List[float]:
        prefix = group + "."
        keys = [k for k in self.sliders if k.startswith(prefix)]
        # Preserve original dict insertion order.
        return [float(self.sliders[k].value()) for k in keys]

    def _swap_leg_sides(self, values) -> List[float]:
        vals = [float(v) for v in values]
        if len(vals) != 18:
            return vals
        return vals[9:18] + vals[0:9]

    def _invert_leg_knees(self, values) -> List[float]:
        vals = [float(v) for v in values]
        if len(vals) != 18:
            return vals
        vals[2] = -vals[2]
        vals[11] = -vals[11]
        return vals

    def _ui_leg_to_runtime(self, values) -> List[float]:
        # MuJoCo knee joints use negative qpos for flexion. Keep UI intuitive:
        # positive knee slider means bend/flex, then adapt to runtime order.
        return self._invert_leg_knees(self._swap_leg_sides(values))

    def _runtime_leg_to_ui(self, values) -> List[float]:
        return self._swap_leg_sides(self._invert_leg_knees(values))

    def payload(self):
        return {
            "manual_actions_enabled": bool(self.enable_override.isChecked()),
            "manual_body_action": self.values_for_group("body"),
            "manual_arm_action": self.values_for_group("arm"),
            "manual_hand_action": self.values_for_group("hand"),
            "manual_leg_action": self._ui_leg_to_runtime(self.values_for_group("leg")),
        }

    def send_actions(self):
        self.local_edit_until = time.time() + 1.2
        payload = self.payload()
        msg = make_set_state_message(**payload)
        ok = send_ipc_message(self.current_host(), self.current_port(), msg)
        if not ok:
            self._clear_active_preset()
            self._set_override_checked_from_runner(False)
        state = "ON" if self.enable_override.isChecked() else "OFF"
        body_vals = payload["manual_body_action"]
        arm_vals = payload["manual_arm_action"]
        hand_vals = payload["manual_hand_action"]
        leg_vals = payload["manual_leg_action"]
        self.status.setText(
            f"{time.strftime('%H:%M:%S')} sent ok={ok} | override={state} | "
            + ("" if ok else "command IPC unavailable; start runner first | ")
            + f"cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port} | "
            f"body={len(body_vals)} max={max([abs(v) for v in body_vals] or [0.0]):.2f} | "
            f"arm={len(arm_vals)} max={max([abs(v) for v in arm_vals] or [0.0]):.2f} | "
            f"hand={len(hand_vals)} min/max={min(hand_vals or [0.0]):.2f}/{max(hand_vals or [0.0]):.2f} | "
            f"leg={len(leg_vals)} max={max([abs(v) for v in leg_vals] or [0.0]):.2f}"
        )

    def level_agent(self):
        """
        Ask runner to level/reset the agent pose/orientation.
        This is different from zeroing sliders: it corrects current body pose.
        """
        self.set_neutral_pose(send=False)
        msg = make_set_state_message(
            manual_actions_enabled=bool(self.enable_override.isChecked()),
            manual_body_action=self.values_for_group("body"),
            manual_arm_action=self.values_for_group("arm"),
            manual_hand_action=self.values_for_group("hand"),
            manual_leg_action=self._ui_leg_to_runtime(self.values_for_group("leg")),
            level_agent_pose=True,
        )
        ok = send_ipc_message(self.current_host(), self.current_port(), msg)
        if not ok:
            self._set_override_checked_from_runner(False)
        self.status.setText(
            f"{time.strftime('%H:%M:%S')} level_agent sent ok={ok} | "
            + ("" if ok else "command IPC unavailable; start runner first | ")
            + f"cmd={self.host}:{self.port} | status={self.status_host}:{self.status_port}"
        )

    def set_neutral_pose(self, send: bool = True):
        """
        Neutral UI pose:
            body/head velocities = 0
            arms = 0 neural raw
            hands = 0.5 direct hand_ctrl neutral
            legs = 0
        """
        for key, slider in self.sliders.items():
            slider.set_value(0.5 if key.startswith("hand.") else 0.0)
        if send:
            self.queue_send()

    def set_fingers(self, value: float):
        # Finger channels are every hand slider except palm pitch/roll.
        skip = ("palm_pitch", "palm_roll", "mcp_yaw")
        for key, slider in self.sliders.items():
            if not key.startswith("hand."):
                continue
            name = key.split(".", 1)[1]
            if any(s in name for s in skip):
                continue
            slider.set_value(float(value))
        self.queue_send()

    def zero_group(self, group: str):
        self._ensure_override_checked()
        self.local_edit_until = time.time() + 1.2
        self._clear_active_preset()
        prefix = group + "."
        for key, slider in self.sliders.items():
            if key.startswith(prefix):
                blocker = QtCore.QSignalBlocker(slider)
                slider.set_zero()
                del blocker
        self.send_actions()

    def zero_all(self):
        self._ensure_override_checked()
        self.local_edit_until = time.time() + 1.2
        self._clear_active_preset()
        for slider in self.sliders.values():
            blocker = QtCore.QSignalBlocker(slider)
            slider.set_zero()
            del blocker
        self.send_actions()

    def closeEvent(self, event):
        self.shutdown_timers()
        # Safety: disable override when closing.
        try:
            blocker = QtCore.QSignalBlocker(self.enable_override)
            self.enable_override.setChecked(False)
            del blocker
            self.send_actions()
        except Exception:
            pass
        self.shutdown_timers()
        super().closeEvent(event)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0C121B;
                color: #DCE8F8;
                font-size: 12px;
            }
            QFrame#toolbar, QFrame#panel {
                background: #141D29;
                border: 1px solid #243145;
                border-radius: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #243145;
                border-radius: 12px;
                background: #141D29;
                padding: 6px;
            }
            QTabBar::tab {
                background: #1D2A3B;
                color: #DCE8F8;
                border: 1px solid #37507A;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 16px;
                margin-right: 3px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #2B66F6;
                color: white;
                border: 1px solid #4F85FF;
            }
            QTabBar::tab:hover {
                background: #243449;
            }
            QLabel#panelTitle {
                color: #EDF4FF;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#sectionTitle {
                color: #9FB6D4;
                font-size: 13px;
                font-weight: 700;
                padding: 4px 0 0 1px;
            }
            QLabel#status {
                color: #B7C5DA;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #0F1722;
                color: #FFFFFF;
                border: 1px solid #304059;
                border-radius: 8px;
                padding: 5px 7px;
                min-height: 24px;
            }
            QCheckBox {
                color: #EDF4FF;
                font-weight: 600;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #DDE7FF;
                background: rgba(255,255,255,20);
            }
            QCheckBox::indicator:checked {
                background: #2B66F6;
                border: 1px solid #4F85FF;
            }
            QPushButton {
                background: #1D2A3B;
                color: white;
                border: 1px solid #37507A;
                border-radius: 10px;
                padding: 7px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #243449;
            }
            QPushButton#presetButton[activePreset="true"] {
                background: #1F8F57;
                border: 1px solid #55D08D;
                color: #FFFFFF;
            }
            QPushButton#presetButton[activePreset="true"]:hover {
                background: #27A765;
            }
            QPushButton#accent {
                background: #2B66F6;
                border: 1px solid #4F85FF;
            }
            QPushButton#danger {
                background: #7A2D4B;
                border: 1px solid #C45A8B;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #0F1722;
                border: 1px solid #304059;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6F84A3;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
                border: 1px solid #9BB2D1;
            }
            QSlider::handle:horizontal:hover {
                background: #9BB2D1;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: #111A27;
                width: 14px;
                margin: 2px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #6F84A3;
                min-height: 34px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9BB2D1;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)


def main():
    parser = argparse.ArgumentParser(description="PyQt manual agent action override over IPC.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=8766)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = AgentActionsWindow(host=args.host, port=args.port, status_host=args.status_host, status_port=args.status_port)
    app.aboutToQuit.connect(win.shutdown_timers)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
