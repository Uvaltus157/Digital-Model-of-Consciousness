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

"""
pyqt_control_panel_ipc.py

Standalone PyQt5 control panel using local IPC messages.

Does NOT import cv2.

Usage:
    python pyqt_control_panel_ipc.py --host 127.0.0.1 --port 8765

Start order:
    1. start main viewer with IPC server
    2. start this PyQt control panel
"""

import argparse
import os
import time
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, fields

from src.platform.ipc.ipc_control_bus import send_ipc_message, make_set_state_message, make_toggle_message, make_action_message
from src.modules.m08_debug_visual_control.module_debug_status_ipc import request_module_debug_status


DEFAULT_WINDOW_CONFIG = {
    "min_width": 860,
    "min_height": 980,
    "width": 940,
    "height": 1040,
}

DEFAULT_MODULE_DEBUG_SCRIPT = "pyqt_module_debug_ipc_status_registry.py"

MODULE_TAB_BUTTONS = {
    "m1": [
        "btn_object_image",
        "btn_object_open3d",
        "btn_object_open3d_rpc",
        "btn_object_open3d_step4",
        "btn_object_open3d_file",
        "btn_save_ply",
        "btn_save_pcd",
        "btn_static_dynamic",
    ],
    "m2": ["btn_event_code"],
    "m3": [],
    "m6": [],
    "m7": ["btn_inner"],
    "m8": ["btn_module_debug"],
    "m14": ["btn_latent"],
}


def start_runner_button_enabled(runner_connected: bool) -> bool:
    return not bool(runner_connected)


@dataclass
class LocalPanelState:
    mujoco_next_run: bool = False
    inner_world: bool = False
    cameras: bool = False
    actions: bool = False
    object_image: bool = False
    event_code_visualizer: bool = False
    object_image_open3d: bool = False
    manual_actions: bool = False
    training: bool = False
    module_debug: bool = False
    latent_semantic: bool = False
    video_sensor_enabled: bool = False
    contact_sensor_enabled: bool = False
    imu_sensor_enabled: bool = False
    static_dynamic_code: bool = False
    connected: bool = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--viewer-script", default="runner.py")
    parser.add_argument("--viewer-config", default="runner")
    parser.add_argument("--terminal", default="xterm")
    parser.add_argument("--module-debug-script", default=DEFAULT_MODULE_DEBUG_SCRIPT)
    parser.add_argument("--module-status-port", type=int, default=8766)
    parser.add_argument("--agent-actions-script", default="pyqt_agent_actions_ipc.py")
    parser.add_argument("--panel-config", default=None)
    args = parser.parse_args()

    try:
        from PyQt5 import QtCore, QtWidgets
    except Exception as e:
        print(f"PyQt5 unavailable: {e}")
        print("Install: pip install PyQt5")
        raise

    class ControlWindow(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.state = LocalPanelState()
            self.module_debug_proc = None
            self.agent_actions_proc = None
            self.open3d_slot_viewer_step4_proc = None
            self.open3d_slot_viewer_rpc_proc = None
            self.open3d_slot_viewer_proc = None
            self.module_debug_external_alive = False
            self.agent_actions_external_alive = False
            self._last_process_scan = 0.0
            self.last_status_ok = False
            self.last_status = {}
            self.setWindowTitle("Conscious Viewer Control - IPC")
            window_cfg = self.load_window_config()
            self.setMinimumWidth(int(window_cfg["min_width"]))
            self.setMinimumHeight(int(window_cfg["min_height"]))
            self.resize(int(window_cfg["width"]), int(window_cfg["height"]))
            self.setStyleSheet("""
                QWidget {
                    background: #0C121B;
                    color: #DCE8F8;
                    font-size: 12px;
                }
                QLabel {
                    background: transparent;
                }
                QGroupBox {
                    background: #101722;
                    border: 1px solid #243145;
                    border-radius: 12px;
                    margin-top: 12px;
                    padding: 12px;
                    font-weight: 800;
                    color: #DCE8F8;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #B7C5DA;
                }
                QPushButton {
                    background: #1D2A3B;
                    color: white;
                    border: 1px solid #37507A;
                    border-radius: 10px;
                    padding: 8px 12px;
                    font-weight: 700;
                }
                QPushButton:hover {
                    background: #263B55;
                }
            """)

            self.status = QtWidgets.QLabel("")
            self.status.setMinimumHeight(26)
            self.status.setStyleSheet("font-weight: bold; color: #B7C5DA; padding: 2px 4px;")

            self.btn_mujoco = QtWidgets.QPushButton()
            self.btn_inner = QtWidgets.QPushButton()
            self.btn_cameras = QtWidgets.QPushButton()
            self.btn_actions = QtWidgets.QPushButton()
           
            self.btn_object_image = QtWidgets.QPushButton()
            self.btn_event_code = QtWidgets.QPushButton()
            self.btn_static_dynamic = QtWidgets.QPushButton()
            self.btn_object_open3d = QtWidgets.QPushButton()
            self.btn_object_open3d_step4 = QtWidgets.QPushButton("Open3D RPC ext")
            self.btn_object_open3d_rpc = QtWidgets.QPushButton("Open3D RPC")
            self.btn_object_open3d_file = QtWidgets.QPushButton("Open3D file")
            self.btn_train = QtWidgets.QPushButton()
            self.btn_module_debug = QtWidgets.QPushButton()
            self.btn_module_debug_pyqt = QtWidgets.QPushButton("PyQt Module Debug")
            self.btn_agent_actions_pyqt = QtWidgets.QPushButton("PyQt Agent Actions Imit")
            self.btn_module_debug_pyqt.setObjectName("pyqtWindowButton")
            self.btn_agent_actions_pyqt.setObjectName("pyqtWindowButton")
            self.btn_latent = QtWidgets.QPushButton()
            self.chk_sensor_video = QtWidgets.QCheckBox("Video")
            self.chk_sensor_contact = QtWidgets.QCheckBox("Tactile")
            self.chk_sensor_imu = QtWidgets.QCheckBox("IMU")

            for b in [self.btn_mujoco, self.btn_inner, self.btn_cameras, self.btn_actions, self.btn_object_image, self.btn_event_code, self.btn_static_dynamic, self.btn_object_open3d, self.btn_train, self.btn_module_debug, self.btn_latent]:
                b.setCheckable(True)
                b.setMinimumHeight(42)
            for cb in [self.chk_sensor_video, self.chk_sensor_contact, self.chk_sensor_imu]:
                cb.setChecked(True)
                cb.setFixedHeight(42)
                cb.setFixedWidth(92)
                cb.setStyleSheet(
                    "QCheckBox {"
                    " background:#141D29;"
                    " color:#DCE8F8;"
                    " border:1px solid #2B3A50;"
                    " border-radius:10px;"
                    " font-weight:800;"
                    " spacing:7px;"
                    " padding:0 9px;"
                    "}"
                    "QCheckBox:hover { background:#1A2636; border:1px solid #4A6386; }"
                    "QCheckBox:disabled { color:#647287; background:#151B24; border:1px solid #263142; }"
                    "QCheckBox::indicator {"
                    " width:15px;"
                    " height:15px;"
                    " border-radius:8px;"
                    " border:1px solid #46566D;"
                    " background:#263142;"
                    "}"
                    "QCheckBox::indicator:checked {"
                    " background:#43D17B;"
                    " border:1px solid #9AF3BB;"
                    "}"
                    "QCheckBox::indicator:unchecked {"
                    " background:#263142;"
                    " border:1px solid #46566D;"
                    "}"
                    "QCheckBox::indicator:disabled {"
                    " background:#1B2330;"
                    " border:1px solid #344052;"
                    "}"
                )

            self.btn_save_ply = QtWidgets.QPushButton("Save inner 3D as PLY")
            self.btn_save_pcd = QtWidgets.QPushButton("Save inner 3D as PCD")
            self.btn_save_model = QtWidgets.QPushButton("Save model")
            self.btn_stop = QtWidgets.QPushButton("Stop")
            self.btn_start_viewer = QtWidgets.QPushButton("Start runner")
            self.btn_stop.setStyleSheet(
                "QPushButton { background:#7A2D4B; border:1px solid #C45A8B; color:white; "
                "font-weight:900; border-radius:10px; }"
                "QPushButton:hover { background:#944060; border:1px solid #FF8FA3; }"
            )
            self.btn_save_ply.setMinimumHeight(42)
            self.btn_save_pcd.setMinimumHeight(42)
            self.btn_save_model.setMinimumHeight(42)
            self.btn_stop.setMinimumHeight(42)
            self.btn_start_viewer.setMinimumHeight(42)
            self.btn_module_debug_pyqt.setMinimumHeight(42)
            self.btn_agent_actions_pyqt.setMinimumHeight(42)
            self.btn_object_open3d_step4.setMinimumHeight(42)
            self.btn_object_open3d_rpc.setMinimumHeight(42)
            self.btn_object_open3d_file.setMinimumHeight(42)
            self.btn_start_viewer.setToolTip("Start the main runner process with the configured runner.yaml")
            self.btn_cameras.setToolTip("Show or hide the input sensors preview window")
            self.chk_sensor_video.setToolTip("Enable or disable video / eyes input")
            self.chk_sensor_contact.setToolTip("Enable or disable tactile / contact input")
            self.chk_sensor_imu.setToolTip("Enable or disable IMU / vestibular body input")
            self.btn_actions.setToolTip("Show or hide the action outputs window")
            self.btn_agent_actions_pyqt.setToolTip("Open or close the PyQt agent action imitation window")
            self.btn_object_image.setToolTip("Show or hide the inner object visualizer window")
            self.btn_event_code.setToolTip("Show or hide the event code visualizer window")
            self.btn_static_dynamic.setToolTip("Show or hide the static/dynamic code debug window")
            self.btn_object_open3d.setToolTip("Show or hide the built-in runner Inner Object Open3D window")
            self.btn_object_open3d_step4.setToolTip("Launch Step4 Open3D viewer with points/mesh/bbox modes")
            self.btn_object_open3d_rpc.setToolTip("Launch live JSON-RPC Open3D Slot Viewer")
            self.btn_object_open3d_file.setToolTip("Launch separate Open3D Slot Viewer from checkpoint/slot_viewer/slot_4d_open3d_latest.npz")
            self.btn_inner.setToolTip("Show or hide the inner world / thoughts window")
            self.btn_latent.setToolTip("Show or hide the latent semantic map window")
            self.btn_mujoco.setToolTip("Enable or disable the MuJoCo viewer on the next runner launch")
            self.btn_train.setToolTip("Enable or disable online training in the runner")
            self.btn_module_debug.setToolTip("Show or hide the runner-owned module debug visualizer")
            self.btn_module_debug_pyqt.setToolTip("Open or close the registry-backed PyQt module debug window")
            self.btn_save_ply.setToolTip("Export the current internal 3D object model as a PLY file")
            self.btn_save_pcd.setToolTip("Export the current internal 3D object model as a PCD file")
            self.btn_save_model.setToolTip("Request the runner to save a checkpoint")
            self.btn_stop.setToolTip("Request the runner to stop")

            hint = QtWidgets.QLabel(
                f"IPC: {args.host}:{args.port}\\n"
                "Hotkeys: I - thoughts, C - cameras/sensors, A - actions, M - manual actions, O - object image, V - event code, Z - static/dynamic code, P - object Open3D, T - train, Q/Esc - stop\\n"
                "PLY/PCD buttons save the current internal 3D object model.\\n"
                "The control panel runs as a separate process without importing cv2.\n"
                "The start button opens the viewer in a separate process."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #8FA4BF; background:#101722; border:1px solid #243145; border-radius:10px; padding:10px;")

            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)
            layout.addWidget(self.status)

            launch_row = QtWidgets.QHBoxLayout()
            launch_row.setSpacing(10)
            launch_row.addWidget(self.btn_start_viewer)
            launch_row.addWidget(self.btn_stop)
            launch_box = QtWidgets.QGroupBox("Startup and Connection")
            launch_box_lay = QtWidgets.QVBoxLayout()
            launch_box_lay.setSpacing(10)
            launch_box_lay.addLayout(launch_row)
            launch_box.setLayout(launch_box_lay)

            viewers_box = QtWidgets.QGroupBox("Core Brain")
            viewers_box.setStyleSheet("""
                QGroupBox {
                    background: #101722;
                    font-weight: bold;
                    border: 1px solid #243145;
                    border-radius: 12px;
                    margin-top: 12px;
                    padding: 12px;
                    color: #DCE8F8;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #B7C5DA;
                }
            """)
            viewers_lay = QtWidgets.QVBoxLayout()
            viewers_lay.setSpacing(10)
            sensor_row = QtWidgets.QHBoxLayout()
            sensor_row.setSpacing(10)
            sensor_row.addWidget(self.btn_cameras, 2)
            sensor_row.addWidget(self.chk_sensor_video, 0)
            sensor_row.addWidget(self.chk_sensor_contact, 0)
            sensor_row.addWidget(self.chk_sensor_imu, 0)
            sensor_row.addWidget(self.btn_train, 2)
            viewers_lay.addLayout(sensor_row)

            module_tabs = QtWidgets.QTabWidget()
            module_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #243145;
                    border-radius: 10px;
                    background: #101722;
                }
                QTabBar::tab {
                    background: #141D29;
                    color: #B7C5DA;
                    border: 1px solid #243145;
                    padding: 7px 11px;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    margin-right: 2px;
                    font-weight: 800;
                }
                QTabBar::tab:selected {
                    background: #243449;
                    color: #FFFFFF;
                    border: 1px solid #4F85FF;
                }
            """)
            def add_module_tab_buttons(tab_lay, button_names):
                row = None
                for idx, button_name in enumerate(button_names):
                    if idx % 2 == 0:
                        row = QtWidgets.QHBoxLayout()
                        row.setSpacing(10)
                        tab_lay.addLayout(row)
                    row.addWidget(getattr(self, button_name))
                if row is not None and len(button_names) % 2 == 1:
                    row.addStretch(1)

            for module_idx in range(1, 16):
                tab = QtWidgets.QWidget()
                tab_lay = QtWidgets.QVBoxLayout(tab)
                tab_lay.setContentsMargins(10, 10, 10, 10)
                tab_lay.setSpacing(10)
                add_module_tab_buttons(tab_lay, MODULE_TAB_BUTTONS.get(f"m{module_idx}", []))
                tab_lay.addStretch(1)
                module_tabs.addTab(tab, f"m{module_idx}")
            viewers_lay.addWidget(module_tabs)
            debug_row = QtWidgets.QHBoxLayout()
            debug_row.setSpacing(10)
            debug_row.addWidget(self.btn_module_debug_pyqt)
            debug_row.addWidget(self.btn_save_model)
            viewers_lay.addLayout(debug_row)
            
            viewers_box.setLayout(viewers_lay)

            model_box = self.make_section("World", [
                self.btn_mujoco,
            ])

            action_row = QtWidgets.QHBoxLayout()
            action_row.setSpacing(10)
            action_row.addWidget(self.btn_actions)
            action_row.addWidget(self.btn_agent_actions_pyqt)
            action_box = QtWidgets.QGroupBox("Actions")
            action_box_lay = QtWidgets.QVBoxLayout()
            action_box_lay.addLayout(action_row)
            action_box.setLayout(action_box_lay)

            layout.addWidget(launch_box)
            layout.addWidget(model_box)
            layout.addWidget(viewers_box)
            layout.addWidget(action_box)
            layout.addWidget(hint)
            self.setLayout(layout)

            self.btn_mujoco.clicked.connect(lambda: self.toggle("mujoco_next_run"))
            self.btn_inner.clicked.connect(lambda: self.toggle("inner_world"))
            self.btn_cameras.clicked.connect(lambda: self.toggle("cameras"))
            self.btn_actions.clicked.connect(lambda: self.toggle("actions"))
            
            self.btn_object_image.clicked.connect(lambda: self.toggle("object_image"))
            self.btn_event_code.clicked.connect(lambda: self.toggle("event_code_visualizer"))
            self.btn_static_dynamic.clicked.connect(lambda: self.toggle("static_dynamic_code"))
            self.btn_object_open3d.clicked.connect(lambda: self.toggle("object_image_open3d"))
            self.btn_object_open3d_step4.clicked.connect(self.launch_inner_object_open3d_step4)
            self.btn_object_open3d_rpc.clicked.connect(self.launch_inner_object_open3d_rpc)
            self.btn_object_open3d_file.clicked.connect(self.launch_inner_object_open3d_file)
            self.btn_train.clicked.connect(lambda: self.toggle("training"))
            self.chk_sensor_video.toggled.connect(lambda checked: self.set_sensor_gate("video", checked))
            self.chk_sensor_contact.toggled.connect(lambda checked: self.set_sensor_gate("contact", checked))
            self.chk_sensor_imu.toggled.connect(lambda checked: self.set_sensor_gate("imu", checked))
            self.btn_module_debug.clicked.connect(lambda: self.toggle("module_debug"))
            self.btn_module_debug_pyqt.clicked.connect(self.open_pyqt_module_debug)
            self.btn_agent_actions_pyqt.clicked.connect(self.open_pyqt_agent_actions)
            self.btn_latent.clicked.connect(lambda: self.toggle("latent_semantic"))
            self.btn_save_ply.clicked.connect(lambda: self.action("save_object_ply"))
            self.btn_save_pcd.clicked.connect(lambda: self.action("save_object_pcd"))
            self.btn_save_model.clicked.connect(lambda: self.action("save_checkpoint"))
            self.btn_stop.clicked.connect(lambda: self.action("stop"))
            self.btn_start_viewer.clicked.connect(self.start_viewer)

            self.status_timer = QtCore.QTimer(self)
            self.status_timer.timeout.connect(self.poll_runner_status)
            self.status_timer.start(500)

            self.refresh_ui()

        def send(self, msg) -> bool:
            ok = send_ipc_message(args.host, args.port, msg)
            if not ok:
                self._reset_to_default()
            self.state.connected = ok
            self.refresh_ui()
            return ok

        def _sync_state_from_status(self, data: dict):
            for f in fields(LocalPanelState):
                if f.name == "connected":
                    continue
                if f.name in data:
                    setattr(self.state, f.name, bool(data.get(f.name, False)))
            sensors = data.get("input_sensors_enabled")
            if not isinstance(sensors, dict) and isinstance(data.get("sleep_sensor_mask"), dict):
                sensors = {k: not bool(v) for k, v in data.get("sleep_sensor_mask", {}).items()}
            if isinstance(sensors, dict):
                self.state.video_sensor_enabled = bool(sensors.get("video", self.state.video_sensor_enabled))
                self.state.contact_sensor_enabled = bool(sensors.get("contact", self.state.contact_sensor_enabled))
                self.state.imu_sensor_enabled = bool(sensors.get("imu", self.state.imu_sensor_enabled))
            if "training_enabled" in data:
                self.state.training = bool(data.get("training_enabled", False))
            self.state.connected = True

        def poll_runner_status(self):
            data = request_module_debug_status(args.host, args.module_status_port, timeout=0.35)
            status_age = float(time.time() - float(data.get("updated_at", 0.0))) if isinstance(data, dict) else 999.0
            if not data or not bool(data.get("ready", True)) or status_age > 2.5:
                self.last_status_ok = False
                self.last_status = {}
                self._reset_to_default()
                self.refresh_ui()
                return

            self.last_status_ok = True
            self.last_status = data
            self._sync_state_from_status(data)
            self.status.setText(f"STATUS IPC: receiving | step={int(data.get('global_step', 0) or 0)}")
            self.refresh_ui()

        def load_window_config(self) -> dict[str, int]:
            cfg = dict(DEFAULT_WINDOW_CONFIG)
            config_path = None
            if args.panel_config:
                config_path = Path(args.panel_config).expanduser()
                if not config_path.is_absolute():
                    config_path = self.resolve_root_path() / config_path
            else:
                config_path = (
                    self.resolve_root_path()
                    / "src"
                    / "modules"
                    / "m08_debug_visual_control"
                    / "config"
                    / "control_panel.yaml"
                )

            try:
                import yaml  # type: ignore

                if config_path.exists():
                    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                    window = data.get("window", {}) if isinstance(data, dict) else {}
                    if isinstance(window, dict):
                        for key in cfg:
                            if key in window:
                                cfg[key] = int(window[key])
            except Exception as e:
                print(f"[control] window config load failed: {e}")

            return cfg

        def make_section(self, title: str, widgets):
            box = QtWidgets.QGroupBox(title)
            box.setStyleSheet("""
                QGroupBox {
                    background: #101722;
                    font-weight: bold;
                    border: 1px solid #243145;
                    border-radius: 12px;
                    margin-top: 12px;
                    padding: 12px;
                    color: #DCE8F8;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #B7C5DA;
                }
            """)
            lay = QtWidgets.QVBoxLayout()
            lay.setSpacing(10)
            for w in widgets:
                lay.addWidget(w)
            box.setLayout(lay)
            return box

        def _runner_dependent_buttons(self):
            return [
                self.btn_mujoco,
                self.btn_inner,
                self.btn_cameras,
                self.chk_sensor_video,
                self.chk_sensor_contact,
                self.chk_sensor_imu,
                self.btn_actions,
                self.btn_object_image,
                self.btn_event_code,
                self.btn_static_dynamic,
                self.btn_object_open3d,
                self.btn_train,
                self.btn_module_debug,
                self.btn_latent,
                self.btn_save_ply,
                self.btn_save_pcd,
                self.btn_save_model,
                self.btn_stop,
            ]

        def _disabled_runner_button_style(self) -> str:
            return (
                "QPushButton { background:#151B24; color:#647287; border:1px solid #263142; "
                "border-radius:10px; padding:8px 12px; font-weight:700; }"
            )

        def _stop_button_style(self) -> str:
            return (
                "QPushButton { background:#7A2D4B; border:1px solid #C45A8B; color:white; "
                "font-weight:900; border-radius:10px; }"
                "QPushButton:hover { background:#944060; border:1px solid #FF8FA3; }"
            )

        def _restore_runner_action_button_styles(self):
            for btn in [self.btn_save_ply, self.btn_save_pcd, self.btn_save_model]:
                btn.setStyleSheet("")
            self.btn_stop.setStyleSheet(self._stop_button_style())

        def _start_runner_button_style(self, enabled: bool) -> str:
            if enabled:
                return (
                    "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                    "border-radius:10px; padding:8px 12px; font-weight:700; }"
                    "QPushButton:hover { background:#263B55; }"
                )
            return self._disabled_runner_button_style()

        def _set_start_runner_enabled(self, connected: bool):
            enabled = start_runner_button_enabled(connected)
            self.btn_start_viewer.setEnabled(enabled)
            self.btn_start_viewer.setStyleSheet(self._start_runner_button_style(enabled))

        def _set_runner_controls_enabled(self, connected: bool):
            for btn in self._runner_dependent_buttons():
                btn.setEnabled(bool(connected))
                if not connected and isinstance(btn, QtWidgets.QPushButton):
                    btn.setStyleSheet(self._disabled_runner_button_style())
            if connected:
                self._restore_runner_action_button_styles()
            self._set_start_runner_enabled(connected)

        def _style_button(self, btn, on: bool, label: str):
            btn.setChecked(on)
            btn.setText(("ON   " if on else "OFF  ") + label)
            if on:
                btn.setStyleSheet(
                    "QPushButton { background:#1F8F57; color:white; border:1px solid #55D08D; "
                    "border-radius:10px; padding:8px 12px; font-weight:900; }"
                    "QPushButton:hover { background:#27A765; border:1px solid #73E9A0; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background:#242C39; color:#B7C5DA; border:1px solid #3A4658; "
                    "border-radius:10px; padding:8px 12px; font-weight:700; }"
                    "QPushButton:hover { background:#324055; color:#FFFFFF; border:1px solid #6F84A3; }"
                )

        def _sleep_mode_active(self) -> bool:
            if not self.last_status_ok or not isinstance(self.last_status, dict):
                return False
            sensor_state = str(self.last_status.get("sensor_state", "")).lower().strip()
            return bool(self.last_status.get("full_sleep", False)) or sensor_state == "sleep"

        def _style_train_button(self, btn, on: bool):
            sleep_mode = self._sleep_mode_active()
            btn.setChecked(on)
            if sleep_mode:
                btn.setText(("ON   " if on else "OFF  ") + "SLEEP: training")
                btn.setStyleSheet(
                    "QPushButton { background:#4A3B19; color:#FFD36D; border:1px solid #FFD36D; "
                    "border-radius:10px; padding:8px 12px; font-weight:900; }"
                    "QPushButton:hover { background:#6A5222; color:#FFFFFF; border:1px solid #FFE39A; }"
                )
                return
            self._style_button(btn, on, "Online training")

        def _style_plain_status_button(self, btn, on: bool, label: str):
            btn.setText(label)
            if on:
                btn.setStyleSheet(
                    "QPushButton { background:#1F8F57; color:white; border:2px solid #C981FF; "
                    "border-radius:10px; padding:8px 12px; font-weight:900; }"
                    "QPushButton:hover { background:#27A765; border:2px solid #D2A8FF; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background:#191426; color:#D2A8FF; border:2px solid #8F63D6; "
                    "border-radius:10px; padding:8px 12px; font-weight:800; }"
                    "QPushButton:hover { background:#2B1E45; color:#FFFFFF; border:2px solid #C981FF; }"
                )

        def _set_connected(self, connected: bool):
            self.state.connected = bool(connected)
            if connected:
                self.status.setStyleSheet(
                    "color:#73E9A0; background:transparent; border:none; "
                    "font-weight:900; padding:2px 4px;"
                )
            else:
                self.status.setStyleSheet(
                    "color:#FF8FA3; background:transparent; border:none; "
                    "font-weight:900; padding:2px 4px;"
                )

        def _reset_to_default(self):
            self.state = LocalPanelState()

        def _process_alive(self, proc) -> bool:
            return proc is not None and proc.poll() is None

        def _script_process_ids(self, script_name: str) -> list[int]:
            needle = str(script_name or "").lower()
            if not needle:
                return []
            pids: list[int] = []

            try:
                import psutil  # type: ignore

                current_pid = os.getpid()
                for proc in psutil.process_iter(["pid", "cmdline"]):
                    try:
                        if int(proc.info.get("pid") or 0) == current_pid:
                            continue
                        cmdline = " ".join(str(x) for x in (proc.info.get("cmdline") or []))
                        if needle in cmdline.lower():
                            pids.append(int(proc.info.get("pid") or 0))
                    except Exception:
                        continue
                return [pid for pid in pids if pid > 0]
            except Exception:
                pass

            try:
                if os.name == "nt":
                    out = subprocess.check_output(
                        ["wmic", "process", "get", "ProcessId,CommandLine"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=1.5,
                    )
                else:
                    out = subprocess.check_output(
                        ["ps", "-eo", "pid,args"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=1.5,
                    )
                current_pid = str(os.getpid())
                for line in out.splitlines():
                    low = line.lower()
                    if needle not in low:
                        continue
                    parts = line.strip().split()
                    if parts and parts[0] == current_pid:
                        continue
                    if parts and parts[-1] == current_pid:
                        continue
                    if os.name == "nt":
                        pid_text = parts[-1] if parts else ""
                    else:
                        pid_text = parts[0] if parts else ""
                    try:
                        pid = int(pid_text)
                        if pid > 0:
                            pids.append(pid)
                    except Exception:
                        continue
                return pids
            except Exception:
                return []

        def _script_process_running(self, script_name: str) -> bool:
            return bool(self._script_process_ids(script_name))

        def _stop_process(self, proc) -> None:
            if proc is None or proc.poll() is not None:
                return
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
                return
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass

        def _stop_script_processes(self, script_name: str) -> int:
            pids = self._script_process_ids(script_name)
            stopped = 0
            for pid in pids:
                if pid == os.getpid():
                    continue
                try:
                    import psutil  # type: ignore

                    proc = psutil.Process(pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.5)
                    except Exception:
                        proc.kill()
                    stopped += 1
                    continue
                except Exception:
                    pass

                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=2.0,
                            check=False,
                        )
                    else:
                        os.kill(pid, 15)
                    stopped += 1
                except Exception:
                    pass
            return stopped

        def _refresh_pyqt_process_status(self, force: bool = False):
            now = time.time()
            if not force and (now - float(getattr(self, "_last_process_scan", 0.0))) < 2.0:
                return
            self._last_process_scan = now
            self.module_debug_external_alive = self._script_process_running(args.module_debug_script)
            self.agent_actions_external_alive = self._script_process_running(args.agent_actions_script)

        def _pyqt_window_alive(self, proc, external_alive: bool) -> bool:
            return bool(self._process_alive(proc) or external_alive)

        def refresh_ui(self):
            s = self.state
            self._refresh_pyqt_process_status()
            self._set_connected(s.connected)
            self._style_button(self.btn_mujoco, s.mujoco_next_run, "MuJoCo viewer")
            self._style_button(self.btn_inner, s.inner_world, "Inner world / thoughts")
            self._style_button(self.btn_cameras, s.cameras, "Input sensors")
            for cb, checked in [
                (self.chk_sensor_video, s.video_sensor_enabled),
                (self.chk_sensor_contact, s.contact_sensor_enabled),
                (self.chk_sensor_imu, s.imu_sensor_enabled),
            ]:
                blocker = QtCore.QSignalBlocker(cb)
                cb.setChecked(bool(checked))
                del blocker
            self._style_button(self.btn_actions, s.actions, "Action outputs")
            
            self._style_button(self.btn_object_image, s.object_image, "Inner Object Visualizer")
            self._style_button(self.btn_event_code, s.event_code_visualizer, "Event Code Visualizer")
            self._style_button(self.btn_static_dynamic, s.static_dynamic_code, "Static/Dynamic Code")
            self._style_button(self.btn_object_open3d, s.object_image_open3d, "Inner Object Open3D")
            self._style_plain_status_button(
                self.btn_object_open3d_step4,
                self._process_alive(self.open3d_slot_viewer_step4_proc),
                "Open3D RPC ext",
            )
            self._style_plain_status_button(
                self.btn_object_open3d_rpc,
                self._process_alive(self.open3d_slot_viewer_rpc_proc),
                "Open3D RPC",
            )
            self._style_plain_status_button(
                self.btn_object_open3d_file,
                self._process_alive(self.open3d_slot_viewer_proc),
                "Open3D file",
            )
            self._style_train_button(self.btn_train, s.training)
            self._style_button(self.btn_module_debug, s.module_debug, "Module debug")
            self._style_plain_status_button(
                self.btn_module_debug_pyqt,
                self._pyqt_window_alive(self.module_debug_proc, self.module_debug_external_alive),
                "PyQt Module Debug",
            )
            self._style_plain_status_button(
                self.btn_agent_actions_pyqt,
                self._pyqt_window_alive(self.agent_actions_proc, self.agent_actions_external_alive),
                "PyQt Agent Actions Imit",
            )
            self._style_button(self.btn_latent, s.latent_semantic, "Latent meaning map")
            if not s.connected:
                self.status.setText("STATUS IPC: no signal")
            self._set_runner_controls_enabled(s.connected)

        def toggle(self, field: str):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            current = bool(getattr(self.state, field))
            new_value = not current
            setattr(self.state, field, new_value)
            self.send(make_set_state_message(**{field: new_value}))

        def set_sensor_gate(self, key: str, checked: bool):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            if key == "video":
                self.state.video_sensor_enabled = bool(checked)
            elif key == "contact":
                self.state.contact_sensor_enabled = bool(checked)
            elif key == "imu":
                self.state.imu_sensor_enabled = bool(checked)
            sensors = {
                "video": bool(self.state.video_sensor_enabled),
                "contact": bool(self.state.contact_sensor_enabled),
                "imu": bool(self.state.imu_sensor_enabled),
            }
            mask = {k: not v for k, v in sensors.items()}
            self.send(make_set_state_message(
                input_sensors_enabled=sensors,
                sleep_sensor_mask=mask,
                video_sensor_enabled=sensors["video"],
                contact_sensor_enabled=sensors["contact"],
                imu_sensor_enabled=sensors["imu"],
                sleep_video_cut=mask["video"],
                sleep_contact_cut=mask["contact"],
                sleep_imu_cut=mask["imu"],
            ))

        def open_pyqt_module_debug(self):
            self._refresh_pyqt_process_status(force=True)
            if self._pyqt_window_alive(self.module_debug_proc, self.module_debug_external_alive):
                self._stop_process(self.module_debug_proc)
                stopped = self._stop_script_processes(args.module_debug_script)
                self.module_debug_proc = None
                self.module_debug_external_alive = False
                self._refresh_pyqt_process_status(force=True)
                self.status.setText(f"PyQt Module Debug stopped ({stopped})")
                self.refresh_ui()
                return

            root = self.resolve_root_path()
            script = self.resolve_panel_script(root, args.module_debug_script)

            if not script.exists():
                self.status.setText(f"module debug script not found: {script}")
                print(f"[control] module debug script not found: {script}")
                return

            cmd = [
                sys.executable,
                str(script),
                "--host", str(args.host),
                "--port", str(args.port),
                "--status-host", str(args.host),
                "--status-port", str(args.module_status_port),
            ]
            try:
                self.module_debug_proc = subprocess.Popen(
                    cmd,
                    cwd=str(root),
                    env=self.child_process_env(root),
                )
                self.status.setText("PyQt Module Debug launched")
                self.refresh_ui()
                print("[control] launched:", " ".join(cmd))
            except Exception as e:
                self.status.setText(f"failed to launch module debug: {e}")
                print(f"[control] failed to launch module debug: {e}")


        def open_pyqt_agent_actions(self):
            self._refresh_pyqt_process_status(force=True)
            if self._pyqt_window_alive(self.agent_actions_proc, self.agent_actions_external_alive):
                self._stop_process(self.agent_actions_proc)
                stopped = self._stop_script_processes(args.agent_actions_script)
                self.agent_actions_proc = None
                self.agent_actions_external_alive = False
                self._refresh_pyqt_process_status(force=True)
                self.status.setText(f"PyQt Agent Actions stopped ({stopped})")
                self.refresh_ui()
                return

            root = self.resolve_root_path()
            script = self.resolve_panel_script(root, args.agent_actions_script)
            
            if not script.exists():
                self.status.setText(f"agent actions script not found: {script}")
                print(f"[control] agent actions script not found: {script}")
                return

            cmd = [
                sys.executable,
                str(script),
                "--host", str(args.host),
                "--port", str(args.port),
                "--status-host", str(args.host),
                "--status-port", str(args.module_status_port),
            ]
            try:
                self.agent_actions_proc = subprocess.Popen(
                    cmd,
                    cwd=str(root),
                    env=self.child_process_env(root),
                )
                self.status.setText("PyQt Agent Actions launched")
                self.refresh_ui()
                print("[control] launched:", " ".join(cmd))
            except Exception as e:
                self.status.setText(f"failed to launch agent actions: {e}")
                print(f"[control] failed to launch agent actions: {e}")

        def launch_inner_object_open3d_step4(self):
            proc = getattr(self, "open3d_slot_viewer_step4_proc", None)
            if proc is not None and proc.poll() is None:
                self._stop_process(proc)
                self.open3d_slot_viewer_step4_proc = None
                self.status.setText("Inner Object Open3D Step4 stopped")
                self.refresh_ui()
                return

            root = self.resolve_root_path()
            viewer = root / "src" / "modules" / "m01_object_imagery" / "open3d_slot_viewer_rpc_step4.py"
            if not viewer.exists():
                self.status.setText(f"Open3D Step4 viewer missing: {viewer}")
                return

            cmd = [
                sys.executable,
                str(viewer),
                "--host",
                "127.0.0.1",
                "--port",
                "8771",
                "--slot",
                "both",
                "--mode",
                "deformed",
            ]
            try:
                self.open3d_slot_viewer_step4_proc = subprocess.Popen(
                    cmd,
                    cwd=str(root),
                    env=self.child_process_env(root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.status.setText(f"launched Inner Object Open3D Step4 | pid={self.open3d_slot_viewer_step4_proc.pid}")
                self.refresh_ui()
            except Exception as e:
                self.status.setText(f"Open3D Step4 launch failed: {e}")

        def launch_inner_object_open3d_rpc(self):
            proc = getattr(self, "open3d_slot_viewer_rpc_proc", None)
            if proc is not None and proc.poll() is None:
                self._stop_process(proc)
                self.open3d_slot_viewer_rpc_proc = None
                self.status.setText("Inner Object Open3D RPC stopped")
                self.refresh_ui()
                return

            root = self.resolve_root_path()
            viewer = root / "src" / "modules" / "m01_object_imagery" / "open3d_slot_viewer_rpc.py"
            if not viewer.exists():
                self.status.setText(f"Open3D RPC viewer missing: {viewer}")
                return

            cmd = [
                sys.executable,
                str(viewer),
                "--host",
                "127.0.0.1",
                "--port",
                "8771",
                "--slot",
                "both",
                "--mode",
                "deformed",
            ]
            try:
                self.open3d_slot_viewer_rpc_proc = subprocess.Popen(
                    cmd,
                    cwd=str(root),
                    env=self.child_process_env(root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.status.setText(f"launched Inner Object Open3D RPC | pid={self.open3d_slot_viewer_rpc_proc.pid}")
                self.refresh_ui()
            except Exception as e:
                self.status.setText(f"Open3D RPC launch failed: {e}")

        def launch_inner_object_open3d_file(self):
            proc = getattr(self, "open3d_slot_viewer_proc", None)
            if proc is not None and proc.poll() is None:
                self._stop_process(proc)
                self.open3d_slot_viewer_proc = None
                self.status.setText("Inner Object Open3D file viewer stopped")
                self.refresh_ui()
                return

            root = self.resolve_root_path()
            viewer = root / "src" / "modules" / "m01_object_imagery" / "open3d_slot_viewer.py"
            export_path = Path("./checkpoint/slot_viewer/slot_4d_open3d_latest.npz")
            if not viewer.exists():
                self.status.setText(f"Open3D file viewer missing: {viewer}")
                return

            cmd = [
                sys.executable,
                str(viewer),
                "--path",
                str(export_path),
                "--slot",
                "both",
                "--mode",
                "deformed",
            ]
            try:
                self.open3d_slot_viewer_proc = subprocess.Popen(
                    cmd,
                    cwd=str(root),
                    env=self.child_process_env(root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.status.setText(
                    "launched Inner Object Open3D file viewer | "
                    f"pid={self.open3d_slot_viewer_proc.pid} | path={export_path}"
                )
                self.refresh_ui()
            except Exception as e:
                self.status.setText(f"Open3D file viewer launch failed: {e}")

        def resolve_root_path(self) -> Path:
            if args.root_path:
                return Path(args.root_path).expanduser().resolve()
            env_root = os.environ.get("PROJECT_ROOT")
            if env_root:
                return Path(env_root).expanduser().resolve()

            for parent in Path(__file__).resolve().parents:
                if (parent / "config" / "runner.yaml").exists():
                    return parent

            return Path(__file__).resolve().parents[3]

        def resolve_panel_script(self, root: Path, script_name: str) -> Path:
            script_path = Path(script_name).expanduser()
            if script_path.is_absolute():
                return script_path

            candidates = [
                root / script_path,
                root / "src" / "modules" / "m08_debug_visual_control" / script_name,
                root / "ctrl_panel" / script_name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        def child_process_env(self, root: Path) -> dict[str, str]:
            env = os.environ.copy()
            env["PROJECT_ROOT"] = str(root)
            existing = env.get("PYTHONPATH", "")
            paths = [str(root)]
            if existing:
                paths.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(paths)
            return env

        def _viewer_base_cmd(self, root: Path, script: Path) -> list[str]:
            return [
                sys.executable,
                str(script),
                "--config-path",
                str(root / "config"),
                "--config-name",
                args.viewer_config,
            ]

        def start_viewer(self):
            if self.state.connected:
                self.status.setText("runner already running")
                self.refresh_ui()
                return

            root = self.resolve_root_path()
            script = root / args.viewer_script

            if not script.exists():
                self._reset_to_default()
                self.status.setText(f"script not found: {script}")
                self.refresh_ui()
                print(f"[control] viewer script not found: {script}")
                return

            # Important:
            # Configs live in root/config; root itself is resolved from this file
            # unless --root-path is explicitly provided.
            viewer_cmd = self._viewer_base_cmd(root, script)
            if args.terminal == "xterm" and os.name != "nt":
                # Use TrueType font. Fixes:
                # xterm: cannot load font "-misc-fixed-..."
                cmd = [
                    "xterm",
                    "-fa",
                    "Monospace",
                    "-fs",
                    "10",
                    "-T",
                    "ConsciousViewer",
                    "-e",
                    *viewer_cmd,
                ]
            elif os.name == "nt" and args.terminal in ("xterm", "cmd", "powershell"):
                cmd = viewer_cmd
            else:
                cmd = [
                    args.terminal,
                    "-T",
                    "ConsciousViewer",
                    "-e",
                    *viewer_cmd,
                ]

            try:
                subprocess.Popen(cmd, cwd=str(root))
                self._reset_to_default()
                self.status.setText("viewer started; waiting for status IPC")
                self.refresh_ui()
                print("[control] started:", " ".join(cmd))
            except Exception as e:
                print(f"[control] terminal launch failed: {e}")
                # Fallback: launch directly without terminal window.
                direct_cmd = viewer_cmd
                try:
                    subprocess.Popen(direct_cmd, cwd=str(root))
                    self._reset_to_default()
                    self.status.setText("viewer started without terminal; waiting for status IPC")
                    self.refresh_ui()
                    print("[control] started without terminal:", " ".join(direct_cmd))
                except Exception as e2:
                    self._reset_to_default()
                    self.status.setText(f"viewer start failed: {e2}")
                    print(f"[control] viewer start failed: {e2}")

        def action(self, action: str):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            self.send(make_action_message(action))

        def closeEvent(self, event):
            try:
                if hasattr(self, "status_timer") and self.status_timer.isActive():
                    self.status_timer.stop()
            except Exception:
                pass
            super().closeEvent(event)

        def keyPressEvent(self, event):
            key = event.key()
            if key == QtCore.Qt.Key_I:
                self.toggle("inner_world")
            elif key == QtCore.Qt.Key_C:
                self.toggle("cameras")
            elif key == QtCore.Qt.Key_A:
                self.toggle("actions")
            elif key == QtCore.Qt.Key_M:
                self.toggle("manual_actions")
            elif key == QtCore.Qt.Key_O:
                self.toggle("object_image")
            elif key == QtCore.Qt.Key_V:
                self.toggle("event_code_visualizer")
            elif key == QtCore.Qt.Key_Z:
                self.toggle("static_dynamic_code")
            elif key == QtCore.Qt.Key_P:
                self.toggle("object_image_open3d")
            elif key == QtCore.Qt.Key_T:
                self.toggle("training")
            elif key == QtCore.Qt.Key_L:
                self.toggle("latent_semantic")
            elif key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
                self.action("stop")
            else:
                super().keyPressEvent(event)

    app = QtWidgets.QApplication([])
    win = ControlWindow()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
