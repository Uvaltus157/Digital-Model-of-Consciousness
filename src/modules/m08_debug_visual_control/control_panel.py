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
import json
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
    "m2": ["btn_event_code", "btn_m2_scenario_imit"],
    "m3": [],
    "m6": [],
    "m7": ["btn_inner"],
    "m8": ["btn_module_debug", "btn_module_lab", "btn_sleep_replay_monitor", "btn_replay_quality_monitor", "btn_m5_learning_quality", "btn_m5_latent_prototype", "btn_m1_object_slot_imit"],
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
            self.module_lab_window = None
            self.module_lab_text = None
            self.m1_object_slot_imit_window = None
            self.m1_object_slot_imit_labels = {}
            self.m1_object_slot_imit_raw = None
            self.m2_scenario_imit_window = None
            self.m2_scenario_imit_labels = {}
            self.m2_scenario_imit_raw = None
            self.m5_latent_prototype_window = None
            self.m5_latent_prototype_labels = {}
            self.m5_latent_prototype_raw = None
            self.m5_learning_quality_window = None
            self.m5_learning_quality_labels = {}
            self.m5_learning_quality_raw = None
            self.replay_quality_monitor_window = None
            self.replay_quality_monitor_labels = {}
            self.replay_quality_monitor_raw = None
            self.sleep_replay_monitor_window = None
            self.sleep_replay_monitor_labels = {}
            self.sleep_replay_monitor_raw = None
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
            self.btn_module_lab = QtWidgets.QPushButton("Run Module Lab")
            self.btn_m1_object_slot_imit = QtWidgets.QPushButton("M1 Object Slot Imit")
            self.btn_m2_scenario_imit = QtWidgets.QPushButton("M2 Scenario Imit")
            self.btn_m5_latent_prototype = QtWidgets.QPushButton("M5 Latent Prototypes")
            self.btn_m5_learning_quality = QtWidgets.QPushButton("M5 Learning Quality")
            self.btn_replay_quality_monitor = QtWidgets.QPushButton("Replay Quality Monitor")
            self.btn_sleep_replay_monitor = QtWidgets.QPushButton("Sleep Replay Monitor")
            self.btn_module_debug_pyqt = QtWidgets.QPushButton("PyQt Module Debug")
            self.btn_agent_actions_pyqt = QtWidgets.QPushButton("PyQt Agent Actions Imit")
            self.btn_module_debug_pyqt.setObjectName("pyqtWindowButton")
            self.btn_agent_actions_pyqt.setObjectName("pyqtWindowButton")
            self.btn_latent = QtWidgets.QPushButton()
            self.chk_sensor_video = QtWidgets.QCheckBox("Video")
            self.chk_sensor_contact = QtWidgets.QCheckBox("Tactile")
            self.chk_sensor_imu = QtWidgets.QCheckBox("IMU")
            self.btn_sleep_replay = QtWidgets.QPushButton("Сон / replay mode")

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
            self.btn_module_lab.setMinimumHeight(42)
            self.btn_m1_object_slot_imit.setMinimumHeight(42)
            self.btn_m2_scenario_imit.setMinimumHeight(42)
            self.btn_m5_learning_quality.setMinimumHeight(42)
            self.btn_m5_latent_prototype.setMinimumHeight(42)
            self.btn_replay_quality_monitor.setMinimumHeight(42)
            self.btn_sleep_replay_monitor.setMinimumHeight(42)
            self.btn_agent_actions_pyqt.setMinimumHeight(42)
            self.btn_sleep_replay.setMinimumHeight(42)
            self.btn_object_open3d_step4.setMinimumHeight(42)
            self.btn_object_open3d_rpc.setMinimumHeight(42)
            self.btn_object_open3d_file.setMinimumHeight(42)
            self.btn_start_viewer.setToolTip("Start the main runner process with the configured runner.yaml")
            self.btn_cameras.setToolTip("Show or hide the input sensors preview window")
            self.chk_sensor_video.setToolTip("Enable or disable video / eyes input")
            self.chk_sensor_contact.setToolTip("Enable or disable tactile / contact input")
            self.chk_sensor_imu.setToolTip("Enable or disable IMU / vestibular body input")
            self.btn_sleep_replay.setToolTip("Toggle full sleep/replay mode by cutting or enabling all external sensors")
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
            self.btn_module_lab.setToolTip("Run module lab contracts/scenarios and show latest result")
            self.btn_sleep_replay_monitor.setToolTip("Open live sleep/replay monitor for M1/M11/M13/M4/M2/M5/M3")
            self.btn_module_lab.setToolTip("Run M8 module lab contracts/scenarios inside the runner via IPC")
            self.btn_m5_learning_quality.setToolTip("Show M5 learning baseline: loss trends, seed response, latent/object stability proxies")
            self.btn_m5_latent_prototype.setToolTip("Inject simulated learned cube/tetrahedron latents into M5 via FocusFeedbackBoundary")
            self.btn_m2_scenario_imit.setToolTip("Inject cube/tetra M2 scenario Gaussian states into Open3D RPC slots")
            self.btn_replay_quality_monitor.setToolTip("Show replay quality/integration metrics: selected episode, identity, pressure/relief deltas")
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
            sensor_row.addWidget(self.btn_sleep_replay, 2)
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
            self.btn_sleep_replay.clicked.connect(self.toggle_sleep_replay_mode)
            self.btn_module_debug.clicked.connect(lambda: self.toggle("module_debug"))
            self.btn_module_lab.clicked.connect(self.open_m8_module_lab_window)
            self.btn_m1_object_slot_imit.clicked.connect(self.open_m1_object_slot_imit_window)
            self.btn_m2_scenario_imit.clicked.connect(self.open_m2_scenario_imit_window)
            self.btn_sleep_replay_monitor.clicked.connect(self.open_sleep_replay_monitor_window)
            self.btn_replay_quality_monitor.clicked.connect(self.open_replay_quality_monitor_window)
            self.btn_m5_learning_quality.clicked.connect(self.open_m5_learning_quality_window)
            self.btn_m5_latent_prototype.clicked.connect(self.open_m5_latent_prototype_window)
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
                self.btn_sleep_replay,
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

        def _window_visible(self, window) -> bool:
            try:
                return bool(window is not None and window.isVisible())
            except Exception:
                return False

        def _clear_sleep_replay_monitor_window_refs(self):
            self.sleep_replay_monitor_window = None
            self.sleep_replay_monitor_labels = {}
            self.sleep_replay_monitor_raw = None
            self.refresh_ui()

        def _clear_replay_quality_monitor_window_refs(self):
            self.replay_quality_monitor_window = None
            self.replay_quality_monitor_labels = {}
            self.replay_quality_monitor_raw = None
            self.refresh_ui()

        def _clear_m5_learning_quality_window_refs(self):
            self.m5_learning_quality_window = None
            self.m5_learning_quality_labels = {}
            self.m5_learning_quality_raw = None
            self.refresh_ui()

        def _clear_m5_latent_prototype_window_refs(self):
            self.m5_latent_prototype_window = None
            self.m5_latent_prototype_labels = {}
            self.m5_latent_prototype_raw = None
            self.refresh_ui()

        def _clear_m1_object_slot_imit_window_refs(self):
            self.m1_object_slot_imit_window = None
            self.m1_object_slot_imit_labels = {}
            self.m1_object_slot_imit_raw = None
            self.refresh_ui()

        def _clear_m2_scenario_imit_window_refs(self):
            self.m2_scenario_imit_window = None
            self.m2_scenario_imit_labels = {}
            self.m2_scenario_imit_raw = None
            self.refresh_ui()

        def _clear_module_lab_window_refs(self):
            self.module_lab_window = None
            self.module_lab_text = None
            self.refresh_ui()

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
            self._style_button(self.btn_sleep_replay, self._sleep_mode_active(), "Сон / replay mode")
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
                self.btn_module_lab,
                self._window_visible(getattr(self, "module_lab_window", None)),
                "Module Lab",
            )
            self._style_plain_status_button(
                self.btn_m1_object_slot_imit,
                self._window_visible(getattr(self, "m1_object_slot_imit_window", None)),
                "M1 Object Slot Imit",
            )
            self._style_plain_status_button(
                self.btn_m2_scenario_imit,
                self._window_visible(getattr(self, "m2_scenario_imit_window", None)),
                "M2 Scenario Imit",
            )
            self._style_plain_status_button(
                self.btn_sleep_replay_monitor,
                self._window_visible(getattr(self, "sleep_replay_monitor_window", None)),
                "Sleep Replay Monitor",
            )
            self._style_plain_status_button(
                self.btn_replay_quality_monitor,
                self._window_visible(getattr(self, "replay_quality_monitor_window", None)),
                "Replay Quality Monitor",
            )
            self._style_plain_status_button(
                self.btn_m5_learning_quality,
                self._window_visible(getattr(self, "m5_learning_quality_window", None)),
                "M5 Learning Quality",
            )
            self._style_plain_status_button(
                self.btn_m5_latent_prototype,
                self._window_visible(getattr(self, "m5_latent_prototype_window", None)),
                "M5 Latent Prototypes",
            )
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
            self.refresh_module_lab_window()
            self.refresh_sleep_replay_monitor_window()
            self.refresh_replay_quality_monitor_window()
            self.refresh_m5_learning_quality_window()
            self.refresh_m5_latent_prototype_window()
            self.refresh_m1_object_slot_imit_window()
            self.refresh_m2_scenario_imit_window()

        def _sleep_replay_monitor_value(self, section: str, key: str, default=""):
            monitor = {}
            if isinstance(getattr(self, "last_status", None), dict):
                monitor = self.last_status.get("sleep_replay_monitor", {}) or {}
            if not isinstance(monitor, dict):
                return default
            sec = monitor.get(section, {})
            if isinstance(sec, dict):
                return sec.get(key, default)
            return default

        def _fmt_monitor_value(self, value):
            if isinstance(value, float):
                return f"{value:.3f}"
            if isinstance(value, bool):
                return "ON" if value else "OFF"
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                return ", ".join(str(x) for x in value)
            return str(value)

        def refresh_sleep_replay_monitor_window(self):
            labels = getattr(self, "sleep_replay_monitor_labels", {}) or {}
            if not labels:
                return
            try:
                if not self.sleep_replay_monitor_window or not self.sleep_replay_monitor_window.isVisible():
                    return
            except Exception:
                return

            monitor = {}
            if isinstance(getattr(self, "last_status", None), dict):
                monitor = self.last_status.get("sleep_replay_monitor", {}) or {}
            if not isinstance(monitor, dict):
                monitor = {}

            header = labels.get("__header__")
            if header is not None:
                step = monitor.get("global_step", self.last_status.get("global_step", 0) if isinstance(self.last_status, dict) else 0)
                state = monitor.get("sensor_state", self.last_status.get("sensor_state", "") if isinstance(self.last_status, dict) else "")
                sleep = monitor.get("full_sleep", self.last_status.get("full_sleep", False) if isinstance(self.last_status, dict) else False)
                header.setText(f"step={step} | state={state} | full_sleep={int(bool(sleep))}")

            for name, label in labels.items():
                if name.startswith("__"):
                    continue
                section, key = name.split(".", 1)
                value = self._sleep_replay_monitor_value(section, key, "")
                label.setText(self._fmt_monitor_value(value))

            raw = getattr(self, "sleep_replay_monitor_raw", None)
            if raw is not None:
                try:
                    raw.setPlainText(json.dumps(monitor, ensure_ascii=False, indent=2))
                except Exception:
                    raw.setPlainText(str(monitor))

        def request_sleep_replay_probe(self, kind: str, intensity: float = 0.8, duration: int = 80):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            payload_kind = str(kind)
            ok = self.send(make_action_message(
                "dream_probe_inject",
                kind=payload_kind,
                intensity=float(intensity),
                duration=int(duration),
                source="m8_sleep_replay_monitor",
            ))
            self.status.setText(
                f"Dream probe requested: {payload_kind}" if ok else "Dream probe request failed"
            )
            self.refresh_ui()

        def open_sleep_replay_monitor_window(self):
            try:
                if self.sleep_replay_monitor_window is not None and self.sleep_replay_monitor_window.isVisible():
                    window = self.sleep_replay_monitor_window
                    window.close()
                    self._clear_sleep_replay_monitor_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 Sleep Replay Monitor")
            dialog.resize(880, 720)
            dialog.setStyleSheet(
                "QDialog { background:#0C121B; color:#DCE8F8; }"
                "QLabel { color:#DCE8F8; background:transparent; }"
                "QGroupBox { background:#101722; border:1px solid #243145; border-radius:12px; "
                "margin-top:12px; padding:10px; color:#B7C5DA; font-weight:800; }"
                "QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 6px; }"
                "QPlainTextEdit { background:#07101A; color:#DCE8F8; border:1px solid #2B3A50; "
                "border-radius:10px; padding:8px; font-family:Consolas, monospace; font-size:11px; }"
                "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                "border-radius:10px; padding:8px 12px; font-weight:800; }"
            )

            main = QtWidgets.QVBoxLayout(dialog)
            main.setContentsMargins(14, 14, 14, 14)
            main.setSpacing(10)

            title = QtWidgets.QLabel("Sleep Replay Monitor: M1 → M5 → M11 → M2 → M5, M4/M13 → M2, M3 blocked")
            title.setWordWrap(True)
            title.setStyleSheet("font-weight:900; color:#D2A8FF;")
            main.addWidget(title)

            header = QtWidgets.QLabel("")
            header.setStyleSheet("font-weight:900; color:#FFD36D;")
            main.addWidget(header)

            probe_row = QtWidgets.QHBoxLayout()
            probe_row.setSpacing(8)
            btn_probe_curiosity = QtWidgets.QPushButton("Probe curiosity")
            btn_probe_stress = QtWidgets.QPushButton("Probe stress")
            btn_probe_replay = QtWidgets.QPushButton("Probe replay seed")
            btn_probe_mixed = QtWidgets.QPushButton("Probe mixed")
            btn_probe_clear = QtWidgets.QPushButton("Clear probe")
            for b in [btn_probe_curiosity, btn_probe_stress, btn_probe_replay, btn_probe_mixed, btn_probe_clear]:
                b.setMinimumHeight(34)
                probe_row.addWidget(b)
            main.addLayout(probe_row)


            labels = {"__header__": header}

            def add_box(title_text, rows):
                box = QtWidgets.QGroupBox(title_text)
                grid = QtWidgets.QGridLayout(box)
                grid.setHorizontalSpacing(12)
                grid.setVerticalSpacing(6)
                for r, (label_text, name) in enumerate(rows):
                    k = QtWidgets.QLabel(label_text)
                    k.setStyleSheet("color:#8FA4BF; font-weight:800;")
                    v = QtWidgets.QLabel("")
                    v.setStyleSheet("color:#FFFFFF; font-weight:900;")
                    v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    grid.addWidget(k, r, 0)
                    grid.addWidget(v, r, 1)
                    labels[name] = v
                return box

            row1 = QtWidgets.QHBoxLayout()
            row1.setSpacing(10)
            row1.addWidget(add_box("M1 sensors", [
                ("state", "m1.state"),
                ("video_on", "m1.video_on"),
                ("contact_on", "m1.contact_on"),
                ("imu_on", "m1.imu_on"),
            ]))
            row1.addWidget(add_box("M11 affect + Δ / trend", [
                ("valence", "m11.valence"),
                ("arousal", "m11.arousal"),
                ("stress", "m11.stress"),
                ("panic", "m11.panic"),
                ("curiosity", "m11.curiosity"),
                ("Δ stress", "m11_delta.stress"),
                ("Δ panic", "m11_delta.panic"),
                ("Δ curiosity", "m11_delta.curiosity"),
                ("trend", "m11_activity.trend"),
                ("change_score", "m11_activity.change_score"),
            ]))
            main.addLayout(row1)

            row2 = QtWidgets.QHBoxLayout()
            row2.setSpacing(10)
            row2.addWidget(add_box("M13 autobiographical retrieval", [
                ("relevance", "m13.relevance"),
                ("episodes", "m13.episodes"),
                ("summary", "m13.summary"),
            ]))
            row2.addWidget(add_box("M4 dynamic identity", [
                ("token", "m4.token"),
                ("gate", "m4.gate"),
                ("stability", "m4.stability"),
                ("novelty", "m4.novelty"),
                ("sentence", "m4.sentence"),
            ]))
            main.addLayout(row2)

            row3 = QtWidgets.QHBoxLayout()
            row3.setSpacing(10)
            row3.addWidget(add_box("M2 event/dream replay", [
                ("replay_gate", "m2.replay_gate"),
                ("should_replay", "m2.should_replay"),
                ("dream_pressure", "m2.dream_pressure"),
                ("event_salience", "m2.event_salience"),
                ("source", "m2.source"),
                ("identity", "m2.identity"),
            ]))
            row3.addWidget(add_box("Dream probe", [
                ("active", "dream_probe.active"),
                ("kind", "dream_probe.kind"),
                ("remaining", "dream_probe.remaining"),
                ("pulse", "dream_probe.pulse"),
                ("intensity", "dream_probe.intensity"),
            ]))
            row3.addWidget(add_box("M5 seed + M3 guard", [
                ("seed_gate", "m5.seed_gate"),
                ("seed_norm", "m5.seed_norm"),
                ("feedback_gate", "m5.feedback_gate"),
                ("m3 sleep_blocked", "m3.sleep_blocked"),
                ("blocked_norm", "m3.blocked_norm"),
                ("blocked_keys", "m3.blocked_keys"),
            ]))
            main.addLayout(row3)

            raw = QtWidgets.QPlainTextEdit()
            raw.setReadOnly(True)
            raw.setMinimumHeight(160)
            main.addWidget(raw)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            close_row.addWidget(btn_close)
            main.addLayout(close_row)

            self.sleep_replay_monitor_window = dialog
            self.sleep_replay_monitor_labels = labels
            self.sleep_replay_monitor_raw = raw
            btn_probe_curiosity.clicked.connect(lambda: self.request_sleep_replay_probe("curiosity", 0.85, 80))
            btn_probe_stress.clicked.connect(lambda: self.request_sleep_replay_probe("stress", 0.85, 80))
            btn_probe_replay.clicked.connect(lambda: self.request_sleep_replay_probe("replay_seed", 0.75, 60))
            btn_probe_mixed.clicked.connect(lambda: self.request_sleep_replay_probe("mixed", 0.75, 80))
            btn_probe_clear.clicked.connect(lambda: self.request_sleep_replay_probe("clear", 0.0, 1))
            btn_close.clicked.connect(dialog.close)
            dialog.finished.connect(lambda _code: self._clear_sleep_replay_monitor_window_refs())

            self.refresh_sleep_replay_monitor_window()
            dialog.show()
            self.refresh_ui()

        def _replay_quality_value(self, section: str, key: str, default=""):
            monitor = {}
            if isinstance(getattr(self, "last_status", None), dict):
                monitor = self.last_status.get("replay_quality_monitor", {}) or {}
            if not isinstance(monitor, dict):
                return default
            if section == "_root":
                return monitor.get(key, default)
            sec = monitor.get(section, {})
            if isinstance(sec, dict):
                return sec.get(key, default)
            return default

        def refresh_replay_quality_monitor_window(self):
            labels = getattr(self, "replay_quality_monitor_labels", {}) or {}
            if not labels:
                return
            try:
                if not self.replay_quality_monitor_window or not self.replay_quality_monitor_window.isVisible():
                    return
            except Exception:
                return

            monitor = {}
            if isinstance(getattr(self, "last_status", None), dict):
                monitor = self.last_status.get("replay_quality_monitor", {}) or {}
            if not isinstance(monitor, dict):
                monitor = {}

            header = labels.get("__header__")
            if header is not None:
                step = monitor.get("global_step", self.last_status.get("global_step", 0) if isinstance(self.last_status, dict) else 0)
                verdict = monitor.get("verdict", "")
                q = monitor.get("quality_score", 0.0)
                ema = monitor.get("quality_ema", 0.0)
                header.setText(f"step={step} | verdict={verdict} | quality={self._fmt_monitor_value(q)} | ema={self._fmt_monitor_value(ema)}")

            for name, label in labels.items():
                if name.startswith("__"):
                    continue
                section, key = name.split(".", 1)
                value = self._replay_quality_value(section, key, "")
                label.setText(self._fmt_monitor_value(value))

            raw = getattr(self, "replay_quality_monitor_raw", None)
            if raw is not None:
                try:
                    raw.setPlainText(json.dumps(monitor, ensure_ascii=False, indent=2))
                except Exception:
                    raw.setPlainText(str(monitor))

        def open_replay_quality_monitor_window(self):
            try:
                if self.replay_quality_monitor_window is not None and self.replay_quality_monitor_window.isVisible():
                    window = self.replay_quality_monitor_window
                    window.close()
                    self._clear_replay_quality_monitor_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 Replay Quality Monitor")
            dialog.resize(920, 760)
            dialog.setStyleSheet(
                "QDialog { background:#0C121B; color:#DCE8F8; }"
                "QLabel { color:#DCE8F8; background:transparent; }"
                "QGroupBox { background:#101722; border:1px solid #243145; border-radius:12px; "
                "margin-top:12px; padding:10px; color:#B7C5DA; font-weight:800; }"
                "QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 6px; }"
                "QPlainTextEdit { background:#07101A; color:#DCE8F8; border:1px solid #2B3A50; "
                "border-radius:10px; padding:8px; font-family:Consolas, monospace; font-size:11px; }"
                "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                "border-radius:10px; padding:8px 12px; font-weight:800; }"
            )

            main = QtWidgets.QVBoxLayout(dialog)
            main.setContentsMargins(14, 14, 14, 14)
            main.setSpacing(10)

            title = QtWidgets.QLabel("Replay Quality Monitor: selected episode/identity + pressure/relief/coherence changes")
            title.setWordWrap(True)
            title.setStyleSheet("font-weight:900; color:#D2A8FF;")
            main.addWidget(title)

            header = QtWidgets.QLabel("")
            header.setStyleSheet("font-weight:900; color:#FFD36D;")
            main.addWidget(header)

            labels = {"__header__": header}

            def add_box(title_text, rows):
                box = QtWidgets.QGroupBox(title_text)
                grid = QtWidgets.QGridLayout(box)
                grid.setHorizontalSpacing(12)
                grid.setVerticalSpacing(6)
                for r, (label_text, name) in enumerate(rows):
                    k = QtWidgets.QLabel(label_text)
                    k.setStyleSheet("color:#8FA4BF; font-weight:800;")
                    v = QtWidgets.QLabel("")
                    v.setStyleSheet("color:#FFFFFF; font-weight:900;")
                    v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    grid.addWidget(k, r, 0)
                    grid.addWidget(v, r, 1)
                    labels[name] = v
                return box

            row1 = QtWidgets.QHBoxLayout()
            row1.setSpacing(10)
            row1.addWidget(add_box("Replay selection", [
                ("verdict", "_root.verdict"),
                ("quality_score", "_root.quality_score"),
                ("quality_ema", "_root.quality_ema"),
                ("source", "_root.replay_source"),
                ("identity", "_root.selected_identity_token"),
                ("episode", "_root.selected_episode_summary"),
            ]))
            row1.addWidget(add_box("M2 replay dynamics", [
                ("replay_gate", "m2.replay_gate"),
                ("should_replay", "m2.should_replay"),
                ("event_salience", "m2.event_salience"),
                ("dream_pressure", "m2.dream_pressure"),
                ("Δ pressure", "m2.dream_pressure_delta"),
                ("pressure trend", "m2.dream_pressure_trend"),
            ]))
            main.addLayout(row1)

            row2 = QtWidgets.QHBoxLayout()
            row2.setSpacing(10)
            row2.addWidget(add_box("Affect integration", [
                ("stress", "affect.stress"),
                ("Δ stress", "affect.stress_delta"),
                ("relief", "affect.relief"),
                ("Δ relief", "affect.relief_delta"),
                ("coherence", "affect.coherence"),
                ("Δ coherence", "affect.coherence_delta"),
                ("expected_affect_delta", "affect.expected_affect_delta"),
            ]))
            row2.addWidget(add_box("Memory / identity support", [
                ("M13 relevance", "m13.relevance"),
                ("M13 episodes", "m13.episodes"),
                ("M4 gate", "m4.dynamic_memory_gate"),
                ("M4 stability", "m4.identity_stability"),
                ("M4 novelty", "m4.identity_novelty"),
                ("M5 seed_gate", "m5.seed_gate"),
                ("M5 seed_norm", "m5.seed_norm"),
            ]))
            main.addLayout(row2)

            raw = QtWidgets.QPlainTextEdit()
            raw.setReadOnly(True)
            raw.setMinimumHeight(240)
            main.addWidget(raw)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            close_row.addWidget(btn_close)
            main.addLayout(close_row)

            self.replay_quality_monitor_window = dialog
            self.replay_quality_monitor_labels = labels
            self.replay_quality_monitor_raw = raw
            btn_close.clicked.connect(dialog.close)
            dialog.finished.connect(lambda _code: self._clear_replay_quality_monitor_window_refs())

            self.refresh_replay_quality_monitor_window()
            dialog.show()
            self.refresh_ui()

        def _m5_learning_quality_value(self, section: str, key: str, default=""):
            monitor = {}
            if isinstance(getattr(self, "last_status", None), dict):
                monitor = self.last_status.get("m5_learning_quality", {}) or {}
            if not isinstance(monitor, dict):
                return default
            if section == "_root":
                return monitor.get(key, default)
            sec = monitor.get(section, {})
            if isinstance(sec, dict):
                return sec.get(key, default)
            return default

        def refresh_m5_learning_quality_window(self):
            labels = getattr(self, "m5_learning_quality_labels", {}) or {}
            if not labels:
                return
            try:
                if not self.m5_learning_quality_window or not self.m5_learning_quality_window.isVisible():
                    return
            except Exception:
                return

            monitor = {}
            if isinstance(getattr(self, "last_status", None), dict):
                monitor = self.last_status.get("m5_learning_quality", {}) or {}
            if not isinstance(monitor, dict):
                monitor = {}

            header = labels.get("__header__")
            if header is not None:
                step = monitor.get("global_step", self.last_status.get("global_step", 0) if isinstance(self.last_status, dict) else 0)
                train_steps = monitor.get("train_steps", 0)
                verdict = monitor.get("verdict", "")
                q = monitor.get("learning_quality", 0.0)
                ema = monitor.get("learning_quality_ema", 0.0)
                header.setText(f"step={step} | train_steps={train_steps} | verdict={verdict} | q={self._fmt_monitor_value(q)} | ema={self._fmt_monitor_value(ema)}")

            for name, label in labels.items():
                if name.startswith("__"):
                    continue
                section, key = name.split(".", 1)
                value = self._m5_learning_quality_value(section, key, "")
                label.setText(self._fmt_monitor_value(value))

            raw = getattr(self, "m5_learning_quality_raw", None)
            if raw is not None:
                try:
                    raw.setPlainText(json.dumps(monitor, ensure_ascii=False, indent=2))
                except Exception:
                    raw.setPlainText(str(monitor))

        def open_m5_learning_quality_window(self):
            try:
                if self.m5_learning_quality_window is not None and self.m5_learning_quality_window.isVisible():
                    window = self.m5_learning_quality_window
                    window.close()
                    self._clear_m5_learning_quality_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 M5 Learning Quality Baseline")
            dialog.resize(940, 760)
            dialog.setStyleSheet(
                "QDialog { background:#0C121B; color:#DCE8F8; }"
                "QLabel { color:#DCE8F8; background:transparent; }"
                "QGroupBox { background:#101722; border:1px solid #243145; border-radius:12px; "
                "margin-top:12px; padding:10px; color:#B7C5DA; font-weight:800; }"
                "QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 6px; }"
                "QPlainTextEdit { background:#07101A; color:#DCE8F8; border:1px solid #2B3A50; "
                "border-radius:10px; padding:8px; font-family:Consolas, monospace; font-size:11px; }"
                "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                "border-radius:10px; padding:8px 12px; font-weight:800; }"
            )

            main = QtWidgets.QVBoxLayout(dialog)
            main.setContentsMargins(14, 14, 14, 14)
            main.setSpacing(10)

            title = QtWidgets.QLabel("M5 Learning Quality Baseline: loss trends, seed response, latent/object stability proxies")
            title.setWordWrap(True)
            title.setStyleSheet("font-weight:900; color:#D2A8FF;")
            main.addWidget(title)

            header = QtWidgets.QLabel("")
            header.setStyleSheet("font-weight:900; color:#FFD36D;")
            main.addWidget(header)

            labels = {"__header__": header}

            def add_box(title_text, rows):
                box = QtWidgets.QGroupBox(title_text)
                grid = QtWidgets.QGridLayout(box)
                grid.setHorizontalSpacing(12)
                grid.setVerticalSpacing(6)
                for r, (label_text, name) in enumerate(rows):
                    k = QtWidgets.QLabel(label_text)
                    k.setStyleSheet("color:#8FA4BF; font-weight:800;")
                    v = QtWidgets.QLabel("")
                    v.setStyleSheet("color:#FFFFFF; font-weight:900;")
                    v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    grid.addWidget(k, r, 0)
                    grid.addWidget(v, r, 1)
                    labels[name] = v
                return box

            row1 = QtWidgets.QHBoxLayout()
            row1.setSpacing(10)
            row1.addWidget(add_box("Training / baseline", [
                ("verdict", "_root.verdict"),
                ("learning_quality", "_root.learning_quality"),
                ("learning_quality_ema", "_root.learning_quality_ema"),
                ("training_enabled", "_root.training_enabled"),
                ("cfg_train_enabled", "_root.cfg_train_enabled"),
                ("last_train_reason", "_root.last_train_reason"),
                ("last_train_error", "_root.last_train_error"),
            ]))
            row1.addWidget(add_box("M5 losses", [
                ("train_loss", "m5_loss.train_loss"),
                ("Δ train_loss", "m5_loss.train_loss_delta"),
                ("train trend", "m5_loss.train_loss_trend"),
                ("prediction_error", "m5_loss.prediction_error"),
                ("Δ prediction", "m5_loss.prediction_error_delta"),
                ("reconstruction_error", "m5_loss.reconstruction_error"),
                ("Δ reconstruction", "m5_loss.reconstruction_error_delta"),
            ]))
            main.addLayout(row1)

            row2 = QtWidgets.QHBoxLayout()
            row2.setSpacing(10)
            row2.addWidget(add_box("M5 latent / seed response", [
                ("latent_coherence", "m5_latent.latent_coherence"),
                ("Δ coherence", "m5_latent.latent_coherence_delta"),
                ("focus_norm", "m5_latent.focus_norm"),
                ("workspace_norm", "m5_latent.workspace_norm"),
                ("obs_embed_norm", "m5_latent.obs_embed_norm"),
                ("seed_gate", "m5_seed_response.seed_gate"),
                ("seed_norm", "m5_seed_response.seed_norm"),
                ("feedback_gate", "m5_seed_response.feedback_gate"),
                ("seed_response", "m5_seed_response.seed_response"),
            ]))
            row2.addWidget(add_box("Object / identity proxy", [
                ("object_recon", "object_identity_proxy.object_recon"),
                ("Δ object_recon", "object_identity_proxy.object_recon_delta"),
                ("identity_stability", "object_identity_proxy.identity_stability"),
                ("Δ stability", "object_identity_proxy.identity_stability_delta"),
                ("identity_novelty", "object_identity_proxy.identity_novelty"),
                ("Δ novelty", "object_identity_proxy.identity_novelty_delta"),
            ]))
            main.addLayout(row2)

            raw = QtWidgets.QPlainTextEdit()
            raw.setReadOnly(True)
            raw.setMinimumHeight(240)
            main.addWidget(raw)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            close_row.addWidget(btn_close)
            main.addLayout(close_row)

            self.m5_learning_quality_window = dialog
            self.m5_learning_quality_labels = labels
            self.m5_learning_quality_raw = raw
            btn_close.clicked.connect(dialog.close)
            dialog.finished.connect(lambda _code: self._clear_m5_learning_quality_window_refs())

            self.refresh_m5_learning_quality_window()
            dialog.show()
            self.refresh_ui()

        def request_m1_object_slot_imit(
            self,
            kind: str,
            *,
            slot: int | None = None,
            cube_slot: int = 1,
            tetra_slot: int = 2,
            selected_slot: int | None = None,
            duration: int = 220,
            alpha: float = 0.5,
        ):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            payload = {
                "kind": str(kind),
                "duration": int(duration),
                "alpha": float(alpha),
                "auto_select_slot": True,
                "source": "m8_m1_object_slot_imit_window",
            }
            if slot is not None:
                payload["slot"] = int(slot)
            if selected_slot is not None:
                payload["selected_slot"] = int(selected_slot)
            if str(kind) == "cube_tetra":
                payload.update({
                    "cube_slot": int(cube_slot),
                    "tetra_slot": int(tetra_slot),
                    "selected_slot": int(selected_slot if selected_slot is not None else tetra_slot),
                })
            ok = self.send(make_action_message("m1_object_slot_imit_inject", **payload))
            self.status.setText(
                f"M1 object slot imit requested: {kind}" if ok else "M1 object slot imit request failed"
            )
            self.refresh_ui()

        def _m1_object_slot_imit_value(self, key: str, default=""):
            data = {}
            if isinstance(getattr(self, "last_status", None), dict):
                data = self.last_status.get("m1_object_slot_imit", {}) or {}
            if not isinstance(data, dict):
                return default
            return data.get(key, default)

        def refresh_m1_object_slot_imit_window(self):
            labels = getattr(self, "m1_object_slot_imit_labels", {}) or {}
            if not labels:
                return
            try:
                if not self.m1_object_slot_imit_window or not self.m1_object_slot_imit_window.isVisible():
                    return
            except Exception:
                return

            data = {}
            if isinstance(getattr(self, "last_status", None), dict):
                data = self.last_status.get("m1_object_slot_imit", {}) or {}
            if not isinstance(data, dict):
                data = {}

            header = labels.get("__header__")
            if header is not None:
                header.setText(
                    f"step={data.get('global_step', 0)} | active={int(bool(data.get('active', False)))} | "
                    f"kind={data.get('kind', '')} | selected_slot={data.get('selected_slot', 0)}"
                )

            for key, label in labels.items():
                if key.startswith("__"):
                    continue
                label.setText(self._fmt_monitor_value(data.get(key, "")))

            raw = getattr(self, "m1_object_slot_imit_raw", None)
            if raw is not None:
                try:
                    raw.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    raw.setPlainText(str(data))

        def open_m1_object_slot_imit_window(self):
            try:
                if self.m1_object_slot_imit_window is not None and self.m1_object_slot_imit_window.isVisible():
                    window = self.m1_object_slot_imit_window
                    window.close()
                    self._clear_m1_object_slot_imit_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 M1 Object Slot Latent Imitator")
            dialog.resize(820, 620)
            dialog.setStyleSheet(
                "QDialog { background:#0C121B; color:#DCE8F8; }"
                "QLabel { color:#DCE8F8; background:transparent; }"
                "QGroupBox { background:#101722; border:1px solid #243145; border-radius:12px; "
                "margin-top:12px; padding:10px; color:#B7C5DA; font-weight:800; }"
                "QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 6px; }"
                "QPlainTextEdit { background:#07101A; color:#DCE8F8; border:1px solid #2B3A50; "
                "border-radius:10px; padding:8px; font-family:Consolas, monospace; font-size:11px; }"
                "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                "border-radius:10px; padding:8px 12px; font-weight:800; }"
            )

            main = QtWidgets.QVBoxLayout(dialog)
            main.setContentsMargins(14, 14, 14, 14)
            main.setSpacing(10)

            title = QtWidgets.QLabel(
                "Simulated M1 object-slot latents: fill target ObjectSlotMemory slots and select Inner Object 3D slot."
            )
            title.setWordWrap(True)
            title.setStyleSheet("font-weight:900; color:#D2A8FF;")
            main.addWidget(title)

            header = QtWidgets.QLabel("")
            header.setStyleSheet("font-weight:900; color:#FFD36D;")
            main.addWidget(header)

            row = QtWidgets.QHBoxLayout()
            btn_fill = QtWidgets.QPushButton("Fill cube slot1 + tetra slot2")
            btn_cube = QtWidgets.QPushButton("Cube → slot1")
            btn_tetra = QtWidgets.QPushButton("Tetra → slot2")
            btn_morph = QtWidgets.QPushButton("Morph → slot3")
            btn_clear = QtWidgets.QPushButton("Clear")
            for b in [btn_fill, btn_cube, btn_tetra, btn_morph, btn_clear]:
                b.setMinimumHeight(36)
                row.addWidget(b)
            main.addLayout(row)

            labels = {"__header__": header}
            box = QtWidgets.QGroupBox("Current M1 object-slot imit")
            grid = QtWidgets.QGridLayout(box)
            rows = [
                ("active", "active"),
                ("kind", "kind"),
                ("remaining", "remaining"),
                ("selected_slot", "selected_slot"),
                ("items", "items"),
                ("last_slots", "last_slots"),
                ("last_names", "last_names"),
                ("selected_slot_z_norm", "selected_slot_z_norm"),
                ("selected_slot_confidence", "selected_slot_confidence"),
                ("slot_metrics", "slot_metrics"),
                ("layout", "layout"),
                ("target", "target"),
                ("source", "source"),
                ("note", "note"),
            ]
            for r, (label_text, key) in enumerate(rows):
                k = QtWidgets.QLabel(label_text)
                k.setStyleSheet("color:#8FA4BF; font-weight:800;")
                v = QtWidgets.QLabel("")
                v.setStyleSheet("color:#FFFFFF; font-weight:900;")
                v.setWordWrap(True)
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                grid.addWidget(k, r, 0)
                grid.addWidget(v, r, 1)
                labels[key] = v
            main.addWidget(box)

            raw = QtWidgets.QPlainTextEdit()
            raw.setReadOnly(True)
            raw.setMinimumHeight(180)
            main.addWidget(raw)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            close_row.addWidget(btn_close)
            main.addLayout(close_row)

            self.m1_object_slot_imit_window = dialog
            self.m1_object_slot_imit_labels = labels
            self.m1_object_slot_imit_raw = raw

            btn_fill.clicked.connect(lambda: self.request_m1_object_slot_imit("cube_tetra", cube_slot=1, tetra_slot=2, selected_slot=2))
            btn_cube.clicked.connect(lambda: self.request_m1_object_slot_imit("cube", slot=1, selected_slot=1))
            btn_tetra.clicked.connect(lambda: self.request_m1_object_slot_imit("tetrahedron", slot=2, selected_slot=2))
            btn_morph.clicked.connect(lambda: self.request_m1_object_slot_imit("morph", slot=3, selected_slot=3, alpha=0.5))
            btn_clear.clicked.connect(lambda: self.request_m1_object_slot_imit("clear", duration=1))
            btn_close.clicked.connect(dialog.close)
            dialog.finished.connect(lambda _code: self._clear_m1_object_slot_imit_window_refs())

            self.refresh_m1_object_slot_imit_window()
            dialog.show()
            self.refresh_ui()

        def request_m2_scenario_imit(self, kind: str, *, slot: int | None = None, density: int = 1, alpha: float = 0.5):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            payload = {
                "kind": str(kind),
                "density": int(density),
                "alpha": float(alpha),
                "source": "m8_m2_scenario_imit_window",
            }
            if slot is not None:
                payload["slot"] = int(slot)
            action = "m2_scenario_imit_clear" if str(kind) == "clear" else "m2_scenario_imit_inject"
            ok = self.send(make_action_message(action, **payload))
            self.status.setText(
                f"M2 scenario imit requested: {kind}" if ok else "M2 scenario imit request failed"
            )
            self.refresh_ui()

        def refresh_m2_scenario_imit_window(self):
            labels = getattr(self, "m2_scenario_imit_labels", {}) or {}
            if not labels:
                return
            try:
                if not self.m2_scenario_imit_window or not self.m2_scenario_imit_window.isVisible():
                    return
            except Exception:
                return

            data = {}
            if isinstance(getattr(self, "last_status", None), dict):
                data = self.last_status.get("m2_scenario_imit", {}) or {}
            if not isinstance(data, dict):
                data = {}

            rpc = data.get("rpc", {}) if isinstance(data.get("rpc", {}), dict) else {}
            streamer = data.get("streamer", {}) if isinstance(data.get("streamer", {}), dict) else {}
            slots = streamer.get("slots", {}) if isinstance(streamer.get("slots", {}), dict) else {}

            header = labels.get("__header__")
            if header is not None:
                header.setText(
                    f"step={data.get('global_step', 0)} | active={int(bool(data.get('active', False)))} | "
                    f"kind={data.get('kind', '')} | rpc={data.get('host', '127.0.0.1')}:{data.get('port', 8771)}"
                )

            values = {
                "active": data.get("active", False),
                "kind": data.get("kind", ""),
                "items": data.get("items", []),
                "target": data.get("target", ""),
                "source": data.get("source", ""),
                "rpc_updated": rpc.get("updated", False),
                "rpc_started": rpc.get("started", streamer.get("started", False)),
                "slot_0_points": rpc.get("slot_0_points", (slots.get("0", {}) or {}).get("raw_points", 0)),
                "slot_1_points": rpc.get("slot_1_points", (slots.get("1", {}) or {}).get("raw_points", 0)),
                "slot_0_name": (slots.get("0", {}) or {}).get("target_name", ""),
                "slot_1_name": (slots.get("1", {}) or {}).get("target_name", ""),
                "layout": data.get("layout", ""),
            }
            for key, label in labels.items():
                if key.startswith("__"):
                    continue
                label.setText(self._fmt_monitor_value(values.get(key, "")))

            raw = getattr(self, "m2_scenario_imit_raw", None)
            if raw is not None:
                try:
                    raw.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    raw.setPlainText(str(data))

        def open_m2_scenario_imit_window(self):
            try:
                if self.m2_scenario_imit_window is not None and self.m2_scenario_imit_window.isVisible():
                    window = self.m2_scenario_imit_window
                    window.close()
                    self._clear_m2_scenario_imit_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 M2 Scenario Imitator for Open3D RPC")
            dialog.resize(860, 620)
            dialog.setStyleSheet(
                "QDialog { background:#0C121B; color:#DCE8F8; }"
                "QLabel { color:#DCE8F8; background:transparent; }"
                "QGroupBox { background:#101722; border:1px solid #243145; border-radius:12px; "
                "margin-top:12px; padding:10px; color:#B7C5DA; font-weight:800; }"
                "QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 6px; }"
                "QPlainTextEdit { background:#07101A; color:#DCE8F8; border:1px solid #2B3A50; "
                "border-radius:10px; padding:8px; font-family:Consolas, monospace; font-size:11px; }"
                "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                "border-radius:10px; padding:8px 12px; font-weight:800; }"
            )

            main = QtWidgets.QVBoxLayout(dialog)
            main.setContentsMargins(14, 14, 14, 14)
            main.setSpacing(10)

            title = QtWidgets.QLabel(
                "M2 scenario imit: publish deterministic cube/tetra Gaussian states to Slot4D JSON-RPC for Open3D RPC."
            )
            title.setWordWrap(True)
            title.setStyleSheet("font-weight:900; color:#D2A8FF;")
            main.addWidget(title)

            header = QtWidgets.QLabel("")
            header.setStyleSheet("font-weight:900; color:#FFD36D;")
            main.addWidget(header)

            row = QtWidgets.QHBoxLayout()
            btn_fill = QtWidgets.QPushButton("Cube slot0 + Tetra slot1")
            btn_cube = QtWidgets.QPushButton("Cube → RPC slot0")
            btn_tetra = QtWidgets.QPushButton("Tetra → RPC slot1")
            btn_morph = QtWidgets.QPushButton("Morph → RPC slot1")
            btn_clear = QtWidgets.QPushButton("Clear")
            for b in [btn_fill, btn_cube, btn_tetra, btn_morph, btn_clear]:
                b.setMinimumHeight(36)
                row.addWidget(b)
            main.addLayout(row)

            labels = {"__header__": header}
            box = QtWidgets.QGroupBox("Current M2 scenario imit / Open3D RPC feed")
            grid = QtWidgets.QGridLayout(box)
            rows = [
                ("active", "active"),
                ("kind", "kind"),
                ("items", "items"),
                ("rpc_started", "rpc_started"),
                ("rpc_updated", "rpc_updated"),
                ("slot_0_points", "slot_0_points"),
                ("slot_0_name", "slot_0_name"),
                ("slot_1_points", "slot_1_points"),
                ("slot_1_name", "slot_1_name"),
                ("layout", "layout"),
                ("target", "target"),
                ("source", "source"),
            ]
            for r, (label_text, key) in enumerate(rows):
                k = QtWidgets.QLabel(label_text)
                k.setStyleSheet("color:#8FA4BF; font-weight:800;")
                v = QtWidgets.QLabel("")
                v.setStyleSheet("color:#FFFFFF; font-weight:900;")
                v.setWordWrap(True)
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                grid.addWidget(k, r, 0)
                grid.addWidget(v, r, 1)
                labels[key] = v
            main.addWidget(box)

            raw = QtWidgets.QPlainTextEdit()
            raw.setReadOnly(True)
            raw.setMinimumHeight(180)
            main.addWidget(raw)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            close_row.addWidget(btn_close)
            main.addLayout(close_row)

            self.m2_scenario_imit_window = dialog
            self.m2_scenario_imit_labels = labels
            self.m2_scenario_imit_raw = raw

            btn_fill.clicked.connect(lambda: self.request_m2_scenario_imit("cube_tetra", density=1))
            btn_cube.clicked.connect(lambda: self.request_m2_scenario_imit("cube", slot=0, density=1))
            btn_tetra.clicked.connect(lambda: self.request_m2_scenario_imit("tetrahedron", slot=1, density=1))
            btn_morph.clicked.connect(lambda: self.request_m2_scenario_imit("morph", slot=1, density=1, alpha=0.5))
            btn_clear.clicked.connect(lambda: self.request_m2_scenario_imit("clear"))
            btn_close.clicked.connect(dialog.close)
            dialog.finished.connect(lambda _code: self._clear_m2_scenario_imit_window_refs())

            self.refresh_m2_scenario_imit_window()
            dialog.show()
            self.refresh_ui()

        def request_m5_latent_prototype(self, kind: str, gate: float = 0.85, duration: int = 120, alpha: float = 0.5):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            ok = self.send(make_action_message(
                "m5_latent_prototype_inject",
                kind=str(kind),
                gate=float(gate),
                duration=int(duration),
                alpha=float(alpha),
                source="m8_m5_latent_prototype_window",
            ))
            self.status.setText(
                f"M5 latent prototype requested: {kind}" if ok else "M5 latent prototype request failed"
            )
            self.refresh_ui()

        def _m5_latent_prototype_value(self, key: str, default=""):
            data = {}
            if isinstance(getattr(self, "last_status", None), dict):
                data = self.last_status.get("m5_latent_prototype", {}) or {}
            if not isinstance(data, dict):
                return default
            return data.get(key, default)

        def refresh_m5_latent_prototype_window(self):
            labels = getattr(self, "m5_latent_prototype_labels", {}) or {}
            if not labels:
                return
            try:
                if not self.m5_latent_prototype_window or not self.m5_latent_prototype_window.isVisible():
                    return
            except Exception:
                return

            data = {}
            if isinstance(getattr(self, "last_status", None), dict):
                data = self.last_status.get("m5_latent_prototype", {}) or {}
            if not isinstance(data, dict):
                data = {}

            header = labels.get("__header__")
            if header is not None:
                header.setText(
                    f"step={data.get('global_step', 0)} | active={int(bool(data.get('active', False)))} | "
                    f"kind={data.get('kind', '')} | remaining={data.get('remaining', 0)}"
                )

            for key, label in labels.items():
                if key.startswith("__"):
                    continue
                label.setText(self._fmt_monitor_value(data.get(key, "")))

            raw = getattr(self, "m5_latent_prototype_raw", None)
            if raw is not None:
                try:
                    raw.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    raw.setPlainText(str(data))

        def open_m5_latent_prototype_window(self):
            try:
                if self.m5_latent_prototype_window is not None and self.m5_latent_prototype_window.isVisible():
                    window = self.m5_latent_prototype_window
                    window.close()
                    self._clear_m5_latent_prototype_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 M5 Latent Prototype Simulator")
            dialog.resize(820, 620)
            dialog.setStyleSheet(
                "QDialog { background:#0C121B; color:#DCE8F8; }"
                "QLabel { color:#DCE8F8; background:transparent; }"
                "QGroupBox { background:#101722; border:1px solid #243145; border-radius:12px; "
                "margin-top:12px; padding:10px; color:#B7C5DA; font-weight:800; }"
                "QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 6px; }"
                "QPlainTextEdit { background:#07101A; color:#DCE8F8; border:1px solid #2B3A50; "
                "border-radius:10px; padding:8px; font-family:Consolas, monospace; font-size:11px; }"
                "QPushButton { background:#1D2A3B; color:white; border:1px solid #37507A; "
                "border-radius:10px; padding:8px 12px; font-weight:800; }"
            )

            main = QtWidgets.QVBoxLayout(dialog)
            main.setContentsMargins(14, 14, 14, 14)
            main.setSpacing(10)

            title = QtWidgets.QLabel("Simulated learned M5 latents: cube / tetrahedron / morph. Diagnostic only; not real trained weights.")
            title.setWordWrap(True)
            title.setStyleSheet("font-weight:900; color:#D2A8FF;")
            main.addWidget(title)

            header = QtWidgets.QLabel("")
            header.setStyleSheet("font-weight:900; color:#FFD36D;")
            main.addWidget(header)

            row = QtWidgets.QHBoxLayout()
            btn_cube = QtWidgets.QPushButton("Inject cube latent")
            btn_tetra = QtWidgets.QPushButton("Inject tetrahedron latent")
            btn_morph = QtWidgets.QPushButton("Inject cube↔tetra morph")
            btn_clear = QtWidgets.QPushButton("Clear")
            for b in [btn_cube, btn_tetra, btn_morph, btn_clear]:
                b.setMinimumHeight(36)
                row.addWidget(b)
            main.addLayout(row)

            labels = {"__header__": header}

            box = QtWidgets.QGroupBox("Current simulated M5 prototype")
            grid = QtWidgets.QGridLayout(box)
            rows = [
                ("active", "active"),
                ("kind", "kind"),
                ("gate", "gate"),
                ("seed_norm", "seed_norm"),
                ("remaining", "remaining"),
                ("cube_similarity", "cube_similarity"),
                ("tetra_similarity", "tetra_similarity"),
                ("layout", "layout"),
                ("target_m5_boundary", "target_m5_boundary"),
                ("source", "source"),
                ("note", "note"),
            ]
            for r, (label_text, key) in enumerate(rows):
                k = QtWidgets.QLabel(label_text)
                k.setStyleSheet("color:#8FA4BF; font-weight:800;")
                v = QtWidgets.QLabel("")
                v.setStyleSheet("color:#FFFFFF; font-weight:900;")
                v.setWordWrap(True)
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                grid.addWidget(k, r, 0)
                grid.addWidget(v, r, 1)
                labels[key] = v
            main.addWidget(box)

            raw = QtWidgets.QPlainTextEdit()
            raw.setReadOnly(True)
            raw.setMinimumHeight(180)
            main.addWidget(raw)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            close_row.addWidget(btn_close)
            main.addLayout(close_row)

            self.m5_latent_prototype_window = dialog
            self.m5_latent_prototype_labels = labels
            self.m5_latent_prototype_raw = raw

            btn_cube.clicked.connect(lambda: self.request_m5_latent_prototype("cube", 0.90, 160, 0.0))
            btn_tetra.clicked.connect(lambda: self.request_m5_latent_prototype("tetrahedron", 0.90, 160, 1.0))
            btn_morph.clicked.connect(lambda: self.request_m5_latent_prototype("morph", 0.85, 160, 0.5))
            btn_clear.clicked.connect(lambda: self.request_m5_latent_prototype("clear", 0.0, 1, 0.5))
            btn_close.clicked.connect(dialog.close)
            dialog.finished.connect(lambda _code: self._clear_m5_latent_prototype_window_refs())

            self.refresh_m5_latent_prototype_window()
            dialog.show()
            self.refresh_ui()

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

        def toggle_sleep_replay_mode(self):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            enable_sleep = not self._sleep_mode_active()
            sensors_enabled = not enable_sleep
            self.state.video_sensor_enabled = sensors_enabled
            self.state.contact_sensor_enabled = sensors_enabled
            self.state.imu_sensor_enabled = sensors_enabled
            sensors = {
                "video": sensors_enabled,
                "contact": sensors_enabled,
                "imu": sensors_enabled,
            }
            mask = {k: not v for k, v in sensors.items()}
            ok = self.send(make_set_state_message(
                input_sensors_enabled=sensors,
                sleep_sensor_mask=mask,
                video_sensor_enabled=sensors_enabled,
                contact_sensor_enabled=sensors_enabled,
                imu_sensor_enabled=sensors_enabled,
                sleep_video_cut=enable_sleep,
                sleep_contact_cut=enable_sleep,
                sleep_imu_cut=enable_sleep,
            ))
            self.status.setText(
                f"Sleep / replay mode {'ON' if enable_sleep else 'OFF'}"
                if ok else "Sleep / replay mode request failed"
            )
            self.refresh_ui()

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

        def run_m8_module_lab(self):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            ok = self.send(make_action_message(
                "module_lab_run",
                module="all",
                source="m8_control_panel",
            ))
            if ok:
                self.status.setText("M8 Module Lab requested")
            else:
                self.status.setText("M8 Module Lab request failed")
            self.refresh_ui()

        def _format_module_lab_result(self, result: dict) -> str:
            if not isinstance(result, dict) or not result:
                return (
                    "Нет результата. Нажми одну из кнопок Run M*.\n\n"
                    "Окно показывает last_module_lab_result из status IPC."
                )
            try:
                return json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                return str(result)

        def refresh_module_lab_window(self):
            text_widget = getattr(self, "module_lab_text", None)
            if text_widget is None:
                return
            try:
                if not self.module_lab_window or not self.module_lab_window.isVisible():
                    return
            except Exception:
                return
            result = {}
            if isinstance(getattr(self, "last_status", None), dict):
                result = self.last_status.get("last_module_lab_result", {}) or {}
            text_widget.setPlainText(self._format_module_lab_result(result))

        def request_m8_module_lab(self, module: str = "all"):
            if not self.state.connected:
                self.status.setText("STATUS IPC: no signal")
                self.refresh_ui()
                return
            label = str(module)
            if getattr(self, "module_lab_text", None) is not None:
                self.module_lab_text.setPlainText(f"Запрос Module Lab: {label}\nЖду status IPC...")
            ok = self.send(make_action_message(
                "module_lab_run",
                module=label,
                source="m8_control_panel",
            ))
            if ok:
                self.status.setText(f"M8 Module Lab requested: {label}")
            else:
                self.status.setText("M8 Module Lab request failed")
            self.refresh_ui()

        def open_m8_module_lab_window(self):
            try:
                if self.module_lab_window is not None and self.module_lab_window.isVisible():
                    window = self.module_lab_window
                    window.close()
                    self._clear_module_lab_window_refs()
                    return
            except Exception:
                pass

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("M8 Module Lab")
            dialog.resize(820, 620)
            dialog.setStyleSheet(
                "QDialog { background: #0C121B; color: #DCE8F8; }"
                "QLabel { color: #B7C5DA; background: transparent; font-weight: 700; }"
                "QPlainTextEdit { background: #07101A; color: #DCE8F8; border: 1px solid #2B3A50; "
                "border-radius: 10px; padding: 10px; font-family: Consolas, monospace; font-size: 11px; }"
                "QPushButton { background: #1D2A3B; color: white; border: 1px solid #37507A; "
                "border-radius: 10px; padding: 8px 12px; font-weight: 800; }"
                "QPushButton:hover { background: #263B55; }"
            )

            lay = QtWidgets.QVBoxLayout(dialog)
            lay.setContentsMargins(14, 14, 14, 14)
            lay.setSpacing(10)

            title = QtWidgets.QLabel(
                "M8 Module Lab: отдельная проверка модулей и всего бессознательного контура"
            )
            title.setWordWrap(True)
            lay.addWidget(title)

            row1 = QtWidgets.QHBoxLayout()
            row1.setSpacing(10)
            btn_m2 = QtWidgets.QPushButton("Run M2 test")
            btn_m4 = QtWidgets.QPushButton("Run M4 test")
            btn_m11 = QtWidgets.QPushButton("Run M11 test")
            btn_m13 = QtWidgets.QPushButton("Run M13 test")
            for b in [btn_m2, btn_m4, btn_m11, btn_m13]:
                b.setMinimumHeight(38)
                row1.addWidget(b)
            lay.addLayout(row1)

            row2 = QtWidgets.QHBoxLayout()
            row2.setSpacing(10)
            btn_m5 = QtWidgets.QPushButton("Run M5Boundary test")
            btn_loop = QtWidgets.QPushButton("Run unconscious loop test")
            btn_scenarios = QtWidgets.QPushButton("Run behavioral scenarios")
            btn_all = QtWidgets.QPushButton("Run all")
            for b in [btn_m5, btn_loop, btn_scenarios, btn_all]:
                b.setMinimumHeight(38)
                row2.addWidget(b)
            lay.addLayout(row2)

            text = QtWidgets.QPlainTextEdit()
            text.setReadOnly(True)
            text.setMinimumHeight(410)
            lay.addWidget(text)

            close_row = QtWidgets.QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QtWidgets.QPushButton("Close")
            btn_close.setMinimumHeight(36)
            close_row.addWidget(btn_close)
            lay.addLayout(close_row)

            self.module_lab_window = dialog
            self.module_lab_text = text

            btn_m2.clicked.connect(lambda: self.request_m8_module_lab("m02"))
            btn_m4.clicked.connect(lambda: self.request_m8_module_lab("m4"))
            btn_m11.clicked.connect(lambda: self.request_m8_module_lab("m11"))
            btn_m13.clicked.connect(lambda: self.request_m8_module_lab("m13"))
            btn_m5.clicked.connect(lambda: self.request_m8_module_lab("m05"))
            btn_loop.clicked.connect(lambda: self.request_m8_module_lab("loop"))
            btn_scenarios.clicked.connect(lambda: self.request_m8_module_lab("scenarios"))
            btn_all.clicked.connect(lambda: self.request_m8_module_lab("all"))
            btn_close.clicked.connect(dialog.close)

            dialog.finished.connect(lambda _code: self._clear_module_lab_window_refs())

            self.refresh_module_lab_window()
            dialog.show()
            self.refresh_ui()

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
