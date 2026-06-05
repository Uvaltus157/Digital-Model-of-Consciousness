#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys

def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / 'src').exists() and ((candidate / '.git').exists() or (candidate / 'config').exists()):
            return candidate
    return Path.cwd().resolve()

ROOT = find_repo_root()
CONTROL_PANEL = ROOT / 'src/modules/m08_debug_visual_control/control_panel.py'

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f'[skip] {label}: already present')
        return text
    if old not in text:
        print(f'[warn] {label}: anchor not found')
        return text
    print(f'[ok] {label}')
    return text.replace(old, new, 1)

def ensure_basics(text: str) -> str:
    if 'import json\n' not in text:
        text = replace_once(text, 'import argparse\n', 'import argparse\nimport json\n', 'import json')
    if '"btn_module_lab"' not in text:
        text = replace_once(text, '"m8": ["btn_module_debug"],', '"m8": ["btn_module_debug", "btn_module_lab"],', 'M8 tab button list')
    if 'self.btn_module_lab' not in text:
        text = replace_once(text, '            self.btn_module_debug = QtWidgets.QPushButton()\n', '            self.btn_module_debug = QtWidgets.QPushButton()\n            self.btn_module_lab = QtWidgets.QPushButton("Module Lab")\n', 'create btn_module_lab')
    if 'self.module_lab_window' not in text:
        text = replace_once(text, '            self.open3d_slot_viewer_proc = None\n', '            self.open3d_slot_viewer_proc = None\n            self.module_lab_window = None\n            self.module_lab_text = None\n', 'module lab window fields')
    if 'self.btn_module_lab.setMinimumHeight(42)' not in text:
        text = replace_once(text, '            self.btn_module_debug_pyqt.setMinimumHeight(42)\n', '            self.btn_module_debug_pyqt.setMinimumHeight(42)\n            self.btn_module_lab.setMinimumHeight(42)\n', 'module lab button height')
    if 'Run module lab contracts/scenarios and show latest result' not in text:
        text = replace_once(text, '            self.btn_module_debug.setToolTip("Show or hide the runner-owned module debug visualizer")\n', '            self.btn_module_debug.setToolTip("Show or hide the runner-owned module debug visualizer")\n            self.btn_module_lab.setToolTip("Run module lab contracts/scenarios and show latest result")\n', 'module lab tooltip')
    text = text.replace('            self.btn_module_lab.clicked.connect(self.run_m8_module_lab)\n', '            self.btn_module_lab.clicked.connect(self.open_m8_module_lab_window)\n')
    if 'self.btn_module_lab.clicked.connect(self.open_m8_module_lab_window)' not in text:
        text = replace_once(text, '            self.btn_module_debug.clicked.connect(lambda: self.toggle("module_debug"))\n', '            self.btn_module_debug.clicked.connect(lambda: self.toggle("module_debug"))\n            self.btn_module_lab.clicked.connect(self.open_m8_module_lab_window)\n', 'module lab signal')
    if 'self.btn_module_lab,' not in text:
        text = replace_once(text, '                self.btn_module_debug,\n', '                self.btn_module_debug,\n                self.btn_module_lab,\n', 'runner dependent module lab button')
    if 'self._style_plain_status_button(self.btn_module_lab, False, "Module Lab")' not in text:
        text = text.replace('            self._style_plain_status_button(self.btn_module_lab, False, "Run Module Lab")\n', '            self._style_plain_status_button(self.btn_module_lab, False, "Module Lab")\n')
        if 'self._style_plain_status_button(self.btn_module_lab, False, "Module Lab")' not in text:
            text = replace_once(text, '            self._style_button(self.btn_module_debug, s.module_debug, "Module debug")\n', '            self._style_button(self.btn_module_debug, s.module_debug, "Module debug")\n            self._style_plain_status_button(self.btn_module_lab, False, "Module Lab")\n', 'module lab refresh style')
    if 'self.refresh_module_lab_window()' not in text:
        text = replace_once(text, '            self._set_runner_controls_enabled(s.connected)\n', '            self._set_runner_controls_enabled(s.connected)\n            self.refresh_module_lab_window()\n', 'refresh module lab window from status')
    return text

def replace_method_block(text: str, methods: str) -> str:
    start = text.find('        def _format_module_lab_result(self, result: dict) -> str:\n')
    end_anchor = '        def action(self, action: str):\n'
    end = text.find(end_anchor)
    if start >= 0 and end > start:
        print('[ok] replaced existing Module Lab window methods')
        return text[:start] + methods + text[end:]
    if end >= 0:
        print('[ok] inserted Module Lab window methods')
        return text[:end] + methods + text[end:]
    print('[warn] cannot find insertion point before action method')
    return text

def patch_control_panel() -> None:
    if not CONTROL_PANEL.exists():
        raise FileNotFoundError(f'missing {CONTROL_PANEL}')
    text = CONTROL_PANEL.read_text(encoding='utf-8')
    old_text = text
    text = ensure_basics(text)
    methods = '        def _format_module_lab_result(self, result: dict) -> str:\n            if not isinstance(result, dict) or not result:\n                return (\n                    "Нет результата. Нажми одну из кнопок Run M*.\\n\\n"\n                    "Окно показывает last_module_lab_result из status IPC."\n                )\n            try:\n                return json.dumps(result, ensure_ascii=False, indent=2)\n            except Exception:\n                return str(result)\n\n        def refresh_module_lab_window(self):\n            text_widget = getattr(self, "module_lab_text", None)\n            if text_widget is None:\n                return\n            try:\n                if not self.module_lab_window or not self.module_lab_window.isVisible():\n                    return\n            except Exception:\n                return\n            result = {}\n            if isinstance(getattr(self, "last_status", None), dict):\n                result = self.last_status.get("last_module_lab_result", {}) or {}\n            text_widget.setPlainText(self._format_module_lab_result(result))\n\n        def request_m8_module_lab(self, module: str = "all"):\n            if not self.state.connected:\n                self.status.setText("STATUS IPC: no signal")\n                self.refresh_ui()\n                return\n            label = str(module)\n            if getattr(self, "module_lab_text", None) is not None:\n                self.module_lab_text.setPlainText(f"Запрос Module Lab: {label}\\nЖду status IPC...")\n            ok = self.send(make_action_message(\n                "module_lab_run",\n                module=label,\n                source="m8_control_panel",\n            ))\n            if ok:\n                self.status.setText(f"M8 Module Lab requested: {label}")\n            else:\n                self.status.setText("M8 Module Lab request failed")\n            self.refresh_ui()\n\n        def open_m8_module_lab_window(self):\n            try:\n                if self.module_lab_window is not None and self.module_lab_window.isVisible():\n                    self.module_lab_window.raise_()\n                    self.module_lab_window.activateWindow()\n                    return\n            except Exception:\n                pass\n\n            dialog = QtWidgets.QDialog(self)\n            dialog.setWindowTitle("M8 Module Lab")\n            dialog.resize(820, 620)\n            dialog.setStyleSheet(\n                "QDialog { background: #0C121B; color: #DCE8F8; }"\n                "QLabel { color: #B7C5DA; background: transparent; font-weight: 700; }"\n                "QPlainTextEdit { background: #07101A; color: #DCE8F8; border: 1px solid #2B3A50; "\n                "border-radius: 10px; padding: 10px; font-family: Consolas, monospace; font-size: 11px; }"\n                "QPushButton { background: #1D2A3B; color: white; border: 1px solid #37507A; "\n                "border-radius: 10px; padding: 8px 12px; font-weight: 800; }"\n                "QPushButton:hover { background: #263B55; }"\n            )\n\n            lay = QtWidgets.QVBoxLayout(dialog)\n            lay.setContentsMargins(14, 14, 14, 14)\n            lay.setSpacing(10)\n\n            title = QtWidgets.QLabel(\n                "M8 Module Lab: отдельная проверка модулей и всего бессознательного контура"\n            )\n            title.setWordWrap(True)\n            lay.addWidget(title)\n\n            row1 = QtWidgets.QHBoxLayout()\n            row1.setSpacing(10)\n            btn_m2 = QtWidgets.QPushButton("Run M2 test")\n            btn_m4 = QtWidgets.QPushButton("Run M4 test")\n            btn_m11 = QtWidgets.QPushButton("Run M11 test")\n            btn_m13 = QtWidgets.QPushButton("Run M13 test")\n            for b in [btn_m2, btn_m4, btn_m11, btn_m13]:\n                b.setMinimumHeight(38)\n                row1.addWidget(b)\n            lay.addLayout(row1)\n\n            row2 = QtWidgets.QHBoxLayout()\n            row2.setSpacing(10)\n            btn_m5 = QtWidgets.QPushButton("Run M5Boundary test")\n            btn_loop = QtWidgets.QPushButton("Run unconscious loop test")\n            btn_scenarios = QtWidgets.QPushButton("Run behavioral scenarios")\n            btn_all = QtWidgets.QPushButton("Run all")\n            for b in [btn_m5, btn_loop, btn_scenarios, btn_all]:\n                b.setMinimumHeight(38)\n                row2.addWidget(b)\n            lay.addLayout(row2)\n\n            text = QtWidgets.QPlainTextEdit()\n            text.setReadOnly(True)\n            text.setMinimumHeight(410)\n            lay.addWidget(text)\n\n            close_row = QtWidgets.QHBoxLayout()\n            close_row.addStretch(1)\n            btn_close = QtWidgets.QPushButton("Close")\n            btn_close.setMinimumHeight(36)\n            close_row.addWidget(btn_close)\n            lay.addLayout(close_row)\n\n            self.module_lab_window = dialog\n            self.module_lab_text = text\n\n            btn_m2.clicked.connect(lambda: self.request_m8_module_lab("m02"))\n            btn_m4.clicked.connect(lambda: self.request_m8_module_lab("m4"))\n            btn_m11.clicked.connect(lambda: self.request_m8_module_lab("m11"))\n            btn_m13.clicked.connect(lambda: self.request_m8_module_lab("m13"))\n            btn_m5.clicked.connect(lambda: self.request_m8_module_lab("m05"))\n            btn_loop.clicked.connect(lambda: self.request_m8_module_lab("loop"))\n            btn_scenarios.clicked.connect(lambda: self.request_m8_module_lab("scenarios"))\n            btn_all.clicked.connect(lambda: self.request_m8_module_lab("all"))\n            btn_close.clicked.connect(dialog.close)\n\n            dialog.finished.connect(lambda _code: setattr(self, "module_lab_text", None))\n\n            self.refresh_module_lab_window()\n            dialog.show()\n\n'
    text = replace_method_block(text, methods)
    if text != old_text:
        CONTROL_PANEL.write_text(text, encoding='utf-8')
        print('[ok] wrote control_panel.py')
    else:
        print('[skip] no changes to control_panel.py')

def main() -> int:
    patch_control_panel()
    print('\nRun:')
    print('  python -m py_compile src/modules/m08_debug_visual_control/control_panel.py')
    print('Then restart the PyQt control panel.')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        raise
