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
FILES = {}
FILES['src/modules/m08_debug_visual_control/module_lab_runtime.py'] = 'from __future__ import annotations\n\nfrom typing import Any, Dict\nimport time\nimport traceback\n\n\ndef _compact(value: Any, *, max_items: int = 6) -> Any:\n    try:\n        import torch\n        if torch.is_tensor(value):\n            flat = value.detach().float().reshape(-1)\n            return {\n                "type": "tensor",\n                "shape": list(value.shape),\n                "sample": [float(x) for x in flat[:max_items].cpu().tolist()],\n            }\n    except Exception:\n        pass\n    if isinstance(value, dict):\n        out = {}\n        for idx, (k, v) in enumerate(value.items()):\n            if idx >= max_items:\n                out["..."] = f"{len(value) - max_items} more"\n                break\n            out[str(k)] = _compact(v, max_items=max_items)\n        return out\n    if isinstance(value, (list, tuple)):\n        return [_compact(v, max_items=max_items) for v in list(value)[:max_items]]\n    if isinstance(value, (str, int, float, bool)) or value is None:\n        return value\n    return str(value)\n\n\ndef run_module_lab_from_payload(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:\n    payload = dict(payload or {})\n    module = str(payload.get("module", "all")).lower().strip() or "all"\n    started = time.time()\n    try:\n        if module in ("behavioral", "scenario", "scenarios", "unconscious_scenarios"):\n            from scripts.module_lab.scenario_unconscious_replay import run_all\n            result = run_all()\n            return {\n                "ok": bool(result.get("status") == "ok"),\n                "module": module,\n                "kind": "behavioral_scenarios",\n                "duration_sec": round(time.time() - started, 4),\n                "result": _compact(result, max_items=12),\n            }\n\n        from scripts.module_lab.run_module_lab import LABS\n        if module == "all":\n            selected = ("m11", "m13", "m4", "m02", "m05", "loop")\n            result = {name: LABS[name]() for name in selected if name in LABS}\n        else:\n            if module not in LABS:\n                raise KeyError(f"unknown module lab {module!r}; available={sorted(LABS)}")\n            result = LABS[module]()\n\n        return {\n            "ok": True,\n            "module": module,\n            "kind": "module_lab",\n            "duration_sec": round(time.time() - started, 4),\n            "result": _compact(result, max_items=12),\n        }\n    except Exception as e:\n        return {\n            "ok": False,\n            "module": module,\n            "kind": "module_lab",\n            "duration_sec": round(time.time() - started, 4),\n            "error": str(e),\n            "traceback": traceback.format_exc(limit=8),\n        }\n\n\n__all__ = ["run_module_lab_from_payload"]\n'
FILES['tests/module_contracts/test_m8_module_lab_runtime_contract.py'] = 'from __future__ import annotations\n\n\ndef test_m8_module_lab_runtime_bridge_runs_loop_or_all():\n    from src.modules.m08_debug_visual_control.module_lab_runtime import run_module_lab_from_payload\n    result = run_module_lab_from_payload({"module": "loop"})\n    assert isinstance(result, dict)\n    assert result["ok"] is True\n    assert result["kind"] == "module_lab"\n    assert result["module"] == "loop"\n\n\ndef test_m8_module_lab_runtime_bridge_runs_behavioral_scenarios():\n    from src.modules.m08_debug_visual_control.module_lab_runtime import run_module_lab_from_payload\n    result = run_module_lab_from_payload({"module": "scenarios"})\n    assert isinstance(result, dict)\n    assert result["ok"] is True\n    assert result["kind"] == "behavioral_scenarios"\n'
FILES['docs/architecture/m8_module_lab_button.md'] = '# M8 Module Lab button\n\nAdds a button to the M8 tab in `pyqt_control_panel_ipc.py` / `control_panel.py`:\n\n```text\nRun Module Lab\n```\n\nThe button sends IPC:\n\n```json\n{\n  "type": "action",\n  "action": "module_lab_run",\n  "payload": {\n    "module": "all",\n    "source": "m8_control_panel"\n  }\n}\n```\n\nRunner stores:\n\n```python\nself.last_module_lab_result\n```\n\nand status IPC exposes:\n\n```text\nlast_module_lab_result\n```\n'

def write_files() -> None:
    for rel, content in FILES.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding='utf-8') == content:
            print(f'[skip] {rel}: unchanged')
            continue
        target.write_text(content, encoding='utf-8')
        print(f'[ok] wrote {rel}')

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f'[skip] {label}: already present')
        return text
    if old not in text:
        print(f'[warn] {label}: anchor not found')
        return text
    print(f'[ok] {label}')
    return text.replace(old, new, 1)

def patch_control_panel() -> None:
    path = ROOT / 'src/modules/m08_debug_visual_control/control_panel.py'
    if not path.exists():
        print('[warn] control_panel.py not found')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    text = replace_once(text, '"m8": ["btn_module_debug"],', '"m8": ["btn_module_debug", "btn_module_lab"],', 'control_panel MODULE_TAB_BUTTONS m8')
    text = replace_once(text, '            self.btn_module_debug = QtWidgets.QPushButton()\n', '            self.btn_module_debug = QtWidgets.QPushButton()\n            self.btn_module_lab = QtWidgets.QPushButton("Run Module Lab")\n', 'control_panel create btn_module_lab')
    text = replace_once(text, '            self.btn_module_debug_pyqt.setMinimumHeight(42)\n', '            self.btn_module_debug_pyqt.setMinimumHeight(42)\n            self.btn_module_lab.setMinimumHeight(42)\n', 'control_panel btn_module_lab height')
    text = replace_once(text, '            self.btn_module_debug.setToolTip("Show or hide the runner-owned module debug visualizer")\n', '            self.btn_module_debug.setToolTip("Show or hide the runner-owned module debug visualizer")\n            self.btn_module_lab.setToolTip("Run M8 module lab contracts/scenarios inside the runner via IPC")\n', 'control_panel btn_module_lab tooltip')
    text = replace_once(text, '            self.btn_module_debug.clicked.connect(lambda: self.toggle("module_debug"))\n', '            self.btn_module_debug.clicked.connect(lambda: self.toggle("module_debug"))\n            self.btn_module_lab.clicked.connect(self.run_m8_module_lab)\n', 'control_panel btn_module_lab signal')
    text = replace_once(text, '                self.btn_module_debug,\n', '                self.btn_module_debug,\n                self.btn_module_lab,\n', 'control_panel runner dependent button')
    text = replace_once(text, '            self._style_button(self.btn_module_debug, s.module_debug, "Module debug")\n', '            self._style_button(self.btn_module_debug, s.module_debug, "Module debug")\n            self._style_plain_status_button(self.btn_module_lab, False, "Run Module Lab")\n', 'control_panel refresh btn_module_lab style')
    insert_method = '        def run_m8_module_lab(self):\n            if not self.state.connected:\n                self.status.setText("STATUS IPC: no signal")\n                self.refresh_ui()\n                return\n            ok = self.send(make_action_message(\n                "module_lab_run",\n                module="all",\n                source="m8_control_panel",\n            ))\n            if ok:\n                self.status.setText("M8 Module Lab requested")\n            else:\n                self.status.setText("M8 Module Lab request failed")\n            self.refresh_ui()\n\n'
    text = replace_once(text, '        def action(self, action: str):\n', insert_method + '        def action(self, action: str):\n', 'control_panel run_m8_module_lab method')
    if text != old_text:
        path.write_text(text, encoding='utf-8')
        print('[ok] wrote control_panel.py')

def patch_action_runtime() -> None:
    path = ROOT / 'src/modules/m03_self_action_causality/action_runtime.py'
    if not path.exists():
        print('[warn] action_runtime.py not found')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    anchor = '        elif action in ("imit_action", "simulate_energy_resonator_action", "energy_resonator_imit_action"):\n            if hasattr(self, "request_energy_resonator_imitation"):\n                self.request_energy_resonator_imitation(payload)\n            else:\n                print("[ipc] imit_action ignored: EnergyResonatorRuntimeMixin is not installed")\n'
    insert = '        elif action in ("imit_action", "simulate_energy_resonator_action", "energy_resonator_imit_action"):\n            if hasattr(self, "request_energy_resonator_imitation"):\n                self.request_energy_resonator_imitation(payload)\n            else:\n                print("[ipc] imit_action ignored: EnergyResonatorRuntimeMixin is not installed")\n        elif action == "module_lab_run":\n            try:\n                from src.modules.m08_debug_visual_control.module_lab_runtime import run_module_lab_from_payload\n\n                result = run_module_lab_from_payload(payload)\n                self.last_module_lab_result = result\n                status = "ok" if bool(result.get("ok", False)) else "fail"\n                print(\n                    "[module_lab] "\n                    f"status={status} module={result.get(\'module\')} "\n                    f"kind={result.get(\'kind\')} duration={result.get(\'duration_sec\')}s"\n                )\n                if not bool(result.get("ok", False)):\n                    print(f"[module_lab] error={result.get(\'error\', \'\')}")\n                if hasattr(self, "write_module_debug_status"):\n                    self.write_module_debug_status()\n            except Exception as e:\n                self.last_module_lab_result = {\n                    "ok": False,\n                    "module": str(payload.get("module", "all")) if isinstance(payload, dict) else "all",\n                    "kind": "module_lab",\n                    "error": str(e),\n                }\n                print(f"[module_lab] failed: {e}")\n                if hasattr(self, "write_module_debug_status"):\n                    self.write_module_debug_status()\n'
    if 'elif action == "module_lab_run":' not in text:
        if anchor in text:
            text = text.replace(anchor, insert, 1)
            print('[ok] action_runtime.py: added module_lab_run IPC action')
        else:
            print('[warn] action_runtime.py: action anchor not found')
    else:
        print('[skip] action_runtime.py: module_lab_run already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')

def patch_module_status_runtime() -> None:
    path = ROOT / 'src/modules/m08_debug_visual_control/module_status_runtime.py'
    if not path.exists():
        print('[warn] module_status_runtime.py not found')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    anchor = '                "last_train_error": str(getattr(self, "last_train_error", "")),\n'
    insert = anchor + '                "last_module_lab_result": dict(getattr(self, "last_module_lab_result", {}) or {}),\n'
    if '"last_module_lab_result"' not in text:
        if anchor in text:
            text = text.replace(anchor, insert, 1)
            print('[ok] module_status_runtime.py: added last_module_lab_result')
        else:
            print('[warn] module_status_runtime.py: status anchor not found')
    else:
        print('[skip] module_status_runtime.py: last_module_lab_result already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')

def main() -> int:
    write_files()
    patch_control_panel()
    patch_action_runtime()
    patch_module_status_runtime()
    print('\nRun:')
    print('  python -m py_compile src/modules/m08_debug_visual_control/control_panel.py')
    print('  python -m py_compile src/modules/m08_debug_visual_control/module_lab_runtime.py')
    print('  python -m py_compile src/modules/m03_self_action_causality/action_runtime.py')
    print('  python -m py_compile src/modules/m08_debug_visual_control/module_status_runtime.py')
    print('  pytest tests/module_contracts/test_m8_module_lab_runtime_contract.py')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        raise
