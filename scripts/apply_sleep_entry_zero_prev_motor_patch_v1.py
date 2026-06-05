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
SLEEP_SENSORS = ROOT / 'src/modules/m06_learning_sleep_consolidation/sleep_sensors.py'
TEST_FILE = 'from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nimport torch\n\n\nclass DummySleepSystem:\n    from src.modules.m06_learning_sleep_consolidation.sleep_sensors import SleepSensorsMixin\n\n    apply_startup_state = SleepSensorsMixin.apply_startup_state\n    input_sensors_enabled_dict_no_startup_apply = SleepSensorsMixin.input_sensors_enabled_dict_no_startup_apply\n    sensor_state_label_no_startup_apply = SleepSensorsMixin.sensor_state_label_no_startup_apply\n    is_full_sleep_mode = SleepSensorsMixin.is_full_sleep_mode\n    sensor_state_label = SleepSensorsMixin.sensor_state_label\n    input_sensors_enabled_dict = SleepSensorsMixin.input_sensors_enabled_dict\n    sleep_sensor_mask_dict = SleepSensorsMixin.sleep_sensor_mask_dict\n    apply_sleep_sensor_state = SleepSensorsMixin.apply_sleep_sensor_state\n    _zero_prev_motor_state_on_sleep_entry = SleepSensorsMixin._zero_prev_motor_state_on_sleep_entry\n\n    def __init__(self) -> None:\n        self.cfg = SimpleNamespace(\n            sleep_sensors=SimpleNamespace(\n                startup_state="awake",\n                enabled=True,\n            )\n        )\n        self.video_sensor_enabled = True\n        self.contact_sensor_enabled = True\n        self.imu_sensor_enabled = True\n        self.prev_embodied_action = torch.ones(1, 15)\n        self.prev_hand_motor = torch.ones(1, 8) * 0.5\n        self._status_writes = 0\n\n    def write_module_debug_status(self):\n        self._status_writes += 1\n\n\ndef test_sleep_entry_zeros_prev_embodied_and_hand_once():\n    system = DummySleepSystem()\n\n    changed = system.apply_sleep_sensor_state({\n        "input_sensors_enabled": {\n            "video": False,\n            "contact": False,\n            "imu": False,\n        }\n    })\n\n    assert changed is True\n    assert system.is_full_sleep_mode() is True\n    assert torch.allclose(system.prev_embodied_action, torch.zeros_like(system.prev_embodied_action))\n    assert torch.allclose(system.prev_hand_motor, torch.zeros_like(system.prev_hand_motor))\n    assert system._status_writes == 1\n    assert system._sleep_replay_prev_motor_reset["reason"] == "entered_full_sleep"\n\n\ndef test_partial_cut_does_not_zero_prev_motor():\n    system = DummySleepSystem()\n\n    changed = system.apply_sleep_sensor_state({\n        "input_sensors_enabled": {\n            "video": False,\n            "contact": True,\n            "imu": True,\n        }\n    })\n\n    assert changed is True\n    assert system.is_full_sleep_mode() is False\n    assert torch.allclose(system.prev_embodied_action, torch.ones_like(system.prev_embodied_action))\n    assert torch.allclose(system.prev_hand_motor, torch.ones_like(system.prev_hand_motor) * 0.5)\n'
DOC_FILE = "# Sleep entry zero prev motor state\n\nWhen entering full sleep/replay mode, the previous awake motor command can still\nbe stored in:\n\n```text\nprev_embodied_action\nprev_hand_motor\n```\n\n`life_runtime.py` uses these values at the very start of the next step before\nthe current step's `sleep_motor_guard` has a chance to run.\n\nThis patch adds a transition guard inside `SleepSensorsMixin.apply_sleep_sensor_state(...)`:\n\n```text\nawake/partial_cut -> sleep\n    zero prev_embodied_action\n    zero prev_hand_motor\n```\n\nIt only fires on a transition into full sleep:\n\n```text\nold_sleep == False\nnew_sleep == True\n```\n\nIt does not fire on partial sensor cuts.\n"
HELPER = '    def _zero_prev_motor_state_on_sleep_entry(self) -> None:\n        """\n        Clear one-step awake motor tail when entering full sleep/replay mode.\n\n        life_runtime reads prev_embodied_action / prev_hand_motor at the very\n        beginning of the next step. Without this reset, the first sleep frame\n        can still execute the last awake command before sleep_motor_guard runs.\n        """\n        zeroed = []\n        norms = {}\n        for attr in ("prev_embodied_action", "prev_hand_motor"):\n            value = getattr(self, attr, None)\n            if not torch.is_tensor(value):\n                continue\n            try:\n                norms[attr] = float(value.detach().float().norm().cpu().item())\n            except Exception:\n                norms[attr] = 0.0\n            setattr(self, attr, torch.zeros_like(value))\n            zeroed.append(attr)\n\n        self._sleep_replay_prev_motor_reset = {\n            "reason": "entered_full_sleep",\n            "zeroed": list(zeroed),\n            "norms": dict(norms),\n        }\n        if zeroed:\n            print(\n                "[sleep_replay] zeroed previous motor tail on sleep entry: "\n                + ", ".join(f"{k}={norms.get(k, 0.0):.4f}" for k in zeroed)\n            )\n\n\n'

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f'[skip] {label}: already present')
        return text
    if old not in text:
        print(f'[warn] {label}: anchor not found')
        return text
    print(f'[ok] {label}')
    return text.replace(old, new, 1)

def patch_sleep_sensors() -> None:
    if not SLEEP_SENSORS.exists():
        raise FileNotFoundError(f'missing {SLEEP_SENSORS}')
    text = SLEEP_SENSORS.read_text(encoding='utf-8')
    old_text = text
    if 'def _zero_prev_motor_state_on_sleep_entry(self)' not in text:
        text = replace_once(text, '    def apply_sleep_sensor_state(self, state: dict) -> bool:\n', HELPER + '    def apply_sleep_sensor_state(self, state: dict) -> bool:\n', 'insert _zero_prev_motor_state_on_sleep_entry')
    else:
        print('[skip] helper already present')
    if 'old_sleep = (not old.get("video", True)' not in text:
        text = replace_once(text, '        old = self.input_sensors_enabled_dict_no_startup_apply()\n', '        old = self.input_sensors_enabled_dict_no_startup_apply()\n        old_sleep = (not old.get("video", True) and not old.get("contact", True) and not old.get("imu", True))\n', 'capture old_sleep')
    else:
        print('[skip] old_sleep already present')
    if 'new_sleep = (not new.get("video", True)' not in text:
        text = replace_once(text, '        new = self.input_sensors_enabled_dict_no_startup_apply()\n        changed = old != new\n', '        new = self.input_sensors_enabled_dict_no_startup_apply()\n        new_sleep = (not new.get("video", True) and not new.get("contact", True) and not new.get("imu", True))\n        changed = old != new\n        if changed and (not old_sleep) and new_sleep:\n            self._zero_prev_motor_state_on_sleep_entry()\n', 'zero prev motor on sleep entry')
    else:
        print('[skip] new_sleep zeroing already present')
    if text != old_text:
        SLEEP_SENSORS.write_text(text, encoding='utf-8')
        print('[ok] wrote sleep_sensors.py')
    else:
        print('[skip] no changes to sleep_sensors.py')

def write_files() -> None:
    files = {
        'tests/module_contracts/test_sleep_entry_zero_prev_motor_contract.py': TEST_FILE,
        'docs/architecture/sleep_entry_zero_prev_motor.md': DOC_FILE,
    }
    for rel, content in files.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding='utf-8') == content:
            print(f'[skip] {rel}: unchanged')
            continue
        target.write_text(content, encoding='utf-8')
        print(f'[ok] wrote {rel}')

def main() -> int:
    patch_sleep_sensors()
    write_files()
    print('\nRun:')
    print('  python -m py_compile src/modules/m06_learning_sleep_consolidation/sleep_sensors.py')
    print('  pytest tests/module_contracts/test_sleep_entry_zero_prev_motor_contract.py')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        raise
