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
STATUS_FILE = ROOT / 'src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py'
CONTROL_FILE = ROOT / 'src/modules/m08_debug_visual_control/control_panel.py'
TEST_FILE = 'from __future__ import annotations\n\nimport torch\n\n\nclass DummySystem:\n    def __init__(self):\n        self.global_step = 1\n        self.video_sensor_enabled = False\n        self.contact_sensor_enabled = False\n        self.imu_sensor_enabled = False\n        self.latest_out = {\n            "affect": {\n                "stress_latent": torch.tensor([[0.10]]),\n                "panic_latent": torch.tensor([[0.05]]),\n                "curiosity_latent": torch.tensor([[0.20]]),\n            },\n            "event_dream_replay": {\n                "replay_context": torch.ones(1, 256),\n                "replay_gate": torch.tensor([[0.1]]),\n            },\n            "sleep_motor_guard": {"blocked": True},\n        }\n\n    def is_full_sleep_mode(self):\n        return True\n\n    def sensor_state_label(self):\n        return "sleep"\n\n    def input_sensors_enabled_dict(self):\n        return {"video": False, "contact": False, "imu": False}\n\n\ndef test_m11_affect_visibility_adds_deltas_and_ranges():\n    from src.modules.m08_debug_visual_control.sleep_replay_monitor_status import build_sleep_replay_monitor_status\n\n    sys = DummySystem()\n    first = build_sleep_replay_monitor_status(sys)\n    assert "m11_delta" in first\n    assert "m11_range" in first\n    assert "m11_activity" in first\n    assert first["m11_activity"]["samples"] == 1\n\n    sys.global_step = 2\n    sys.latest_out["affect"]["stress_latent"] = torch.tensor([[0.14]])\n    sys.latest_out["affect"]["panic_latent"] = torch.tensor([[0.08]])\n    sys.latest_out["affect"]["curiosity_latent"] = torch.tensor([[0.30]])\n\n    second = build_sleep_replay_monitor_status(sys)\n\n    assert second["m11_delta"]["stress"] > 0\n    assert second["m11_delta"]["panic"] > 0\n    assert second["m11_delta"]["curiosity"] > 0\n    assert second["m11_activity"]["change_score"] > 0\n    assert second["m11_activity"]["trend"] in ("↑", "↓", "→", "↕")\n    assert second["m11_range"]["stress_min"] <= second["m11_range"]["stress_max"]\n'
DOC_FILE = '# M11 affect visibility in Sleep Replay Monitor\n\nM11 affect can change very slowly because `EmotionalDriveConfig.ema_decay` is high\nand live values are rounded in the UI. This does not mean M11 is inactive.\n\nThis patch adds monitor-only diagnostics:\n\n```text\nm11_delta:\n    valence_delta\n    arousal_delta\n    stress_delta\n    panic_delta\n    curiosity_delta\n\nm11_range:\n    stress_min / stress_max\n    panic_min / panic_max\n    curiosity_min / curiosity_max\n\nm11_activity:\n    change_score\n    trend\n    samples\n```\n\nIt does not change M11 emotional dynamics. It only makes small changes visible.\n'

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f'[skip] {label}: already present')
        return text
    if old not in text:
        print(f'[warn] {label}: anchor not found')
        return text
    print(f'[ok] {label}')
    return text.replace(old, new, 1)

def patch_status_file() -> None:
    if not STATUS_FILE.exists():
        raise FileNotFoundError(f'missing {STATUS_FILE}')
    text = STATUS_FILE.read_text(encoding='utf-8')
    old_text = text
    helper = '\ndef _trend(delta: float, eps: float = 0.002) -> str:\n    if delta > eps:\n        return "↑"\n    if delta < -eps:\n        return "↓"\n    return "→"\n\n\ndef _init_affect_visibility_state(system: Any) -> Dict[str, Any]:\n    state = getattr(system, "_sleep_replay_m11_visibility_state", None)\n    if not isinstance(state, dict):\n        state = {\n            "prev": {},\n            "mins": {},\n            "maxs": {},\n            "samples": 0,\n        }\n        setattr(system, "_sleep_replay_m11_visibility_state", state)\n    return state\n\n\ndef _attach_m11_visibility(system: Any, payload: Dict[str, Any]) -> None:\n    m11 = payload.get("m11", {})\n    if not isinstance(m11, dict):\n        return\n\n    keys = ("valence", "arousal", "stress", "panic", "curiosity")\n    current = {k: float(m11.get(k, 0.0) or 0.0) for k in keys}\n\n    state = _init_affect_visibility_state(system)\n    prev = state.get("prev", {}) if isinstance(state.get("prev"), dict) else {}\n    mins = state.get("mins", {}) if isinstance(state.get("mins"), dict) else {}\n    maxs = state.get("maxs", {}) if isinstance(state.get("maxs"), dict) else {}\n    samples = int(state.get("samples", 0) or 0) + 1\n\n    delta = {}\n    for key, value in current.items():\n        old = float(prev.get(key, value))\n        delta[key] = float(value - old)\n        mins[key] = min(float(mins.get(key, value)), value)\n        maxs[key] = max(float(maxs.get(key, value)), value)\n\n    state["prev"] = dict(current)\n    state["mins"] = dict(mins)\n    state["maxs"] = dict(maxs)\n    state["samples"] = samples\n\n    affect_keys = ("stress", "panic", "curiosity")\n    change_score = float(sum(abs(delta[k]) for k in affect_keys))\n    signed_score = float(sum(delta[k] for k in affect_keys))\n    if change_score < 0.002:\n        trend = "→"\n    elif signed_score > 0.002:\n        trend = "↑"\n    elif signed_score < -0.002:\n        trend = "↓"\n    else:\n        trend = "↕"\n\n    payload["m11_delta"] = {\n        "valence": float(delta["valence"]),\n        "arousal": float(delta["arousal"]),\n        "stress": float(delta["stress"]),\n        "panic": float(delta["panic"]),\n        "curiosity": float(delta["curiosity"]),\n    }\n    payload["m11_range"] = {\n        "stress_min": float(mins.get("stress", current["stress"])),\n        "stress_max": float(maxs.get("stress", current["stress"])),\n        "panic_min": float(mins.get("panic", current["panic"])),\n        "panic_max": float(maxs.get("panic", current["panic"])),\n        "curiosity_min": float(mins.get("curiosity", current["curiosity"])),\n        "curiosity_max": float(maxs.get("curiosity", current["curiosity"])),\n    }\n    payload["m11_activity"] = {\n        "change_score": change_score,\n        "trend": trend,\n        "stress_trend": _trend(delta["stress"]),\n        "panic_trend": _trend(delta["panic"]),\n        "curiosity_trend": _trend(delta["curiosity"]),\n        "samples": samples,\n    }\n\n'
    if 'def _attach_m11_visibility(system: Any, payload: Dict[str, Any]) -> None:' not in text:
        text = replace_once(text, 'def build_sleep_replay_monitor_status(system: Any) -> Dict[str, Any]:\n', helper + '\ndef build_sleep_replay_monitor_status(system: Any) -> Dict[str, Any]:\n', 'insert M11 visibility helpers')
    else:
        print('[skip] M11 visibility helpers already present')
    if '_attach_m11_visibility(system, payload)' not in text:
        text = replace_once(text, '    return {\n        "global_step": int(getattr(system, "global_step", 0)),\n', '    payload = {\n        "global_step": int(getattr(system, "global_step", 0)),\n', 'convert return dict to payload dict')
        text = replace_once(text, '        "trace_present": bool(trace),\n    }\n', '        "trace_present": bool(trace),\n    }\n    _attach_m11_visibility(system, payload)\n    return payload\n', 'attach M11 visibility before return')
    else:
        print('[skip] _attach_m11_visibility call already present')
    if text != old_text:
        STATUS_FILE.write_text(text, encoding='utf-8')
        print('[ok] wrote sleep_replay_monitor_status.py')
    else:
        print('[skip] no status file changes')

def patch_control_panel() -> None:
    if not CONTROL_FILE.exists():
        print('[warn] control_panel.py missing')
        return
    text = CONTROL_FILE.read_text(encoding='utf-8')
    old_text = text
    if 'm11_delta.stress' not in text:
        text = replace_once(text, '                ("curiosity", "m11.curiosity"),\n            ]))\n            main.addLayout(row1)\n', '                ("curiosity", "m11.curiosity"),\n                ("Δ stress", "m11_delta.stress"),\n                ("Δ panic", "m11_delta.panic"),\n                ("Δ curiosity", "m11_delta.curiosity"),\n                ("trend", "m11_activity.trend"),\n                ("change_score", "m11_activity.change_score"),\n            ]))\n            main.addLayout(row1)\n', 'add M11 delta rows to monitor')
    else:
        print('[skip] M11 delta rows already present')
    if 'M11 min/max' not in text:
        # Add compact min/max into the raw title area by extending M11 box title if exact old title is present.
        text = text.replace('row1.addWidget(add_box("M11 affect", [', 'row1.addWidget(add_box("M11 affect + Δ / trend", [', 1)
    if text != old_text:
        CONTROL_FILE.write_text(text, encoding='utf-8')
        print('[ok] wrote control_panel.py')
    else:
        print('[skip] no control_panel changes')

def write_files() -> None:
    files = {
        'tests/module_contracts/test_m11_affect_visibility_monitor_contract.py': TEST_FILE,
        'docs/architecture/m11_affect_visibility_monitor.md': DOC_FILE,
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
    patch_status_file()
    patch_control_panel()
    write_files()
    print('\nRun:')
    print('  python -m py_compile src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py')
    print('  python -m py_compile src/modules/m08_debug_visual_control/control_panel.py')
    print('  pytest tests/module_contracts/test_m11_affect_visibility_monitor_contract.py')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        raise
