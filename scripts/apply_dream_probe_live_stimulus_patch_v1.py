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
DREAM_PROBE_RUNTIME = 'from __future__ import annotations\n\n"""\nLive diagnostic stimulus for sleep/replay debugging.\n\nThis is not a training fixture and not a MuJoCo simulator. It is a short-lived\nruntime probe controlled by M8 IPC buttons. The probe perturbs internal M5/M11\ninputs so the Sleep Replay Monitor can show visible deltas.\n\nRules:\n    - does not feed raw M1 into M2\n    - does not mutate out["focus_context"] directly\n    - can create a temporary M2/M5 seed through the existing seed bus\n"""\n\nfrom typing import Any, Dict, Optional\n\nimport torch\n\n\ndef _clamp(x: Any, lo: float, hi: float, default: float) -> float:\n    try:\n        v = float(x)\n    except Exception:\n        v = float(default)\n    return max(float(lo), min(float(hi), v))\n\n\ndef _device_from_out(out: Optional[Dict[str, Any]]) -> torch.device:\n    if isinstance(out, dict):\n        for v in out.values():\n            if torch.is_tensor(v):\n                return v.device\n            if isinstance(v, dict):\n                for vv in v.values():\n                    if torch.is_tensor(vv):\n                        return vv.device\n    return torch.device("cpu")\n\n\ndef _scalar_tensor(value: float, device: torch.device) -> torch.Tensor:\n    return torch.tensor([[float(value)]], dtype=torch.float32, device=device)\n\n\ndef _scalar(x: Any, default: float = 0.0) -> float:\n    try:\n        if torch.is_tensor(x):\n            if x.numel() == 0:\n                return float(default)\n            return float(x.detach().float().reshape(-1)[0].cpu().item())\n        if x is None:\n            return float(default)\n        return float(x)\n    except Exception:\n        return float(default)\n\n\nclass DreamProbeRuntimeMixin:\n    def request_dream_probe(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:\n        payload = dict(payload or {})\n        kind = str(payload.get("kind", payload.get("probe", "curiosity"))).lower().strip()\n        if kind in ("stop", "clear", "off", "none"):\n            self._dream_probe_state = {\n                "active": False,\n                "kind": "clear",\n                "remaining": 0,\n                "duration": 0,\n                "intensity": 0.0,\n                "source": str(payload.get("source", "ipc")),\n                "started_step": int(getattr(self, "global_step", 0)),\n            }\n            print("[dream_probe] cleared")\n            if hasattr(self, "write_module_debug_status"):\n                self.write_module_debug_status()\n            return dict(self._dream_probe_state)\n\n        duration = max(1, int(_clamp(payload.get("duration", 60), 1, 500, 60)))\n        intensity = _clamp(payload.get("intensity", 0.75), 0.0, 1.5, 0.75)\n\n        if kind in ("replay", "seed", "replay_seed", "m5_seed"):\n            kind = "replay_seed"\n        elif kind in ("stress", "panic", "fear", "uncertainty"):\n            kind = "stress"\n        elif kind in ("mixed", "dream", "pulse"):\n            kind = "mixed"\n        else:\n            kind = "curiosity"\n\n        state = {\n            "active": True,\n            "kind": kind,\n            "remaining": int(duration),\n            "duration": int(duration),\n            "intensity": float(intensity),\n            "source": str(payload.get("source", "ipc")),\n            "started_step": int(getattr(self, "global_step", 0)),\n            "last_pulse": 0.0,\n        }\n        self._dream_probe_state = state\n\n        if kind == "replay_seed":\n            self._inject_probe_replay_seed(float(intensity), source="dream_probe_request")\n\n        print(\n            f"[dream_probe] requested kind={kind} intensity={intensity:.3f} "\n            f"duration={duration} source={state[\'source\']}"\n        )\n        if hasattr(self, "write_module_debug_status"):\n            self.write_module_debug_status()\n        return dict(state)\n\n    def _inject_probe_replay_seed(self, intensity: float, *, source: str = "dream_probe") -> None:\n        latest = getattr(self, "latest_out", {}) or {}\n        device = _device_from_out(latest)\n        seed = None\n        for key in ("focus_context", "workspace_out", "obs_embed"):\n            value = latest.get(key) if isinstance(latest, dict) else None\n            if torch.is_tensor(value):\n                seed = value.detach().clone().float()\n                break\n        if seed is None:\n            seed = torch.zeros(1, 256, dtype=torch.float32, device=device)\n            seed[:, 0] = float(intensity)\n        if seed.ndim == 1:\n            seed = seed.unsqueeze(0)\n        if seed.shape[-1] != 256:\n            fixed = torch.zeros(seed.shape[0], 256, dtype=seed.dtype, device=seed.device)\n            n = min(int(seed.shape[-1]), 256)\n            fixed[:, :n] = seed[:, :n]\n            seed = fixed\n\n        gate = _scalar_tensor(float(intensity), seed.device)\n        self._event_dream_next_focus_seed = seed\n        self._event_dream_next_focus_gate = gate\n        self._dream_probe_last_seed = {\n            "seed_gate": float(intensity),\n            "seed_norm": float(seed.detach().float().norm().cpu().item()),\n            "source": str(source),\n        }\n\n    def apply_dream_probe_to_out(self, out: Dict[str, Any], obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n        del obs\n        if not isinstance(out, dict):\n            return out\n\n        state = getattr(self, "_dream_probe_state", None)\n        if not isinstance(state, dict) or not bool(state.get("active", False)):\n            return out\n\n        remaining = int(state.get("remaining", 0) or 0)\n        duration = max(1, int(state.get("duration", 1) or 1))\n        if remaining <= 0:\n            state["active"] = False\n            state["remaining"] = 0\n            return out\n\n        intensity = float(state.get("intensity", 0.75) or 0.75)\n        phase = max(0.0, min(1.0, float(remaining) / float(duration)))\n        pulse = float(intensity * phase)\n        kind = str(state.get("kind", "curiosity"))\n\n        device = _device_from_out(out)\n        values = out.setdefault("values", {})\n        if not isinstance(values, dict):\n            values = {}\n            out["values"] = values\n\n        object_imagery = out.setdefault("object_imagery", {})\n        if not isinstance(object_imagery, dict):\n            object_imagery = {}\n            out["object_imagery"] = object_imagery\n\n        reflection = out.setdefault("preconscious_reflection_out", {})\n        if not isinstance(reflection, dict):\n            reflection = {}\n            out["preconscious_reflection_out"] = reflection\n\n        if kind in ("curiosity", "mixed"):\n            current_curiosity = _scalar(values.get("curiosity"), 0.0)\n            values["curiosity"] = _scalar_tensor(max(current_curiosity, pulse), device)\n\n        if kind in ("stress", "mixed"):\n            # Raise uncertainty seen by M11 without bypassing M11:\n            # low coherence/object/self confidence -> higher stress/fear/panic.\n            low_conf = max(0.0, 1.0 - pulse)\n            current_coh = _scalar(values.get("coherence"), 1.0)\n            values["coherence"] = _scalar_tensor(min(current_coh, low_conf), device)\n            object_imagery["object_confidence"] = _scalar_tensor(low_conf, device)\n            reflection["model_confidence"] = _scalar_tensor(low_conf, device)\n            if "self_core" not in out or not isinstance(out.get("self_core"), dict):\n                out["self_core"] = {}\n            out["self_core"]["self_confidence"] = _scalar_tensor(low_conf, device)\n\n        if kind == "replay_seed":\n            self._inject_probe_replay_seed(pulse, source="dream_probe_apply")\n            current_curiosity = _scalar(values.get("curiosity"), 0.0)\n            values["curiosity"] = _scalar_tensor(max(current_curiosity, min(1.0, 0.25 + pulse)), device)\n\n        state["remaining"] = int(max(0, remaining - 1))\n        state["last_pulse"] = float(pulse)\n        state["active"] = bool(state["remaining"] > 0)\n\n        out["dream_probe"] = {\n            "active": bool(state.get("active", False)),\n            "kind": kind,\n            "remaining": int(state.get("remaining", 0)),\n            "duration": int(duration),\n            "intensity": float(intensity),\n            "pulse": float(pulse),\n            "source": str(state.get("source", "")),\n        }\n        self._dream_probe_state = state\n        return out\n\n\n__all__ = ["DreamProbeRuntimeMixin"]\n'
TEST_FILE = 'from __future__ import annotations\n\nimport torch\n\n\nclass DummyProbeSystem:\n    from src.modules.m02_event_dream_replay.dream_probe_runtime import DreamProbeRuntimeMixin\n\n    request_dream_probe = DreamProbeRuntimeMixin.request_dream_probe\n    apply_dream_probe_to_out = DreamProbeRuntimeMixin.apply_dream_probe_to_out\n    _inject_probe_replay_seed = DreamProbeRuntimeMixin._inject_probe_replay_seed\n\n    def __init__(self):\n        self.global_step = 10\n        self.latest_out = {\n            "focus_context": torch.ones(1, 256) * 0.25,\n        }\n        self.status_writes = 0\n\n    def write_module_debug_status(self):\n        self.status_writes += 1\n\n\ndef test_curiosity_probe_changes_values_without_focus_mutation():\n    sys = DummyProbeSystem()\n    sys.request_dream_probe({"kind": "curiosity", "intensity": 0.8, "duration": 3})\n\n    out = {\n        "values": {\n            "curiosity": torch.tensor([[0.1]]),\n            "coherence": torch.tensor([[0.9]]),\n        },\n        "focus_context": torch.zeros(1, 256),\n    }\n    old_focus = out["focus_context"].clone()\n    out2 = sys.apply_dream_probe_to_out(out)\n\n    assert out2["dream_probe"]["kind"] == "curiosity"\n    assert float(out2["values"]["curiosity"].item()) >= 0.7\n    assert torch.allclose(out2["focus_context"], old_focus)\n    assert sys._dream_probe_state["remaining"] == 2\n\n\ndef test_stress_probe_lowers_confidence_for_m11_uncertainty():\n    sys = DummyProbeSystem()\n    sys.request_dream_probe({"kind": "stress", "intensity": 0.9, "duration": 2})\n\n    out = {\n        "values": {"coherence": torch.tensor([[0.95]])},\n        "object_imagery": {"object_confidence": torch.tensor([[0.95]])},\n        "preconscious_reflection_out": {"model_confidence": torch.tensor([[0.95]])},\n    }\n    out2 = sys.apply_dream_probe_to_out(out)\n\n    assert out2["dream_probe"]["kind"] == "stress"\n    assert float(out2["values"]["coherence"].item()) < 0.2\n    assert float(out2["object_imagery"]["object_confidence"].item()) < 0.2\n    assert float(out2["preconscious_reflection_out"]["model_confidence"].item()) < 0.2\n\n\ndef test_replay_seed_probe_uses_existing_m5_seed_bus():\n    sys = DummyProbeSystem()\n    state = sys.request_dream_probe({"kind": "replay_seed", "intensity": 0.6, "duration": 4})\n\n    assert state["kind"] == "replay_seed"\n    assert torch.is_tensor(sys._event_dream_next_focus_seed)\n    assert torch.is_tensor(sys._event_dream_next_focus_gate)\n    assert tuple(sys._event_dream_next_focus_seed.shape) == (1, 256)\n    assert float(sys._event_dream_next_focus_gate.item()) == 0.6\n    assert sys.status_writes == 1\n'
DOC_FILE = '# Dream Probe / Replay Probe\n\nAdds live diagnostic stimulus buttons for sleep/replay debugging.\n\nThis is not a permanent model feature. It is a runtime diagnostic probe:\n\n```text\nM8 Sleep Replay Monitor\n    Probe curiosity\n    Probe stress\n    Probe replay seed\n    Probe mixed\n    Clear probe\n```\n\nIPC action:\n\n```text\ndream_probe_inject\n```\n\nPayload examples:\n\n```json\n{"kind": "curiosity", "intensity": 0.85, "duration": 80}\n{"kind": "stress", "intensity": 0.85, "duration": 80}\n{"kind": "replay_seed", "intensity": 0.75, "duration": 60}\n{"kind": "mixed", "intensity": 0.75, "duration": 80}\n{"kind": "clear"}\n```\n\nRules:\n\n```text\n- no raw M1 → M2 path\n- no direct out["focus_context"] mutation\n- stress/curiosity probe affects M11 inputs before EmotionalDrive.compute(...)\n- replay_seed probe uses the existing M2/M5 seed bus:\n  _event_dream_next_focus_seed + _event_dream_next_focus_gate\n```\n'

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f'[skip] {label}: already present')
        return text
    if old not in text:
        print(f'[warn] {label}: anchor not found')
        return text
    print(f'[ok] {label}')
    return text.replace(old, new, 1)

def write_files() -> None:
    files = {
        'src/modules/m02_event_dream_replay/dream_probe_runtime.py': DREAM_PROBE_RUNTIME,
        'tests/module_contracts/test_dream_probe_runtime_contract.py': TEST_FILE,
        'docs/architecture/dream_probe_live_stimulus.md': DOC_FILE,
    }
    for rel, content in files.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding='utf-8') == content:
            print(f'[skip] {rel}: unchanged')
            continue
        target.write_text(content, encoding='utf-8')
        print(f'[ok] wrote {rel}')

def patch_runner() -> None:
    path = ROOT / 'src/apps/runner.py'
    if not path.exists():
        print('[warn] runner.py missing')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    if 'DreamProbeRuntimeMixin' not in text:
        text = replace_once(text, 'from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin\n', 'from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin\nfrom src.modules.m02_event_dream_replay.dream_probe_runtime import DreamProbeRuntimeMixin\n', 'runner import DreamProbeRuntimeMixin')
        text = replace_once(text, '    EventDreamReplayRuntimeMixin,\n', '    EventDreamReplayRuntimeMixin,\n    DreamProbeRuntimeMixin,\n', 'runner inherit DreamProbeRuntimeMixin')
    else:
        print('[skip] runner DreamProbeRuntimeMixin already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')
        print('[ok] wrote runner.py')

def patch_action_runtime() -> None:
    path = ROOT / 'src/modules/m03_self_action_causality/action_runtime.py'
    if not path.exists():
        print('[warn] action_runtime.py missing')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    block = '''        elif action in ("dream_probe_inject", "dream_probe_clear"):\n            if hasattr(self, "request_dream_probe"):\n                if action == "dream_probe_clear":\n                    payload = {"kind": "clear", **dict(payload or {})}\n                self.request_dream_probe(payload)\n            else:\n                print("[dream_probe] ignored: DreamProbeRuntimeMixin is not installed")\n'''
    if 'dream_probe_inject' not in text:
        text = replace_once(text, '        elif action == "module_lab_run":\n', block + '        elif action == "module_lab_run":\n', 'action_runtime dream_probe action')
    else:
        print('[skip] action_runtime dream_probe already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')
        print('[ok] wrote action_runtime.py')

def patch_life_runtime() -> None:
    path = ROOT / 'src/apps/life_runtime.py'
    if not path.exists():
        print('[warn] life_runtime.py missing')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    insert = '''        if hasattr(self, "apply_dream_probe_to_out"):\n            try:\n                out = self.apply_dream_probe_to_out(out, obs)\n            except Exception as e:\n                if not hasattr(self, "_dream_probe_warned"):\n                    print(f"[dream_probe] apply skipped: {e}")\n                    self._dream_probe_warned = True\n\n'''
    if 'apply_dream_probe_to_out' not in text:
        text = replace_once(text, '        emotion = self.emotional_drive.compute(out, obs)\n', insert + '        emotion = self.emotional_drive.compute(out, obs)\n', 'life_runtime apply dream probe before M11')
    else:
        print('[skip] life_runtime dream probe already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')
        print('[ok] wrote life_runtime.py')

def patch_monitor_status() -> None:
    path = ROOT / 'src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py'
    if not path.exists():
        print('[warn] sleep_replay_monitor_status.py missing; apply sleep_replay_monitor patch first')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    if '"dream_probe":' not in text:
        text = replace_once(text, '        "trace_present": bool(trace),\n', '        "dream_probe": dict(out.get("dream_probe", getattr(system, "_dream_probe_state", {}) or {}) or {}),\n        "trace_present": bool(trace),\n', 'monitor status add dream_probe')
    else:
        print('[skip] monitor status dream_probe already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')
        print('[ok] wrote sleep_replay_monitor_status.py')

def patch_control_panel() -> None:
    path = ROOT / 'src/modules/m08_debug_visual_control/control_panel.py'
    if not path.exists():
        print('[warn] control_panel.py missing')
        return
    text = path.read_text(encoding='utf-8')
    old_text = text
    probe_methods = '        def request_sleep_replay_probe(self, kind: str, intensity: float = 0.8, duration: int = 80):\n            if not self.state.connected:\n                self.status.setText("STATUS IPC: no signal")\n                self.refresh_ui()\n                return\n            payload_kind = str(kind)\n            ok = self.send(make_action_message(\n                "dream_probe_inject",\n                kind=payload_kind,\n                intensity=float(intensity),\n                duration=int(duration),\n                source="m8_sleep_replay_monitor",\n            ))\n            self.status.setText(\n                f"Dream probe requested: {payload_kind}" if ok else "Dream probe request failed"\n            )\n            self.refresh_ui()\n\n'
    if 'def request_sleep_replay_probe(self, kind: str' not in text:
        text = replace_once(text, '        def open_sleep_replay_monitor_window(self):\n', probe_methods + '        def open_sleep_replay_monitor_window(self):\n', 'control panel add request_sleep_replay_probe')
    else:
        print('[skip] request_sleep_replay_probe already present')
    probe_button_block = '            probe_row = QtWidgets.QHBoxLayout()\n            probe_row.setSpacing(8)\n            btn_probe_curiosity = QtWidgets.QPushButton("Probe curiosity")\n            btn_probe_stress = QtWidgets.QPushButton("Probe stress")\n            btn_probe_replay = QtWidgets.QPushButton("Probe replay seed")\n            btn_probe_mixed = QtWidgets.QPushButton("Probe mixed")\n            btn_probe_clear = QtWidgets.QPushButton("Clear probe")\n            for b in [btn_probe_curiosity, btn_probe_stress, btn_probe_replay, btn_probe_mixed, btn_probe_clear]:\n                b.setMinimumHeight(34)\n                probe_row.addWidget(b)\n            main.addLayout(probe_row)\n\n'
    if 'Probe curiosity' not in text:
        text = replace_once(text, '            main.addWidget(header)\n', '            main.addWidget(header)\n\n' + probe_button_block, 'control panel add probe buttons')
    else:
        print('[skip] probe buttons already present')
    if 'dream_probe.kind' not in text:
        text = replace_once(text, '            row3.addWidget(add_box("M5 seed + M3 guard", [\n', '            row3.addWidget(add_box("Dream probe", [\n                ("active", "dream_probe.active"),\n                ("kind", "dream_probe.kind"),\n                ("remaining", "dream_probe.remaining"),\n                ("pulse", "dream_probe.pulse"),\n                ("intensity", "dream_probe.intensity"),\n            ]))\n            row3.addWidget(add_box("M5 seed + M3 guard", [\n', 'control panel add dream_probe box')
    else:
        print('[skip] dream_probe box already present')
    probe_connections = '            btn_probe_curiosity.clicked.connect(lambda: self.request_sleep_replay_probe("curiosity", 0.85, 80))\n            btn_probe_stress.clicked.connect(lambda: self.request_sleep_replay_probe("stress", 0.85, 80))\n            btn_probe_replay.clicked.connect(lambda: self.request_sleep_replay_probe("replay_seed", 0.75, 60))\n            btn_probe_mixed.clicked.connect(lambda: self.request_sleep_replay_probe("mixed", 0.75, 80))\n            btn_probe_clear.clicked.connect(lambda: self.request_sleep_replay_probe("clear", 0.0, 1))\n'
    if 'btn_probe_curiosity.clicked.connect' not in text:
        text = replace_once(text, '            btn_close.clicked.connect(dialog.close)\n', probe_connections + '            btn_close.clicked.connect(dialog.close)\n', 'control panel probe button signals')
    else:
        print('[skip] probe button signals already present')
    if text != old_text:
        path.write_text(text, encoding='utf-8')
        print('[ok] wrote control_panel.py')

def main() -> int:
    write_files()
    patch_runner()
    patch_action_runtime()
    patch_life_runtime()
    patch_monitor_status()
    patch_control_panel()
    print('\nRun:')
    print('  python -m py_compile src/modules/m02_event_dream_replay/dream_probe_runtime.py')
    print('  python -m py_compile src/apps/runner.py')
    print('  python -m py_compile src/modules/m03_self_action_causality/action_runtime.py')
    print('  python -m py_compile src/apps/life_runtime.py')
    print('  python -m py_compile src/modules/m08_debug_visual_control/sleep_replay_monitor_status.py')
    print('  python -m py_compile src/modules/m08_debug_visual_control/control_panel.py')
    print('  pytest tests/module_contracts/test_dream_probe_runtime_contract.py')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        raise
