#!/usr/bin/env python3
from __future__ import annotations

"""Install unconscious live-debug runtime hooks and behavioral scenarios."""

from pathlib import Path
import sys


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").exists() and ((candidate / ".git").exists() or (candidate / "config").exists()):
            return candidate
    return Path.cwd().resolve()


ROOT = find_repo_root()
FILES = {}

FILES['src/modules/m02_event_dream_replay/unconscious_loop_trace.py'] = 'from __future__ import annotations\n\n"""\nLive trace for the strictly unconscious DMoC loop.\n\nTarget architecture:\n\n    M1 -> M5 -> M11 -> M2 -> FocusFeedbackBoundary -> M5\n          |             ^\n          v             |\n         M4 ------------+\n                       M13 -> M2\n\nThis file only observes runtime packets. It does not change architecture.\n"""\n\nfrom typing import Any, Dict, Optional\n\nimport torch\n\n\ndef _scalar(x: Any, default: float = 0.0) -> float:\n    try:\n        if torch.is_tensor(x):\n            if x.numel() == 0:\n                return float(default)\n            return float(x.detach().float().reshape(-1)[0].cpu().item())\n        if x is None:\n            return float(default)\n        return float(x)\n    except Exception:\n        return float(default)\n\n\ndef _norm(x: Any, default: float = 0.0) -> float:\n    try:\n        if torch.is_tensor(x):\n            if x.numel() == 0:\n                return float(default)\n            return float(x.detach().float().norm(dim=-1).reshape(-1)[0].cpu().item())\n        return float(default)\n    except Exception:\n        return float(default)\n\n\ndef build_unconscious_loop_trace_packet(out: Dict, *, sleep_mode: bool = False, sensor_state: str = "") -> Dict[str, Any]:\n    emotion = out.get("emotion", {}) if isinstance(out.get("emotion"), dict) else {}\n    affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}\n    m13 = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}\n    m4 = out.get("long_dynamic_memory", {}) if isinstance(out.get("long_dynamic_memory"), dict) else {}\n    m2 = out.get("event_dream_replay", {}) if isinstance(out.get("event_dream_replay"), dict) else {}\n    m5_feedback = out.get("focus_feedback", {}) if isinstance(out.get("focus_feedback"), dict) else {}\n    attention = out.get("attention", {}) if isinstance(out.get("attention"), dict) else {}\n    sleep_motor = out.get("sleep_motor_guard", {}) if isinstance(out.get("sleep_motor_guard"), dict) else {}\n\n    seed = m2.get("next_focus_context_seed", m2.get("replay_context"))\n    seed_gate = m2.get("next_focus_context_seed_gate", m2.get("replay_gate"))\n\n    return {\n        "sleep": bool(sleep_mode),\n        "sensor_state": str(sensor_state),\n        "m11": {\n            "valence": _scalar(emotion.get("emotional_valence"), _scalar(affect.get("valence"), 0.0)),\n            "arousal": _scalar(emotion.get("emotional_arousal"), _scalar(affect.get("arousal"), 0.0)),\n            "stress": _scalar(affect.get("stress_latent"), 0.0),\n            "panic": _scalar(affect.get("panic_latent"), 0.0),\n            "curiosity": _scalar(affect.get("curiosity_latent"), 0.0),\n            "comfort": _scalar(affect.get("comfort_latent"), 0.0),\n            "relief": _scalar(affect.get("relief_latent"), 0.0),\n        },\n        "m13": {\n            "relevance": _scalar(m13.get("retrieval_relevance"), 0.0),\n            "episodes": _scalar(m13.get("episode_count", m13.get("retrieved_episode_count")), 0.0),\n            "summary": str(m13.get("summary", m13.get("last_summary", ""))),\n        },\n        "m4": {\n            "token": str(m4.get("identity_token", "")),\n            "gate": _scalar(m4.get("dynamic_memory_gate"), 0.0),\n            "stability": _scalar(m4.get("identity_stability"), 0.0),\n            "novelty": _scalar(m4.get("identity_novelty"), 0.0),\n            "sentence": str(m4.get("selected_sentence", "")),\n        },\n        "m2": {\n            "replay_gate": _scalar(m2.get("replay_gate"), 0.0),\n            "should_replay": _scalar(m2.get("should_replay"), 0.0),\n            "dream_pressure": _scalar(m2.get("dream_pressure"), 0.0),\n            "event_salience": _scalar(m2.get("event_salience"), 0.0),\n            "source": str(m2.get("replay_source", "")),\n            "identity": str(m2.get("selected_identity_token", "")),\n        },\n        "m5_seed": {\n            "seed_norm": _norm(seed, 0.0),\n            "seed_gate": _scalar(seed_gate, 0.0),\n            "feedback_gate": _scalar(\n                m5_feedback.get("total_gate", attention.get("focus_feedback_gate")),\n                0.0,\n            ),\n            "feedback_active": _scalar(m5_feedback.get("active"), 0.0),\n        },\n        "m3": {\n            "sleep_blocked": bool(sleep_motor.get("blocked", False)),\n            "stage": str(sleep_motor.get("stage", "")),\n            "blocked_norm": _scalar(sleep_motor.get("blocked_motor_norm"), 0.0),\n        },\n    }\n\n\ndef format_unconscious_loop_trace(packet: Dict[str, Any], *, step: int = 0) -> str:\n    m11 = packet["m11"]\n    m13 = packet["m13"]\n    m4 = packet["m4"]\n    m2 = packet["m2"]\n    seed = packet["m5_seed"]\n    m3 = packet["m3"]\n    return (\n        f"[unconscious_loop step={int(step)}] "\n        f"sleep={int(packet[\'sleep\'])} state={packet[\'sensor_state\']} | "\n        f"m11: val={m11[\'valence\']:.2f} ar={m11[\'arousal\']:.2f} "\n        f"stress={m11[\'stress\']:.2f} panic={m11[\'panic\']:.2f} cur={m11[\'curiosity\']:.2f} | "\n        f"m13: rel={m13[\'relevance\']:.2f} eps={m13[\'episodes\']:.0f} | "\n        f"m4: token={m4[\'token\']} gate={m4[\'gate\']:.2f} stab={m4[\'stability\']:.2f} nov={m4[\'novelty\']:.2f} | "\n        f"m2: gate={m2[\'replay_gate\']:.2f} should={m2[\'should_replay\']:.0f} "\n        f"pressure={m2[\'dream_pressure\']:.2f} sal={m2[\'event_salience\']:.2f} src={m2[\'source\']} | "\n        f"m5_seed: gate={seed[\'seed_gate\']:.2f} norm={seed[\'seed_norm\']:.2f} fb={seed[\'feedback_gate\']:.3f} | "\n        f"m3_sleep_block={int(m3[\'sleep_blocked\'])}"\n    )\n\n\nclass UnconsciousLoopTraceRuntimeMixin:\n    def _unconscious_trace_cfg(self) -> Any:\n        return getattr(getattr(self, "cfg", None), "event_dream_replay", None)\n\n    def _unconscious_trace_enabled(self) -> bool:\n        cfg = self._unconscious_trace_cfg()\n        return bool(getattr(cfg, "unconscious_trace_enabled", True))\n\n    def build_unconscious_loop_trace_packet(self, out: Dict, obs: Optional[Dict] = None) -> Dict[str, Any]:\n        del obs\n        sleep_mode = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False\n        try:\n            state = self.sensor_state_label() if hasattr(self, "sensor_state_label") else ""\n        except Exception:\n            state = ""\n        packet = build_unconscious_loop_trace_packet(out, sleep_mode=sleep_mode, sensor_state=state)\n        out["unconscious_loop_trace"] = packet\n        return packet\n\n    def maybe_print_unconscious_loop_trace(self, out: Dict, obs: Optional[Dict] = None) -> None:\n        if not self._unconscious_trace_enabled():\n            return\n        cfg = self._unconscious_trace_cfg()\n        every = int(getattr(cfg, "unconscious_trace_every_steps", getattr(cfg, "print_every_steps", 30)))\n        if every <= 0:\n            return\n        step = int(getattr(self, "global_step", 0))\n        if step % every != 0:\n            return\n        packet = self.build_unconscious_loop_trace_packet(out, obs)\n        print(format_unconscious_loop_trace(packet, step=step))\n\n\n__all__ = [\n    "UnconsciousLoopTraceRuntimeMixin",\n    "build_unconscious_loop_trace_packet",\n    "format_unconscious_loop_trace",\n]\n'

FILES['src/modules/m03_self_action_causality/sleep_motor_guard.py'] = 'from __future__ import annotations\n\n"""\nMotor guard for sleep/replay mode.\n\nDuring full sleep:\n    - M5 may still imagine/propose actions internally.\n    - M3/body must not execute those actions in the external MuJoCo world.\n\nThis guard zeros executable motor tensors while preserving proposed actions under\n`out["sleep_motor_guard"]["imagined_*"]` for debugging.\n"""\n\nfrom typing import Any, Dict\n\nimport torch\n\n\nMOTOR_KEYS = (\n    "embodied_targets",\n    "hand_ctrl",\n    "leg_ctrl",\n)\n\n\ndef _norm(x: Any) -> float:\n    try:\n        if torch.is_tensor(x):\n            return float(x.detach().float().norm().cpu().item())\n        return 0.0\n    except Exception:\n        return 0.0\n\n\ndef block_motor_outputs_for_sleep(out: Dict, *, sleep_mode: bool, stage: str = "") -> Dict:\n    if not isinstance(out, dict):\n        return out\n\n    packet = {\n        "blocked": bool(sleep_mode),\n        "stage": str(stage),\n        "blocked_motor_norm": 0.0,\n        "blocked_keys": [],\n    }\n\n    if not bool(sleep_mode):\n        packet["reason"] = "awake_or_partial_sensor_cut"\n        out["sleep_motor_guard"] = packet\n        return out\n\n    total_norm = 0.0\n    for key in MOTOR_KEYS:\n        value = out.get(key)\n        if torch.is_tensor(value):\n            out[f"imagined_{key}"] = value.detach().clone()\n            total_norm += _norm(value)\n            out[key] = torch.zeros_like(value)\n            packet["blocked_keys"].append(key)\n\n    # Keep action proposal visible, but mark it non-executable.\n    if torch.is_tensor(out.get("action_ids")):\n        out["imagined_action_ids"] = out["action_ids"].detach().clone()\n\n    packet["blocked_motor_norm"] = float(total_norm)\n    packet["reason"] = "full_sleep_mode_blocks_external_motor_execution"\n    out["sleep_motor_guard"] = packet\n    return out\n\n\nclass SleepMotorGuardRuntimeMixin:\n    def apply_sleep_motor_guard(self, out: Dict, *, stage: str = "") -> Dict:\n        sleep_mode = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False\n        return block_motor_outputs_for_sleep(out, sleep_mode=sleep_mode, stage=stage)\n\n\n__all__ = [\n    "MOTOR_KEYS",\n    "SleepMotorGuardRuntimeMixin",\n    "block_motor_outputs_for_sleep",\n]\n'

FILES['scripts/module_lab/scenario_unconscious_replay.py'] = 'from __future__ import annotations\n\n"""\nBehavioral scenarios for the unconscious sleep/replay loop.\n\nRun:\n    python scripts/module_lab/scenario_unconscious_replay.py --json\n"""\n\nimport argparse\nimport json\nfrom typing import Any, Dict\n\nimport torch\n\nfrom scripts.module_lab.module_fixture_factory import (\n    FakePassportManager,\n    make_fake_event_memory,\n    make_fake_inner_object,\n    make_fake_m13_memory,\n    make_fake_m4_identity,\n    make_fake_m5_out,\n    scalar,\n)\n\n\ndef f(x: Any) -> float:\n    if torch.is_tensor(x):\n        return float(x.detach().reshape(-1)[0].cpu().item())\n    try:\n        return float(x)\n    except Exception:\n        return 0.0\n\n\ndef compute_m2_packet(*, panic: float, stress: float, curiosity: float, m13_relevance: float, m4_gate: float, dream_mode: bool = True) -> Dict[str, Any]:\n    from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig\n\n    out = make_fake_m5_out(curiosity=curiosity)\n    out["affect"] = {\n        "panic_latent": scalar(panic),\n        "stress_latent": scalar(stress),\n        "curiosity_latent": scalar(curiosity),\n        "expected_affect_delta": scalar(0.10),\n    }\n    out["autobiographical_memory"] = make_fake_m13_memory(relevance=m13_relevance)\n    out["long_dynamic_memory"] = make_fake_m4_identity(gate=m4_gate)\n\n    m2 = EventDreamReplay(EventDreamReplayConfig(\n        replay_context_dim=256,\n        event_code_dim=8,\n        blend_replay_into_focus=False,\n        use_m13_context=True,\n        use_m4_context=True,\n        seed_to_m5_boundary=True,\n    ))\n    return m2.compute(out=out, event_memory=make_fake_event_memory(), dream_mode=dream_mode)\n\n\ndef scenario_calm_no_replay() -> Dict[str, Any]:\n    packet = compute_m2_packet(\n        panic=0.0,\n        stress=0.0,\n        curiosity=0.0,\n        m13_relevance=0.0,\n        m4_gate=0.0,\n        dream_mode=False,\n    )\n    return {\n        "name": "calm_no_replay",\n        "dream_pressure": f(packet["dream_pressure"]),\n        "event_salience": f(packet["event_salience"]),\n        "should_replay": f(packet["should_replay"]),\n        "expect": "low replay pressure",\n        "pass": f(packet["dream_pressure"]) < 0.50,\n    }\n\n\ndef scenario_curiosity_replay() -> Dict[str, Any]:\n    packet = compute_m2_packet(\n        panic=0.1,\n        stress=0.2,\n        curiosity=0.9,\n        m13_relevance=0.7,\n        m4_gate=0.8,\n        dream_mode=True,\n    )\n    return {\n        "name": "curiosity_replay",\n        "dream_pressure": f(packet["dream_pressure"]),\n        "event_salience": f(packet["event_salience"]),\n        "should_replay": f(packet["should_replay"]),\n        "identity": packet.get("selected_identity_token", ""),\n        "expect": "high curiosity + M13/M4 context activates replay",\n        "pass": f(packet["dream_pressure"]) >= 0.35 and f(packet["event_salience"]) >= 0.25,\n    }\n\n\ndef scenario_bad_prediction_dream() -> Dict[str, Any]:\n    packet = compute_m2_packet(\n        panic=0.9,\n        stress=0.9,\n        curiosity=0.3,\n        m13_relevance=0.5,\n        m4_gate=0.6,\n        dream_mode=True,\n    )\n    return {\n        "name": "bad_prediction_dream",\n        "dream_pressure": f(packet["dream_pressure"]),\n        "event_salience": f(packet["event_salience"]),\n        "should_replay": f(packet["should_replay"]),\n        "expect": "panic/stress increases dream pressure",\n        "pass": f(packet["dream_pressure"]) >= 0.50,\n    }\n\n\ndef scenario_m4_identity_context() -> Dict[str, Any]:\n    from src.modules.m04_long_dynamic_memory.long_dynamic_memory import LongDynamicMemory, LongDynamicMemoryConfig\n\n    out = make_fake_m5_out()\n    obj = make_fake_inner_object()\n    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=256))\n    packet = m4.compute(\n        out=out,\n        obj=obj,\n        passport_manager=FakePassportManager(context_dim=256),\n        event_memory=None,\n        dream_mode=True,\n        global_step=1,\n    )\n    return {\n        "name": "object_identity_replay",\n        "identity_token": packet.get("identity_token", ""),\n        "dynamic_memory_gate": f(packet["dynamic_memory_gate"]),\n        "identity_stability": f(packet["identity_stability"]),\n        "identity_novelty": f(packet["identity_novelty"]),\n        "expect": "stable object identity is available for M2",\n        "pass": bool(packet.get("identity_token")) and f(packet["dynamic_memory_gate"]) > 0.0,\n    }\n\n\nSCENARIOS = [\n    scenario_calm_no_replay,\n    scenario_curiosity_replay,\n    scenario_bad_prediction_dream,\n    scenario_m4_identity_context,\n]\n\n\ndef run_all() -> Dict[str, Any]:\n    results = [fn() for fn in SCENARIOS]\n    return {\n        "status": "ok" if all(r["pass"] for r in results) else "fail",\n        "scenarios": results,\n    }\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--json", action="store_true")\n    args = parser.parse_args()\n    result = run_all()\n    print(json.dumps(result, ensure_ascii=False, indent=2))\n\n\nif __name__ == "__main__":\n    main()\n'

FILES['tests/module_contracts/test_unconscious_behavioral_scenarios.py'] = 'from __future__ import annotations\n\n\ndef test_unconscious_behavioral_scenarios_pass():\n    from scripts.module_lab.scenario_unconscious_replay import run_all\n\n    result = run_all()\n    assert result["status"] == "ok", result\n    names = {item["name"] for item in result["scenarios"]}\n    assert "calm_no_replay" in names\n    assert "curiosity_replay" in names\n    assert "bad_prediction_dream" in names\n    assert "object_identity_replay" in names\n\n\ndef test_bad_prediction_has_more_pressure_than_calm():\n    from scripts.module_lab.scenario_unconscious_replay import scenario_bad_prediction_dream, scenario_calm_no_replay\n\n    calm = scenario_calm_no_replay()\n    bad = scenario_bad_prediction_dream()\n    assert bad["dream_pressure"] >= calm["dream_pressure"]\n'

FILES['tests/module_contracts/test_sleep_motor_guard_contract.py'] = 'from __future__ import annotations\n\nimport torch\n\n\ndef test_sleep_motor_guard_blocks_executable_motor_outputs():\n    from src.modules.m03_self_action_causality.sleep_motor_guard import block_motor_outputs_for_sleep\n\n    out = {\n        "embodied_targets": torch.ones(1, 15),\n        "hand_ctrl": torch.ones(1, 8) * 0.5,\n        "leg_ctrl": torch.ones(1, 18) * 0.25,\n        "action_ids": torch.tensor([2]),\n    }\n\n    blocked = block_motor_outputs_for_sleep(out, sleep_mode=True, stage="test")\n\n    assert torch.allclose(blocked["embodied_targets"], torch.zeros_like(blocked["embodied_targets"]))\n    assert torch.allclose(blocked["hand_ctrl"], torch.zeros_like(blocked["hand_ctrl"]))\n    assert torch.allclose(blocked["leg_ctrl"], torch.zeros_like(blocked["leg_ctrl"]))\n    assert "imagined_embodied_targets" in blocked\n    assert "imagined_hand_ctrl" in blocked\n    assert "imagined_leg_ctrl" in blocked\n    assert "imagined_action_ids" in blocked\n    assert blocked["sleep_motor_guard"]["blocked"] is True\n    assert blocked["sleep_motor_guard"]["blocked_motor_norm"] > 0.0\n\n\ndef test_sleep_motor_guard_keeps_awake_outputs():\n    from src.modules.m03_self_action_causality.sleep_motor_guard import block_motor_outputs_for_sleep\n\n    embodied = torch.ones(1, 15)\n    hand = torch.ones(1, 8) * 0.5\n    out = {\n        "embodied_targets": embodied.clone(),\n        "hand_ctrl": hand.clone(),\n    }\n\n    awake = block_motor_outputs_for_sleep(out, sleep_mode=False, stage="test")\n\n    assert torch.allclose(awake["embodied_targets"], embodied)\n    assert torch.allclose(awake["hand_ctrl"], hand)\n    assert awake["sleep_motor_guard"]["blocked"] is False\n'

FILES['tests/module_contracts/test_unconscious_loop_trace_contract.py'] = 'from __future__ import annotations\n\nfrom scripts.module_lab.module_fixture_factory import make_unconscious_loop_out\n\n\ndef test_unconscious_loop_trace_packet_and_format():\n    from src.modules.m02_event_dream_replay.unconscious_loop_trace import (\n        build_unconscious_loop_trace_packet,\n        format_unconscious_loop_trace,\n    )\n\n    out = make_unconscious_loop_out()\n    out["event_dream_replay"] = {\n        "replay_gate": out["long_dynamic_memory"]["dynamic_memory_gate"],\n        "should_replay": out["long_dynamic_memory"]["dynamic_memory_gate"],\n        "dream_pressure": out["long_dynamic_memory"]["dynamic_memory_gate"],\n        "event_salience": out["long_dynamic_memory"]["dynamic_memory_gate"],\n        "replay_context": out["focus_context"],\n        "replay_source": "test",\n        "selected_identity_token": "obj:test",\n    }\n\n    packet = build_unconscious_loop_trace_packet(out, sleep_mode=True, sensor_state="sleep")\n    assert packet["sleep"] is True\n    assert "m11" in packet\n    assert "m13" in packet\n    assert "m4" in packet\n    assert "m2" in packet\n    assert "m5_seed" in packet\n\n    line = format_unconscious_loop_trace(packet, step=1)\n    assert "[unconscious_loop step=1]" in line\n    assert "m11:" in line\n    assert "m2:" in line\n    assert "m5_seed:" in line\n'

FILES['docs/architecture/unconscious_live_debug.md'] = '# Unconscious live debug v1\n\nAdds runtime visibility for the unconscious sleep/replay loop.\n\n## Architecture\n\n```text\nM1 -> M5 -> M11 -> M2 -> FocusFeedbackBoundary -> M5\n      |             ^\n      v             |\n     M4 ------------+\n                   M13 -> M2\n\nM5 -> M3 -> body/world -> M1\n```\n\n## Live trace\n\nThe runtime can print:\n\n```text\n[unconscious_loop step=120]\nsleep=1 state=sleep\nm11: val=-0.12 ar=0.64 stress=0.48 panic=0.20 cur=0.72\nm13: rel=0.61 eps=20\nm4: token=obj_003 gate=0.77 stab=0.82 nov=0.14\nm2: gate=1.00 should=1 pressure=0.69 sal=0.74 src=event\nm5_seed: gate=0.08 norm=11.4 fb=0.03\nm3_sleep_block=1\n```\n\n## Motor guard\n\nIn full sleep mode:\n\n```text\nvideo/contact/imu disabled -> is_full_sleep_mode() == True\n```\n\nM5 may still propose actions internally, but `SleepMotorGuardRuntimeMixin`\nzeros external executable motor tensors:\n\n```text\nembodied_targets\nhand_ctrl\nleg_ctrl\n```\n\nThe original proposed values are kept as:\n\n```text\nimagined_embodied_targets\nimagined_hand_ctrl\nimagined_leg_ctrl\nimagined_action_ids\n```\n\n## Behavioral scenarios\n\nRun:\n\n```bash\npython scripts/module_lab/scenario_unconscious_replay.py --json\n```\n\nIt checks:\n\n```text\ncalm_no_replay\ncuriosity_replay\nbad_prediction_dream\nobject_identity_replay\n```\n'

FILES['README_unconscious_live_debug_patch_v1.md'] = '# unconscious_live_debug_patch_v1\n\nAdds:\n\n```text\nsrc/modules/m02_event_dream_replay/unconscious_loop_trace.py\nsrc/modules/m03_self_action_causality/sleep_motor_guard.py\nscripts/module_lab/scenario_unconscious_replay.py\ntests/module_contracts/test_unconscious_behavioral_scenarios.py\ntests/module_contracts/test_sleep_motor_guard_contract.py\ntests/module_contracts/test_unconscious_loop_trace_contract.py\ndocs/architecture/unconscious_live_debug.md\n```\n\nInstalls runtime hooks into:\n\n```text\nsrc/apps/runner.py\nsrc/apps/life_runtime.py\n```\n\nInstall:\n\n```bash\nunzip -o unconscious_live_debug_patch_v1.zip -d .\npython scripts/apply_unconscious_live_debug_patch_v1.py\n```\n\nCheck:\n\n```bash\npython scripts/module_lab/scenario_unconscious_replay.py --json\npytest tests/module_contracts\n```\n'


def write_new_files() -> None:
    for rel, content in FILES.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") == content:
            print(f"[skip] {rel}: unchanged")
            continue
        target.write_text(content, encoding="utf-8")
        print(f"[ok] wrote {rel}")


def patch_runner() -> None:
    path = ROOT / "src/apps/runner.py"
    if not path.exists():
        print("[warn] runner.py not found; runtime mixins not installed")
        return
    text = path.read_text(encoding="utf-8")
    old = text

    imp1 = "from src.modules.m02_event_dream_replay.unconscious_loop_trace import UnconsciousLoopTraceRuntimeMixin\n"
    if imp1 not in text:
        anchor = "from src.modules.m02_event_dream_replay.event_dream_runtime import EventDreamReplayRuntimeMixin\n"
        if anchor in text:
            text = text.replace(anchor, anchor + imp1, 1)
            print("[ok] runner.py: imported UnconsciousLoopTraceRuntimeMixin")
        else:
            print("[warn] runner.py: event_dream import anchor missing")

    imp2 = "from src.modules.m03_self_action_causality.sleep_motor_guard import SleepMotorGuardRuntimeMixin\n"
    if imp2 not in text:
        anchor = "from src.modules.m03_self_action_causality.action_runtime import ActionRuntimeMixin\n"
        if anchor in text:
            text = text.replace(anchor, anchor + imp2, 1)
            print("[ok] runner.py: imported SleepMotorGuardRuntimeMixin")
        else:
            print("[warn] runner.py: action_runtime import anchor missing")

    if "UnconsciousLoopTraceRuntimeMixin," not in text:
        anchor = "    EventDreamReplayRuntimeMixin,\n"
        if anchor in text:
            text = text.replace(anchor, anchor + "    UnconsciousLoopTraceRuntimeMixin,\n", 1)
            print("[ok] runner.py: added UnconsciousLoopTraceRuntimeMixin to UnifiedSystem")
        else:
            print("[warn] runner.py: EventDreamReplayRuntimeMixin class anchor missing")

    if "SleepMotorGuardRuntimeMixin," not in text:
        anchor = "    ActionRuntimeMixin,\n"
        if anchor in text:
            text = text.replace(anchor, anchor + "    SleepMotorGuardRuntimeMixin,\n", 1)
            print("[ok] runner.py: added SleepMotorGuardRuntimeMixin to UnifiedSystem")
        else:
            print("[warn] runner.py: ActionRuntimeMixin class anchor missing")

    if text != old:
        path.write_text(text, encoding="utf-8")


def patch_life_runtime() -> None:
    path = ROOT / "src/apps/life_runtime.py"
    if not path.exists():
        print("[warn] life_runtime.py not found; runtime hooks not installed")
        return
    text = path.read_text(encoding="utf-8")
    old = text

    # Add motor guard for pre_observe output before it is sent to MuJoCo/body.
    if 'apply_sleep_motor_guard(out0, stage="pre_observe")' not in text:
        anchor = '        self.apply_bird_leg_controls(out0["leg_ctrl"])\n'
        insert = (
            '        if hasattr(self, "apply_sleep_motor_guard"):\n'
            '            out0 = self.apply_sleep_motor_guard(out0, stage="pre_observe")\n'
        )
        if anchor in text:
            text = text.replace(anchor, insert + anchor, 1)
            print("[ok] life_runtime.py: inserted pre_observe sleep motor guard")
        else:
            print("[warn] life_runtime.py: pre_observe bird_leg anchor missing")

    # Add motor guard for main output before prev_* is stored and before side effects.
    if 'apply_sleep_motor_guard(out, stage="main")' not in text:
        anchor = '        if hasattr(self, "maybe_print_energy_resonator_trace"):\n'
        insert = (
            '        if hasattr(self, "apply_sleep_motor_guard"):\n'
            '            out = self.apply_sleep_motor_guard(out, stage="main")\n'
        )
        if anchor in text:
            text = text.replace(anchor, insert + anchor, 1)
            print("[ok] life_runtime.py: inserted main sleep motor guard")
        else:
            print("[warn] life_runtime.py: main energy trace anchor missing")

    # Add live trace after M2/M13 debug traces, or after event dream trace if present.
    if "maybe_print_unconscious_loop_trace" not in text:
        anchor = (
            '        if hasattr(self, "maybe_print_autobiographical_memory_trace"):\n'
            '            self.maybe_print_autobiographical_memory_trace(out)\n'
        )
        insert = (
            '        if hasattr(self, "maybe_print_unconscious_loop_trace"):\n'
            '            self.maybe_print_unconscious_loop_trace(out, obs)\n'
        )
        if anchor in text:
            text = text.replace(anchor, anchor + insert, 1)
            print("[ok] life_runtime.py: inserted unconscious loop live trace")
        else:
            # fallback: after event dream trace
            anchor2 = (
                '        if hasattr(self, "maybe_print_event_dream_replay_trace"):\n'
                '            self.maybe_print_event_dream_replay_trace(out)\n'
            )
            if anchor2 in text:
                text = text.replace(anchor2, anchor2 + insert, 1)
                print("[ok] life_runtime.py: inserted unconscious loop live trace after event dream trace")
            else:
                print("[warn] life_runtime.py: no trace anchor found")

    if text != old:
        path.write_text(text, encoding="utf-8")


def main() -> int:
    write_new_files()
    patch_runner()
    patch_life_runtime()

    print("\nRun:")
    print("  python scripts/module_lab/scenario_unconscious_replay.py --json")
    print("  pytest tests/module_contracts")
    print("\nCompile new files:")
    for rel in FILES:
        if rel.endswith(".py"):
            print(f"  python -m py_compile {rel}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
