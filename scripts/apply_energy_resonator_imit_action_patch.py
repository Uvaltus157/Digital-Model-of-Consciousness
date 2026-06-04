#!/usr/bin/env python3
from __future__ import annotations

"""
Apply Energy Resonator "Имит action" integration.

Run from repository root:

    python scripts/apply_energy_resonator_imit_action_patch.py

This script is idempotent: running it again should skip already-applied edits.
"""

from pathlib import Path
import sys


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").exists() and (candidate / "config").exists():
            return candidate
        if (candidate / "src").exists() and (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


ROOT = find_repo_root()


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str, marker: str) -> bool:
    text = read(path)
    if marker in text:
        print(f"[skip] {path}: {marker} already present")
        return False
    if old not in text:
        raise RuntimeError(
            f"Anchor not found in {path}.\n"
            f"Looked for:\n{old}\n"
        )
    write(path, text.replace(old, new, 1))
    print(f"[ok] {path}: inserted {marker}")
    return True


def main() -> int:
    # 1) runner.py import
    replace_once(
        "src/apps/runner.py",
        old=(
            "from src.modules.m10_global_conscious_broadcast.broadcast_runtime import GlobalBroadcastRuntimeMixin\n"
            "from src.modules.m12_metacognition_monitor.metacognition_runtime import MetacognitionRuntimeMixin\n"
        ),
        new=(
            "from src.modules.m10_global_conscious_broadcast.broadcast_runtime import GlobalBroadcastRuntimeMixin\n"
            "from src.modules.m11_motivational_homeostasis.imit.energy_resonator_runtime import EnergyResonatorRuntimeMixin\n"
            "from src.modules.m12_metacognition_monitor.metacognition_runtime import MetacognitionRuntimeMixin\n"
        ),
        marker="EnergyResonatorRuntimeMixin",
    )

    # 2) runner.py mixin in UnifiedSystem
    replace_once(
        "src/apps/runner.py",
        old=(
            "    ThoughtChainRuntimeMixin,\n"
            "    GlobalBroadcastRuntimeMixin,\n"
            "    InnerSpeechRuntimeMixin,\n"
        ),
        new=(
            "    ThoughtChainRuntimeMixin,\n"
            "    GlobalBroadcastRuntimeMixin,\n"
            "    EnergyResonatorRuntimeMixin,\n"
            "    InnerSpeechRuntimeMixin,\n"
        ),
        marker="    EnergyResonatorRuntimeMixin,",
    )

    # 3) life_runtime.py bridge call
    replace_once(
        "src/apps/life_runtime.py",
        old=(
            "        out[\"inner_object\"] = self.compute_inner_object_image(obs, out)\n"
            "        self._compute_long_dynamic_memory(obs, out)\n"
            "        out[\"self_core\"] = self.compute_self_core(obs, out)\n"
            "        self._apply_conscious_action_guard(obs, out)\n"
        ),
        new=(
            "        out[\"inner_object\"] = self.compute_inner_object_image(obs, out)\n"
            "        self._compute_long_dynamic_memory(obs, out)\n"
            "        if hasattr(self, \"compute_energy_resonator\"):\n"
            "            try:\n"
            "                self.compute_energy_resonator(obs, out)\n"
            "            except Exception as e:\n"
            "                if not hasattr(self, \"_energy_resonator_warned\"):\n"
            "                    print(f\"[energy_resonator] compute skipped: {e}\")\n"
            "                    self._energy_resonator_warned = True\n"
            "        out[\"self_core\"] = self.compute_self_core(obs, out)\n"
            "        if hasattr(self, \"maybe_print_energy_resonator_trace\"):\n"
            "            self.maybe_print_energy_resonator_trace(out)\n"
            "        self._apply_conscious_action_guard(obs, out)\n"
        ),
        marker="compute_energy_resonator",
    )

    # 4) action_runtime.py action handler
    replace_once(
        "src/modules/m03_self_action_causality/action_runtime.py",
        old=(
            "        elif action == \"ping\":\n"
            "            print(\"[ipc] ping received\")\n"
            "        elif is_adaptive_gesture_command(action):\n"
        ),
        new=(
            "        elif action == \"ping\":\n"
            "            print(\"[ipc] ping received\")\n"
            "        elif action in (\"imit_action\", \"simulate_energy_resonator_action\", \"energy_resonator_imit_action\"):\n"
            "            if hasattr(self, \"request_energy_resonator_imitation\"):\n"
            "                self.request_energy_resonator_imitation(payload)\n"
            "            else:\n"
            "                print(\"[ipc] imit_action ignored: EnergyResonatorRuntimeMixin is not installed\")\n"
            "        elif is_adaptive_gesture_command(action):\n"
        ),
        marker="request_energy_resonator_imitation",
    )

    # 5) action_runtime.py high level action seen
    replace_once(
        "src/modules/m03_self_action_causality/action_runtime.py",
        old=(
            "                if action.startswith(\"gesture_\") or action.startswith(\"fly_to_\") or action.startswith(\"stop_fly_to_\"):\n"
            "                    high_level_action_seen = True\n"
        ),
        new=(
            "                if (\n"
            "                    action.startswith(\"gesture_\")\n"
            "                    or action.startswith(\"fly_to_\")\n"
            "                    or action.startswith(\"stop_fly_to_\")\n"
            "                    or action in (\"imit_action\", \"simulate_energy_resonator_action\", \"energy_resonator_imit_action\")\n"
            "                ):\n"
            "                    high_level_action_seen = True\n"
        ),
        marker="simulate_energy_resonator_action",
    )

    # 6) pyqt_agent_actions_ipc.py button in scenarios
    replace_once(
        "src/modules/m08_debug_visual_control/pyqt_agent_actions_ipc.py",
        old=(
            "        scenarios = [\n"
            "            (\"Fly to cube + palpate\", \"fly_to_cube_palpate\", self.gesture_fly_to_cube_and_palpate),\n"
        ),
        new=(
            "        scenarios = [\n"
            "            (\"Имит action\", \"imit_action\", self.gesture_imit_action),\n"
            "            (\"Fly to cube + palpate\", \"fly_to_cube_palpate\", self.gesture_fly_to_cube_and_palpate),\n"
        ),
        marker="gesture_imit_action",
    )

    # 7) pyqt_agent_actions_ipc.py method
    replace_once(
        "src/modules/m08_debug_visual_control/pyqt_agent_actions_ipc.py",
        old=(
            "    def gesture_touch_object_pose(self):\n"
            "        self.send_action_command(\"gesture_touch_object_pose\")\n"
            "\n"
            "    def gesture_fly_to_cube_and_palpate(self):\n"
        ),
        new=(
            "    def gesture_touch_object_pose(self):\n"
            "        self.send_action_command(\"gesture_touch_object_pose\")\n"
            "\n"
            "    def gesture_imit_action(self):\n"
            "        self.send_action_command(\n"
            "            \"imit_action\",\n"
            "            active_key=\"imit_action\",\n"
            "            mode=\"conscious\",\n"
            "            steps=1,\n"
            "            blend_into_focus_context=True,\n"
            "            focus_blend_weight=0.015,\n"
            "        )\n"
            "\n"
            "    def gesture_fly_to_cube_and_palpate(self):\n"
        ),
        marker="    def gesture_imit_action(self):",
    )

    print("\nDone. Now run:")
    print("  python -m py_compile src/modules/m11_motivational_homeostasis/imit/energy_resonator.py")
    print("  python -m py_compile src/modules/m11_motivational_homeostasis/imit/energy_resonator_runtime.py")
    print("  python -m py_compile scripts/run_energy_resonator_demo.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
