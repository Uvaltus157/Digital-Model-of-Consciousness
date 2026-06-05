#!/usr/bin/env python3
from __future__ import annotations

"""
Apply unconscious sleep/replay loop patch v1.

Goal:
    Fix the strictly unconscious loop:

        M11 affect
             ↓
        M2 event_dream_replay  ←  M13 autobiographical memory
             ↑
             │
        M4 long dynamic identity
             ↓
        M5 FocusFeedbackBoundary
             ↓
        M5 inner playback
             ↓
        M11 affect

Important:
    M2 must NOT directly overwrite/blend out["focus_context"] by default.
    M2 must send replay_context to the SAME M5 FocusFeedbackBoundary seed input
    that M15 uses in the conscious loop.

Run from repository root:

    python scripts/apply_unconscious_sleep_loop_patch_v1.py
"""

from pathlib import Path
import sys


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").exists() and ((candidate / ".git").exists() or (candidate / "config").exists()):
            return candidate
    return Path.cwd().resolve()


ROOT = find_repo_root()


def p(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    file = p(path)
    if not file.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return file.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    p(path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str, marker: str) -> bool:
    text = read(path)
    if marker in text:
        print(f"[skip] {path}: marker already present: {marker}")
        return False
    if old not in text:
        raise RuntimeError(f"Anchor not found in {path}.\nLooked for:\n{old}")
    write(path, text.replace(old, new, 1))
    print(f"[ok] {path}: inserted {marker}")
    return True


def main() -> int:
    # ------------------------------------------------------------------
    # 1) M2 config: add M4 context and M5 seed boundary settings.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        old=(
            "@dataclass\n"
            "class EventDreamReplayConfig:\n"
            "    enabled: bool = True\n"
            "    replay_context_dim: int = 256\n"
            "    event_code_dim: int = 8\n"
            "    replay_threshold: float = 0.35\n"
            "    focus_blend: float = 0.15\n"
            "    blend_replay_into_focus: bool = True\n"
            "    use_m13_context: bool = True\n"
            "    use_event_memory: bool = True\n"
            "    max_recent_events_scan: int = 16\n"
        ),
        new=(
            "@dataclass\n"
            "class EventDreamReplayConfig:\n"
            "    enabled: bool = True\n"
            "    replay_context_dim: int = 256\n"
            "    event_code_dim: int = 8\n"
            "    replay_threshold: float = 0.35\n"
            "    focus_blend: float = 0.15\n"
            "    # Legacy path. Keep available, but default runtime should use the\n"
            "    # M5 FocusFeedbackBoundary seed input instead of direct focus mutation.\n"
            "    blend_replay_into_focus: bool = False\n"
            "    use_m13_context: bool = True\n"
            "    use_m4_context: bool = True\n"
            "    m4_context_weight: float = 0.20\n"
            "    use_event_memory: bool = True\n"
            "    max_recent_events_scan: int = 16\n"
            "    seed_to_m5_boundary: bool = True\n"
            "    seed_gate_gain: float = 1.0\n"
            "    apply_stage: str = \"pre_observe\"  # both | pre_observe | main\n"
            "    seed_only_in_sleep: bool = True\n"
        ),
        marker="use_m4_context",
    )

    # ------------------------------------------------------------------
    # 2) M2 compute: read M4 long dynamic identity context.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        old=(
            "        memory13 = out.get(\"autobiographical_memory\", {}) if isinstance(out.get(\"autobiographical_memory\"), dict) else {}\n"
            "\n"
            "        event, event_idx = self._select_replay_event(event_memory) if bool(c.use_event_memory) else ({}, -1)\n"
            "        event_vec = self._event_vector(event, device)\n"
            "        m13_context = pad_or_trim_replay(memory13.get(\"retrieved_context\"), int(c.replay_context_dim), device=device)\n"
            "        focus_context = pad_or_trim_replay(out.get(\"focus_context\"), int(c.replay_context_dim), device=device)\n"
        ),
        new=(
            "        memory13 = out.get(\"autobiographical_memory\", {}) if isinstance(out.get(\"autobiographical_memory\"), dict) else {}\n"
            "        memory4 = out.get(\"long_dynamic_memory\", {}) if isinstance(out.get(\"long_dynamic_memory\"), dict) else {}\n"
            "\n"
            "        event, event_idx = self._select_replay_event(event_memory) if bool(c.use_event_memory) else ({}, -1)\n"
            "        event_vec = self._event_vector(event, device)\n"
            "        m13_context = pad_or_trim_replay(memory13.get(\"retrieved_context\"), int(c.replay_context_dim), device=device)\n"
            "        m4_context = pad_or_trim_replay(memory4.get(\"dynamic_identity_context\"), int(c.replay_context_dim), device=device)\n"
            "        focus_context = pad_or_trim_replay(out.get(\"focus_context\"), int(c.replay_context_dim), device=device)\n"
        ),
        marker="memory4 = out.get(\"long_dynamic_memory\"",
    )

    # ------------------------------------------------------------------
    # 3) M2 compute: include M4 identity in salience and replay_context.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        old=(
            "        memory_relevance = _scalar(memory13.get(\"retrieval_relevance\"), 0.0)\n"
            "        event_delta = _scalar(event.get(\"delta_norm\"), 0.0)\n"
            "        event_contact = _scalar(event.get(\"contact_norm\"), 0.0)\n"
            "        event_action = _scalar(event.get(\"action_norm\"), 0.0)\n"
        ),
        new=(
            "        memory_relevance = _scalar(memory13.get(\"retrieval_relevance\"), 0.0)\n"
            "        identity_stability = _scalar(memory4.get(\"identity_stability\"), 0.0)\n"
            "        identity_novelty = _scalar(memory4.get(\"identity_novelty\"), 0.0)\n"
            "        dynamic_memory_gate = _scalar(memory4.get(\"dynamic_memory_gate\"), 0.0)\n"
            "        event_delta = _scalar(event.get(\"delta_norm\"), 0.0)\n"
            "        event_contact = _scalar(event.get(\"contact_norm\"), 0.0)\n"
            "        event_action = _scalar(event.get(\"action_norm\"), 0.0)\n"
        ),
        marker="identity_stability = _scalar(memory4.get",
    )

    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        old=(
            "                + 0.22 * memory_relevance\n"
            "                + 0.30 * event_delta\n"
        ),
        new=(
            "                + 0.22 * memory_relevance\n"
            "                + 0.12 * identity_stability\n"
            "                + 0.10 * identity_novelty\n"
            "                + 0.10 * dynamic_memory_gate\n"
            "                + 0.30 * event_delta\n"
        ),
        marker="+ 0.12 * identity_stability",
    )

    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        old=(
            "        m13_weight = 0.35 if bool(c.use_m13_context) else 0.0\n"
            "        event_weight = 0.45 if event else 0.0\n"
            "        focus_weight = max(0.0, 1.0 - m13_weight - event_weight)\n"
            "        replay_context = focus_weight * focus_context + event_weight * event_vec + m13_weight * m13_context\n"
        ),
        new=(
            "        m13_weight = 0.35 if bool(c.use_m13_context) else 0.0\n"
            "        event_weight = 0.45 if event else 0.0\n"
            "        m4_weight = float(c.m4_context_weight) if bool(getattr(c, \"use_m4_context\", True)) and dynamic_memory_gate > 0.0 else 0.0\n"
            "        focus_weight = max(0.0, 1.0 - m13_weight - event_weight - m4_weight)\n"
            "        replay_context = (\n"
            "            focus_weight * focus_context\n"
            "            + event_weight * event_vec\n"
            "            + m13_weight * m13_context\n"
            "            + m4_weight * m4_context\n"
            "        )\n"
        ),
        marker="m4_weight = float(c.m4_context_weight)",
    )

    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        old=(
            "            \"selected_episode_summary\": str(memory13.get(\"summary\", memory13.get(\"last_summary\", \"\"))),\n"
            "            \"replay_source\": source,\n"
        ),
        new=(
            "            \"selected_episode_summary\": str(memory13.get(\"summary\", memory13.get(\"last_summary\", \"\"))),\n"
            "            \"selected_identity_token\": str(memory4.get(\"identity_token\", \"\")),\n"
            "            \"selected_identity_sentence\": str(memory4.get(\"selected_sentence\", \"\")),\n"
            "            \"identity_stability\": torch.tensor([[identity_stability]], dtype=torch.float32, device=device),\n"
            "            \"identity_novelty\": torch.tensor([[identity_novelty]], dtype=torch.float32, device=device),\n"
            "            \"dynamic_memory_gate\": torch.tensor([[dynamic_memory_gate]], dtype=torch.float32, device=device),\n"
            "            \"replay_source\": source,\n"
        ),
        marker="selected_identity_token",
    )

    # ------------------------------------------------------------------
    # 4) Runtime config: default to M5 seed boundary, not direct focus blend.
    # ------------------------------------------------------------------
    replace_once(
        "src/shared/config.py",
        old=(
            "@dataclass\n"
            "class EventDreamReplayRuntimeConfig:\n"
            "    enabled: bool = True\n"
            "    replay_context_dim: int = 256\n"
            "    event_code_dim: int = 8\n"
            "    replay_threshold: float = 0.35\n"
            "    focus_blend: float = 0.15\n"
            "    blend_replay_into_focus: bool = True\n"
            "    use_m13_context: bool = True\n"
            "    use_event_memory: bool = True\n"
            "    max_recent_events_scan: int = 16\n"
            "    print_every_steps: int = 30\n"
        ),
        new=(
            "@dataclass\n"
            "class EventDreamReplayRuntimeConfig:\n"
            "    enabled: bool = True\n"
            "    replay_context_dim: int = 256\n"
            "    event_code_dim: int = 8\n"
            "    replay_threshold: float = 0.35\n"
            "    focus_blend: float = 0.15\n"
            "    blend_replay_into_focus: bool = False\n"
            "    use_m13_context: bool = True\n"
            "    use_m4_context: bool = True\n"
            "    m4_context_weight: float = 0.20\n"
            "    use_event_memory: bool = True\n"
            "    max_recent_events_scan: int = 16\n"
            "    seed_to_m5_boundary: bool = True\n"
            "    seed_gate_gain: float = 1.0\n"
            "    apply_stage: str = \"pre_observe\"  # both | pre_observe | main\n"
            "    seed_only_in_sleep: bool = True\n"
            "    print_every_steps: int = 30\n"
        ),
        marker="m4_context_weight: float = 0.20",
    )

    # ------------------------------------------------------------------
    # 5) EventDreamReplayRuntimeMixin: pass new config fields.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_runtime.py",
        old=(
            "            use_m13_context=bool(getattr(cfg_obj, \"use_m13_context\", True)),\n"
            "            use_event_memory=bool(getattr(cfg_obj, \"use_event_memory\", True)),\n"
            "            max_recent_events_scan=int(getattr(cfg_obj, \"max_recent_events_scan\", 16)),\n"
        ),
        new=(
            "            use_m13_context=bool(getattr(cfg_obj, \"use_m13_context\", True)),\n"
            "            use_m4_context=bool(getattr(cfg_obj, \"use_m4_context\", True)),\n"
            "            m4_context_weight=float(getattr(cfg_obj, \"m4_context_weight\", 0.20)),\n"
            "            use_event_memory=bool(getattr(cfg_obj, \"use_event_memory\", True)),\n"
            "            max_recent_events_scan=int(getattr(cfg_obj, \"max_recent_events_scan\", 16)),\n"
            "            seed_to_m5_boundary=bool(getattr(cfg_obj, \"seed_to_m5_boundary\", True)),\n"
            "            seed_gate_gain=float(getattr(cfg_obj, \"seed_gate_gain\", 1.0)),\n"
            "            apply_stage=str(getattr(cfg_obj, \"apply_stage\", \"pre_observe\")),\n"
            "            seed_only_in_sleep=bool(getattr(cfg_obj, \"seed_only_in_sleep\", True)),\n"
        ),
        marker="seed_to_m5_boundary=bool",
    )

    # ------------------------------------------------------------------
    # 6) EventDreamReplayRuntimeMixin: add seed bus methods.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_runtime.py",
        old=(
            "        ))\n"
            "        print(\"[event_dream_replay] initialized\")\n"
            "\n"
            "    def compute_event_dream_replay(self, obs: dict, out: dict):\n"
        ),
        new=(
            "        ))\n"
            "        if not hasattr(self, \"_event_dream_next_focus_seed\"):\n"
            "            self._event_dream_next_focus_seed = None\n"
            "            self._event_dream_next_focus_gate = None\n"
            "            self._event_dream_seed_step = -1\n"
            "        print(\"[event_dream_replay] initialized\")\n"
            "\n"
            "    def _event_dream_stage_allowed(self, stage: str) -> bool:\n"
            "        cfg = getattr(getattr(self, \"cfg\", None), \"event_dream_replay\", None)\n"
            "        apply_stage = str(getattr(cfg, \"apply_stage\", \"pre_observe\"))\n"
            "        if apply_stage not in (\"both\", \"pre_observe\", \"main\"):\n"
            "            apply_stage = \"pre_observe\"\n"
            "        return apply_stage == \"both\" or str(stage) == apply_stage\n"
            "\n"
            "    def get_event_dream_focus_seed(self, stage: str = \"model_step\"):\n"
            "        cfg = getattr(getattr(self, \"cfg\", None), \"event_dream_replay\", None)\n"
            "        if not bool(getattr(cfg, \"enabled\", True)):\n"
            "            return None, None\n"
            "        if bool(getattr(cfg, \"seed_only_in_sleep\", True)):\n"
            "            is_sleep = bool(self.is_full_sleep_mode()) if hasattr(self, \"is_full_sleep_mode\") else False\n"
            "            if not is_sleep:\n"
            "                return None, None\n"
            "        if not self._event_dream_stage_allowed(stage):\n"
            "            return None, None\n"
            "        seed = getattr(self, \"_event_dream_next_focus_seed\", None)\n"
            "        gate = getattr(self, \"_event_dream_next_focus_gate\", None)\n"
            "        if not torch.is_tensor(seed):\n"
            "            return None, None\n"
            "        return seed.detach(), gate.detach() if torch.is_tensor(gate) else gate\n"
            "\n"
            "    def get_m5_focus_seed(self, stage: str = \"model_step\"):\n"
            "        \"\"\"Common M5 FocusFeedbackBoundary seed bus.\n"
            "\n"
            "        Priority:\n"
            "          1. M2 dream/replay seed in sleep mode.\n"
            "          2. M15 conscious loop seed when available.\n"
            "        Both enter M5 through the same focus_context_seed/gate inputs.\n"
            "        \"\"\"\n"
            "        seed, gate = self.get_event_dream_focus_seed(stage=stage)\n"
            "        if torch.is_tensor(seed):\n"
            "            return seed, gate\n"
            "        if hasattr(self, \"get_conscious_loop_focus_seed\"):\n"
            "            return self.get_conscious_loop_focus_seed(stage=stage)\n"
            "        return None, None\n"
            "\n"
            "    def _store_event_dream_m5_seed(self, packet: dict) -> None:\n"
            "        self._event_dream_next_focus_seed = None\n"
            "        self._event_dream_next_focus_gate = None\n"
            "        cfg = self.event_dream_replay.cfg\n"
            "        if not bool(getattr(cfg, \"seed_to_m5_boundary\", True)):\n"
            "            return\n"
            "        replay_context = packet.get(\"replay_context\")\n"
            "        replay_gate = packet.get(\"replay_gate\", packet.get(\"should_replay\"))\n"
            "        dream_pressure = packet.get(\"dream_pressure\")\n"
            "        if not torch.is_tensor(replay_context):\n"
            "            return\n"
            "        if torch.is_tensor(replay_gate):\n"
            "            gate = replay_gate.detach().float()\n"
            "        else:\n"
            "            gate = torch.tensor([[float(replay_gate or 0.0)]], dtype=torch.float32, device=replay_context.device)\n"
            "        if torch.is_tensor(dream_pressure):\n"
            "            gate = gate * dream_pressure.detach().float().to(gate.device)\n"
            "        gate = gate * float(getattr(cfg, \"seed_gate_gain\", 1.0))\n"
            "        if gate.ndim == 0:\n"
            "            gate = gate.reshape(1, 1)\n"
            "        elif gate.ndim == 1:\n"
            "            gate = gate.reshape(-1, 1)\n"
            "        if float(gate.detach().reshape(-1)[0].cpu().item()) <= 0.0:\n"
            "            return\n"
            "        self._event_dream_next_focus_seed = replay_context.detach()\n"
            "        self._event_dream_next_focus_gate = gate.detach()\n"
            "        self._event_dream_seed_step = int(getattr(self, \"global_step\", -1))\n"
            "        packet[\"next_focus_context_seed\"] = self._event_dream_next_focus_seed\n"
            "        packet[\"next_focus_context_seed_gate\"] = self._event_dream_next_focus_gate\n"
            "        packet[\"target_m5_boundary\"] = \"FocusFeedbackBoundary(workspace_seed + preconscious_seed)\"\n"
            "        packet[\"seed_source\"] = \"m02_event_dream_replay\"\n"
            "\n"
            "    def compute_event_dream_replay(self, obs: dict, out: dict):\n"
        ),
        marker="def get_m5_focus_seed",
    )

    # ------------------------------------------------------------------
    # 7) EventDreamReplayRuntimeMixin: store seed after compute.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_runtime.py",
        old=(
            "        )\n"
            "        out[\"event_dream_replay\"] = packet\n"
            "\n"
            "        focus = out.get(\"focus_context\")\n"
        ),
        new=(
            "        )\n"
            "        out[\"event_dream_replay\"] = packet\n"
            "        self._store_event_dream_m5_seed(packet)\n"
            "\n"
            "        focus = out.get(\"focus_context\")\n"
        ),
        marker="self._store_event_dream_m5_seed(packet)",
    )

    # ------------------------------------------------------------------
    # 8) EventDreamReplayRuntimeMixin: direct blend becomes legacy only.
    # ------------------------------------------------------------------
    replace_once(
        "src/modules/m02_event_dream_replay/event_dream_runtime.py",
        old=(
            "            bool(cfg.blend_replay_into_focus)\n"
            "            and torch.is_tensor(focus)\n"
        ),
        new=(
            "            bool(cfg.blend_replay_into_focus)\n"
            "            and not bool(getattr(cfg, \"seed_to_m5_boundary\", True))\n"
            "            and torch.is_tensor(focus)\n"
        ),
        marker="and not bool(getattr(cfg, \"seed_to_m5_boundary\", True))",
    )

    # ------------------------------------------------------------------
    # 9) model_step: ask common M5 seed bus first.
    # ------------------------------------------------------------------
    replace_once(
        "src/apps/unified_conscious_viewer.py",
        old=(
            "    def model_step(self, obs: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor], action_override=None, write_memory: bool = True, model_stage: str = \"main\", focus_context_seed=None, focus_context_seed_gate=None) -> Dict:\n"
            "        if focus_context_seed is None and hasattr(self, \"get_conscious_loop_focus_seed\"):\n"
            "            focus_context_seed, focus_context_seed_gate = self.get_conscious_loop_focus_seed(stage=model_stage)\n"
        ),
        new=(
            "    def model_step(self, obs: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor], action_override=None, write_memory: bool = True, model_stage: str = \"main\", focus_context_seed=None, focus_context_seed_gate=None) -> Dict:\n"
            "        if focus_context_seed is None:\n"
            "            if hasattr(self, \"get_m5_focus_seed\"):\n"
            "                focus_context_seed, focus_context_seed_gate = self.get_m5_focus_seed(stage=model_stage)\n"
            "            elif hasattr(self, \"get_conscious_loop_focus_seed\"):\n"
            "                focus_context_seed, focus_context_seed_gate = self.get_conscious_loop_focus_seed(stage=model_stage)\n"
        ),
        marker="get_m5_focus_seed",
    )

    # ------------------------------------------------------------------
    # 10) life_runtime: label the two M5 stages.
    # ------------------------------------------------------------------
    replace_once(
        "src/apps/life_runtime.py",
        old="            out0 = self.model_step(obs0, self.state)\n",
        new="            out0 = self.model_step(obs0, self.state, model_stage=\"pre_observe\")\n",
        marker="model_stage=\"pre_observe\"",
    )

    replace_once(
        "src/apps/life_runtime.py",
        old="            out = self.model_step(obs, self.state)\n",
        new="            out = self.model_step(obs, self.state, model_stage=\"main\")\n",
        marker="model_stage=\"main\"",
    )

    # ------------------------------------------------------------------
    # 11) life_runtime: after M11 affect, retrieve M13 and run M2.
    # ------------------------------------------------------------------
    replace_once(
        "src/apps/life_runtime.py",
        old=(
            "        if isinstance(emotion.get(\"affect\"), dict):\n"
            "            out[\"affect\"] = emotion[\"affect\"]\n"
            "        if self.cfg.emotional_drive.inject_into_env_reward:\n"
        ),
        new=(
            "        if isinstance(emotion.get(\"affect\"), dict):\n"
            "            out[\"affect\"] = emotion[\"affect\"]\n"
            "\n"
            "        # Strict unconscious sleep/replay loop:\n"
            "        #   M11 affect -> M13 retrieval + M4 identity -> M2 replay selector\n"
            "        #   -> next M5 FocusFeedbackBoundary seed.\n"
            "        if hasattr(self, \"compute_autobiographical_retrieval\"):\n"
            "            try:\n"
            "                self.compute_autobiographical_retrieval(obs, out)\n"
            "            except Exception as e:\n"
            "                if not hasattr(self, \"_autobiographical_retrieval_warned\"):\n"
            "                    print(f\"[autobiographical_memory] retrieval skipped: {e}\")\n"
            "                    self._autobiographical_retrieval_warned = True\n"
            "        if hasattr(self, \"compute_event_dream_replay\"):\n"
            "            try:\n"
            "                self.compute_event_dream_replay(obs, out)\n"
            "            except Exception as e:\n"
            "                if not hasattr(self, \"_event_dream_replay_warned\"):\n"
            "                    print(f\"[event_dream_replay] compute skipped: {e}\")\n"
            "                    self._event_dream_replay_warned = True\n"
            "        if hasattr(self, \"maybe_print_event_dream_replay_trace\"):\n"
            "            self.maybe_print_event_dream_replay_trace(out)\n"
            "        if hasattr(self, \"maybe_print_autobiographical_memory_trace\"):\n"
            "            self.maybe_print_autobiographical_memory_trace(out)\n"
            "\n"
            "        if self.cfg.emotional_drive.inject_into_env_reward:\n"
        ),
        marker="Strict unconscious sleep/replay loop",
    )

    print("\nDone. Suggested checks:")
    checks = [
        "src/modules/m02_event_dream_replay/event_dream_replay.py",
        "src/modules/m02_event_dream_replay/event_dream_runtime.py",
        "src/apps/unified_conscious_viewer.py",
        "src/apps/life_runtime.py",
        "src/shared/config.py",
    ]
    for path in checks:
        print(f"  python -m py_compile {path}")
    print("\nSmoke check:")
    print("  python - <<'PY'")
    print("  from src.modules.m02_event_dream_replay.event_dream_replay import EventDreamReplay, EventDreamReplayConfig")
    print("  from src.apps.runner import UnifiedSystem")
    print("  print('imports ok')")
    print("  PY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
