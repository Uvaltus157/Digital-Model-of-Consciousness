#!/usr/bin/env python3
from __future__ import annotations

"""
Anchor-fix patcher for conscious_loop v2.

Use this after the previous v2 patcher failed with:

    RuntimeError: Anchor not found in
    src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py

Reason:
    Current conscious_dreamer_memory_thought.py imports explicit classes from
    conscious_dreamer_core, not `from ...conscious_dreamer_full import *`.

Run from repository root:

    python scripts/apply_conscious_loop_patch_v2_1_anchor_fix.py

This script is idempotent and continues the same v2 integration using anchors
that match the current file layout.
"""

from pathlib import Path
import sys


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").exists() and ((candidate / "config").exists() or (candidate / ".git").exists()):
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


def insert_after(path: str, anchor: str, insert: str, marker: str) -> bool:
    text = read(path)
    if marker in text:
        print(f"[skip] {path}: marker already present: {marker}")
        return False
    if anchor not in text:
        raise RuntimeError(f"Anchor not found in {path}.\nLooked for:\n{anchor}")
    write(path, text.replace(anchor, anchor + insert, 1))
    print(f"[ok] {path}: inserted {marker}")
    return True


def main() -> int:
    # ---------------------------------------------------------------------
    # runner.py, config.py, unified_conscious_viewer.py, life_runtime.py
    # are included too, because the old script may have stopped halfway.
    # All edits are idempotent.
    # ---------------------------------------------------------------------

    # runner import
    insert_after(
        "src/apps/runner.py",
        anchor="from src.modules.m15_counterfactual_imagination_planning.thought_chain_runtime import ThoughtChainRuntimeMixin\n",
        insert="from src.modules.m15_counterfactual_imagination_planning.conscious_loop_runtime import ConsciousLoopRuntimeMixin\n",
        marker="ConsciousLoopRuntimeMixin",
    )

    # runner mixin order
    replace_once(
        "src/apps/runner.py",
        old=(
            "    ThoughtChainRuntimeMixin,\n"
            "    GlobalBroadcastRuntimeMixin,\n"
        ),
        new=(
            "    ThoughtChainRuntimeMixin,\n"
            "    ConsciousLoopRuntimeMixin,\n"
            "    GlobalBroadcastRuntimeMixin,\n"
        ),
        marker="    ConsciousLoopRuntimeMixin,",
    )

    # config class
    replace_once(
        "src/shared/config.py",
        old="@dataclass\nclass EventDreamReplayRuntimeConfig:\n",
        new=(
            "@dataclass\n"
            "class ConsciousLoopRuntimeConfig:\n"
            "    enabled: bool = True\n"
            "    feedback_gain: float = 0.22\n"
            "    min_gate: float = 0.00\n"
            "    max_gate: float = 0.22\n"
            "    require_self_binding: bool = True\n"
            "    use_metacognition_gate: bool = True\n"
            "    apply_stage: str = \"both\"  # both | pre_observe | main\n"
            "    print_every_steps: int = 30\n"
            "\n"
            "\n"
            "@dataclass\n"
            "class EventDreamReplayRuntimeConfig:\n"
        ),
        marker="class ConsciousLoopRuntimeConfig",
    )

    # config field
    replace_once(
        "src/shared/config.py",
        old=(
            "    self_core: SelfCoreRuntimeConfig = field(default_factory=SelfCoreRuntimeConfig)\n"
            "    thought_chain: ThoughtChainRuntimeConfig = field(default_factory=ThoughtChainRuntimeConfig)\n"
            "    event_dream_replay: EventDreamReplayRuntimeConfig = field(default_factory=EventDreamReplayRuntimeConfig)\n"
        ),
        new=(
            "    self_core: SelfCoreRuntimeConfig = field(default_factory=SelfCoreRuntimeConfig)\n"
            "    thought_chain: ThoughtChainRuntimeConfig = field(default_factory=ThoughtChainRuntimeConfig)\n"
            "    conscious_loop: ConsciousLoopRuntimeConfig = field(default_factory=ConsciousLoopRuntimeConfig)\n"
            "    event_dream_replay: EventDreamReplayRuntimeConfig = field(default_factory=EventDreamReplayRuntimeConfig)\n"
        ),
        marker="conscious_loop: ConsciousLoopRuntimeConfig",
    )

    # ---------------------------------------------------------------------
    # M5 current-file anchors
    # ---------------------------------------------------------------------
    m5 = "src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py"

    # Current file imports explicit classes from conscious_dreamer_core.
    insert_after(
        m5,
        anchor=(
            "from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (\n"
            "    AttentionController,\n"
            "    ConsciousDreamerCore,\n"
            "    ConsciousDreamerCoreConfig,\n"
            "    RSSMCore,\n"
            "    Workspace,\n"
            "    BodyContextModel,\n"
            "    ObjectRepresentation,\n"
            "    PreconsciousReflectionLoop,\n"
            "    ImaginationCore,\n"
            "    DecoderHeads,\n"
            "    VisionEncoder,\n"
            "    MLPEncoder,\n"
            ")\n"
        ),
        insert=(
            "from src.modules.m05_world_model_attention_workspace.models.focus_feedback_boundary import FocusFeedbackBoundary\n"
        ),
        marker="FocusFeedbackBoundary",
    )

    # Instantiate FocusFeedbackBoundary after decoder creation.
    insert_after(
        m5,
        anchor="        self.decoder = DecoderHeads(l.rssm_dim, d.image_channels, d.image_height, d.image_width)\n",
        insert=(
            "        self.focus_feedback_boundary = FocusFeedbackBoundary(\n"
            "            focus_context_dim=c.workspace_dim,\n"
            "            workspace_seed_dim=c.workspace_dim,\n"
            "            thought_dim=c.thought_dim,\n"
            "            hidden_dim=max(c.workspace_dim, c.thought_dim),\n"
            "        )\n"
        ),
        marker="self.focus_feedback_boundary = FocusFeedbackBoundary",
    )

    # Step signature.
    replace_once(
        m5,
        old=(
            "    def step(self, left, right, pose, body_state, state, tactile=None, hand_motor=None, embodied_action=None, depth=None, object_state=None, action_override=None, write_memory: bool = True) -> Dict:\n"
        ),
        new=(
            "    def step(self, left, right, pose, body_state, state, tactile=None, hand_motor=None, embodied_action=None, depth=None, object_state=None, action_override=None, write_memory: bool = True, focus_context_seed=None, focus_context_seed_gate=None) -> Dict:\n"
        ),
        marker="focus_context_seed=None",
    )

    # Focus feedback boundary after AttentionController.
    replace_once(
        m5,
        old=(
            "        attn = self.attention([vision, pose_latent, body_latent, tactile_latent, motor_latent, object_state_latent, action_latent])\n"
            "        fused = self.fusion(torch.cat([attn[\"workspace_seed\"], attn[\"context\"]], dim=-1))\n"
        ),
        new=(
            "        attn = self.attention([vision, pose_latent, body_latent, tactile_latent, motor_latent, object_state_latent, action_latent])\n"
            "        focus_feedback = self.focus_feedback_boundary(\n"
            "            workspace_seed=attn[\"workspace_seed\"],\n"
            "            focus_context_seed=focus_context_seed,\n"
            "            focus_context_seed_gate=focus_context_seed_gate,\n"
            "        )\n"
            "        attn = dict(attn)\n"
            "        attn[\"raw_workspace_seed\"] = attn[\"workspace_seed\"]\n"
            "        attn[\"workspace_seed\"] = focus_feedback[\"workspace_seed\"]\n"
            "        attn[\"focus_feedback\"] = focus_feedback\n"
            "        fused = self.fusion(torch.cat([attn[\"workspace_seed\"], attn[\"context\"]], dim=-1))\n"
        ),
        marker="focus_feedback = self.focus_feedback_boundary",
    )

    # Apply feedback to preconscious_seed after Workspace output.
    replace_once(
        m5,
        old=(
            "        preconscious_seed = ws[\"preconscious_seed\"]\n"
            "        obj = self.object_repr(ws[\"workspace\"], body_context, tactile_latent, vision, object_state_latent)\n"
        ),
        new=(
            "        preconscious_seed = ws[\"preconscious_seed\"]\n"
            "        if isinstance(focus_feedback, dict):\n"
            "            adjusted_preconscious_seed = self.focus_feedback_boundary.apply_preconscious_seed(preconscious_seed, focus_feedback)\n"
            "            if torch.is_tensor(adjusted_preconscious_seed):\n"
            "                preconscious_seed = adjusted_preconscious_seed\n"
            "                ws = dict(ws)\n"
            "                ws[\"preconscious_seed\"] = preconscious_seed\n"
            "                ws[\"report\"] = self.workspace.report(torch.cat([ws[\"workspace\"], preconscious_seed], dim=-1))\n"
            "        obj = self.object_repr(ws[\"workspace\"], body_context, tactile_latent, vision, object_state_latent)\n"
        ),
        marker="adjusted_preconscious_seed",
    )

    # Attention debug fields.
    replace_once(
        m5,
        old=(
            "            \"attention\": {\n"
            "                \"tokens\": attn[\"tokens\"],\n"
            "                \"modality_weights\": attn[\"modality_weights\"],\n"
            "                \"attn_matrix\": attn[\"attn_matrix\"],\n"
            "            },\n"
        ),
        new=(
            "            \"attention\": {\n"
            "                \"tokens\": attn[\"tokens\"],\n"
            "                \"modality_weights\": attn[\"modality_weights\"],\n"
            "                \"attn_matrix\": attn[\"attn_matrix\"],\n"
            "                \"focus_feedback_gate\": attn.get(\"focus_feedback\", {}).get(\"total_gate\") if isinstance(attn.get(\"focus_feedback\"), dict) else None,\n"
            "                \"focus_feedback_learned_gate\": attn.get(\"focus_feedback\", {}).get(\"learned_gate\") if isinstance(attn.get(\"focus_feedback\"), dict) else None,\n"
            "            },\n"
        ),
        marker="focus_feedback_learned_gate",
    )

    # Top-level output debug packet.
    replace_once(
        m5,
        old=(
            "            \"workspace_out\": ws[\"workspace\"],\n"
            "            \"focus_context\": focus_context,\n"
        ),
        new=(
            "            \"workspace_out\": ws[\"workspace\"],\n"
            "            \"focus_context\": focus_context,\n"
            "            \"focus_feedback\": {\n"
            "                \"active\": focus_feedback.get(\"active\") if isinstance(focus_feedback, dict) else None,\n"
            "                \"external_gate\": focus_feedback.get(\"external_gate\") if isinstance(focus_feedback, dict) else None,\n"
            "                \"learned_gate\": focus_feedback.get(\"learned_gate\") if isinstance(focus_feedback, dict) else None,\n"
            "                \"total_gate\": focus_feedback.get(\"total_gate\") if isinstance(focus_feedback, dict) else None,\n"
            "                \"seed_norm\": focus_feedback.get(\"seed_norm\") if isinstance(focus_feedback, dict) else None,\n"
            "            },\n"
        ),
        marker="\"focus_feedback\": {",
    )

    # ---------------------------------------------------------------------
    # model_step passes seed into M5.
    # ---------------------------------------------------------------------
    replace_once(
        "src/apps/unified_conscious_viewer.py",
        old=(
            "    def model_step(self, obs: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor], action_override=None, write_memory: bool = True) -> Dict:\n"
            "        return self.model.step(\n"
        ),
        new=(
            "    def model_step(self, obs: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor], action_override=None, write_memory: bool = True, model_stage: str = \"main\", focus_context_seed=None, focus_context_seed_gate=None) -> Dict:\n"
            "        if focus_context_seed is None and hasattr(self, \"get_conscious_loop_focus_seed\"):\n"
            "            focus_context_seed, focus_context_seed_gate = self.get_conscious_loop_focus_seed(stage=model_stage)\n"
            "        return self.model.step(\n"
        ),
        marker="model_stage: str = \"main\"",
    )

    replace_once(
        "src/apps/unified_conscious_viewer.py",
        old=(
            "            object_state=obs[\"object_state\"],\n"
            "            action_override=action_override,\n"
            "            write_memory=write_memory,\n"
            "        )\n"
        ),
        new=(
            "            object_state=obs[\"object_state\"],\n"
            "            action_override=action_override,\n"
            "            write_memory=write_memory,\n"
            "            focus_context_seed=focus_context_seed,\n"
            "            focus_context_seed_gate=focus_context_seed_gate,\n"
            "        )\n"
        ),
        marker="focus_context_seed=focus_context_seed",
    )

    # life_runtime stage labels.
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

    # late feedback after affect. This is kept separate from any earlier M9 block.
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
            "        if hasattr(self, \"compute_metacognition\"):\n"
            "            try:\n"
            "                self.compute_metacognition(obs, out)\n"
            "            except Exception as e:\n"
            "                if not hasattr(self, \"_metacognition_runtime_warned\"):\n"
            "                    print(f\"[metacognition] compute skipped: {e}\")\n"
            "                    self._metacognition_runtime_warned = True\n"
            "        if hasattr(self, \"compute_conscious_loop_feedback\"):\n"
            "            try:\n"
            "                self.compute_conscious_loop_feedback(obs, out)\n"
            "            except Exception as e:\n"
            "                if not hasattr(self, \"_conscious_loop_warned\"):\n"
            "                    print(f\"[conscious_loop] feedback skipped: {e}\")\n"
            "                    self._conscious_loop_warned = True\n"
            "        if hasattr(self, \"maybe_print_conscious_loop_trace\"):\n"
            "            self.maybe_print_conscious_loop_trace(out)\n"
            "        if hasattr(self, \"maybe_print_metacognition_trace\"):\n"
            "            self.maybe_print_metacognition_trace(out)\n"
            "        if self.cfg.emotional_drive.inject_into_env_reward:\n"
        ),
        marker="compute_conscious_loop_feedback",
    )

    print("\nDone. Now run:")
    print("  python -m py_compile src/modules/m05_world_model_attention_workspace/models/focus_feedback_boundary.py")
    print("  python -m py_compile src/modules/m15_counterfactual_imagination_planning/conscious_loop_runtime.py")
    print("  python -m py_compile src/apps/life_runtime.py")
    print("  python -m py_compile src/apps/runner.py")
    print("  python -m py_compile src/apps/unified_conscious_viewer.py")
    print("  python -m py_compile src/modules/m05_world_model_attention_workspace/models/conscious_dreamer_memory_thought.py")
    print("  python -m py_compile src/shared/config.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
