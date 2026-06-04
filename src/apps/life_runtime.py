from __future__ import annotations

import time

import numpy as np
import torch

from src.apps.life_depth_focus import LifeDepthFocusMixin
from src.apps.life_report_helpers import LifeReportHelpersMixin
from src.apps.life_stats_runtime import LifeStatsRuntimeMixin


class LifeRuntimeMixin(
    LifeDepthFocusMixin,
    LifeReportHelpersMixin,
    LifeStatsRuntimeMixin,
):
    """One live DMoC runtime step.

    The large helper surface previously living in this file is split into:
      - life_depth_focus.py       focused depth-map observation helpers
      - life_report_helpers.py    scalar/self-confidence/inner-report helpers
      - life_stats_runtime.py     stats, visualizer updates, logging, checkpoints
    """

    def _apply_conscious_action_guard(self, obs: dict, out: dict) -> None:
        if not hasattr(self, "compute_conscious_action"):
            return
        try:
            self.compute_conscious_action(obs, out)
        except Exception as e:
            if not hasattr(self, "_semantic_action_warned"):
                print(f"[semantic_action] compute skipped: {e}")
                self._semantic_action_warned = True

    def _compute_long_dynamic_memory(self, obs: dict, out: dict) -> None:
        if not hasattr(self, "compute_long_dynamic_memory"):
            return
        try:
            self.compute_long_dynamic_memory(obs, out)
        except Exception as e:
            if not hasattr(self, "_long_dynamic_memory_runtime_warned"):
                print(f"[long_dynamic_memory] compute skipped: {e}")
                self._long_dynamic_memory_runtime_warned = True

    def _write_autobiographical_episode(self, obs: dict, out: dict) -> None:
        if not hasattr(self, "write_autobiographical_episode"):
            return
        try:
            self.write_autobiographical_episode(obs, out)
        except Exception as e:
            if not hasattr(self, "_autobiographical_write_warned"):
                print(f"[autobiographical_memory] write skipped: {e}")
                self._autobiographical_write_warned = True

    def life_step(self) -> None:
        if hasattr(self, "log_tetra_live_step_started"):
            self.log_tetra_live_step_started()

        # Read IPC commands first, before deciding what to render/train.
        self.poll_ipc_control_messages()
        self.update_fly_to_cube_palpate_scenario()

        prev_action = int(self.state["prev_action_ids"].item())

        self.world.manual_control_active = bool(getattr(self, "_ipc_manual_actions_enabled", False))
        dyn0 = self.apply_dynamic_agent_rig_control(self.prev_embodied_action[0].detach().cpu().numpy())
        mocap_safe_emb0 = np.zeros_like(self.prev_embodied_action[0].detach().cpu().numpy()) if dyn0 is not None else self.prev_embodied_action[0].detach().cpu().numpy()
        obs0 = self.world.observe(
            action_id=prev_action,
            embodied_targets=mocap_safe_emb0,
            hand_controls=self.prev_hand_motor[0].detach().cpu().numpy(),
        )
        obs0 = self.gate_observation_for_sleep(obs0)
        obs0 = self.apply_focused_depth_observation(obs0)

        with torch.no_grad():
            out0 = self.model_step(obs0, self.state, model_stage="pre_observe")

        # Imitate neural output at the exact model output boundary.
        # This is where PyQt sliders replace real neural outputs.
        out0 = self.apply_pyqt_neural_output_override(out0, stage="pre_observe")
        # Anti-freeze exploration before the second observe().
        out0 = self.apply_exploration_motor(out0, novelty_score=1.0)
        out0 = self.protect_manual_body_output(out0)
        out0 = self.apply_manual_arm_action_override(out0)
        out0 = self.apply_manual_hand_action_dimension_override(out0)
        out0["leg_ctrl"] = self.compute_leg_control(out0)
        out0 = self.apply_manual_leg_action_override(out0)
        self.apply_bird_leg_controls(out0["leg_ctrl"])

        self.world.set_attention_drive(
            focus_idx=int(out0["focus"]["focus_idx"].item()),
            curiosity_drive=float(out0["values"]["curiosity"].item()),
            planned_action_id=int(out0["action_ids"].item()),
        )

        dyn1 = self.apply_dynamic_agent_rig_control(out0["embodied_targets"][0].detach().cpu().numpy())
        mocap_safe_emb1 = np.zeros_like(out0["embodied_targets"][0].detach().cpu().numpy()) if dyn1 is not None else out0["embodied_targets"][0].detach().cpu().numpy()
        self.world.manual_control_active = bool(getattr(self, "_ipc_manual_actions_enabled", False))
        obs = self.world.observe(
            action_id=int(out0["action_ids"].item()),
            embodied_targets=mocap_safe_emb1,
            hand_controls=out0["hand_ctrl"][0].detach().cpu().numpy(),
        )
        obs = self.gate_observation_for_sleep(obs)
        obs = self.apply_focused_depth_observation(obs)
        if dyn1 is not None:
            out0["dynamic_agent_rig"] = dyn1
        self.maybe_print_vestibular_trace(obs)

        with torch.no_grad():
            out = self.model_step(obs, self.state, model_stage="main")

        # Imitate neural output at the exact model output boundary.
        # This is where PyQt sliders replace real neural outputs.
        out = self.apply_pyqt_neural_output_override(out, stage="main")

        novelty_score = self.novelty.score(
            out["obs_embed"],
            out["workspace_out"],
            out["imagined"]["imagined_value"],
        )

        # Apply anti-freeze exploration to the stored next action as well.
        out = self.apply_exploration_motor(out, novelty_score=float(novelty_score))
        out = self.protect_manual_body_output(out)
        out = self.apply_manual_arm_action_override(out)
        out = self.apply_manual_hand_action_dimension_override(out)
        out["leg_ctrl"] = self.compute_leg_control(out)
        out = self.apply_manual_leg_action_override(out)
        out["inner_object"] = self.compute_inner_object_image(obs, out)
        self._compute_long_dynamic_memory(obs, out)
        if hasattr(self, "compute_energy_resonator"):
            try:
                self.compute_energy_resonator(obs, out)
            except Exception as e:
                if not hasattr(self, "_energy_resonator_warned"):
                    print(f"[energy_resonator] compute skipped: {e}")
                    self._energy_resonator_warned = True
        out["self_core"] = self.compute_self_core(obs, out)
        if hasattr(self, "maybe_print_energy_resonator_trace"):
            self.maybe_print_energy_resonator_trace(out)
        self._apply_conscious_action_guard(obs, out)
        self.maybe_print_self_core_trace(out)

        # Emotion must be computed BEFORE replay.add(), otherwise training never sees
        # the intrinsic emotional reward. EmotionalDrive may reuse the cached packet
        # created earlier for M15/M10/M9 without a second EMA update.
        self.prev_embodied_action = out["embodied_targets"].detach()
        self.prev_hand_motor = out["hand_ctrl"].detach()
        self.state = out["state"]
        self.latest_out = out

        chosen_action = int(out["action_ids"].item())
        obs["next_action_id"] = torch.tensor([chosen_action], device=self.device, dtype=torch.long)

        self.replay.add({
            "left": obs["left"].detach().cpu(),
            "right": obs["right"].detach().cpu(),
            "pose": obs["pose"].detach().cpu(),
            "body_state": obs["body_state"].detach().cpu(),
            "tactile": obs["tactile"].detach().cpu(),
            "hand_motor": self.prev_hand_motor.detach().cpu(),
            "leg_ctrl": out.get("leg_ctrl", self.prev_leg_motor).detach().cpu(),
            "embodied_action": self.prev_embodied_action.detach().cpu(),
            "object_state": obs["object_state"].detach().cpu(),
            "reward": obs["reward"].detach().cpu(),
            "done": obs["done"].detach().cpu(),
            "action_id": obs["next_action_id"].detach().cpu(),
            "depth": obs["depth"].detach().cpu(),
        })

        # Human-readable inner speech for visualizer. M7 may be absent while it
        # is being split out, so never require symbolic_report at runtime level.
        decoded_report, target_report, inner_report_confidence = self._life_runtime_inner_report(obs, out)
        out["decoded_report"] = decoded_report
        out["target_report"] = target_report

        emotion = self.emotional_drive.compute(out, obs)
        out["emotion"] = emotion
        if isinstance(emotion.get("affect"), dict):
            out["affect"] = emotion["affect"]

        # Strict unconscious sleep/replay loop:
        #   M11 affect -> M13 retrieval + M4 identity -> M2 replay selector
        #   -> next M5 FocusFeedbackBoundary seed.
        if hasattr(self, "compute_autobiographical_retrieval"):
            try:
                self.compute_autobiographical_retrieval(obs, out)
            except Exception as e:
                if not hasattr(self, "_autobiographical_retrieval_warned"):
                    print(f"[autobiographical_memory] retrieval skipped: {e}")
                    self._autobiographical_retrieval_warned = True
        if hasattr(self, "compute_event_dream_replay"):
            try:
                self.compute_event_dream_replay(obs, out)
            except Exception as e:
                if not hasattr(self, "_event_dream_replay_warned"):
                    print(f"[event_dream_replay] compute skipped: {e}")
                    self._event_dream_replay_warned = True
        if hasattr(self, "maybe_print_event_dream_replay_trace"):
            self.maybe_print_event_dream_replay_trace(out)
        if hasattr(self, "maybe_print_autobiographical_memory_trace"):
            self.maybe_print_autobiographical_memory_trace(out)

        if hasattr(self, "compute_metacognition"):
            try:
                self.compute_metacognition(obs, out)
            except Exception as e:
                if not hasattr(self, "_metacognition_runtime_warned"):
                    print(f"[metacognition] compute skipped: {e}")
                    self._metacognition_runtime_warned = True
        if hasattr(self, "compute_conscious_loop_feedback"):
            try:
                self.compute_conscious_loop_feedback(obs, out)
            except Exception as e:
                if not hasattr(self, "_conscious_loop_warned"):
                    print(f"[conscious_loop] feedback skipped: {e}")
                    self._conscious_loop_warned = True
        if hasattr(self, "maybe_print_conscious_loop_trace"):
            self.maybe_print_conscious_loop_trace(out)
        if hasattr(self, "maybe_print_metacognition_trace"):
            self.maybe_print_metacognition_trace(out)
        if self.cfg.emotional_drive.inject_into_env_reward:
            obs["reward"] = obs["reward"] + emotion["intrinsic_reward"].detach() * float(self.cfg.emotional_drive.reward_weight)

        self._write_autobiographical_episode(obs, out)

        self_confidence = self._life_runtime_self_confidence(out)
        self.latest_stats = self.build_latest_life_stats(
            obs=obs,
            out=out,
            emotion=emotion,
            novelty_score=float(novelty_score),
            chosen_action=chosen_action,
            decoded_report=decoded_report,
            target_report=target_report,
            inner_report_confidence=float(inner_report_confidence),
            self_confidence=float(self_confidence),
        )

        self.finalize_life_step_side_effects(
            obs=obs,
            out=out,
            emotion=emotion,
            decoded_report=decoded_report,
            target_report=target_report,
            inner_report_confidence=float(inner_report_confidence),
        )

        self.global_step += 1
        sleep_sec = getattr(self.cfg.life, "sleep_sec", None)
        if sleep_sec is None:
            sleep_sec = 1.0 / max(float(getattr(self.cfg.life, "fps", 15.0)), 1e-6)
        time.sleep(max(0.0, float(sleep_sec)))


__all__ = ["LifeRuntimeMixin"]
