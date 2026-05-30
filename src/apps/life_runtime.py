from __future__ import annotations

import time

import numpy as np
import torch


class LifeRuntimeMixin:
    def _floating_depth_focus_params(self) -> tuple[float | None, float | None, str]:
        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        if not isinstance(scenario, dict):
            return None, None, ""
        is_active = bool(getattr(self, "_fly_to_cube_palpate_active", False) or scenario.get("active", False))
        if not is_active:
            return None, None, ""
        if str(scenario.get("scenario", "")) != "fly_to_tetrahedron_inspect":
            return None, None, ""
        try:
            focus_depth = float(scenario.get("gaze_distance", 0.0))
            if not np.isfinite(focus_depth) or focus_depth <= 0.0:
                return None, None, ""
            half_range = max(0.10, float(scenario.get("depth_focus_half_range", 0.85)))
            return focus_depth, half_range, str(scenario.get("gaze_target", ""))
        except Exception:
            return None, None, ""

    def apply_focused_depth_observation(self, obs: dict) -> dict:
        """
        During floating-object inspection the learning input should be a focused
        depth map, not the raw metric renderer depth. Keep depth_raw for debug
        views that still want meters.
        """
        if not isinstance(obs, dict) or "depth" not in obs:
            return obs

        focus_depth, half_range, focus_label = self._floating_depth_focus_params()
        if focus_depth is None:
            return obs

        depth = obs.get("depth")
        if not torch.is_tensor(depth):
            return obs

        try:
            raw = depth.detach()
            valid = torch.isfinite(raw) & (raw > 1e-6)
            if not bool(valid.any().detach().cpu().item()):
                return obs

            lo = max(0.0, float(focus_depth) - float(half_range))
            hi = float(focus_depth) + float(half_range)
            focused = (torch.clamp(raw.float(), min=lo, max=hi) - lo) / max(hi - lo, 1e-6)
            focused = torch.where(valid, focused, torch.ones_like(focused))

            out = dict(obs)
            out["depth_raw"] = raw
            out["depth"] = focused.clamp(0.0, 1.0)
            out["depth_focus_applied"] = True
            out["depth_focus_depth"] = float(focus_depth)
            out["depth_focus_half_range"] = float(half_range)
            out["depth_focus_label"] = str(focus_label)
            return out
        except Exception as e:
            if not hasattr(self, "_focused_depth_warned"):
                print(f"[depth_focus] failed to focus observation depth: {e}")
                self._focused_depth_warned = True
            return obs

    def _life_runtime_scalar(self, value, default: float = 0.0) -> float:
        """Safely convert optional model/runtime outputs to a Python float."""
        if value is None:
            return float(default)
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().float().reshape(-1)[0].cpu().item())
            return float(value)
        except Exception:
            return float(default)

    def _life_runtime_nested_get(self, data: dict, *keys: str):
        cur = data
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def _life_runtime_self_confidence(self, out: dict) -> float:
        """
        Runtime fallback for confidence while M7/M12/self-core boundaries are
        being split. Prefer the old reflection field when present, then try
        explicit self-core confidence, then derive a conservative self-core
        score from agency/ownership/continuity.
        """
        direct = self._life_runtime_nested_get(out, "reflection_out", "self_confidence")
        if direct is not None:
            return self._life_runtime_scalar(direct, 0.0)

        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        for key in ("self_confidence", "confidence", "reflective_confidence"):
            if key in self_core:
                return self._life_runtime_scalar(self_core.get(key), 0.0)

        parts = [
            self_core.get("agency_score"),
            self_core.get("body_ownership_score"),
            self_core.get("self_continuity_score"),
        ]
        vals = [self._life_runtime_scalar(v, float("nan")) for v in parts if v is not None]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            return float(np.clip(np.mean(vals), 0.0, 1.0))
        return 0.0

    def _life_runtime_inner_report(self, obs: dict, out: dict) -> tuple[str, str, float]:
        """
        Build human-readable inner speech without assuming M7 has already added
        out["symbolic_report"]. Missing reports become empty strings/confidence 0.
        """
        symbolic = out.get("symbolic_report", {}) if isinstance(out.get("symbolic_report"), dict) else {}
        confidence = self._life_runtime_scalar(symbolic.get("confidence"), 0.0)

        decoded_report = ""
        token_ids = symbolic.get("text_token_ids")
        if token_ids is not None and hasattr(self, "speech_vocab") and self.speech_vocab is not None:
            try:
                ids = token_ids
                if torch.is_tensor(ids) and ids.ndim > 1:
                    ids = ids[0]
                decoded_report = str(self.speech_vocab.decode(ids, skip_special=True))
            except Exception:
                decoded_report = ""

        if not decoded_report:
            for key in ("text", "decoded_text", "report_text"):
                value = symbolic.get(key)
                if value:
                    decoded_report = str(value)
                    break

        target_report = ""
        if hasattr(self, "speech_teacher") and self.speech_teacher is not None:
            try:
                target_report = str(self.speech_teacher.build_report(obs, out))
            except Exception:
                target_report = ""

        return decoded_report, target_report, confidence

    def life_step(self) -> None:
        if hasattr(self, "log_tetra_live_step_started"):
            self.log_tetra_live_step_started()

        # Read IPC commands first, before deciding what to render/train.
        self.poll_ipc_control_messages()
        self.update_fly_to_cube_palpate_scenario()

        prev_action = int(self.state["prev_action_ids"].item())

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
            out0 = self.model_step(obs0, self.state)

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
            out = self.model_step(obs, self.state)

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
        out["self_core"] = self.compute_self_core(obs, out)
        self.maybe_print_self_core_trace(out)

        # Emotion must be computed BEFORE replay.add(), otherwise training never sees
        # the intrinsic emotional reward.
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
        if self.cfg.emotional_drive.inject_into_env_reward:
            obs["reward"] = obs["reward"] + emotion["intrinsic_reward"].detach() * float(self.cfg.emotional_drive.reward_weight)

        self_confidence = self._life_runtime_self_confidence(out)

        self.latest_stats = {
            "step": self.global_step,
            "training": bool(self.training_enabled) and not (self.is_full_sleep_mode() if hasattr(self, "is_full_sleep_mode") else False),
            "sensor_state": self.sensor_state_label() if hasattr(self, "sensor_state_label") else "unknown",
            "quality": self.quality.get(),
            "train_steps": self.train_steps,
            "focus_idx": int(out["focus"]["focus_idx"].item()),
            "action": chosen_action,
            "curiosity": float(out["values"]["curiosity"].item()),
            "coherence": float(out["values"]["coherence"].item()),
            "self_confidence": self_confidence,
            "inner_report_confidence": inner_report_confidence,
            "decoded_report": decoded_report,
            "target_report": target_report,
            "novelty_score": float(novelty_score),
            "emotion_valence": float(emotion["emotional_valence"]),
            "emotion_arousal": float(emotion["emotional_arousal"]),
            "intrinsic_reward": float(emotion["intrinsic_reward"].detach().cpu().item()),
            "meaning_progress": float(emotion["meaning_progress"]),
            "gap_fill_reward": float(emotion["gap_fill_reward"]),
            "misunderstanding": float(emotion["misunderstanding"]),
            "confusion_increase": float(emotion["confusion_increase"]),
            "coherence_loss": float(emotion["coherence_loss"]),
            "alignment_loss": float(emotion["alignment_loss"]),
            "chaotic_touch": float(emotion["chaotic_touch"]),
            "multimodal_alignment": float(emotion["multimodal_alignment"]),
            "contact_pleasure": float(emotion["contact_pleasure"]),
            "uncertainty": float(emotion["uncertainty"]),
            "affect_latent_norm": float(out.get("affect", {}).get("affect_latents", torch.zeros(1, device=self.device)).norm(dim=-1).mean().detach().cpu().item()) if isinstance(out.get("affect"), dict) and torch.is_tensor(out.get("affect", {}).get("affect_latents")) else 0.0,
            "affect_panic": float(out.get("affect", {}).get("panic_latent", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("affect"), dict) and torch.is_tensor(out.get("affect", {}).get("panic_latent")) else 0.0,
            "affect_comfort": float(out.get("affect", {}).get("comfort_latent", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("affect"), dict) and torch.is_tensor(out.get("affect", {}).get("comfort_latent")) else 0.0,
            "exploration_active": bool(out.get("exploration", {}).get("active", False)),
            "exploration_boost": float(out.get("exploration", {}).get("boost", 0.0)),
            "dynamic_rig_enabled": bool(self.dynamic_agent_rig_controller is not None),
            "rig_z": float(self.world.data.qpos[self.dynamic_agent_rig_controller.qpos_adr + 2]) if self.dynamic_agent_rig_controller is not None else 0.0,
            "hover_height": float(self.cfg.dynamic_agent_rig.hover_height),
            "hover_target_z": float(getattr(self.dynamic_agent_rig_controller, "hover_target_z", 0.0)) if self.dynamic_agent_rig_controller is not None else 0.0,
            "manual_override": bool(getattr(self, "_ipc_manual_actions_enabled", False)),
            "object_decoder_loss": float(getattr(self, "latest_object_decoder_stats", {}).get("object_decoder_loss", 0.0) or 0.0),
            "object_rgb_loss": float(getattr(self, "latest_object_decoder_stats", {}).get("object_rgb_loss", 0.0) or 0.0),
            "object_depth_loss": float(getattr(self, "latest_object_decoder_stats", {}).get("object_depth_loss", 0.0) or 0.0),
            "object_mask_loss": float(getattr(self, "latest_object_decoder_stats", {}).get("object_mask_loss", 0.0) or 0.0),
            "inner_action_active": bool(out.get("inner_object", {}).get("inner_action_active", False)) if isinstance(out.get("inner_object"), dict) else False,
            "inner_action_confidence": float(out.get("inner_object", {}).get("inner_action_confidence", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_action_confidence")) else 0.0,
            "inner_trust_value": float(out.get("inner_object", {}).get("inner_trust_value", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_trust_value")) else 0.0,
            "inner_trust_alpha": float(out.get("inner_object", {}).get("inner_trust_alpha", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_trust_alpha")) else 0.0,
            "inner_trust_allowed": bool(float(out.get("inner_object", {}).get("inner_trust_allowed", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) > 0.5) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_trust_allowed")) else False,
            "inner_trust_applied_to_policy": bool(float(out.get("inner_object", {}).get("inner_trust_applied_to_policy", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) > 0.5) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_trust_applied_to_policy")) else False,
            "inner_trust_reason": str(out.get("inner_object", {}).get("inner_trust_reason", "")) if isinstance(out.get("inner_object"), dict) else "",
            "passport_active": bool(float(out.get("inner_object", {}).get("passport_active", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) > 0.5) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_active")) else False,
            "passport_token": str(out.get("inner_object", {}).get("passport_token", "")) if isinstance(out.get("inner_object"), dict) else "",
            "passport_count": float(out.get("inner_object", {}).get("passport_count", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_count")) else 0.0,
            "passport_replay_active": bool(float(out.get("inner_object", {}).get("passport_replay_active", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) > 0.5) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_replay_active")) else False,
            "passport_second_order_decoded": bool(float(out.get("inner_object", {}).get("passport_second_order_decoded", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) > 0.5) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_second_order_decoded")) else False,
            "passport_debug_active": bool(float(out.get("inner_object", {}).get("passport_debug_active", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) > 0.5) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_debug_active")) else False,
            "passport_debug_z_distance": float(out.get("inner_object", {}).get("passport_debug_z_distance", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_debug_z_distance")) else 0.0,
            "passport_debug_z_cosine": float(out.get("inner_object", {}).get("passport_debug_z_cosine", torch.zeros(1, device=self.device)).reshape(-1)[0].detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("passport_debug_z_cosine")) else 0.0,
            "inner_real_action_trace_path": str(out.get("inner_object", {}).get("inner_real_action_trace_path", "")) if isinstance(out.get("inner_object"), dict) else "",
            "inner_real_action_body_norm": float(out.get("inner_object", {}).get("inner_action_body", torch.zeros(1, device=self.device)).norm(dim=-1).mean().detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_action_body")) else 0.0,
            "inner_real_action_hand_norm": float(out.get("inner_object", {}).get("inner_action_hand", torch.zeros(1, device=self.device)).norm(dim=-1).mean().detach().cpu().item()) if isinstance(out.get("inner_object"), dict) and torch.is_tensor(out.get("inner_object", {}).get("inner_action_hand")) else 0.0,
            "pyqt_manual_actions_enabled": bool(getattr(self, "_ipc_manual_actions_enabled", False)),
            "manual_body_len": int(len(self._ipc_manual_body_action)) if getattr(self, "_ipc_manual_body_action", None) is not None else 0,
            "manual_arm_len": int(len(self._ipc_manual_arm_action)) if getattr(self, "_ipc_manual_arm_action", None) is not None else 0,
            "manual_hand_len": int(len(self._ipc_manual_hand_action)) if getattr(self, "_ipc_manual_hand_action", None) is not None else 0,
            "manual_leg_len": int(len(self._ipc_manual_leg_action)) if getattr(self, "_ipc_manual_leg_action", None) is not None else 0,
            "fly_to_cube_palpate_active": bool(getattr(self, "_fly_to_cube_palpate_active", False)),
            "fly_to_cube_palpate_phase": str(getattr(self, "_fly_to_cube_palpate_status", {}).get("phase", "")) if isinstance(getattr(self, "_fly_to_cube_palpate_status", {}), dict) else "",
            "fly_to_cube_palpate_dist": float(getattr(self, "_fly_to_cube_palpate_status", {}).get("body_dist", 0.0)) if isinstance(getattr(self, "_fly_to_cube_palpate_status", {}), dict) else 0.0,
            "fly_to_cube_palpate_touch": float(getattr(self, "_fly_to_cube_palpate_status", {}).get("tactile_sum", 0.0)) if isinstance(getattr(self, "_fly_to_cube_palpate_status", {}), dict) else 0.0,
            "module_debug": bool(getattr(self, "show_module_debug_window", False)),
            "module_training_flags": dict(getattr(self, "module_training_gate", None).flags) if hasattr(self, "module_training_gate") else {},
            "module_training_seq": int(getattr(self, "last_module_training_seq", 0)),
            "module_trainable_total": int(getattr(self, "module_training_gate", None).count_trainable().get("total", 0)) if hasattr(self, "module_training_gate") else 0,
            "contact_damping": bool(self.cfg.dynamic_agent_rig.contact_angular_damping_enabled),
            "vestibular_norm": float(obs.get("vestibular", torch.zeros(1, 24, device=self.device)).norm(dim=-1).mean().detach().cpu().item()),
            "balance_reward": float(obs.get("balance_reward", torch.zeros(1, device=self.device)).mean().detach().cpu().item()),
            "flight_z": float(getattr(self.world, "cam_pos", np.zeros(3))[2]),
            "roll_deg": float(getattr(self.world, "roll_deg", 0.0)),
            "head_yaw": float(getattr(self.world, "head_ctrl", np.zeros(3))[0]),
            "head_pitch": float(getattr(self.world, "head_ctrl", np.zeros(3))[1]),
            "head_roll": float(getattr(self.world, "head_ctrl", np.zeros(3))[2]),
            "self_agency": float(out.get("self_core", {}).get("agency_score", torch.zeros(1, device=self.device)).mean().detach().cpu().item()) if isinstance(out.get("self_core"), dict) else 0.0,
            "self_ownership": float(out.get("self_core", {}).get("body_ownership_score", torch.zeros(1, device=self.device)).mean().detach().cpu().item()) if isinstance(out.get("self_core"), dict) else 0.0,
            "self_continuity": float(out.get("self_core", {}).get("self_continuity_score", torch.zeros(1, device=self.device)).mean().detach().cpu().item()) if isinstance(out.get("self_core"), dict) else 0.0,
            "self_core_present": bool(hasattr(self, "self_core") and self.self_core is not None),
            "self_core_trainable": int(sum(p.numel() for p in self.self_core.parameters() if p.requires_grad)) if hasattr(self, "self_core") and self.self_core is not None else 0,
            "flight_z_min": float(self.cfg.mocap_flight_bounds.min_z),
            "flight_z_max": float(self.cfg.mocap_flight_bounds.max_z),
            "body_state_dim": int(obs.get("body_state", torch.zeros(1, 0, device=self.device)).shape[-1]),
            "leg_ctrl_norm": float(out.get("leg_ctrl", self.prev_leg_motor).norm(dim=-1).mean().detach().cpu().item()),
            "embodied_dim": int(self.cfg.embodied_dim),
            "touch_sum": float(obs["tactile"].sum().item()),
            "object_repr_norm": float(out["object_repr"].norm(dim=-1).mean().item()),
            "memory_used": float(out["memory"]["memory_usage"].sum().item()),
            "modality_weights": out["attention"]["modality_weights"][0].detach().cpu().numpy(),
            "sphere": self.world.get_object_pos("sphere"),
        }

        self.write_module_debug_status()
        # Read IPC again right before drawing, so buttons feel responsive.
        self.poll_ipc_control_messages()
        self.update_camera_preview_window(obs)
        self.update_action_output_window()
        
        # Legacy cv2 manual action window removed; PyQt Agent Actions sends IPC instead.
        self.update_inner_object_window(obs, out)
        self.update_inner_object_open3d_window(out)
        self.update_inner_world_window(out)
        self.update_latent_semantic_window(out)

        if self.global_step % self.cfg.life.report_every_steps == 0:
            serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.latest_stats.items()}
            self.log_event({"step": self.global_step, "event": "life_tick", **serial})

        if hasattr(self, "log_tetra_life_tick_diagnostics"):
            self.log_tetra_life_tick_diagnostics()

        # Periodic checkpoint save.
        # This is the missing link for checkpoint_load.save_path:
        # checkpointing.py knows where to save, but life_step must actually call it.
        try:
            self.maybe_save_checkpoint(force=False, owner="life")
        except AttributeError:
            # Older CheckpointingMixin without maybe_save_checkpoint.
            pass
        except Exception as e:
            if self.global_step % max(1, self.cfg.life.report_every_steps) == 0:
                print(f"[checkpoint] periodic save failed: {e}")

        self.global_step += 1
