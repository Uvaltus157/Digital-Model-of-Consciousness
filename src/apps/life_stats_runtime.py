from __future__ import annotations

import numpy as np
import torch


class LifeStatsRuntimeMixin:
    def _life_tensor_float(self, value, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().float().reshape(-1)[0].cpu().item())
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _life_tensor_bool(self, value, default: bool = False) -> bool:
        return bool(self._life_tensor_float(value, 1.0 if default else 0.0) > 0.5)

    def build_latest_life_stats(
        self,
        *,
        obs: dict,
        out: dict,
        emotion: dict,
        novelty_score: float,
        chosen_action: int,
        decoded_report: str,
        target_report: str,
        inner_report_confidence: float,
        self_confidence: float,
    ) -> dict:
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        inner_object = out.get("inner_object", {}) if isinstance(out.get("inner_object"), dict) else {}
        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        metacog = out.get("metacognition", {}) if isinstance(out.get("metacognition"), dict) else {}
        memory13 = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}
        conscious_action = out.get("conscious_action", {}) if isinstance(out.get("conscious_action"), dict) else {}
        broadcast = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}
        thought_chain = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}

        stats = {
            "step": self.global_step,
            "training": bool(self.training_enabled) and not (self.is_full_sleep_mode() if hasattr(self, "is_full_sleep_mode") else False),
            "sensor_state": self.sensor_state_label() if hasattr(self, "sensor_state_label") else "unknown",
            "quality": self.quality.get(),
            "train_steps": self.train_steps,
            "focus_idx": int(out["focus"]["focus_idx"].item()),
            "action": int(chosen_action),
            "curiosity": float(out["values"]["curiosity"].item()),
            "coherence": float(out["values"]["coherence"].item()),
            "self_confidence": float(self_confidence),
            "inner_report_confidence": float(inner_report_confidence),
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
            "affect_latent_norm": float(affect.get("affect_latents", torch.zeros(1, device=self.device)).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(affect.get("affect_latents")) else 0.0,
            "affect_panic": self._life_tensor_float(affect.get("panic_latent"), 0.0),
            "affect_comfort": self._life_tensor_float(affect.get("comfort_latent"), 0.0),
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
            "inner_action_active": bool(inner_object.get("inner_action_active", False)),
            "inner_action_confidence": self._life_tensor_float(inner_object.get("inner_action_confidence"), 0.0),
            "inner_trust_value": self._life_tensor_float(inner_object.get("inner_trust_value"), 0.0),
            "inner_trust_alpha": self._life_tensor_float(inner_object.get("inner_trust_alpha"), 0.0),
            "inner_trust_allowed": self._life_tensor_bool(inner_object.get("inner_trust_allowed"), False),
            "inner_trust_applied_to_policy": self._life_tensor_bool(inner_object.get("inner_trust_applied_to_policy"), False),
            "inner_trust_reason": str(inner_object.get("inner_trust_reason", "")),
            "passport_active": self._life_tensor_bool(inner_object.get("passport_active"), False),
            "passport_token": str(inner_object.get("passport_token", "")),
            "passport_count": self._life_tensor_float(inner_object.get("passport_count"), 0.0),
            "passport_replay_active": self._life_tensor_bool(inner_object.get("passport_replay_active"), False),
            "passport_second_order_decoded": self._life_tensor_bool(inner_object.get("passport_second_order_decoded"), False),
            "passport_debug_active": self._life_tensor_bool(inner_object.get("passport_debug_active"), False),
            "passport_debug_z_distance": self._life_tensor_float(inner_object.get("passport_debug_z_distance"), 0.0),
            "passport_debug_z_cosine": self._life_tensor_float(inner_object.get("passport_debug_z_cosine"), 0.0),
            "inner_real_action_trace_path": str(inner_object.get("inner_real_action_trace_path", "")),
            "inner_real_action_body_norm": float(inner_object.get("inner_action_body", torch.zeros(1, device=self.device)).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(inner_object.get("inner_action_body")) else 0.0,
            "inner_real_action_hand_norm": float(inner_object.get("inner_action_hand", torch.zeros(1, device=self.device)).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(inner_object.get("inner_action_hand")) else 0.0,
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
            "self_agency": self._life_tensor_float(self_core.get("agency_score"), 0.0),
            "self_ownership": self._life_tensor_float(self_core.get("body_ownership_score"), 0.0),
            "self_continuity": self._life_tensor_float(self_core.get("self_continuity_score"), 0.0),
            "self_core_present": bool(hasattr(self, "self_core") and self.self_core is not None),
            "self_core_trainable": int(sum(p.numel() for p in self.self_core.parameters() if p.requires_grad)) if hasattr(self, "self_core") and self.self_core is not None else 0,
            "flight_z_min": float(self.cfg.mocap_flight_bounds.min_z),
            "flight_z_max": float(self.cfg.mocap_flight_bounds.max_z),
            "body_state_dim": int(obs.get("body_state", torch.zeros(1, 0, device=self.device)).shape[-1]),
            "leg_ctrl_norm": float(out.get("leg_ctrl", self.prev_leg_motor).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(out.get("leg_ctrl", self.prev_leg_motor)) else 0.0,
            "embodied_dim": int(self.cfg.embodied_dim),
            "touch_sum": float(obs.get("tactile", torch.zeros(1, device=self.device)).sum().item()),
            "object_repr_norm": float(out.get("object_repr", torch.zeros(1, device=self.device)).norm(dim=-1).mean().item()) if torch.is_tensor(out.get("object_repr")) else 0.0,
            "memory_used": float(out.get("memory", {}).get("memory_usage", torch.zeros(1, device=self.device)).sum().item()) if isinstance(out.get("memory"), dict) and torch.is_tensor(out.get("memory", {}).get("memory_usage")) else 0.0,
            "modality_weights": out.get("attention", {}).get("modality_weights", torch.zeros(1, 1, device=self.device))[0].detach().cpu().numpy() if isinstance(out.get("attention"), dict) and torch.is_tensor(out.get("attention", {}).get("modality_weights")) else np.zeros(1, dtype=np.float32),
            "sphere": self.world.get_object_pos("sphere") if hasattr(self.world, "get_object_pos") else np.zeros(3, dtype=np.float32),
            "m10_selected_source": str(broadcast.get("selected_source", "")),
            "m10_priority": self._life_tensor_float(broadcast.get("priority"), 0.0),
            "m10_gate": self._life_tensor_float(broadcast.get("broadcast_gate"), 0.0),
            "m12_confidence": self._life_tensor_float(metacog.get("metacognitive_confidence"), 0.0),
            "m12_doubt": self._life_tensor_float(metacog.get("doubt"), 0.0),
            "m12_verify": self._life_tensor_float(metacog.get("verification_need"), 0.0),
            "m12_hold": self._life_tensor_float(metacog.get("action_hold"), 0.0),
            "m13_episode_count": self._life_tensor_float(memory13.get("episode_count", memory13.get("retrieved_episode_count")), 0.0),
            "m13_retrieval_relevance": self._life_tensor_float(memory13.get("retrieval_relevance"), 0.0),
            "m13_blended_into_focus": self._life_tensor_bool(memory13.get("blended_into_focus"), False),
            "m13_last_summary": str(memory13.get("last_summary", memory13.get("summary", ""))),
            "m14_action_scale": self._life_tensor_float(conscious_action.get("applied_action_scale"), 1.0),
            "m14_reason": str(conscious_action.get("reason", "")),
            "m15_best_chain_score": self._life_tensor_float(thought_chain.get("best_chain_score"), 0.0),
            "m15_predicted_affect_delta": self._life_tensor_float(thought_chain.get("predicted_affect_delta"), 0.0),
        }

        inner_diag = getattr(self, "_latest_inner_speech_diagnostics", {})
        if isinstance(inner_diag, dict):
            stats.update(inner_diag)
        return stats

    def finalize_life_step_side_effects(self, *, obs: dict, out: dict, emotion: dict, decoded_report: str, target_report: str, inner_report_confidence: float) -> None:
        self.write_module_debug_status()
        self.poll_ipc_control_messages()

        calls = [
            ("maybe_print_action_signal_trace", (out,), {}),
            ("maybe_print_emotional_drive_trace", (emotion,), {}),
            ("maybe_print_thought_chain_trace", (out,), {}),
            ("maybe_print_inner_speech_trace", (out,), {}),
            ("maybe_print_global_broadcast_trace", (out,), {}),
            ("maybe_print_metacognition_trace", (out,), {}),
            ("maybe_print_autobiographical_memory_trace", (out,), {}),
            ("maybe_print_semantic_action_trace", (out,), {}),
            ("maybe_print_inner_object_trace", (out,), {}),
            ("maybe_update_inner_world", (obs, out, decoded_report, target_report, inner_report_confidence), {}),
            ("maybe_update_camera_preview", (obs, out), {"show_window": self.show_camera_preview_window}),
            ("maybe_update_action_outputs", (obs, out), {}),
            ("maybe_update_inner_object_visualizer", (obs, out), {}),
            ("maybe_update_inner_object_open3d", (obs, out), {}),
            ("maybe_update_latent_semantic_map", (obs, out), {}),
            ("maybe_update_static_dynamic_code_visualizer", (obs, out), {}),
            ("maybe_update_event_code_visualizer", (obs, out), {}),
        ]
        for name, args, kwargs in calls:
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    warn = f"_{name}_warned"
                    if not hasattr(self, warn):
                        print(f"[{name}] skipped: {e}")
                        setattr(self, warn, True)

        if self.global_step % self.cfg.life.report_every_steps == 0:
            serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.latest_stats.items()}
            self.log_event({"step": self.global_step, "event": "life_tick", **serial})

        if hasattr(self, "log_tetra_life_tick_diagnostics"):
            self.log_tetra_life_tick_diagnostics()

        try:
            self.maybe_save_checkpoint(force=False, owner="life")
        except AttributeError:
            pass
        except Exception as e:
            if self.global_step % max(1, self.cfg.life.report_every_steps) == 0:
                print(f"[checkpoint] periodic save failed: {e}")

        self.train_once_if_ready()


__all__ = ["LifeStatsRuntimeMixin"]
