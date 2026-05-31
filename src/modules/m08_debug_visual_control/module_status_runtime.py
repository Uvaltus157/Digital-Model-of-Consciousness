from __future__ import annotations

import time
import torch.optim as optim


class ModuleStatusRuntimeMixin:
    def _module_debug_body_xpos(self, body_name: str):
        world = getattr(self, "world", None)
        if world is None or not hasattr(world, "model") or not hasattr(world, "data"):
            return []
        try:
            import mujoco

            bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
            if bid >= 0:
                return world.data.xpos[bid].tolist()
        except Exception:
            pass
        return []

    def _module_debug_body_mocap_pos(self, body_name: str):
        world = getattr(self, "world", None)
        if world is None or not hasattr(world, "data"):
            return []
        try:
            mid = world._mocap_id_for_body(str(body_name), fallback=-1)
            if int(mid) >= 0:
                return world.data.mocap_pos[int(mid)].tolist()
        except Exception:
            pass
        return []

    def _module_debug_actuator_ctrls(self, names):
        world = getattr(self, "world", None)
        if world is None or not hasattr(world, "model") or not hasattr(world, "data"):
            return {}
        try:
            import mujoco

            out = {}
            for name in names:
                aid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name))
                if aid >= 0:
                    out[str(name)] = float(world.data.ctrl[aid])
                else:
                    out[str(name)] = None
            return out
        except Exception:
            return {}

    def _module_debug_joint_qpos(self, names):
        world = getattr(self, "world", None)
        if world is None or not hasattr(world, "model") or not hasattr(world, "data"):
            return {}
        try:
            import mujoco

            out = {}
            for name in names:
                jid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                if jid >= 0:
                    adr = int(world.model.jnt_qposadr[jid])
                    out[str(name)] = float(world.data.qpos[adr])
                else:
                    out[str(name)] = None
            return out
        except Exception:
            return {}

    def rebuild_optimizer_from_trainable_modules(self):
        params = [p for _name, p in self.module_training_gate.trainable_named_parameters()]
        if not params:
            print("[module_debug] no trainable parameters selected; optimizer keeps one frozen dummy-free state")
            return
        self.optimizer = optim.AdamW(
            params,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        counts = self.module_training_gate.count_trainable()
        print(f"[module_debug] optimizer rebuilt | trainable_total={counts.get('total', 0):,} | flags={self.module_training_gate.flags}")


    def write_module_debug_status(self):
        try:
            if not hasattr(self, "module_training_gate"):
                return

            counts = self.module_training_gate.count_trainable()
            full_sleep = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
            sensor_state = self.sensor_state_label() if hasattr(self, "sensor_state_label") else "unknown"
            training_enabled = bool(getattr(self, "training_enabled", False))
            cfg_train_enabled = bool(getattr(self.cfg.train, "enabled", False))
            effective_training = bool(training_enabled and cfg_train_enabled and not full_sleep)

            payload = {
                "updated_at": time.time(),
                "global_step": int(getattr(self, "global_step", 0)),
                "train_steps": int(getattr(self, "train_steps", 0)),

                # Window/runtime toggles reflected back to PyQt control panels.
                "mujoco_next_run": bool(getattr(getattr(self, "cfg", None), "viewer", None).allow_mujoco_window) if getattr(getattr(self, "cfg", None), "viewer", None) is not None else False,
                "inner_world": bool(getattr(self, "show_inner_world_window", False)),
                "cameras": bool(getattr(self, "show_camera_preview_window", False)),
                "depth": True,
                "actions": bool(getattr(self, "show_action_outputs_window", False)),
                "manual_actions": bool(getattr(self, "_ipc_manual_actions_enabled", False)),
                "object_image": bool(getattr(self, "show_inner_object_window", False)),
                "event_code_visualizer": bool(getattr(self, "show_event_code_visualizer_window", False)),
                "object_image_open3d": bool(getattr(self, "show_inner_object_open3d_window", False)),
                "static_dynamic_code": bool(getattr(self, "show_static_dynamic_code_window", False)),
                "latent_semantic": bool(getattr(self, "show_latent_semantic_window", False)),
                "module_debug": bool(getattr(self, "show_module_debug_window", False)),

                # Training state.
                "training_enabled": training_enabled,       # user/runtime switch
                "cfg_train_enabled": cfg_train_enabled,     # config switch
                "effective_training": effective_training,   # actually allowed now
                "training": effective_training,             # backward-compatible UI field
                "full_sleep": full_sleep,
                "sensor_state": sensor_state,

                "last_module_training_seq": int(getattr(self, "last_module_training_seq", 0)),
                "replay_len": int(len(self.replay)) if hasattr(self, "replay") else 0,
                "replay_min_ready": int(getattr(self.cfg.replay, "min_ready", 0)),
                "replay_ready": bool((len(self.replay) if hasattr(self, "replay") else 0) >= int(getattr(self.cfg.replay, "min_ready", 0))),
                "last_train_reason": str(getattr(self, "last_train_reason", "")),
                "last_train_loss": float(getattr(self, "last_train_loss", 0.0) or 0.0),
                "last_train_error": str(getattr(self, "last_train_error", "")),
                "object_decoder_stats": dict(getattr(self, "latest_object_decoder_stats", {})),
                "long_dynamic_memory_learning": dict(getattr(self, "latest_long_dynamic_memory_stats", {})),
                "long_dynamic_memory_status": {
                    "present": bool(hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None),
                    "train_flag": bool(getattr(getattr(self, "module_training_gate", None), "flags", {}).get("long_dynamic_memory", False)),
                    "params": int(sum(p.numel() for p in self.long_dynamic_object_memory.parameters())) if hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None else 0,
                    "trainable": int(sum(p.numel() for p in self.long_dynamic_object_memory.parameters() if p.requires_grad)) if hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None else 0,
                    "loss": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("loss", 0.0) or 0.0),
                    "loss_ema": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("loss_ema", 0.0) or 0.0),
                    "reward_proxy": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("reward_proxy", 0.0) or 0.0),
                    "recon": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("recon", 0.0) or 0.0),
                    "smooth": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("smooth", 0.0) or 0.0),
                    "z_static_norm": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("z_static_norm", 0.0) or 0.0),
                    "z_dynamic_norm": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("z_dynamic_norm", 0.0) or 0.0),
                },

                # Self/module status.
                "self_core_present": bool(hasattr(self, "self_core") and self.self_core is not None),
                "self_core_enabled": bool(getattr(self.cfg.self_core, "enabled", False)),
                "self_core_trainable": int(sum(p.numel() for p in self.self_core.parameters() if p.requires_grad)) if hasattr(self, "self_core") and self.self_core is not None else 0,

                # Manual action status.
                "manual_actions_enabled": bool(getattr(self, "_ipc_manual_actions_enabled", False)),
                "manual_body_action": self._ipc_manual_body_action.tolist() if getattr(self, "_ipc_manual_body_action", None) is not None else [],
                "manual_arm_action": self._ipc_manual_arm_action.tolist() if getattr(self, "_ipc_manual_arm_action", None) is not None else [],
                "manual_hand_action": self._ipc_manual_hand_action.tolist() if getattr(self, "_ipc_manual_hand_action", None) is not None else [],
                "manual_leg_action": self._ipc_manual_leg_action.tolist() if getattr(self, "_ipc_manual_leg_action", None) is not None else [],
                "manual_body_len": int(len(self._ipc_manual_body_action)) if getattr(self, "_ipc_manual_body_action", None) is not None else 0,
                "manual_arm_len": int(len(self._ipc_manual_arm_action)) if getattr(self, "_ipc_manual_arm_action", None) is not None else 0,
                "manual_hand_len": int(len(self._ipc_manual_hand_action)) if getattr(self, "_ipc_manual_hand_action", None) is not None else 0,
                "manual_leg_len": int(len(self._ipc_manual_leg_action)) if getattr(self, "_ipc_manual_leg_action", None) is not None else 0,
                "world_cam_pos": getattr(getattr(self, "world", None), "cam_pos", []).tolist() if hasattr(getattr(self, "world", None), "cam_pos") else [],
                "world_yaw_deg": float(getattr(getattr(self, "world", None), "yaw_deg", 0.0) or 0.0),
                "world_pitch_deg": float(getattr(getattr(self, "world", None), "pitch_deg", 0.0) or 0.0),
                "world_roll_deg": float(getattr(getattr(self, "world", None), "roll_deg", 0.0) or 0.0),
                "agent_rig_xpos": self._module_debug_body_xpos("agent_rig"),
                "agent_rig_mocap_pos": self._module_debug_body_mocap_pos("agent_rig"),
                "arm_actuator_ctrl": self._module_debug_actuator_ctrls([
                    "act_left_shoulder_yaw",
                    "act_left_shoulder_pitch",
                    "act_left_elbow",
                    "act_right_shoulder_yaw",
                    "act_right_shoulder_pitch",
                    "act_right_elbow",
                ]),
                "arm_joint_qpos": self._module_debug_joint_qpos([
                    "left_shoulder_yaw",
                    "left_shoulder_pitch",
                    "left_elbow_hinge",
                    "right_shoulder_yaw",
                    "right_shoulder_pitch",
                    "right_elbow_hinge",
                ]),
                "hand_actuator_ctrl_sample": self._module_debug_actuator_ctrls([
                    "act_left_palm_roll",
                    "act_left_palm_pitch",
                    "act_left_index_mcp",
                    "act_left_index_pip",
                    "act_right_palm_roll",
                    "act_right_palm_pitch",
                    "act_right_index_mcp",
                    "act_right_index_pip",
                ]),
                "hand_joint_qpos_sample": self._module_debug_joint_qpos([
                    "left_palm_roll",
                    "left_palm_pitch",
                    "left_index_mcp",
                    "left_index_pip",
                    "right_palm_roll",
                    "right_palm_pitch",
                    "right_index_mcp",
                    "right_index_pip",
                ]),
                "leg_actuator_ctrl_sample": self._module_debug_actuator_ctrls([
                    "act_left_hip_yaw",
                    "act_left_hip_pitch",
                    "act_left_knee",
                    "act_right_hip_yaw",
                    "act_right_hip_pitch",
                    "act_right_knee",
                ]),
                "leg_joint_qpos_sample": self._module_debug_joint_qpos([
                    "left_hip_yaw",
                    "left_hip_pitch",
                    "left_knee",
                    "right_hip_yaw",
                    "right_hip_pitch",
                    "right_knee",
                ]),
                "latest_embodied": getattr(getattr(self, "world", None), "latest_embodied", []).tolist() if hasattr(getattr(self, "world", None), "latest_embodied") else [],
                "latest_hand_ctrl": getattr(getattr(self, "world", None), "latest_hand_ctrl", []).tolist() if hasattr(getattr(self, "world", None), "latest_hand_ctrl") else [],

                # Action preset/scenario status for PyQt Agent Actions.
                "adaptive_gesture_status": dict(getattr(self, "_adaptive_gesture_status", {}) or {}),
                "adaptive_scenario_status": dict(getattr(self, "_fly_to_cube_palpate_status", {}) or {}),
                "adaptive_scenario_active": bool(getattr(self, "_fly_to_cube_palpate_active", False)),

                # Sleep/sensor state for PyQt checkboxes.
                "input_sensors_enabled": self.input_sensors_enabled_dict() if hasattr(self, "input_sensors_enabled_dict") else {
                    "video": bool(getattr(self, "video_sensor_enabled", True)),
                    "contact": bool(getattr(self, "contact_sensor_enabled", True)),
                    "imu": bool(getattr(self, "imu_sensor_enabled", True)),
                },
                "sleep_sensor_mask": self.sleep_sensor_mask_dict() if hasattr(self, "sleep_sensor_mask_dict") else {
                    "video": not bool(getattr(self, "video_sensor_enabled", True)),
                    "contact": not bool(getattr(self, "contact_sensor_enabled", True)),
                    "imu": not bool(getattr(self, "imu_sensor_enabled", True)),
                },
                "video_sensor_enabled": bool(getattr(self, "video_sensor_enabled", True)),
                "contact_sensor_enabled": bool(getattr(self, "contact_sensor_enabled", True)),
                "imu_sensor_enabled": bool(getattr(self, "imu_sensor_enabled", True)),
                "sleep_video_cut": not bool(getattr(self, "video_sensor_enabled", True)),
                "sleep_contact_cut": not bool(getattr(self, "contact_sensor_enabled", True)),
                "sleep_imu_cut": not bool(getattr(self, "imu_sensor_enabled", True)),


                # Long dynamic object memory status/checker.
                "long_dynamic_memory_status": {
                    "present": bool(hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None),
                    "train_flag": bool(getattr(self.module_training_gate, "flags", {}).get("long_dynamic_memory", False)) if hasattr(self, "module_training_gate") else False,
                    "trainable": int(sum(p.numel() for p in self.long_dynamic_object_memory.parameters() if p.requires_grad)) if hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None else 0,
                    "params": int(sum(p.numel() for p in self.long_dynamic_object_memory.parameters())) if hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None else 0,
                    "ready": bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("dynamic_ready", False)),
                    "write": bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("slot_update_allowed", False)),
                    "motion_ok": bool(float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_motion_ok", 0.0)) > 0.5),
                    "dz": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_dz", 0.0) or 0.0),
                    "dz_threshold": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_dz_threshold", 0.0) or 0.0),
                    "confidence": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_confidence", 0.0) or 0.0),
                    "ready_streak": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_ready_streak", 0.0) or 0.0),
                    "steps": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_steps", 0.0) or 0.0),
                    "active_steps": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_active_steps", 0.0) or 0.0),
                    "last_loss": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("loss", 0.0) or 0.0),
                    "last_recon": float((getattr(self, "latest_long_dynamic_memory_stats", {}) or {}).get("recon", 0.0) or 0.0),
                    "dynamic_object_confidence": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_confidence", 0.0) or 0.0),
                    "object_formed_confidence": float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_confidence", 0.0) or 0.0) if (bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("dynamic_ready", False)) and bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("slot_update_allowed", False))) else 0.0,
                    "object_formed_ready": bool(bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("dynamic_ready", False)) and bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("slot_update_allowed", False))),
                    "checker_static_ok": bool(
                        not bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("dynamic_ready", False))
                        and not bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("slot_update_allowed", False))
                        and float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_dz", 0.0) or 0.0)
                            < max(float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_dz_threshold", 0.003) or 0.003), 1e-9)
                    ),
                    "checker_dynamic_ok": bool(
                        bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("dynamic_ready", False))
                        and bool((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("slot_update_allowed", False))
                        and float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_dz", 0.0) or 0.0)
                            >= max(float((getattr(self, "_inner_object_dynamic_debug", {}) or {}).get("long_dynamic_dz_threshold", 0.003) or 0.003), 1e-9)
                    ),
                },

                # Module training checkboxes and trainable counts.
                "module_training": dict(self.module_training_gate.flags),
                "trainable_counts": counts,
                "quality": float(self.quality.get()) if hasattr(self, "quality") else 0.0,
            }

            if getattr(self, "latest_stats", None) is not None:
                for k in [
                    "self_agency",
                    "self_ownership",
                    "self_continuity",
                    "novelty_score",
                    "curiosity",
                    "coherence",
                    "inner_report_confidence",
                    "leg_ctrl_norm",
                    "vestibular_norm",
                    "balance_reward",
                    "object_decoder_loss",
                    "object_rgb_loss",
                    "object_depth_loss",
                    "object_mask_loss",
                ]:
                    if k in self.latest_stats:
                        try:
                            payload[k] = float(self.latest_stats[k])
                        except Exception:
                            pass

            inner_diag = getattr(self, "_latest_inner_speech_diagnostics", {})
            if isinstance(inner_diag, dict):
                for k in [
                    "inner_speech_source",
                    "uses_thought_chain",
                    "uses_self_bound_context",
                    "uses_affect_latents",
                    "thought_chain_planning_readiness",
                ]:
                    if k in inner_diag:
                        payload[k] = inner_diag[k]
                        if getattr(self, "latest_stats", None) is not None:
                            try:
                                self.latest_stats[k] = inner_diag[k]
                            except Exception:
                                pass

            if getattr(self, "module_status_server", None) is not None:
                self.module_status_server.update_status(payload)
        except Exception as e:
            if getattr(self, "global_step", 0) % 200 == 0:
                print(f"[module_debug] status IPC update failed: {e}")


    def set_module_training_flags(self, flags: dict):
        if not hasattr(self, "module_training_gate"):
            return
        before = dict(self.module_training_gate.flags)
        self.module_training_gate.set_flags(flags)
        if before != self.module_training_gate.flags:
            self.rebuild_optimizer_from_trainable_modules()
        self.write_module_debug_status()


    def toggle_module_training_flag(self, key: str):
        if not hasattr(self, "module_training_gate"):
            return
        if key in self.module_training_gate.flags:
            self.module_training_gate.toggle(key)
            self.rebuild_optimizer_from_trainable_modules()
            self.write_module_debug_status()
