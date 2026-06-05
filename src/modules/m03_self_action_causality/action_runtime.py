from __future__ import annotations

import numpy as np
import torch
from src.modules.m03_self_action_causality.adaptive_gesture_controller import ensure_adaptive_gesture_controller, is_adaptive_gesture_command
from src.platform.gui.opencv_gui_thread import close_cv2_window
from src.modules.m15_counterfactual_imagination_planning.adaptive_scenario_controller import AdaptiveScenarioController


class ActionRuntimeMixin:
    def _sanitize_manual_vector(self, value, size: int, fill: float = 0.0, lo: float | None = None, hi: float | None = None) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.full(int(size), float(fill), dtype=np.float32)
        n = min(int(size), int(arr.size))
        if n > 0:
            out[:n] = arr[:n]
        out = np.nan_to_num(out, nan=float(fill), posinf=float(fill), neginf=float(fill)).astype(np.float32)
        if lo is not None or hi is not None:
            out = np.clip(out, -np.inf if lo is None else float(lo), np.inf if hi is None else float(hi)).astype(np.float32)
        return out

    def _close_inner_world_gui_windows(self) -> None:
        try:
            close_cv2_window("dreamer inner world v3")
            close_cv2_window("dreamer inner world")
        except Exception:
            pass

    def _close_latent_semantic_gui_window(self) -> None:
        try:
            if hasattr(self, "latent_semantic_viz") and self.latent_semantic_viz is not None:
                self.latent_semantic_viz.close()
        except Exception:
            pass
        try:
            close_cv2_window(self.cfg.latent_semantic_map.window_name)
        except Exception:
            pass

    def _close_event_code_visualizer_window(self) -> None:
        try:
            if hasattr(self, "event_code_viz") and self.event_code_viz is not None:
                self.event_code_viz.close()
        except Exception:
            pass
        try:
            close_cv2_window(self.cfg.event_code_visualizer.window_name)
        except Exception:
            pass

    def apply_ipc_set_state(self, state: dict):
        self.apply_sleep_sensor_state(state)
        manual_keys = {
            "manual_actions_enabled",
            "manual_body_action",
            "manual_arm_action",
            "manual_hand_action",
            "manual_leg_action",
        }
        if any(k in state for k in manual_keys):
            # Any direct manual slider update takes control back from the scripted scenario.
            ctrl = getattr(self, "adaptive_scenario_controller", None)
            if ctrl is not None and getattr(ctrl, "active", None) is not None:
                self.stop_fly_to_cube_palpate_scenario("manual slider input")
            else:
                self._fly_to_cube_palpate_active = False
                self._set_fly_to_cube_status(active=False, phase="manual_override", reason="manual slider input")
            self._adaptive_gesture_status = {"active": False, "reason": "manual slider input"}
        if "mujoco_next_run" in state:
            self.cfg.viewer.allow_mujoco_window = bool(state["mujoco_next_run"])
        if "inner_world" in state:
            self.show_inner_world_window = bool(state["inner_world"])
            if not self.show_inner_world_window:
                self._close_inner_world_gui_windows()
            print(f"[ipc] inner_world={self.show_inner_world_window}")
        if "cameras" in state:
            self.show_camera_preview_window = bool(state["cameras"])
            self.camera_preview_armed = bool(state["cameras"])
            if not self.show_camera_preview_window:
                try:
                    close_cv2_window(self.cfg.camera_preview.window_name)
                    close_cv2_window(self._action_window_name())
                    self._action_window_created = False
                except Exception:
                    pass
            # _camera_preview_gui_thread_close_on_toggle
            if not self.show_camera_preview_window:
                try:
                    close_cv2_window(self.cfg.camera_preview.window_name)
                except Exception:
                    pass
            print(f"[ipc] cameras={self.show_camera_preview_window} armed={self.camera_preview_armed}")
        if hasattr(self.cfg, "camera_preview"):
            self.cfg.camera_preview.show_depth = True
        if "actions" in state:
            self.show_action_outputs_window = bool(state["actions"])
            if not self.show_action_outputs_window:
                self.close_action_outputs_window()
            print(f"[ipc] actions={self.show_action_outputs_window}")
    
        if "module_training" in state:
            mt = state.get("module_training", {})
            seq = state.get("module_training_seq", None)
            if seq is not None:
                try:
                    self.last_module_training_seq = int(seq)
                except Exception:
                    pass
            if isinstance(mt, dict):
                self.set_module_training_flags(mt)
                print(f"[ipc] module_training seq={self.last_module_training_seq} flags={self.module_training_gate.flags}")
        if "manual_actions" in state:
            # Backward-compatible alias only. No cv2 window is opened anymore.
            self.show_manual_action_override_window = False
            self._ipc_manual_actions_enabled = bool(state["manual_actions"])
            print(f"[ipc] manual_actions alias -> manual_actions_enabled={self._ipc_manual_actions_enabled}")

        if "manual_actions_enabled" in state:
            old_enabled = bool(getattr(self, "_ipc_manual_actions_enabled", False))
            self._ipc_manual_actions_enabled = bool(state["manual_actions_enabled"])
            if self._ipc_manual_actions_enabled and not old_enabled:
                print("[ipc] manual control enabled: neural outputs now overridden by PyQt sliders")
            print(f"[ipc] manual_actions_enabled={self._ipc_manual_actions_enabled}")

        if "manual_body_action" in state:
            try:
                self._ipc_manual_body_action = self._sanitize_manual_vector(state["manual_body_action"], 9, fill=0.0)
                self._last_manual_body_action = self._ipc_manual_body_action
            except Exception as e:
                print(f"[ipc] bad manual_body_action: {e}")

        if "manual_arm_action" in state:
            try:
                self._ipc_manual_arm_action = self._sanitize_manual_vector(state["manual_arm_action"], 6, fill=0.0, lo=-1.0, hi=1.0)
            except Exception as e:
                print(f"[ipc] bad manual_arm_action: {e}")

        if "manual_hand_action" in state:
            try:
                self._ipc_manual_hand_action = self._sanitize_manual_vector(state["manual_hand_action"], 44, fill=0.5, lo=0.0, hi=1.0)
            except Exception as e:
                print(f"[ipc] bad manual_hand_action: {e}")

        if "manual_leg_action" in state:
            try:
                self._ipc_manual_leg_action = self._sanitize_manual_vector(state["manual_leg_action"], 18, fill=0.0, lo=-1.0, hi=1.0)
            except Exception as e:
                print(f"[ipc] bad manual_leg_action: {e}")
        if "object_image" in state:
            self.show_inner_object_window = bool(state["object_image"])
            if not self.show_inner_object_window:
                try:
                    self.inner_object_viz.close()
                except Exception:
                    pass
            print(f"[ipc] object_image={self.show_inner_object_window}")
        if "inner_object_dream_slot_index" in state:
            try:
                slot_idx = int(state["inner_object_dream_slot_index"])
                n_slots = int(getattr(self.cfg.object_image, "num_slots", 10))
                slot_idx = max(0, min(slot_idx, max(0, n_slots - 1)))
                if getattr(self, "inner_object_viz", None) is not None:
                    self.inner_object_viz.requested_dream_slot_index = slot_idx
                self._ipc_inner_object_dream_slot_index = slot_idx
                print(f"[ipc] inner_object_dream_slot_index={slot_idx}")
            except Exception as e:
                print(f"[ipc] bad inner_object_dream_slot_index: {e}")
        if "event_code_visualizer" in state:
            self.show_event_code_visualizer_window = bool(state["event_code_visualizer"])
            if not self.show_event_code_visualizer_window:
                self._close_event_code_visualizer_window()
            print(f"[ipc] event_code_visualizer={self.show_event_code_visualizer_window}")
        if "level_agent_pose" in state and bool(state["level_agent_pose"]):
            self.level_agent_pose()
            print("[ipc] level_agent_pose requested")

        if "object_image_open3d" in state:
            self.show_inner_object_open3d_window = bool(state["object_image_open3d"])
            if not self.show_inner_object_open3d_window:
                try:
                    self.inner_object_open3d_viz.close()
                except Exception:
                    pass
            print(f"[ipc] object_image_open3d={self.show_inner_object_open3d_window}")
        if "training" in state:
            self.training_enabled = bool(state["training"])
            print(f"[ipc] training={self.training_enabled}")
            self.write_module_debug_status()
        if "latent_semantic" in state:
            self.show_latent_semantic_window = bool(state["latent_semantic"])
            if not self.show_latent_semantic_window:
                self._close_latent_semantic_gui_window()
            print(f"[ipc] latent_semantic={self.show_latent_semantic_window}")

        if "static_dynamic_code" in state:
            self.show_static_dynamic_code_window = bool(state["static_dynamic_code"])
            if not self.show_static_dynamic_code_window:
                try:
                    self.static_dynamic_code_viz.close()
                except Exception:
                    pass
            print(f"[ipc] static_dynamic_code={self.show_static_dynamic_code_window}")


    def apply_ipc_action(self, action: str, payload: dict | None = None):
        payload = dict(payload or {})
        if action == "close_aux":
            self.show_inner_world_window = False
            self.show_camera_preview_window = False
            self.camera_preview_armed = False
            self.show_action_outputs_window = False
            try:
                close_cv2_window(self.cfg.camera_preview.window_name)
            except Exception:
                pass
            try:
                close_cv2_window("dreamer inner world v3")
            except Exception:
                pass
            try:
                close_cv2_window("dreamer inner world")
            except Exception:
                pass
            try:
                self.inner_object_open3d_viz.close()

            except Exception:
                pass

            self.show_latent_semantic_window = False
            try:
                self.latent_semantic_viz.close()
            except Exception:
                pass

            self.show_static_dynamic_code_window = False
            try:
                self.static_dynamic_code_viz.close()
            except Exception:
                pass

            self.show_event_code_visualizer_window = False
            self._close_event_code_visualizer_window()
        elif action == "stop":
            self.shutdown = True
        elif action == "ping":
            print("[ipc] ping received")
        elif action in ("imit_action", "simulate_energy_resonator_action", "energy_resonator_imit_action"):
            if hasattr(self, "request_energy_resonator_imitation"):
                self.request_energy_resonator_imitation(payload)
            else:
                print("[ipc] imit_action ignored: EnergyResonatorRuntimeMixin is not installed")
        elif action in ("dream_probe_inject", "dream_probe_clear"):
            print(f"[dream_probe][ipc] action={action} payload={payload}")
            if hasattr(self, "request_dream_probe"):
                if action == "dream_probe_clear":
                    payload = {"kind": "clear", **dict(payload or {})}
                self.request_dream_probe(payload)
            else:
                print("[dream_probe] ignored: DreamProbeRuntimeMixin is not installed")
        elif action == "module_lab_run":
            try:
                from src.modules.m08_debug_visual_control.module_lab_runtime import run_module_lab_from_payload

                result = run_module_lab_from_payload(payload)
                self.last_module_lab_result = result
                status = "ok" if bool(result.get("ok", False)) else "fail"
                print(
                    "[module_lab] "
                    f"status={status} module={result.get('module')} "
                    f"kind={result.get('kind')} duration={result.get('duration_sec')}s"
                )
                if not bool(result.get("ok", False)):
                    print(f"[module_lab] error={result.get('error', '')}")
                if hasattr(self, "write_module_debug_status"):
                    self.write_module_debug_status()
            except Exception as e:
                self.last_module_lab_result = {
                    "ok": False,
                    "module": str(payload.get("module", "all")) if isinstance(payload, dict) else "all",
                    "kind": "module_lab",
                    "error": str(e),
                }
                print(f"[module_lab] failed: {e}")
                if hasattr(self, "write_module_debug_status"):
                    self.write_module_debug_status()
        elif is_adaptive_gesture_command(action):
            self.start_adaptive_gesture_command(action)
        elif action == "fly_to_cube_palpate":
            self.start_fly_to_cube_palpate_scenario()
        elif action == "fly_to_small_cube_grasp_rotate":
            self.start_fly_to_small_cube_grasp_rotate_scenario()
        elif action == "fly_to_tetrahedron_inspect":
            self.start_fly_to_tetrahedron_inspect_scenario(payload)
        elif action == "stop_fly_to_tetrahedron_inspect":
            self.stop_fly_to_tetrahedron_inspect_scenario("button toggle")
        elif action == "stop_adaptive_scenario":
            self.stop_fly_to_cube_palpate_scenario(str(payload.get("reason", "ipc stop")))
        elif action == "save_object_ply":
            self.export_inner_object_model("ply")
        elif action == "save_object_pcd":
            self.export_inner_object_model("pcd")
        elif action == "save_checkpoint":
            path = payload.get("path", None)
            if hasattr(self, "save_checkpoint"):
                checkpoint_path = str(path or self.checkpoint_path_for_save())
                if hasattr(self, "_tetra_diag_write"):
                    self._tetra_diag_write(
                        "ipc_save_weights_requested",
                        path=checkpoint_path,
                        reason=str(payload.get("reason", "ipc_action")),
                    )
                success = bool(self.save_checkpoint(path))
                if hasattr(self, "_tetra_diag_write"):
                    self._tetra_diag_write(
                        "ipc_save_weights_done",
                        path=checkpoint_path,
                        success=success,
                        reason=str(payload.get("reason", "ipc_action")),
                    )
                    if success:
                        try:
                            renderer = getattr(self, "slot_4d_playback_renderer", None)
                            previews = getattr(renderer, "last_preview", {}) or {}
                            metrics_by_slot = getattr(renderer, "last_metrics", {}) or {}

                            def has_tensor(slot_id: int, key: str) -> bool:
                                value = (previews.get(slot_id) or {}).get(key)
                                return bool(torch.is_tensor(value) and value.numel() > 0)

                            def tensor_shape(slot_id: int, key: str):
                                value = (previews.get(slot_id) or {}).get(key)
                                return tuple(int(x) for x in value.shape) if torch.is_tensor(value) else ()

                            def metric(slot_id: int, key: str, default=None):
                                m = metrics_by_slot.get(slot_id)
                                if isinstance(m, dict):
                                    return m.get(key, default)
                                return getattr(m, key, default)

                            def target_for_slot(slot_id: int, default: str) -> str:
                                target_by_slot = getattr(self, "_dynamic_slot_target_by_slot", {}) or {}
                                target = str(target_by_slot.get(int(slot_id), "") or "").strip()
                                if target:
                                    return target
                                target = str(metric(slot_id, "target_name", "") or "").strip()
                                if target and target != "unknown":
                                    return target
                                diag_targets = getattr(self, "_tetra_diag_slot_targets", {}) or {}
                                target = str(diag_targets.get(int(slot_id), "") or "").strip()
                                return target or default

                            slot0_ok = bool(metric(0, "render_valid", False)) and has_tensor(0, "rgb") and has_tensor(0, "depth") and has_tensor(0, "alpha")
                            slot1_ok = bool(metric(1, "render_valid", False)) and has_tensor(1, "rgb") and has_tensor(1, "depth") and has_tensor(1, "alpha")
                            if slot0_ok and slot1_ok:
                                self._tetra_diag_write(
                                    "SUCCESS_INNER_OBJECT_4D_PREVIEW_IMAGE_VISIBLE",
                                    slot_0_target=target_for_slot(0, "tetrahedron"),
                                    slot_0_has_rgb=1,
                                    slot_0_has_depth=1,
                                    slot_0_has_alpha=1,
                                    slot_0_render_valid=1,
                                    slot_0_rgb_shape=tensor_shape(0, "rgb"),
                                    slot_1_target=target_for_slot(1, "cube"),
                                    slot_1_has_rgb=1,
                                    slot_1_has_depth=1,
                                    slot_1_has_alpha=1,
                                    slot_1_render_valid=1,
                                    slot_1_rgb_shape=tensor_shape(1, "rgb"),
                                    panel="4D / 3DGS PREVIEW IMAGE",
                                    placement="row2_right_of_decoders",
                                    weights_saved_via_ipc=1,
                                    checkpoint_path=checkpoint_path,
                                )
                        except Exception:
                            pass
    def _ensure_adaptive_scenario_controller(self) -> AdaptiveScenarioController:
        if not hasattr(self, "adaptive_scenario_controller") or self.adaptive_scenario_controller is None:
            self.adaptive_scenario_controller = AdaptiveScenarioController(self)
        return self.adaptive_scenario_controller


    def _set_fly_to_cube_status(self, **kwargs) -> None:
        status = dict(getattr(self, "_fly_to_cube_palpate_status", {}) or {})
        status.update(kwargs)
        self._fly_to_cube_palpate_status = status


    def start_fly_to_cube_palpate_scenario(self) -> None:
        self._adaptive_gesture_status = {"active": False, "reason": "scenario started"}
        ctrl = self._ensure_adaptive_scenario_controller()
        ctrl.start("fly_to_cube_palpate")


    def start_fly_to_small_cube_grasp_rotate_scenario(self) -> None:
        self._adaptive_gesture_status = {"active": False, "reason": "scenario started"}
        ctrl = self._ensure_adaptive_scenario_controller()
        ctrl.start("fly_to_small_cube_grasp_rotate")


    def start_fly_to_tetrahedron_inspect_scenario(self, options: dict | None = None) -> None:
        self._adaptive_gesture_status = {"active": False, "reason": "scenario started"}
        ctrl = self._ensure_adaptive_scenario_controller()
        if isinstance(options, dict) and isinstance(options.get("options"), dict):
            options = dict(options.get("options") or {})
        ctrl.start("fly_to_tetrahedron_inspect", options=options)


    def stop_fly_to_cube_palpate_scenario(self, reason: str = "stopped") -> None:
        ctrl = self._ensure_adaptive_scenario_controller()
        ctrl.stop(reason)


    def stop_fly_to_tetrahedron_inspect_scenario(self, reason: str = "stopped") -> None:
        ctrl = self._ensure_adaptive_scenario_controller()
        active = getattr(ctrl, "active", None)
        if active is not None and getattr(active, "name", "") == "fly_to_tetrahedron_inspect":
            ctrl.stop(reason)
        else:
            self._fly_to_cube_palpate_active = False
            self._set_fly_to_cube_status(
                active=False,
                scenario="fly_to_tetrahedron_inspect",
                phase="stopped",
                reason=reason,
            )
        self._adaptive_gesture_status = {"active": False, "reason": reason}
        self._ipc_manual_actions_enabled = False


    def update_fly_to_cube_palpate_scenario(self) -> None:
        ctrl = self._ensure_adaptive_scenario_controller()
        ctrl.update()


    def start_adaptive_gesture_command(self, action: str) -> None:
        ctrl_scenario = getattr(self, "adaptive_scenario_controller", None)
        if ctrl_scenario is not None and getattr(ctrl_scenario, "active", None) is not None:
            ctrl_scenario.stop("gesture command")
            self._fly_to_cube_palpate_active = False
            self._fly_to_cube_palpate_status = dict(getattr(ctrl_scenario, "active", None).status) if getattr(ctrl_scenario, "active", None) is not None else {
                "active": False,
                "phase": "done",
                "reason": "gesture command",
            }
        ctrl = ensure_adaptive_gesture_controller(self)
        ctrl.start(action)


    def update_adaptive_gesture_controller(self) -> None:
        ctrl = getattr(self, "adaptive_gesture_controller", None)
        if ctrl is not None:
            ctrl.update()



    def poll_ipc_control_messages(self):
        if self.ipc_server is None:
            return
        if self.global_step % max(1, self.cfg.ipc_control.poll_every_steps) != 0:
            return

        high_level_action_seen = False
        manual_state_keys = {
            "manual_actions",
            "manual_actions_enabled",
            "manual_body_action",
            "manual_arm_action",
            "manual_hand_action",
            "manual_leg_action",
        }
        for msg in self.ipc_server.drain():
            msg_type = msg.get("type")
            if msg_type == "set_state":
                state = msg.get("state", {})
                if (
                    high_level_action_seen
                    and isinstance(state, dict)
                    and any(k in state for k in manual_state_keys)
                ):
                    print("[ipc] skipped stale manual set_state after high-level action")
                    continue
                self.apply_ipc_set_state(state)
            elif msg_type == "action":
                payload = msg.get("payload", {})
                action = str(msg.get("action", ""))
                self.apply_ipc_action(action, payload if isinstance(payload, dict) else {})
                if action.startswith("gesture_") or action.startswith("fly_to_") or action.startswith("stop_fly_to_"):
                    high_level_action_seen = True
            elif msg_type == "toggle":
                key = str(msg.get("key", ""))
                if key == "inner_world":
                    self.show_inner_world_window = not self.show_inner_world_window
                    if not self.show_inner_world_window:
                        try:
                            self._close_inner_world_gui_windows()
                        except Exception:
                            pass
                elif key == "cameras":
                    self.show_camera_preview_window = not self.show_camera_preview_window
                    self.camera_preview_armed = bool(self.show_camera_preview_window)
                elif key == "actions":
                    self.show_action_outputs_window = not self.show_action_outputs_window
                    if not self.show_action_outputs_window:
                        self.close_action_outputs_window()
                elif key == "manual_actions":
                    self._ipc_manual_actions_enabled = not bool(getattr(self, "_ipc_manual_actions_enabled", False))
                    self.show_manual_action_override_window = False
                    if not self._ipc_manual_actions_enabled:
                        self.stop_fly_to_cube_palpate_scenario("manual override disabled")
                    print(f"[ipc] manual_actions toggle alias -> manual_actions_enabled={self._ipc_manual_actions_enabled}")
     
                elif key == "object_image":
                    self.show_inner_object_window = not self.show_inner_object_window
                    if not self.show_inner_object_window:
                        try:
                            self.inner_object_viz.close()
                        except Exception:
                            pass
                elif key == "event_code_visualizer":
                    self.show_event_code_visualizer_window = not bool(getattr(self, "show_event_code_visualizer_window", False))
                    if not self.show_event_code_visualizer_window:
                        self._close_event_code_visualizer_window()
                elif key == "object_image_open3d":
                    self.show_inner_object_open3d_window = not self.show_inner_object_open3d_window
                    if not self.show_inner_object_open3d_window:
                        try:
                            self.inner_object_open3d_viz.close()
                        except Exception:
                            pass
                elif key == "training":
                    self.training_enabled = not self.training_enabled
                elif key == "mujoco_next_run":
                    print("poll_ipc_control_messages: cmd -> 'mujoco_next_run' == ", self.cfg.viewer.allow_mujoco_window)
                    self.cfg.viewer.allow_mujoco_window = not self.cfg.viewer.allow_mujoco_window
                elif key == "latent_semantic":
                    self.show_latent_semantic_window = not self.show_latent_semantic_window
                    if not self.show_latent_semantic_window:
                        try:
                            self.latent_semantic_viz.close()
                        except Exception:
                            pass

                elif key == "static_dynamic_code":
                    self.show_static_dynamic_code_window = not bool(getattr(self, "show_static_dynamic_code_window", False))
                    if not self.show_static_dynamic_code_window:
                        try:
                            self.static_dynamic_code_viz.close()
                        except Exception:
                            pass


    def _action_is_stuck(self, action_id: int) -> bool:
        if not hasattr(self, "_last_action_id"):
            self._last_action_id = action_id
            self._same_action_steps = 0
            return False
        if int(action_id) == int(self._last_action_id):
            self._same_action_steps += 1
        else:
            self._last_action_id = action_id
            self._same_action_steps = 0
        return self._same_action_steps > max(5, self.cfg.exploration.action_cycle_period)


    def apply_exploration_motor(self, out: dict, novelty_score: float = 1.0):
        """
        Bootstrap / anti-freeze motion.

        It does not replace the model policy; it only adds a soft exploratory
        component when the policy is near-zero, during warmup, or when novelty is low.
        """
        cfg = self.cfg.exploration
        if not cfg.enabled:
            return out

        emb = out["embodied_targets"].clone()
        hand = out["hand_ctrl"].clone()
        action_id = int(out["action_ids"].item())

        emb_norm = float(emb.detach().norm(dim=-1).mean().cpu().item())
        hand_norm = float(hand.detach().norm(dim=-1).mean().cpu().item())

        warmup = self.global_step < cfg.warmup_steps
        low_motion = emb_norm < cfg.min_embodied_norm
        low_hand = hand_norm < cfg.min_hand_norm
        low_novelty = novelty_score < cfg.low_novelty_threshold
        stuck_action = self._action_is_stuck(action_id)

        if not (warmup or low_motion or low_hand or low_novelty or stuck_action):
            return out

        t = float(self.global_step)
        boost = cfg.stuck_boost if (low_novelty or stuck_action) else 1.0
        base_amp = float(cfg.base_amp) * boost
        hand_amp = float(cfg.hand_amp) * boost

        # Embodied vector convention used by MujocoLiveWorld:
        # first dims behave as base movement / lift / yaw / pitch controls.
        pattern = torch.zeros_like(emb)
        if pattern.shape[-1] > 0:
            pattern[:, 0] = base_amp * np.sin(t * 0.035)
        if pattern.shape[-1] > 1:
            pattern[:, 1] = base_amp * 0.75 * np.cos(t * 0.027)
        if pattern.shape[-1] > 2:
            pattern[:, 2] = base_amp * 0.18 * np.sin(t * 0.019)
        if pattern.shape[-1] > 3:
            pattern[:, 3] = base_amp * 0.50 * np.sin(t * 0.021)
        if pattern.shape[-1] > 4:
            pattern[:, 4] = base_amp * 0.35 * np.cos(t * 0.017)

        if warmup or low_motion or low_novelty:
            emb = torch.clamp(emb + pattern.to(emb.device, emb.dtype), -1.0, 1.0)

        # Fingers/hands: small waves so tactile sensors can discover contacts.
        if warmup or low_hand or low_novelty:
            hpat = torch.zeros_like(hand)
            dims = hpat.shape[-1]
            for i in range(dims):
                hpat[:, i] = hand_amp * np.sin(t * (0.025 + 0.0015 * i) + i * 0.37)
            hand = torch.clamp(hand + hpat.to(hand.device, hand.dtype), -1.0, 1.0)

        if cfg.cycle_action_when_stuck and (warmup or stuck_action or low_novelty):
            action_id = int((self.global_step // max(1, cfg.action_cycle_period)) % self.cfg.action_dim)
            out["action_ids"] = torch.tensor([action_id], device=emb.device, dtype=torch.long)

        out["embodied_targets"] = emb
        out["hand_ctrl"] = hand

        out["exploration"] = {
            "active": True,
            "warmup": bool(warmup),
            "low_motion": bool(low_motion),
            "low_hand": bool(low_hand),
            "low_novelty": bool(low_novelty),
            "stuck_action": bool(stuck_action),
            "emb_norm_before": emb_norm,
            "hand_norm_before": hand_norm,
            "boost": float(boost),
        }
        return out


    def protect_manual_body_output(self, out: dict):
        """
        If manual override is active, keep embodied_targets[0:6] equal to the
        imitated neural-output vector after any safety/exploration post-processing.
        """
        manual = out.get("manual_action_override")
        if not isinstance(manual, dict) or "manual_body_action" not in manual:
            return out
        vec = manual["manual_body_action"]
        emb = out["embodied_targets"].clone()
        emb[0, 0:5] = torch.tensor(vec[:5], device=emb.device, dtype=emb.dtype)
        if emb.shape[-1] > 11:
            emb[0, 11] = torch.tensor(vec[5], device=emb.device, dtype=emb.dtype)
        else:
            emb[0, 5] = torch.tensor(vec[5], device=emb.device, dtype=emb.dtype)
        if emb.shape[-1] >= 15 and len(vec) >= 9:
            emb[0, 12:15] = torch.tensor(vec[6:9], device=emb.device, dtype=emb.dtype)
        out["embodied_targets"] = emb
        self._last_manual_body_action = np.asarray(vec, dtype=np.float32)
        self.trace_action_signal("after_exploration_protect", emb[0, :6].detach().cpu().numpy())
        return out


    def apply_manual_leg_action_override(self, out: dict) -> dict:
        if not getattr(self, "_ipc_manual_actions_enabled", False):
            return out
        if self._ipc_manual_leg_action is None:
            return out
        if "leg_ctrl" not in out or out["leg_ctrl"] is None:
            return out
        leg_vec = np.asarray(self._ipc_manual_leg_action, dtype=np.float32).reshape(-1)
        leg = out["leg_ctrl"].clone()
        n = min(leg.shape[-1], leg_vec.shape[0])
        leg[0, :n] = torch.tensor(leg_vec[:n], device=leg.device, dtype=leg.dtype)
        out["leg_ctrl"] = leg
        out["manual_leg_action"] = leg_vec.astype(np.float32)
        self.trace_action_signal("manual_leg_action", leg_vec[:min(12, len(leg_vec))])
        return out


    def level_agent_pose(self):
        """
        Reset/level the agent body pose for manual control.

        This is intentionally different from zero sliders:
        - zero sliders only means zero desired velocity/output
        - level_agent_pose corrects current world pose/orientation state
        """
        try:
            if hasattr(self, "world"):
                # For mocap/camera rig world.
                if hasattr(self.world, "roll_deg"):
                    self.world.roll_deg = 0.0
                if hasattr(self.world, "yaw_deg"):
                    self.world.yaw_deg = 0.0
                if hasattr(self.world, "pitch_deg"):
                    self.world.pitch_deg = 0.0
                if hasattr(self.world, "head_ctrl"):
                    self.world.head_ctrl[:] = 0.0

                # Keep current x/y but put z in safe hover range when available.
                if hasattr(self.world, "cam_pos"):
                    try:
                        if not np.all(np.isfinite(np.asarray(self.world.cam_pos, dtype=np.float64))):
                            self.world.cam_pos[:] = np.asarray(getattr(self.world.cfg, "start_pos", [-3.0, -3.0, 2.2]), dtype=np.float64)
                        z = float(getattr(self.cfg.dynamic_agent_rig, "hover_height", self.world.cam_pos[2]))
                        min_z = float(getattr(self.cfg.mocap_flight_bounds, "min_z", 0.05)) if hasattr(self.cfg, "mocap_flight_bounds") else 0.05
                        max_z = float(getattr(self.cfg.mocap_flight_bounds, "max_z", 10.0)) if hasattr(self.cfg, "mocap_flight_bounds") else 10.0
                        self.world.cam_pos[2] = float(np.clip(z, min_z, max_z))
                    except Exception:
                        pass

                # Clear smoothed arm/hand/body memories so neutral command takes effect quickly.
                if hasattr(self.world, "prev_arm_ctrl"):
                    self.world.prev_arm_ctrl[:] = 0.0
                if hasattr(self.world, "hand_bridge") and hasattr(self.world.hand_bridge, "prev_ctrl"):
                    self.world.hand_bridge.prev_ctrl[:] = 0.5

                try:
                    self.world._update_rig_pose()
                except Exception:
                    pass

            if hasattr(self, "prev_embodied_action"):
                self.prev_embodied_action.zero_()
            if hasattr(self, "prev_hand_motor"):
                self.prev_hand_motor.fill_(0.5)
            if hasattr(self, "prev_leg_motor"):
                self.prev_leg_motor.zero_()
            self._ipc_manual_body_action = np.zeros(9, dtype=np.float32)
            self._ipc_manual_arm_action = np.zeros(6, dtype=np.float32)
            self._ipc_manual_hand_action = np.full(44, 0.5, dtype=np.float32)
            self._ipc_manual_leg_action = np.zeros(18, dtype=np.float32)
        except Exception as e:
            print(f"[level_agent_pose] failed: {e}")

    def apply_manual_arm_action_override(self, out: dict) -> dict:
        """
        Force _ipc_manual_arm_action into possible arm output channels.
        """
        if not getattr(self, "_ipc_manual_actions_enabled", False):
            return out

        arm_vec = getattr(self, "_ipc_manual_arm_action", None)
        if arm_vec is None:
            return out

        try:
            arm_vec = np.asarray(arm_vec, dtype=np.float32).reshape(-1)
        except Exception:
            return out

        if arm_vec.size <= 0:
            return out

        for key in ("arm_ctrl", "arm_action", "arm_targets", "manual_arm_ctrl"):
            if key in out and out[key] is not None:
                try:
                    t = out[key].clone()
                    n = min(t.shape[-1], arm_vec.shape[0])
                    t[0, :n] = torch.tensor(arm_vec[:n], device=t.device, dtype=t.dtype)
                    out[key] = t
                except Exception:
                    pass

        if "embodied_targets" in out and out["embodied_targets"] is not None:
            try:
                emb = out["embodied_targets"].clone()
                if emb.shape[-1] >= 11:
                    n = min(6, arm_vec.shape[0], emb.shape[-1] - 5)
                    emb[0, 5:5+n] = torch.tensor(arm_vec[:n], device=emb.device, dtype=emb.dtype)
                    out["embodied_targets"] = emb
            except Exception:
                pass

        out["manual_arm_action"] = arm_vec.astype(np.float32)
        try:
            self._last_manual_arm_action = arm_vec.astype(np.float32)
        except Exception:
            pass

        try:
            self.trace_action_signal("manual_arm_action", arm_vec[:min(6, len(arm_vec))])
        except Exception:
            pass
        return out


    def apply_manual_hand_action_dimension_override(self, out: dict) -> dict:
        """
        Force manual hand action to the correct hand_ctrl dimension.

        New hand order has 44 dims:
            per hand = palm_roll, palm_pitch + 5 * (mcp_yaw, mcp, pip, dip)
        """
        if not getattr(self, "_ipc_manual_actions_enabled", False):
            return out
        hand_vec = getattr(self, "_ipc_manual_hand_action", None)
        if hand_vec is None or "hand_ctrl" not in out or out["hand_ctrl"] is None:
            return out

        try:
            hand_vec = np.asarray(hand_vec, dtype=np.float32).reshape(-1)
        except Exception:
            return out

        if hand_vec.size <= 0:
            return out

        try:
            h = out["hand_ctrl"]
            batch = int(h.shape[0]) if len(h.shape) >= 2 else 1
            current_dim = int(h.shape[-1])
            target_dim = max(current_dim, int(hand_vec.size))

            if current_dim != target_dim:
                new_h = torch.full((batch, target_dim), 0.5, device=h.device, dtype=h.dtype)
                n_old = min(current_dim, target_dim)
                new_h[:, :n_old] = h[:, :n_old]
                h = new_h
            else:
                h = h.clone()

            n = min(target_dim, int(hand_vec.size))
            h[0, :n] = torch.tensor(hand_vec[:n], device=h.device, dtype=h.dtype)
            out["hand_ctrl"] = h
            out["manual_hand_action"] = hand_vec.astype(np.float32)
            self._last_manual_hand_action = hand_vec.astype(np.float32)
            self.trace_action_signal("manual_hand_action", hand_vec[:min(12, len(hand_vec))])
        except Exception as e:
            try:
                print(f"[manual_hand_dim] failed: {e}")
            except Exception:
                pass

        return out



    def apply_pyqt_neural_output_override(self, out: dict, stage: str = "") -> dict:
        """
        THE IMPORTANT OVERRIDE POINT.

        This is the explicit boundary where real neural network outputs are
        replaced by PyQt IPC slider values.

        It must be called immediately after self.model_step(...), before:
            - exploration motor
            - protect_manual_body_output
            - compute_leg_control / apply_bird_leg_controls
            - world.observe(... physics)
            - replay.add(...)

        Replaced outputs:
            out["embodied_targets"] : body + head
            out["hand_ctrl"]        : palms + fingers
            out["leg_ctrl"]         : legs + toes, after leg head computes it
        """
        if not getattr(self, "_ipc_manual_actions_enabled", False):
            return out

        out = self.apply_manual_action_override(out)
        out = self.apply_manual_hand_action_dimension_override(out)
        out = self.apply_manual_arm_action_override(out)

        if isinstance(out, dict):
            out.setdefault("pyqt_neural_override", {})
            out["pyqt_neural_override"].update({
                "active": True,
                "stage": stage,
                "body_len": int(len(self._ipc_manual_body_action)) if self._ipc_manual_body_action is not None else 0,
                "hand_len": int(len(self._ipc_manual_hand_action)) if self._ipc_manual_hand_action is not None else 0,
                "leg_len": int(len(self._ipc_manual_leg_action)) if self._ipc_manual_leg_action is not None else 0,
            })
        return out


    def apply_dynamic_agent_rig_control(self, embodied_action):
        # mocap_contacts mode: central rig is controlled by mocap pose path.
        # Returning None lets world.observe() receive embodied_targets normally.
        return None


    def trace_action_signal(self, key: str, value):
        if not self.cfg.action_trace.enabled:
            return
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            self._action_trace[key] = arr[:6].copy()
        except Exception:
            self._action_trace[key] = str(value)


    def maybe_print_vestibular_trace(self, obs: dict):
        if not getattr(self.cfg, "vestibular", None) or not self.cfg.vestibular.enabled:
            return
        if self.global_step % max(1, self.cfg.vestibular.print_every_steps) != 0:
            return
        vest = obs.get("vestibular", None)
        if vest is None:
            print(f"[vestibular step={self.global_step}] no obs['vestibular']")
            return
        v = vest.detach().cpu().numpy().reshape(-1)
        lg = v[0:3]
        la = v[3:6]
        rg = v[6:9]
        ra = v[9:12]
        gc = v[12:15]
        gd = v[15:18]
        ac = v[18:21]
        ad = v[21:24]
        print(
            f"[vestibular step={self.global_step}] "
            f"Lgyro={np.round(lg,3).tolist()} Rgyro={np.round(rg,3).tolist()} "
            f"Gcommon={np.round(gc,3).tolist()} Gdiff={np.round(gd,3).tolist()} "
            f"Lacc={np.round(la,3).tolist()} Racc={np.round(ra,3).tolist()} "
            f"Acommon={np.round(ac,3).tolist()} Adiff={np.round(ad,3).tolist()}"
        )


    def maybe_print_action_trace(self, dyn_info=None):
        if not self.cfg.action_trace.enabled:
            return
        if self.global_step % max(1, self.cfg.action_trace.print_every_steps) != 0:
            return

        parts = [f"[action_trace step={self.global_step}]"]
        for key in [
            "manual_slider",
            "model_output_after_override",
            "after_exploration_protect",
            "before_dynamic_controller",
        ]:
            val = self._action_trace.get(key, None)
            if isinstance(val, np.ndarray):
                parts.append(f"{key}={np.round(val, 3).tolist()}")
            elif val is not None:
                parts.append(f"{key}={val}")

        if dyn_info is not None:
            try:
                parts.append(f"dyn.target_lin={np.round(dyn_info.get('target_lin', []), 3).tolist()}")
                parts.append(f"dyn.target_ang={np.round(dyn_info.get('target_ang', []), 3).tolist()}")
                parts.append(f"dyn.force={np.round(dyn_info.get('force', []), 3).tolist()}")
                parts.append(f"dyn.torque={np.round(dyn_info.get('torque', []), 3).tolist()}")
                parts.append(f"z={float(dyn_info.get('z', 0.0)):.3f}")
                parts.append(f"hover_target_z={float(dyn_info.get('hover_target_z', 0.0)):.3f}")
                parts.append(f"contact_level={float(dyn_info.get('estimated_contact_level', dyn_info.get('contact_level', 0.0))):.3f}")
            except Exception as e:
                parts.append(f"dyn_info_error={e}")

        if self.dynamic_agent_rig_controller is not None:
            try:
                qa = self.dynamic_agent_rig_controller.qpos_adr
                va = self.dynamic_agent_rig_controller.qvel_adr
                qpos = self.world.data.qpos[qa:qa + 7].copy()
                qvel = self.world.data.qvel[va:va + 6].copy()
                parts.append(f"qpos_xyz={np.round(qpos[:3], 3).tolist()}")
                parts.append(f"qvel={np.round(qvel, 3).tolist()}")
            except Exception as e:
                parts.append(f"qpos_qvel_error={e}")
        else:
            parts.append("dynamic_agent_rig_controller=None")

        print(" | ".join(parts))


    def get_manual_body_action_vector(self):
        """
        IPC-only manual override source.

        The legacy cv2 manual_actions window is removed. Values now come only
        from the PyQt Agent Actions window through:
            manual_actions_enabled
            manual_body_action
        """
        if getattr(self, "_ipc_manual_actions_enabled", False) and self._ipc_manual_body_action is not None:
            return np.asarray(self._ipc_manual_body_action, dtype=np.float32)
        return None


    def apply_manual_action_override(self, out: dict):
        """
        Neural-output-level override.

        This must be called immediately after model_step(), before exploration,
        leg/head modules, replay, visualization, and physics. It imitates the
        actual neural output out["embodied_targets"][0:6].
        """
        manual_vec = self.get_manual_body_action_vector()
        if manual_vec is None:
            return out

        neural_vec = out["embodied_targets"][0, :6].detach().cpu().numpy()
        self.trace_action_signal("manual_slider", manual_vec)
        emb = out["embodied_targets"].clone()
        # Manual slider vector layout:
        # 0: vx, 1: vy, 2: vz, 3: body_yaw, 4: body_pitch, 5: body_roll
        # 6: head_yaw, 7: head_pitch, 8: head_roll
        #
        # Model embodied layout:
        # 0:5 body transl/yaw/pitch, 5:10 old arms, 11 body_roll, 12:15 head.
        emb[0, 0:5] = torch.tensor(manual_vec[:5], device=emb.device, dtype=emb.dtype)
        if emb.shape[-1] > 11:
            emb[0, 11] = torch.tensor(manual_vec[5], device=emb.device, dtype=emb.dtype)
        else:
            emb[0, 5] = torch.tensor(manual_vec[5], device=emb.device, dtype=emb.dtype)
        if emb.shape[-1] >= 15 and len(manual_vec) >= 9:
            emb[0, 12:15] = torch.tensor(manual_vec[6:9], device=emb.device, dtype=emb.dtype)
        # Arms are NOT part of hand_ctrl. They live in embodied_targets[5:11].
        # Layout: L shoulder yaw/pitch/elbow, R shoulder yaw/pitch/elbow.
        if getattr(self, "_ipc_manual_actions_enabled", False) and self._ipc_manual_arm_action is not None and emb.shape[-1] >= 11:
            arm_vec = self._sanitize_manual_vector(self._ipc_manual_arm_action, 6, fill=0.0, lo=-1.0, hi=1.0)
            n_arm = min(6, arm_vec.shape[0])
            emb[0, 5:5 + n_arm] = torch.tensor(arm_vec[:n_arm], device=emb.device, dtype=emb.dtype)
            out["manual_arm_action"] = arm_vec.astype(np.float32)
            self.trace_action_signal("manual_arm_action", arm_vec[:min(6, len(arm_vec))])

        out["embodied_targets"] = emb

        # Hands/palms/fingers follow realistic_hand_mjcf.both_hand_control_names().
        #
        # Important:
        #   PyQt hand sliders are already direct 0..1 hand_ctrl values.
        #   RealisticHandBridge expects hand_ctrl in 0..1.
        if getattr(self, "_ipc_manual_actions_enabled", False) and self._ipc_manual_hand_action is not None and "hand_ctrl" in out:
            hand_vec = self._sanitize_manual_vector(self._ipc_manual_hand_action, 44, fill=0.5, lo=0.0, hi=1.0)
            hand = out["hand_ctrl"].clone()
            n = min(hand.shape[-1], hand_vec.shape[0])
            hand[0, :n] = torch.tensor(hand_vec[:n], device=hand.device, dtype=hand.dtype)
            out["hand_ctrl"] = hand
            out["manual_hand_action"] = hand_vec.astype(np.float32)
            self.trace_action_signal("manual_hand_action", hand_vec[:min(12, len(hand_vec))])

        self.trace_action_signal("model_output_after_override", emb[0, :6].detach().cpu().numpy())
        out["manual_action_override"] = {
            "active": True,
            "override_level": "model_output_embodied_targets_0_6",
            "neural_body_action": neural_vec.astype(np.float32),
            "manual_body_action": manual_vec.astype(np.float32),
        }
        return out


    def update_manual_action_override_window(self, out: dict | None = None):
        """
        Legacy cv2 manual action window removed.

        Manual control is now handled by PyQt Agent Actions over IPC.
        """
        return
