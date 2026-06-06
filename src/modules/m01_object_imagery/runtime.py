from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.modules.m01_object_imagery.models.object_inner_imagery_3d import summarize_vision_tensors, pad_or_trim
from src.modules.m02_event_dream_replay.event_latent_codec import EventLatentCodecConfig, EventLatentSentenceMemory
from src.modules.m02_event_dream_replay.event_code_visualizer import EventCodeVisualizerV2, EventCodeVisualizerV2Config
from src.modules.m08_debug_visual_control.static_dynamic_code_debug_runtime import StaticDynamicCodeDebugRuntimeMixin
from src.modules.m04_long_dynamic_memory.passport_debug_runtime import PassportDebugRuntimeMixin
from src.modules.m03_self_action_causality.inner_real_action_trace_runtime import InnerRealActionTraceRuntimeMixin
from src.modules.m12_metacognition_monitor.inner_trust_gate_runtime import InnerTrustGateRuntimeMixin
from src.modules.m03_self_action_causality.inner_outcome_evaluator_runtime import InnerOutcomeEvaluatorRuntimeMixin
from src.modules.m03_self_action_causality.inner_action_decoder_runtime import InnerActionDecoderRuntimeMixin
from src.modules.m15_counterfactual_imagination_planning.inner_scenario_mind_runtime import InnerScenarioMindRuntimeMixin
from src.modules.m04_long_dynamic_memory.dynamic_object_passport_runtime import DynamicObjectPassportRuntimeMixin
from src.platform.gui.highgui_event_pump_runtime import pump_highgui_events
from src.modules.m02_event_dream_replay.models.neural_event_decoder import NeuralEventDecoder, NeuralEventDecoderConfig


class ObjectImageryRuntimeMixin(StaticDynamicCodeDebugRuntimeMixin, DynamicObjectPassportRuntimeMixin, PassportDebugRuntimeMixin, InnerScenarioMindRuntimeMixin, InnerActionDecoderRuntimeMixin, InnerOutcomeEvaluatorRuntimeMixin, InnerTrustGateRuntimeMixin, InnerRealActionTraceRuntimeMixin):
    def _input_sensor_enabled(self, name: str) -> bool:
        """
        Central sensor-gate query used by inner-object runtime/debug.

        This prevents debug preview paths from bypassing sleep/partial-cut gates.
        """
        if name == "video":
            return bool(getattr(self, "video_sensor_enabled", True))
        if name == "contact":
            return bool(getattr(self, "contact_sensor_enabled", True))
        if name == "imu":
            return bool(getattr(self, "imu_sensor_enabled", True))
        return True

    def _zero_like_or_zeros(self, value, *shape):
        try:
            if value is not None:
                return torch.zeros_like(value)
        except Exception:
            pass
        return torch.zeros(*shape, device=self.device, dtype=torch.float32)

    def _safe_obs_for_inner_object_viz(self, obs: dict) -> dict:
        """
        Return an obs copy for visualizer/teacher logic with disabled sensors
        removed from any bypass path.

        - video OFF: do not let object_state/geometry pretend that camera sees.
        - imu OFF: do not let pose/geometry pretend that body pose is sensed.
        - contact OFF: tactile is zeroed and contact previews are not added.
        """
        if not isinstance(obs, dict):
            return obs

        out = dict(obs)
        out["input_sensors_enabled"] = {
            "video": self._input_sensor_enabled("video"),
            "contact": self._input_sensor_enabled("contact"),
            "imu": self._input_sensor_enabled("imu"),
        }
        out["sleep_sensor_mask"] = {k: not bool(v) for k, v in out["input_sensors_enabled"].items()}

        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        scenario_active = bool(
            isinstance(scenario, dict)
            and (getattr(self, "_fly_to_cube_palpate_active", False) or scenario.get("active", False))
        )
        if scenario_active and isinstance(scenario, dict) and str(scenario.get("scenario", "")) == "fly_to_tetrahedron_inspect":
            try:
                focus_depth = float(scenario.get("gaze_distance", 0.0))
                half_range = float(scenario.get("depth_focus_half_range", 0.85))
                if np.isfinite(focus_depth) and focus_depth > 0.0:
                    out["depth_focus_depth"] = focus_depth
                    out["depth_focus_half_range"] = max(0.10, half_range)
                    out["depth_focus_label"] = str(scenario.get("gaze_target", ""))
            except Exception:
                pass

        if not self._input_sensor_enabled("video"):
            for key in ("left", "right", "depth"):
                if key in out:
                    try:
                        out[key] = torch.zeros_like(out[key])
                    except Exception:
                        pass
            # Prevent geometry visibility from acting as hidden vision.
            out["object_state"] = None

        if not self._input_sensor_enabled("contact"):
            for key in ("tactile", "contact", "contacts", "contact_sensors"):
                if key in out:
                    try:
                        out[key] = torch.zeros_like(out[key])
                    except Exception:
                        pass

        if not self._input_sensor_enabled("imu"):
            for key in ("pose", "vestibular", "imu", "gyro", "accel"):
                if key in out:
                    try:
                        out[key] = torch.zeros_like(out[key])
                    except Exception:
                        pass
            # Without pose/IMU, geometry visibility must not be used.
            out["pose"] = None

        return out



    # -------------------------------------------------------------------------
    # Level-2 event code visualizer
    # -------------------------------------------------------------------------
    def _event_code_visualizer_enabled(self) -> bool:
        cfg_viz = getattr(self.cfg, "event_code_visualizer", None)
        if not bool(getattr(cfg_viz, "enabled", False)):
            return False
        return bool(getattr(self, "show_event_code_visualizer_window", False))

    def _ensure_event_code_visualizer(self) -> None:
        if hasattr(self, "event_code_viz") and self.event_code_viz is not None:
            return
        cfg_viz = getattr(self.cfg, "event_code_visualizer", None)
        self.event_code_viz = EventCodeVisualizerV2(EventCodeVisualizerV2Config(
            enabled=bool(getattr(cfg_viz, "enabled", True)),
            window_name=str(getattr(cfg_viz, "window_name", "event code / slot vocabulary")),
            width=int(getattr(cfg_viz, "width", 1500)),
            height=int(getattr(cfg_viz, "height", 980)),
            show_every_steps=int(getattr(cfg_viz, "show_every_steps", 1)),
            delay_ms=int(getattr(cfg_viz, "delay_ms", 1)),
            max_slots=int(getattr(cfg_viz, "max_slots", getattr(self.cfg.object_image, "num_slots", 10))),
            max_events=int(getattr(cfg_viz, "max_events", 14)),
        ))

    def update_event_code_visualizer_window(self, obj: dict | None) -> None:
        """
        Draw second-level semantic-code window.

        Level 1 = object image / slot latent heatmap.
        Level 2 = slot vocabulary + event code sentences.
        """
        if not self._event_code_visualizer_enabled():
            try:
                if hasattr(self, "event_code_viz") and self.event_code_viz is not None:
                    self.event_code_viz.close()
            except Exception:
                pass
            return

        cfg_viz = getattr(self.cfg, "event_code_visualizer", None)
        if int(getattr(self, "global_step", 0)) % max(1, int(getattr(cfg_viz, "show_every_steps", 1))) != 0:
            return

        try:
            self._ensure_event_code_visualizer()
            self.event_code_viz.draw(
                obj=obj,
                event_memory=getattr(self, "event_latent_memory", None),
                global_step=int(getattr(self, "global_step", 0)),
            )
        except Exception as e:
            if not hasattr(self, "_event_code_viz_warned"):
                print(f"[event_code_viz] update failed: {e}")
                self._event_code_viz_warned = True


    # -------------------------------------------------------------------------
    # Event latent code memory
    # -------------------------------------------------------------------------
    def _ensure_event_latent_memory(self) -> None:
        if hasattr(self, "event_latent_memory") and self.event_latent_memory is not None:
            return

        cfg_event = getattr(self.cfg, "event_memory", None)
        cfg_obj = getattr(self.cfg, "object_image", None)

        self.event_latent_memory = EventLatentSentenceMemory(EventLatentCodecConfig(
            enabled=bool(getattr(cfg_event, "enabled", True)),
            latent_dim=int(getattr(cfg_obj, "latent_dim", 128)),
            max_events=int(getattr(cfg_event, "max_events", 512)),
            delta_threshold=float(getattr(cfg_event, "delta_threshold", 0.015)),
            action_threshold=float(getattr(cfg_event, "action_threshold", 0.010)),
            contact_threshold=float(getattr(cfg_event, "contact_threshold", 0.010)),
            record_in_sleep=bool(getattr(cfg_event, "record_in_sleep", True)),
            keep_z_snapshots=bool(getattr(cfg_event, "keep_z_snapshots", True)),
            use_slot_vocabulary=bool(getattr(cfg_event, "use_slot_vocabulary", True)),
            slot_token_prefix=str(getattr(cfg_event, "slot_token_prefix", "OBJ")),
            compose_semantic_sentences=bool(getattr(cfg_event, "compose_semantic_sentences", True)),
            sentence_language=str(getattr(cfg_event, "sentence_language", "code")),
            use_sentence_memory=bool(getattr(cfg_event, "use_sentence_memory", True)),
            max_sentences=int(getattr(cfg_event, "max_sentences", getattr(cfg_event, "max_events", 512))),
            max_episodes=int(getattr(cfg_event, "max_episodes", 64)),
            episode_gap_steps=int(getattr(cfg_event, "episode_gap_steps", 25)),
            new_episode_on_slot_change=bool(getattr(cfg_event, "new_episode_on_slot_change", False)),
            use_scenario_decoder=bool(getattr(cfg_event, "use_scenario_decoder", True)),
            scenario_max_replay_steps=int(getattr(cfg_event, "scenario_max_replay_steps", 32)),
            scenario_interpolate_steps=int(getattr(cfg_event, "scenario_interpolate_steps", 3)),
            scenario_loop=bool(getattr(cfg_event, "scenario_loop", True)),
        ))

    def _update_event_latent_memory(self, prev_state: dict, obj: dict, obs: dict, out: dict, dream_mode: bool) -> dict:
        """
        First step toward code-sentences for dynamic object events.

        It observes:
            z_before(slot) -> z_after(slot)
            + action/contact/vision/touch context

        and writes a compact event sentence, e.g.
            EVT t=120 SLOT_1 KIND=contact_transition DZ_MID ACT_LOW TOUCH_HIGH ...
        """
        try:
            self._ensure_event_latent_memory()
            if not bool(getattr(self.event_latent_memory.cfg, "enabled", True)):
                return obj

            ev = self.event_latent_memory.encode_step(
                prev_state=prev_state,
                obj=obj,
                obs=obs,
                out=out,
                dream_mode=bool(dream_mode),
                global_step=int(getattr(self, "global_step", 0)),
            )
            if isinstance(ev, dict) and ev:
                obj.update(ev)
                if ev.get("event_active"):
                    try:
                        obj["event_code_text"] = str(ev.get("event_code_sentence", ""))
                    except Exception:
                        pass
            return obj
        except Exception as e:
            if not hasattr(self, "_event_latent_memory_warned"):
                print(f"[event_memory] update failed: {e}")
                self._event_latent_memory_warned = True
            return obj


    # -------------------------------------------------------------------------
    # Progressive semantic image memory
    # -------------------------------------------------------------------------
    def _ensure_progressive_semantic_memory_state(self) -> None:
        """
        Runtime-only state for progressive semantic slot formation.

        Correct interpretation:
            slot 0    = first whole-scene semantic image
            slot 1..N = later internal semantic images formed from sensor flow

        A slot is not a hand-written object label.
        A slot is not "cube", "sphere", "cylinder", or "ground".
        A slot is an internal image/state that became stable enough to be kept.
        """
        if not hasattr(self, "_semantic_next_slot"):
            self._semantic_next_slot = 1
        if not hasattr(self, "_semantic_current_slot"):
            self._semantic_current_slot = None
        if not hasattr(self, "_semantic_event_was_active"):
            self._semantic_event_was_active = False
        if not hasattr(self, "_semantic_last_discovery_step"):
            self._semantic_last_discovery_step = -10**9
        if not hasattr(self, "_semantic_scene_reference"):
            self._semantic_scene_reference = None
        if not hasattr(self, "_semantic_scene_slot_created"):
            self._semantic_scene_slot_created = False
        if not hasattr(self, "_semantic_dynamic_seen_count"):
            self._semantic_dynamic_seen_count = 0

        # Generic dynamic object slot identity state.
        if not hasattr(self, "_dynamic_object_slot_by_target"):
            self._dynamic_object_slot_by_target = {}
        if not hasattr(self, "_dynamic_slot_target_by_slot"):
            self._dynamic_slot_target_by_slot = {}
        if not hasattr(self, "_dynamic_protected_slots"):
            self._dynamic_protected_slots = set()
        if not hasattr(self, "_dynamic_next_free_slot"):
            self._dynamic_next_free_slot = 0

    def _tensor_activity(self, x) -> float:
        try:
            if x is None:
                return 0.0
            return float(x.detach().float().abs().mean().cpu().item())
        except Exception:
            return 0.0

    def _scene_novelty_score(self, scene_prop: torch.Tensor) -> float:
        """
        Simple sensory novelty score against the previous whole-scene summary.

        This is not object identity. It only detects that the current sensory
        image differs enough to be considered a possible new semantic event.
        """
        try:
            current = scene_prop.detach()
            if current.ndim == 1:
                current = current.unsqueeze(0)

            if self._semantic_scene_reference is None:
                self._semantic_scene_reference = current.clone()
                return 0.0

            ref = self._semantic_scene_reference.to(device=current.device, dtype=current.dtype)
            diff = torch.mean(torch.abs(current - ref)).detach()
            # Slow update of background scene reference.
            self._semantic_scene_reference = (0.995 * ref + 0.005 * current).detach()
            return float(diff.cpu().item())
        except Exception:
            return 0.0

    def _dynamic_sensory_input_score(self, obs: dict, scene_prop: torch.Tensor) -> tuple[float, float, float]:
        """
        Dynamic sensory input is what can start meaning.

        A static frame is only a picture. It should not create a semantic slot.
        A semantic image begins when the system receives change over time:
            - video/depth novelty across frames,
            - contact/tactile activity,
            - embodied motor interaction,
            - later: prediction error and multi-view consistency.

        Returns:
            total_dynamic_score, scene_novelty, interaction_score
        """
        self._ensure_progressive_semantic_memory_state()
        scene_novelty = self._scene_novelty_score(scene_prop)
        interaction = self._interaction_activity_score(obs)

        # IMU/body signal can also contribute to "viewpoint over time".
        imu_score = 0.0
        try:
            body = obs.get("body_state")
            if body is not None:
                imu_score = 0.10 * self._tensor_activity(body)
        except Exception:
            imu_score = 0.0

        total = float(scene_novelty + interaction + imu_score)
        return total, float(scene_novelty), float(interaction)

    def _dynamic_target_motion_allowed(self) -> tuple[bool, str]:
        """
        Distinguish object motion from ego/viewpoint motion for scripted probes.

        During floating-object inspection, the camera/body can move while the
        tetrahedron is static. That should update observation state, but must
        not become a dynamic object slot until the target itself rotates.
        """
        status = getattr(self, "_fly_to_cube_palpate_status", None)
        if not isinstance(status, dict):
            return True, "no_scripted_target"
        if str(status.get("scenario", "")) != "fly_to_tetrahedron_inspect":
            return True, "other_scenario"
        if not bool(status.get("active", False)):
            return True, "scenario_inactive"
        if str(status.get("phase", "")) != "inspect":
            return False, "inspect_not_reached"
        if bool(status.get("rotate_tetrahedron", False)):
            return True, "tetrahedron_rotating"
        if bool(status.get("rotate_cube", False)):
            return True, "cube_rotating"
        if bool(status.get("fly_cube", False)):
            return True, "cube_flying"
        return False, "static_scripted_target"

    def _dynamic_current_target_name(self) -> str:
        status = getattr(self, "_fly_to_cube_palpate_status", None)
        if not isinstance(status, dict):
            return "unknown"
        name = str(status.get("gaze_target", "") or "").strip().lower()
        if name in ("tetra", "tetrahedron"):
            return "tetrahedron"
        if name == "cube":
            return "cube"
        return name or "unknown"

    def _dynamic_target_slot_index(self, target_name: str) -> int:
        # Generic dynamic object slot policy:
        # known target -> same slot; new target -> next free protected slot.
        self._ensure_progressive_semantic_memory_state()

        target_name = str(target_name or "unknown").strip().lower() or "unknown"
        n_slots = max(1, int(getattr(self.cfg.object_image, "num_slots", 10)))

        if target_name in ("", "unknown", "none"):
            return 0

        slot_by_target = getattr(self, "_dynamic_object_slot_by_target", {})
        target_by_slot = getattr(self, "_dynamic_slot_target_by_slot", {})
        protected = getattr(self, "_dynamic_protected_slots", set())

        if target_name in slot_by_target:
            slot = int(slot_by_target[target_name])
            is_new = False
            matched_existing_slot = slot
            decision_reason = "matched_existing_dynamic_object"
        else:
            used = {int(k) for k in target_by_slot.keys()}
            blocked = used | {int(s) for s in protected}
            slot = None

            for i in range(n_slots):
                if i not in blocked:
                    slot = i
                    break

            if slot is None:
                for i in range(n_slots):
                    if i not in protected:
                        slot = i
                        break

            if slot is None:
                slot = max(0, n_slots - 1)

            slot_by_target[target_name] = int(slot)
            target_by_slot[int(slot)] = target_name
            self._dynamic_object_slot_by_target = slot_by_target
            self._dynamic_slot_target_by_slot = target_by_slot
            self._dynamic_next_free_slot = max(int(getattr(self, "_dynamic_next_free_slot", 0)), int(slot) + 1)
            is_new = True
            matched_existing_slot = "none"
            decision_reason = "new_dynamic_object_next_free_slot"

        # Once a dynamic identity receives a slot, keep it protected from unrelated targets.
        protected.add(int(slot))
        self._dynamic_protected_slots = protected

        try:
            if hasattr(self, "log_dynamic_object_slot_policy"):
                self.log_dynamic_object_slot_policy(
                    target_name=target_name,
                    is_new_object=bool(is_new),
                    matched_existing_slot=matched_existing_slot,
                    allocated_slot=int(slot),
                    next_free_slot=int(getattr(self, "_dynamic_next_free_slot", int(slot) + 1)),
                    protected_slots=sorted(int(s) for s in protected),
                    overwrite_allowed=False,
                    decision_reason=decision_reason,
                )
            if hasattr(self, "log_slot_protection_state"):
                self.log_slot_protection_state(
                    slot_id=int(slot),
                    target_name=target_name,
                    formed_conf=1.0,
                    z_dynamic_norm=0.0,
                    protected=True,
                    reason="dynamic_target_slot_assigned",
                )
        except Exception:
            pass

        return int(slot)

    def _interaction_activity_score(self, obs: dict) -> float:
        """
        Generic embodied interaction score.

        This does not mean "specific object". It only says that the sensory stream
        currently contains an embodied event strong enough to possibly form a new
        internal image.
        """
        tactile_score = self._tensor_activity(obs.get("tactile"))

        motor_score = 0.0
        try:
            if hasattr(self, "prev_hand_motor") and self.prev_hand_motor is not None:
                motor_score += self._tensor_activity(self.prev_hand_motor)
            if hasattr(self, "prev_embodied_action") and self.prev_embodied_action is not None:
                motor_score += 0.25 * self._tensor_activity(self.prev_embodied_action)
        except Exception:
            pass

        return float(tactile_score + 0.10 * motor_score)

    def _allocate_or_keep_semantic_slot(self, semantic_event_active: bool) -> int | None:
        """
        Allocate slots sequentially only when a new semantic event appears.

        Rising edge:
            no semantic event -> semantic event
        allocates next slot.

        While the event continues, the same slot can keep being refined.

        When the event stops, no extra slot receives the whole scene.
        """
        self._ensure_progressive_semantic_memory_state()

        if not semantic_event_active:
            self._semantic_event_was_active = False
            self._semantic_current_slot = None
            return None

        if not self._semantic_event_was_active:
            n_slots = max(1, int(getattr(self.cfg.object_image, "num_slots", 10)))
            cooldown = int(getattr(self.cfg.object_image, "semantic_discovery_cooldown_steps", 12))
            can_allocate = (int(getattr(self, "global_step", 0)) - int(self._semantic_last_discovery_step)) >= cooldown

            if can_allocate and n_slots > 1:
                slot = int(self._semantic_next_slot)
                slot = max(1, min(slot, n_slots - 1))
                self._semantic_current_slot = slot
                self._semantic_next_slot = 1 + (slot % max(1, n_slots - 1))
                self._semantic_last_discovery_step = int(getattr(self, "global_step", 0))
                print(f"[inner_object][semantic_memory] new semantic image -> slot {slot}")
            elif self._semantic_current_slot is None and n_slots > 1:
                self._semantic_current_slot = 1

        self._semantic_event_was_active = True
        return self._semantic_current_slot

    def _ensure_bchw(self, x: torch.Tensor, channels=None) -> torch.Tensor:
        x = x.float().to(self.device)
        if x.ndim == 2:
            x = x.view(1, 1, x.shape[0], x.shape[1])
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim == 4 and channels is not None and x.shape[1] > channels:
            x = x[:, :channels]
        return x

    def _semantic_candidate_from_sensory_stream(self, obs: dict, scene_prop: torch.Tensor) -> torch.Tensor:
        """
        Build one candidate semantic image from the current sensory stream.

        This is not a fixed grid filling all slots.
        This is not object_state.
        This is not ground.
        This is a single candidate formed only when the system has a reason
        to create/refine a new internal image.

        For now we use the most salient RGB/depth region as bootstrap.
        Later this should be replaced by:
            - motion consistency,
            - contact binding,
            - active manipulation,
            - prediction-error clustering,
            - temporal coherence.
        """
        left = obs.get("left")
        right = obs.get("right", left)
        depth = obs.get("depth")
        if left is None:
            return scene_prop

        left = self._ensure_bchw(left, channels=3)
        right = self._ensure_bchw(right, channels=3)
        depth = self._ensure_bchw(depth, channels=1) if depth is not None else left[:, :1] * 0.0

        h, w = left.shape[-2], left.shape[-1]
        if right.shape[-2:] != (h, w):
            right = F.interpolate(right, size=(h, w), mode="bilinear", align_corners=False)
        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)

        rows = max(1, int(getattr(self.cfg.object_image, "semantic_candidate_search_rows", 3)))
        cols = max(1, int(getattr(self.cfg.object_image, "semantic_candidate_search_cols", 4)))

        global_mean = left.mean(dim=(-2, -1), keepdim=True)
        best_score = None
        best_crop = None

        for r in range(rows):
            y0 = int(round(r * h / rows))
            y1 = int(round((r + 1) * h / rows))
            for c in range(cols):
                x0 = int(round(c * w / cols))
                x1 = int(round((c + 1) * w / cols))
                if y1 <= y0 or x1 <= x0:
                    continue

                l_crop = left[:, :, y0:y1, x0:x1]
                r_crop = right[:, :, y0:y1, x0:x1]
                d_crop = depth[:, :, y0:y1, x0:x1]

                contrast = torch.abs(l_crop - global_mean).mean(dim=(1, 2, 3), keepdim=True)
                depth_activity = d_crop.float().std(dim=(1, 2, 3), keepdim=True)
                score = contrast + depth_activity

                score_value = float(score.mean().detach().cpu().item())
                if best_score is None or score_value > best_score:
                    best_score = score_value
                    best_crop = (l_crop, r_crop, d_crop, x0, x1, y0, y1, score)

        if best_crop is None:
            return scene_prop

        l_crop, r_crop, d_crop, x0, x1, y0, y1, score = best_crop
        prop = summarize_vision_tensors(l_crop, r_crop, d_crop).to(self.device)
        if prop.ndim == 1:
            prop = prop.unsqueeze(0)

        # Sensor-position code only. This is not semantic object labeling.
        if prop.shape[-1] >= 4:
            cx = ((x0 + x1) * 0.5 / max(1, w)) * 2.0 - 1.0
            cy = ((y0 + y1) * 0.5 / max(1, h)) * 2.0 - 1.0
            area = ((x1 - x0) * (y1 - y0)) / max(1, w * h)
            act = torch.tanh(score.view(-1, 1)).to(device=self.device, dtype=prop.dtype)
            geom = torch.cat([
                torch.full_like(act, float(cx)),
                torch.full_like(act, float(cy)),
                torch.full_like(act, float(area)),
                act,
            ], dim=-1)
            prop[:, -4:] = geom

        return prop


    # -------------------------------------------------------------------------
    # Long dynamic object memory: z_static stream -> z_dynamic_object
    # -------------------------------------------------------------------------
    def _ensure_long_dynamic_object_memory(self, z_static: torch.Tensor) -> None:
        if hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None:
            return
        from src.modules.m04_long_dynamic_memory.models.long_dynamic_object_memory import LongDynamicObjectMemory
        cfg_obj = getattr(self.cfg, "object_image", None)
        input_dim = int(z_static.shape[-1])
        hidden_dim = int(getattr(cfg_obj, "long_dynamic_memory_hidden_dim", getattr(cfg_obj, "latent_dim", 128)))
        context_dim = int(getattr(cfg_obj, "long_dynamic_memory_context_dim", 6))
        ema_alpha = float(getattr(cfg_obj, "long_dynamic_memory_ema_alpha", 0.08))
        residual_scale = float(getattr(cfg_obj, "long_dynamic_memory_residual_scale", 0.35))
        self.long_dynamic_object_memory = LongDynamicObjectMemory(
            input_dim=input_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            ema_alpha=ema_alpha,
            residual_scale=residual_scale,
        ).to(self.device)
        self._long_dynamic_object_memory_state = None
        self._long_dynamic_object_ready_streak = 0

        try:
            if hasattr(self, "module_training_gate"):
                self.module_training_gate.apply()
                self.rebuild_optimizer_from_trainable_modules()
            elif hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.add_param_group({"params": self.long_dynamic_object_memory.parameters()})
        except Exception as e:
            if not hasattr(self, "_long_dynamic_memory_opt_warned"):
                print(f"[long_dynamic_memory] optimizer/module gate attach skipped: {e}")
                self._long_dynamic_memory_opt_warned = True

        print("[long_dynamic_memory] lazy initialized")
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                existing = set()
                for group in self.optimizer.param_groups:
                    for p in group.get("params", []):
                        existing.add(id(p))
                params = [p for p in self.long_dynamic_object_memory.parameters() if id(p) not in existing]
                if params:
                    self.optimizer.add_param_group({"params": params})
                    print(f"[long_dynamic_memory] added params to optimizer: {sum(p.numel() for p in params):,}")
        except Exception as e:
            if not hasattr(self, "_long_dynamic_memory_opt_warned"):
                print(f"[long_dynamic_memory] optimizer add skipped: {e}")
                self._long_dynamic_memory_opt_warned = True

    def _long_dynamic_context_vector(self, obs: dict, dynamic_score: float, scene_novelty: float, interaction: float) -> torch.Tensor:
        def act(key: str) -> float:
            try:
                return self._tensor_activity(obs.get(key))
            except Exception:
                return 0.0
        hand_activity = 0.0
        try:
            if hasattr(self, "prev_hand_motor") and self.prev_hand_motor is not None:
                hand_activity = self._tensor_activity(self.prev_hand_motor)
        except Exception:
            pass
        return torch.tensor(
            [[float(dynamic_score), float(scene_novelty), float(interaction), act("body_state"), act("tactile"), float(hand_activity)]],
            device=self.device,
            dtype=torch.float32,
        )


    def _long_dynamic_depth_motion_score(self, obs: dict) -> float:
        try:
            if not isinstance(obs, dict) or "depth" not in obs:
                return 0.0
            x = obs.get("depth")
            if not torch.is_tensor(x):
                return 0.0

            cur = x.detach().float()
            if cur.ndim == 4:
                cur = cur[0]
            cur = cur.reshape(-1)
            cur = torch.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

            prev = getattr(self, "_long_dynamic_prev_depth_frame", None)
            self._long_dynamic_prev_depth_frame = cur.detach().clone()

            if prev is None or not torch.is_tensor(prev) or prev.numel() != cur.numel():
                return 0.0

            prev = torch.nan_to_num(prev.to(cur.device, cur.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            return float(torch.mean(torch.abs(cur - prev)).detach().cpu().item())
        except Exception:
            return 0.0


    def _long_dynamic_object_memory_step(self, obs: dict, z_static: torch.Tensor, dynamic_score: float, scene_novelty: float, interaction: float, dynamic_input_active: bool):
        self._ensure_long_dynamic_object_memory(z_static)
        target_name = self._dynamic_current_target_name()
        prev_target_name = str(getattr(self, "_long_dynamic_memory_target_name", "") or "")
        if prev_target_name and target_name != prev_target_name:
            self._long_dynamic_object_memory_state = None
            self._long_dynamic_object_ready_streak = 0
            self._long_dynamic_prev_depth_frame = None
        self._long_dynamic_memory_target_name = target_name
        ctx = self._long_dynamic_context_vector(obs, dynamic_score, scene_novelty, interaction)
        z_dynamic, confidence, state, diag = self.long_dynamic_object_memory(
            z_static=z_static,
            context=ctx,
            state=getattr(self, "_long_dynamic_object_memory_state", None),
            dynamic_active=bool(dynamic_input_active),
        )
        self._long_dynamic_object_memory_state = state
        try:
            self._last_long_dynamic_training_tensors = {
                "z_static": z_static,
                "z_dynamic": z_dynamic,
                "confidence": confidence,
            }
        except Exception:
            pass
        cfg_obj = getattr(self.cfg, "object_image", None)
        min_steps = int(getattr(cfg_obj, "long_dynamic_memory_min_steps", 8))
        stable_steps = int(getattr(cfg_obj, "long_dynamic_memory_stable_steps", 4))
        conf_thr = float(getattr(cfg_obj, "long_dynamic_memory_confidence_threshold", 0.05))
        require_dynamic = bool(getattr(cfg_obj, "long_dynamic_memory_require_dynamic_input", True))
        pixel_motion = float(diag.get("long_dynamic_pixel_motion", 0.0))
        depth_motion = self._long_dynamic_depth_motion_score(obs)
        diag["long_dynamic_depth_motion"] = float(depth_motion)

        conf_value = float(confidence.detach().float().mean().cpu().item())
        active_ok = bool(dynamic_input_active) or not require_dynamic

        # STRICT MOTION GATE:
        # A static picture must never become READY/WRITE just because the GRU
        # confidence is non-zero or because the first frame looked novel.
        #
        # Ready now requires actual temporal change in z_static for several
        # consecutive steps:
        #     static image -> dz ~= 0 -> READY=0 WRITE=0 props=0
        try:
            dz_value = float(diag.get("long_dynamic_dz", 0.0))
        except Exception:
            dz_value = 0.0

        dz_thr = float(getattr(cfg_obj, "long_dynamic_memory_dz_threshold", 0.003))
        min_active_steps = int(getattr(cfg_obj, "long_dynamic_memory_min_active_steps", stable_steps))
        pixel_motion_value = float(diag.get("long_dynamic_pixel_motion", 0.0))
        pixel_thr = float(getattr(cfg_obj, "long_dynamic_memory_pixel_motion_threshold", 0.0015))
        depth_thr = float(getattr(cfg_obj, "long_dynamic_memory_depth_motion_threshold", 0.0008))
        try:
            depth_motion_value = float(diag.get("long_dynamic_depth_motion", 0.0))
        except Exception:
            depth_motion_value = 0.0
        motion_ok = bool((dz_value >= dz_thr) or (pixel_motion_value >= pixel_thr) or (depth_motion_value >= depth_thr))

        if active_ok and motion_ok and conf_value >= conf_thr:
            self._long_dynamic_object_ready_streak = int(getattr(self, "_long_dynamic_object_ready_streak", 0)) + 1
        else:
            self._long_dynamic_object_ready_streak = 0

        ready = (
            int(getattr(state, "steps", 0)) >= min_steps
            and int(getattr(state, "dynamic_steps", 0)) >= min_active_steps
            and int(getattr(self, "_long_dynamic_object_ready_streak", 0)) >= stable_steps
            and motion_ok
        )

        diag.update({
            "long_dynamic_ready": float(1.0 if ready else 0.0),
            "long_dynamic_ready_streak": float(getattr(self, "_long_dynamic_object_ready_streak", 0)),
            "long_dynamic_min_steps": float(min_steps),
            "long_dynamic_stable_steps": float(stable_steps),
            "long_dynamic_min_active_steps": float(min_active_steps),
            "long_dynamic_conf_threshold": float(conf_thr),
            "long_dynamic_dz_threshold": float(dz_thr),
            "long_dynamic_depth_motion": float(depth_motion_value),
            "long_dynamic_depth_motion_threshold": float(depth_thr),
            "long_dynamic_motion_ok": float(1.0 if motion_ok else 0.0),
        })
        return z_dynamic, confidence, bool(ready), diag


    def build_inner_object_vision_proposals(self, obs: dict) -> torch.Tensor:
        """
        Build semantic proposals from long dynamic memory.

        Static frame never writes a slot. Dynamic input is accumulated in
        LongDynamicObjectMemory, and only z_dynamic_object can update memory.
        """
        self._ensure_progressive_semantic_memory_state()
        scene = summarize_vision_tensors(obs["left"], obs["right"], obs.get("depth")).to(self.device)
        self._slot_reconstruction_last_obs = obs
        if scene.ndim == 1:
            scene = scene.unsqueeze(0)
        self._inner_object_proposal_target_slots = []
        self._inner_object_proposal_kinds = []
        self._inner_object_proposal_target_names = []
        self._inner_object_dynamic_debug = {
            "dynamic_ready": False,
            "dynamic_source": "static_frame_no_write",
            "slot_update_allowed": False,
        }
        if hasattr(self, "get_m1_imit_inner_object_proposals"):
            m1_imit = self.get_m1_imit_inner_object_proposals(scene)
            if isinstance(m1_imit, dict) and torch.is_tensor(m1_imit.get("proposals")):
                self._inner_object_proposal_target_slots = list(m1_imit.get("target_slots", []))
                self._inner_object_proposal_kinds = list(m1_imit.get("proposal_kinds", []))
                self._inner_object_proposal_target_names = list(m1_imit.get("target_names", []))
                self._inner_object_dynamic_debug = {"dynamic_ready": True, "dynamic_source": "m1_object_slot_imit", "slot_update_allowed": True, "m1_imit_active": True, "m1_imit_details": m1_imit.get("details", [])}
                return m1_imit["proposals"]
        if not bool(getattr(self, "video_sensor_enabled", True)):
            return scene.unsqueeze(1)
        dynamic_score, scene_novelty, interaction = self._dynamic_sensory_input_score(obs, scene)
        dynamic_thr = float(getattr(self.cfg.object_image, "semantic_dynamic_threshold", 0.025))
        novelty_thr = float(getattr(self.cfg.object_image, "semantic_novelty_threshold", 0.08))
        interaction_thr = float(getattr(self.cfg.object_image, "semantic_interaction_threshold", 0.03))
        raw_dynamic_input_active = dynamic_score > dynamic_thr or scene_novelty > novelty_thr or interaction > interaction_thr
        target_motion_allowed, target_motion_reason = self._dynamic_target_motion_allowed()
        dynamic_input_active = bool(raw_dynamic_input_active and target_motion_allowed)
        z_dynamic, confidence, dynamic_ready, dyn_diag = self._long_dynamic_object_memory_step(
            obs=obs,
            z_static=scene,
            dynamic_score=float(dynamic_score),
            scene_novelty=float(scene_novelty),
            interaction=float(interaction),
            dynamic_input_active=bool(dynamic_input_active),
        )
        target_name = self._dynamic_current_target_name()
        self._inner_object_dynamic_debug = {
            "dynamic_input_active": bool(dynamic_input_active),
            "dynamic_target_name": str(target_name),
            "dynamic_input_raw_active": bool(raw_dynamic_input_active),
            "dynamic_target_motion_allowed": bool(target_motion_allowed),
            "dynamic_target_motion_reason": str(target_motion_reason),
            "dynamic_ready": bool(dynamic_ready),
            "dynamic_source": "long_dynamic_memory" if dynamic_ready else "observing_not_ready",
            "slot_update_allowed": bool(dynamic_ready),
            "dynamic_score": float(dynamic_score),
            "scene_novelty": float(scene_novelty),
            "interaction": float(interaction),
            **dict(dyn_diag),
        }
        if not dynamic_ready:
            return scene.unsqueeze(1)
        self._semantic_dynamic_seen_count += 1
        proposals = [z_dynamic]
        self._inner_object_proposal_target_slots.append(self._dynamic_target_slot_index(target_name))
        self._inner_object_proposal_target_names.append(str(target_name))
        self._inner_object_proposal_kinds.append("dynamic_object")
        if not bool(self._semantic_scene_slot_created):
            self._semantic_scene_slot_created = True
            print("[inner_object][long_dynamic_memory] first dynamic object image -> slot 0")
        write_event_slots = bool(getattr(self.cfg.object_image, "long_dynamic_write_event_slots", False))
        semantic_event_active = (interaction > interaction_thr) or (scene_novelty > novelty_thr)
        if write_event_slots:
            target_slot = self._allocate_or_keep_semantic_slot(semantic_event_active)
            if target_slot is not None:
                proposals.append(z_dynamic)
                self._inner_object_proposal_target_slots.append(int(target_slot))
                self._inner_object_proposal_kinds.append("dynamic_event")
        else:
            self._allocate_or_keep_semantic_slot(False)
        max_props = max(1, int(getattr(self.cfg.object_image, "max_object_proposals", 10)))
        proposals = proposals[:max_props]
        self._inner_object_proposal_target_slots = self._inner_object_proposal_target_slots[:len(proposals)]
        self._inner_object_proposal_kinds = self._inner_object_proposal_kinds[:len(proposals)]
        return torch.stack(proposals, dim=1)

    def _memory_update_forced_slot(
        self,
        state: dict,
        z_update: torch.Tensor,
        vision_strength: torch.Tensor,
        touch_strength: torch.Tensor,
        slot_index: int,
    ) -> dict:
        """
        Update exactly one slot and keep all other slots unchanged.

        This prevents the whole scene from washing through all slots.
        """
        slot_index = int(slot_index)

        before_z = state.get("z_obj_slots")
        before_c = state.get("confidence_slots")
        before_age = state.get("slot_age")

        slot = self.inner_object_system.memory(
            state,
            z_update,
            vision_strength,
            touch_strength,
            freeze_update=False,
            force_slot_index=slot_index,
        )

        try:
            z_new = slot.get("z_obj_slots")
            c_new = slot.get("confidence_slots")
            age_new = slot.get("slot_age")

            if torch.is_tensor(before_z) and torch.is_tensor(z_new):
                mask = torch.ones(z_new.shape[1], dtype=torch.bool, device=z_new.device)
                mask[slot_index] = False
                z_new[:, mask, :] = before_z.to(device=z_new.device, dtype=z_new.dtype)[:, mask, :]
                slot["z_obj_slots"] = z_new

            if torch.is_tensor(before_c) and torch.is_tensor(c_new):
                mask = torch.ones(c_new.shape[1], dtype=torch.bool, device=c_new.device)
                mask[slot_index] = False
                c_new[:, mask, :] = before_c.to(device=c_new.device, dtype=c_new.dtype)[:, mask, :]
                slot["confidence_slots"] = c_new

            if torch.is_tensor(before_age) and torch.is_tensor(age_new):
                mask = torch.ones(age_new.shape[1], dtype=torch.bool, device=age_new.device)
                mask[slot_index] = False
                age_new[:, mask, :] = before_age.to(device=age_new.device, dtype=age_new.dtype)[:, mask, :]
                slot["slot_age"] = age_new

        except Exception as e:
            if not hasattr(self, "_progressive_restore_warned"):
                print(f"[inner_object][semantic_memory] non-target restore failed: {e}")
                self._progressive_restore_warned = True

        try:
            z_slots = slot["z_obj_slots"]
            c_slots = slot["confidence_slots"]
            age = slot["slot_age"]
            b = z_slots.shape[0]
            idx = torch.full((b,), slot_index, device=z_slots.device, dtype=torch.long)
            batch_idx = torch.arange(b, device=z_slots.device)
            slot["z_obj"] = z_slots[batch_idx, idx]
            slot["confidence"] = c_slots[batch_idx, idx]
            slot["active_slot_age"] = age[batch_idx, idx]
            slot["active_slot_index"] = idx.view(b, 1)
        except Exception:
            pass

        return slot

    def _run_progressive_inner_object_system(
        self,
        prev_state: dict,
        vision: torch.Tensor,
        tactile: torch.Tensor,
        body: torch.Tensor,
        hand: torch.Tensor,
        leg: torch.Tensor,
        dream_mode: bool,
    ) -> dict:
        """
        Run inner object system with progressive semantic slot policy.

        If dream:
            use dream_step path unless an explicit M1 imit proposal targets
            ObjectSlotMemory slots.

        If awake:
            proposal 0 -> slot 0 = whole scene image
            proposal 1 -> next semantic slot, only if a semantic event exists
        """
        if vision.ndim == 2:
            vision = vision.unsqueeze(1)

        target_slots = list(getattr(self, "_inner_object_proposal_target_slots", []))
        proposal_kinds = list(getattr(self, "_inner_object_proposal_kinds", []))
        proposal_target_names = list(getattr(self, "_inner_object_proposal_target_names", []))

        # Hard guard: static_frame / dynamic_scene / one-frame crop must not update slots.
        allowed_kinds = {"dynamic_object", "dynamic_event", "dream_replay", "m1_imit_dynamic_object"}
        if target_slots:
            keep = []
            for i, _slot in enumerate(target_slots):
                kind = str(proposal_kinds[i]) if i < len(proposal_kinds) else ""
                if kind in allowed_kinds:
                    keep.append(i)
            if len(keep) != len(target_slots):
                if vision.ndim == 3 and keep:
                    idx = torch.tensor(keep, device=vision.device, dtype=torch.long)
                    vision = vision.index_select(1, idx)
                elif vision.ndim == 3 and not keep:
                    vision = vision[:, :0, :]
                target_slots = [int(target_slots[i]) for i in keep]
                proposal_kinds = [str(proposal_kinds[i]) for i in keep]
                proposal_target_names = [str(proposal_target_names[i]) for i in keep if i < len(proposal_target_names)]

        has_m1_imit_target = any(str(kind) == "m1_imit_dynamic_object" for kind in proposal_kinds)
        if dream_mode and not has_m1_imit_target:
            return self.inner_object_system(
                prev_state,
                vision,
                tactile,
                body,
                hand,
                leg,
                freeze_memory_update=False,
                dream_mode=True,
            )

        # No dynamic sensory input -> no semantic write.
        # Read and decode existing memory only.
        #
        # IMPORTANT:
        #   `vision` is a 12-dim sensory summary, not the 128-dim object latent.
        #   ObjectSlotMemory.read_state() uses z_template.shape[-1] to coerce
        #   slot dimensionality. Passing vision here creates z_obj with dim=12,
        #   then ObjectImaginationHead2D crashes:
        #       mat1 and mat2 shapes cannot be multiplied (1x12 and 128x256)
        #
        #   Therefore the read template must always be latent_dim-sized.
        if not target_slots:
            latent_dim = int(getattr(self.cfg.object_image, "latent_dim", 128))
            ref = prev_state.get("z_obj")
            if torch.is_tensor(ref) and ref.ndim >= 2 and int(ref.shape[-1]) == latent_dim:
                z_template = ref.to(self.device)
            else:
                if vision.ndim == 3:
                    batch = int(vision.shape[0])
                    device = vision.device
                    dtype = vision.dtype
                else:
                    batch = int(vision.shape[0]) if vision.ndim > 1 else 1
                    device = vision.device
                    dtype = vision.dtype
                z_template = torch.zeros(batch, latent_dim, device=device, dtype=dtype)

            slot = self.inner_object_system.memory.read_state(prev_state, z_template)
            if not bool(getattr(self, "video_sensor_enabled", True)):
                c_slots = slot.get("confidence_slots")
                if torch.is_tensor(c_slots):
                    c_zero = torch.zeros_like(c_slots)
                    slot["confidence_slots"] = c_zero
                    active_idx = slot.get("active_slot_index")
                    try:
                        if torch.is_tensor(active_idx):
                            idx = active_idx.to(device=c_zero.device).long().reshape(-1).clamp(0, c_zero.shape[1] - 1)
                            batch_idx = torch.arange(c_zero.shape[0], device=c_zero.device)
                            slot["confidence"] = c_zero[batch_idx, idx]
                        else:
                            slot["confidence"] = torch.zeros(c_zero.shape[0], 1, device=c_zero.device, dtype=c_zero.dtype)
                    except Exception:
                        slot["confidence"] = torch.zeros(c_zero.shape[0], 1, device=c_zero.device, dtype=c_zero.dtype)

            # Extra safety: never send a non-latent vector into the decoder.
            z_obj = slot.get("z_obj")
            if not torch.is_tensor(z_obj) or int(z_obj.shape[-1]) != latent_dim:
                z_obj = torch.zeros(z_template.shape[0], latent_dim, device=z_template.device, dtype=z_template.dtype)
                slot["z_obj"] = z_obj

            extra = {k: v for k, v in slot.items() if k != "z_obj"}
            decoded = self.inner_object_system.decode_z(slot["z_obj"], extra)
            try:
                decoded["semantic_slot_policy"] = torch.tensor([[2.0]], device=slot["z_obj"].device, dtype=slot["z_obj"].dtype)
                decoded["semantic_proposal_count"] = torch.tensor([[0.0]], device=slot["z_obj"].device, dtype=slot["z_obj"].dtype)
                dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
                decoded["long_dynamic_ready"] = torch.tensor([[float(bool(dbg.get("dynamic_ready", False)))]], device=slot["z_obj"].device, dtype=slot["z_obj"].dtype)
                decoded["long_dynamic_slot_update_allowed"] = torch.tensor([[float(bool(dbg.get("slot_update_allowed", False)))]], device=slot["z_obj"].device, dtype=slot["z_obj"].dtype)
            except Exception:
                pass
            decoded = self._attach_long_dynamic_debug_tensors(decoded, slot["z_obj"])
            return decoded

        state = prev_state
        last_slot = None

        p_count = int(vision.shape[1])
        for pi in range(p_count):
            v_i = vision[:, pi, :]
            target_slot = int(target_slots[pi]) if pi < len(target_slots) else 0

            source = str(proposal_kinds[pi]) if pi < len(proposal_kinds) else "dynamic_object"
            target_name_for_recon = str(proposal_target_names[pi]) if pi < len(proposal_target_names) else str(getattr(self, "_inner_object_dynamic_debug", {}).get("dynamic_target_name", "dynamic_object"))

            if source == "m1_imit_dynamic_object":
                # M1 imit proposal is already a z_obj-like latent.
                # Do NOT pass it through fusion(...), because fusion expects raw
                # sensory summaries and can make the target slot stay empty.
                latent_dim = int(getattr(self.cfg.object_image, "latent_dim", 128))
                z_update = v_i
                if torch.is_tensor(z_update) and int(z_update.shape[-1]) != latent_dim:
                    z_update = pad_or_trim(z_update, latent_dim)

                bsz = int(z_update.shape[0]) if torch.is_tensor(z_update) and z_update.ndim >= 2 else 1
                vision_strength = torch.ones(bsz, 1, device=z_update.device, dtype=z_update.dtype)
                touch_strength = torch.zeros(bsz, 1, device=z_update.device, dtype=z_update.dtype)
                fused = {
                    "z_update": z_update,
                    "vision_strength": vision_strength,
                    "touch_strength": touch_strength,
                    "touch_activity": touch_strength,
                    "vision_activity": vision_strength,
                }
            else:
                fused = self.inner_object_system.fusion(v_i, tactile, body, hand, leg)

            if source == "m1_imit_dynamic_object":
                try:
                    print(
                        "[inner_object][slot_write_attempt] "
                        f"pi={pi} source={source} target_slot={target_slot} "
                        f"v_shape={tuple(v_i.shape)} v_norm={float(v_i.detach().float().norm().cpu().item()):.4f}"
                    )
                except Exception:
                    pass

            slot = self._memory_update_forced_slot(
                state,
                fused["z_update"],
                fused["vision_strength"],
                fused["touch_strength"],
                target_slot,
            )
            if source == "m1_imit_dynamic_object":
                try:
                    z_slots = slot.get("z_obj_slots")
                    c_slots = slot.get("confidence_slots")
                    if torch.is_tensor(z_slots) and torch.is_tensor(c_slots):
                        print(
                            "[inner_object][slot_write_done] "
                            f"target_slot={target_slot} "
                            f"z_norm={float(z_slots[:, target_slot, :].detach().float().norm().cpu().item()):.4f} "
                            f"conf={float(c_slots[:, target_slot, :].detach().float().mean().cpu().item()):.4f}"
                        )
                except Exception:
                    pass
            if hasattr(self, "log_tetra_slot_write"):
                self.log_tetra_slot_write(target_slot, source, v_i, slot)
            self._slot_observation_reconstruction_step(target_slot, target_name_for_recon, source)
            self._slot_gaussian_reconstruction_step(target_slot, target_name_for_recon, source)
            self._slot_4d_timeline_step(target_slot, target_name_for_recon, source)
            self._slot_4d_deformation_step(target_slot, target_name_for_recon, source)
            self._slot_4d_playback_step(target_slot, target_name_for_recon, source)
            self._slot_4d_open3d_export_step(target_slot, target_name_for_recon, source)
            self._slot_4d_jsonrpc_stream_step(target_slot, target_name_for_recon, source)
            self._slot_object_memory_step(target_slot, target_name_for_recon, source)

            slot.update({
                "vision_strength": fused.get("vision_strength"),
                "touch_strength": fused.get("touch_strength"),
                "touch_activity": fused.get("touch_activity"),
                "vision_activity": fused.get("vision_activity"),
            })

            state = self.inner_object_system._state_from_slot_output(slot)
            last_slot = slot

        if last_slot is None:
            return self.inner_object_system(
                prev_state,
                vision[:, 0, :] if vision.ndim == 3 else vision,
                tactile,
                body,
                hand,
                leg,
                freeze_memory_update=False,
                dream_mode=False,
            )

        # Decode the last updated semantic image.
        extra = {k: v for k, v in last_slot.items() if k != "z_obj"}
        decoded = self.inner_object_system.decode_z(last_slot["z_obj"], extra)

        try:
            decoded["semantic_slot_policy"] = torch.tensor([[1.0]], device=last_slot["z_obj"].device, dtype=last_slot["z_obj"].dtype)
            decoded["semantic_updated_slot"] = decoded.get("active_slot_index")
            decoded["semantic_proposal_count"] = torch.tensor([[float(p_count)]], device=last_slot["z_obj"].device, dtype=last_slot["z_obj"].dtype)
            dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
            decoded["long_dynamic_ready"] = torch.tensor([[float(bool(dbg.get("dynamic_ready", False)))]], device=last_slot["z_obj"].device, dtype=last_slot["z_obj"].dtype)
            decoded["long_dynamic_slot_update_allowed"] = torch.tensor([[float(bool(dbg.get("slot_update_allowed", False)))]], device=last_slot["z_obj"].device, dtype=last_slot["z_obj"].dtype)
            if source == "m1_imit_dynamic_object":
                decoded["debug_imit_fallback_shape"] = True
                decoded["debug_imit_source"] = "m1_object_slot_imit"
                decoded["debug_imit_shape_kind"] = target_name_for_recon
        except Exception:
            pass

        decoded = self._attach_long_dynamic_debug_tensors(decoded, last_slot["z_obj"])
        return decoded





    # -------------------------------------------------------------------------
    # Step 1 toward 3DGS/4D: per-slot observation buffer + point cloud
    # -------------------------------------------------------------------------
    def _ensure_slot_observation_reconstruction(self) -> None:
        if hasattr(self, "slot_observation_buffer") and hasattr(self, "slot_pointcloud_reconstructor"):
            return
        from src.modules.m01_object_imagery.slot_observation_reconstruction import SlotObservationBuffer, SlotPointCloudReconstructor
        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_observation_buffer = SlotObservationBuffer(max_frames_per_slot=int(getattr(cfg_obj, "slot_observation_max_frames", 96)))
        self.slot_pointcloud_reconstructor = SlotPointCloudReconstructor(max_points_per_slot=int(getattr(cfg_obj, "slot_pointcloud_max_points", 24000)), stride=int(getattr(cfg_obj, "slot_pointcloud_depth_stride", 6)))
        self._slot_reconstruction_latest_metrics = {}
        print("[slot_recon] observation buffer + point cloud reconstructor initialized")

    def _slot_observation_reconstruction_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        obs = getattr(self, "_slot_reconstruction_last_obs", None)
        if not isinstance(obs, dict):
            return
        try:
            self._ensure_slot_observation_reconstruction()
            dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
            formed_conf = float(dbg.get("long_dynamic_confidence", 0.0) or 0.0) if bool(dbg.get("dynamic_ready", False)) else 0.0
            z_dynamic_norm = float(dbg.get("long_dynamic_z_dynamic_norm", 0.0) or 0.0)
            live_step = int(getattr(self, "live_step", getattr(self, "global_step", 0)))
            observation = self.slot_pointcloud_reconstructor.observation_from_runtime(slot_id=int(slot_id), target_name=str(target_name or "dynamic_object"), runtime_obs=obs, live_step=live_step, formed_conf=formed_conf, z_dynamic_norm=z_dynamic_norm)
            frame_count = self.slot_observation_buffer.add(observation)
            metrics = self.slot_pointcloud_reconstructor.integrate(observation)
            metrics["frame_count"] = int(frame_count)
            metrics["slot_id"] = int(slot_id)
            metrics["target_name"] = str(target_name or "dynamic_object")
            self._slot_reconstruction_latest_metrics = metrics
            if hasattr(self, "log_slot_observation_buffer"):
                self.log_slot_observation_buffer(slot_id=int(slot_id), target_name=str(target_name or "dynamic_object"), frame_count=int(frame_count), depth_valid=bool(metrics.get("depth_valid", False)), live_step=live_step)
            if hasattr(self, "log_slot_pointcloud_reconstruction"):
                self.log_slot_pointcloud_reconstruction(**metrics)
        except Exception as e:
            if not hasattr(self, "_slot_reconstruction_warned"):
                print(f"[slot_recon] update failed: {e}")
                self._slot_reconstruction_warned = True



    # -------------------------------------------------------------------------
    # Step 2A toward 3DGS/4D: per-slot low-res Gaussian reconstructor
    # -------------------------------------------------------------------------
    def _ensure_slot_gaussian_reconstruction(self) -> None:
        if hasattr(self, "slot_gaussian_reconstructor"):
            return
        from src.modules.m01_object_imagery.slot_gaussian_reconstruction import SlotGaussianReconstructor

        cfg_obj = getattr(self.cfg, "object_image", None)

        # Step 2B backend switch:
        #   object_image.slot_gaussian_renderer_backend = torch_lowres | cuda_3dgs | auto
        slot_gaussian_renderer_backend = str(getattr(cfg_obj, "slot_gaussian_renderer_backend", "auto"))
        slot_gaussian_cuda_allow_fallback = bool(getattr(cfg_obj, "slot_gaussian_cuda_allow_fallback", True))

        self.slot_gaussian_reconstructor = SlotGaussianReconstructor(
            image_size=int(getattr(cfg_obj, "slot_gaussian_image_size", 64)),
            max_gaussians=int(getattr(cfg_obj, "slot_gaussian_max_gaussians", 768)),
            max_render_gaussians=int(getattr(cfg_obj, "slot_gaussian_max_render_gaussians", 256)),
            lr=float(getattr(cfg_obj, "slot_gaussian_lr", 3.0e-3)),
            train_steps_per_update=int(getattr(cfg_obj, "slot_gaussian_train_steps_per_update", 1)),
            depth_weight=float(getattr(cfg_obj, "slot_gaussian_depth_weight", 0.35)),
            renderer_backend=slot_gaussian_renderer_backend,
            allow_fallback=slot_gaussian_cuda_allow_fallback,
            preview_every_steps=int(getattr(cfg_obj, "slot_gaussian_preview_every_steps", 1)),
            device=self.device,
        )

        self._slot_gaussian_latest_metrics = {}
        self._slot_gaussian_success_logged = False

        try:
            if hasattr(self, "log_slot_gaussian_cuda_backend"):
                self.log_slot_gaussian_cuda_backend(**self.slot_gaussian_reconstructor.backend_status())
            if hasattr(self, "log_tetra_repair_applied") and not bool(getattr(self, "_slot_gaussian_step2b_repair_logged", False)):
                self.log_tetra_repair_applied(
                    repair_stage="slot_gaussian_step2b_backend_metrics",
                    reason="correct_torch_lowres_fallback_flag_and_import_error_metrics",
                    changed_area="slot_gaussian_cuda_adapter,slot_gaussian_reconstruction",
                )
                self._slot_gaussian_step2b_repair_logged = True
        except Exception:
            pass

        print(
            "[slot_3dgs] Step 2B Gaussian renderer "
            f"backend={slot_gaussian_renderer_backend} "
            f"fallback={slot_gaussian_cuda_allow_fallback}"
        )

    def _slot_gaussian_reconstruction_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            if not hasattr(self, "slot_observation_buffer") or not hasattr(self, "slot_pointcloud_reconstructor"):
                return

            self._ensure_slot_gaussian_reconstruction()

            slot_id = int(slot_id)
            target_name = str(target_name or "dynamic_object")
            observation = self.slot_observation_buffer.latest(slot_id)
            if observation is None:
                return

            points = getattr(self.slot_pointcloud_reconstructor, "points", {}).get(slot_id)
            colors = getattr(self.slot_pointcloud_reconstructor, "colors", {}).get(slot_id)

            metrics = self.slot_gaussian_reconstructor.train_step(
                slot_id=slot_id,
                target_name=target_name,
                observation=observation,
                points=points,
                colors=colors,
            )
            self._slot_gaussian_latest_metrics = dict(metrics)

            if hasattr(self, "log_slot_gaussian_init") and bool(metrics.get("initialized", False)):
                n_src = 0 if points is None else int(getattr(points, "shape", [0])[0])
                self.log_slot_gaussian_init(
                    slot_id=slot_id,
                    target_name=target_name,
                    gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
                    source_points=n_src,
                )

            if hasattr(self, "log_slot_gaussian_train"):
                self.log_slot_gaussian_train(**metrics)
            if hasattr(self, "log_slot_gaussian_render"):
                self.log_slot_gaussian_render(**metrics)
            if hasattr(self, "log_slot_gaussian_cuda_backend"):
                self.log_slot_gaussian_cuda_backend(**metrics)
            if hasattr(self, "log_slot_gaussian_preview_frame"):
                self.log_slot_gaussian_preview_frame(**metrics)

            self._maybe_log_slot_gaussian_step2b_success()

        except Exception as e:
            if not hasattr(self, "_slot_gaussian_warned"):
                print(f"[slot_3dgs] update failed: {e}")
                self._slot_gaussian_warned = True

    def _maybe_log_slot_gaussian_step2_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_gaussian_success_logged", False)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return
            metrics_by_slot = getattr(self.slot_gaussian_reconstructor, "last_metrics", {}) or {}
            m0 = metrics_by_slot.get(0)
            m1 = metrics_by_slot.get(1)
            if m0 is None or m1 is None:
                return
            ok0 = int(getattr(m0, "gaussian_count", 0)) > 0 and int(getattr(m0, "updates", 0)) > 0
            ok1 = int(getattr(m1, "gaussian_count", 0)) > 0 and int(getattr(m1, "updates", 0)) > 0
            if not (ok0 and ok1):
                return
            if hasattr(self, "log_success_slot_gaussian_step2"):
                self.log_success_slot_gaussian_step2(
                    slot_0_target=str(getattr(m0, "target_name", "tetrahedron")),
                    slot_0_gaussian_count=int(getattr(m0, "gaussian_count", 0)),
                    slot_0_recon_loss=float(getattr(m0, "total_loss", 0.0)),
                    slot_0_updates=int(getattr(m0, "updates", 0)),
                    slot_1_target=str(getattr(m1, "target_name", "cube")),
                    slot_1_gaussian_count=int(getattr(m1, "gaussian_count", 0)),
                    slot_1_recon_loss=float(getattr(m1, "total_loss", 0.0)),
                    slot_1_updates=int(getattr(m1, "updates", 0)),
                )
                self._slot_gaussian_success_logged = True
        except Exception:
            pass


    def _maybe_log_slot_gaussian_step2b_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_gaussian_success_logged", False)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return

            metrics_by_slot = getattr(self.slot_gaussian_reconstructor, "last_metrics", {}) or {}
            m0 = metrics_by_slot.get(0)
            m1 = metrics_by_slot.get(1)
            if m0 is None or m1 is None:
                return

            ok0 = int(getattr(m0, "gaussian_count", 0)) > 0 and int(getattr(m0, "updates", 0)) > 0
            ok1 = int(getattr(m1, "gaussian_count", 0)) > 0 and int(getattr(m1, "updates", 0)) > 0
            if not (ok0 and ok1):
                return

            if hasattr(self, "log_success_slot_gaussian_cuda_step2b"):
                self.log_success_slot_gaussian_cuda_step2b(
                    slot_0_target=str(getattr(m0, "target_name", "tetrahedron")),
                    slot_0_gaussian_count=int(getattr(m0, "gaussian_count", 0)),
                    slot_0_recon_loss=float(getattr(m0, "total_loss", 0.0)),
                    slot_0_updates=int(getattr(m0, "updates", 0)),
                    slot_0_backend=str(getattr(m0, "backend", "torch_lowres")),
                    slot_0_preview_fps=float(getattr(m0, "preview_fps", 0.0)),
                    slot_1_target=str(getattr(m1, "target_name", "cube")),
                    slot_1_gaussian_count=int(getattr(m1, "gaussian_count", 0)),
                    slot_1_recon_loss=float(getattr(m1, "total_loss", 0.0)),
                    slot_1_updates=int(getattr(m1, "updates", 0)),
                    slot_1_backend=str(getattr(m1, "backend", "torch_lowres")),
                    slot_1_preview_fps=float(getattr(m1, "preview_fps", 0.0)),
                    fallback_used=bool(getattr(m0, "fallback_used", False) or getattr(m1, "fallback_used", False)),
                )
                self._slot_gaussian_success_logged = True
        except Exception:
            pass



    # -------------------------------------------------------------------------
    # Step 3A toward 4D reconstruction: per-slot Gaussian timeline
    # -------------------------------------------------------------------------
    def _ensure_slot_4d_reconstruction(self) -> None:
        if hasattr(self, "slot_4d_reconstructor"):
            return
        from src.modules.m01_object_imagery.slot_4d_reconstruction import Slot4DReconstructor

        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_4d_reconstructor = Slot4DReconstructor(
            max_frames_per_slot=int(getattr(cfg_obj, "slot_4d_timeline_max_frames", 256)),
            sample_points=int(getattr(cfg_obj, "slot_4d_sample_points", 128)),
        )
        self._slot_4d_latest_metrics = {}
        self._slot_4d_success_logged = False
        print("[slot_4d] Step 3A per-slot Gaussian timeline initialized")

    def _slot_4d_timeline_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            if not bool(getattr(getattr(self.cfg, "object_image", None), "slot_4d_timeline_enabled", True)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return
            self._ensure_slot_4d_reconstruction()

            dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
            live_step = int(getattr(self, "live_step", getattr(self, "global_step", 0)))
            formed_conf = float(dbg.get("long_dynamic_confidence", 0.0) or 0.0) if bool(dbg.get("dynamic_ready", False)) else 0.0
            z_dynamic_norm = float(dbg.get("long_dynamic_z_dynamic_norm", 0.0) or 0.0)

            metrics = self.slot_4d_reconstructor.add_from_gaussian_reconstructor(
                slot_id=int(slot_id),
                target_name=str(target_name or "dynamic_object"),
                live_step=live_step,
                gaussian_reconstructor=self.slot_gaussian_reconstructor,
                formed_conf=formed_conf,
                z_dynamic_norm=z_dynamic_norm,
            )
            self._slot_4d_latest_metrics = dict(metrics)

            if hasattr(self, "log_slot_4d_frame"):
                self.log_slot_4d_frame(**metrics)
            if hasattr(self, "log_slot_4d_timeline"):
                self.log_slot_4d_timeline(**metrics)

            self._maybe_log_slot_4d_timeline_success()
        except Exception as e:
            if not hasattr(self, "_slot_4d_warned"):
                print(f"[slot_4d] update failed: {e}")
                self._slot_4d_warned = True

    def _maybe_log_slot_4d_timeline_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_4d_success_logged", False)):
                return
            if not hasattr(self, "slot_4d_reconstructor"):
                return

            s = self.slot_4d_reconstructor.summary()
            slot0 = dict(s.get("slot_0", {}) or {})
            slot1 = dict(s.get("slot_1", {}) or {})

            slot0_target = str(slot0.get("target_name", "unknown"))
            slot1_target = str(slot1.get("target_name", "unknown"))
            slot0_frames = int(slot0.get("frame_count", 0) or 0)
            slot1_frames = int(slot1.get("frame_count", 0) or 0)
            slot0_gauss = int(slot0.get("gaussian_count", 0) or 0)
            slot1_gauss = int(slot1.get("gaussian_count", 0) or 0)

            if not (
                slot0_target == "tetrahedron"
                and slot1_target == "cube"
                and slot0_frames > 0
                and slot1_frames > 0
                and slot0_gauss > 0
                and slot1_gauss > 0
            ):
                return

            if hasattr(self, "log_success_slot_4d_timeline_step3a"):
                self.log_success_slot_4d_timeline_step3a(
                    slot_0_target=slot0_target,
                    slot_0_timeline_frames=slot0_frames,
                    slot_0_gaussian_count=slot0_gauss,
                    slot_0_temporal_span=int(slot0.get("temporal_span", 0) or 0),
                    slot_1_target=slot1_target,
                    slot_1_timeline_frames=slot1_frames,
                    slot_1_gaussian_count=slot1_gauss,
                    slot_1_temporal_span=int(slot1.get("temporal_span", 0) or 0),
                )
                self._slot_4d_success_logged = True
        except Exception:
            pass



    # -------------------------------------------------------------------------
    # Step 3B: neural 4D deformation field from timeline frame pairs
    # -------------------------------------------------------------------------
    def _ensure_slot_4d_deformation(self) -> None:
        if hasattr(self, "slot_4d_deformation_trainer"):
            return
        from src.modules.m01_object_imagery.slot_4d_deformation import Slot4DDeformationTrainer

        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_4d_deformation_trainer = Slot4DDeformationTrainer(
            hidden_dim=int(getattr(cfg_obj, "slot_4d_deformation_hidden_dim", 96)),
            lr=float(getattr(cfg_obj, "slot_4d_deformation_lr", 2.0e-3)),
            train_steps_per_update=int(getattr(cfg_obj, "slot_4d_deformation_train_steps_per_update", 1)),
            min_frames=int(getattr(cfg_obj, "slot_4d_deformation_min_frames", 2)),
            delta_reg_weight=float(getattr(cfg_obj, "slot_4d_deformation_delta_reg_weight", 1.0e-4)),
            device=self.device,
        )
        self._slot_4d_deformation_latest_metrics = {}
        self._slot_4d_deformation_success_logged = False
        print("[slot_4d] Step 3B neural deformation trainer initialized")

    def _slot_4d_deformation_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if not bool(getattr(cfg_obj, "slot_4d_deformation_enabled", True)):
                return
            if not hasattr(self, "slot_4d_reconstructor"):
                return
            self._ensure_slot_4d_deformation()

            timeline = getattr(self.slot_4d_reconstructor, "timeline", None)
            if timeline is None:
                return

            metrics = self.slot_4d_deformation_trainer.train_from_timeline(
                slot_id=int(slot_id),
                target_name=str(target_name or "dynamic_object"),
                timeline=timeline,
            )
            self._slot_4d_deformation_latest_metrics = dict(metrics)

            if hasattr(self, "log_slot_4d_deformation_model"):
                self.log_slot_4d_deformation_model(
                    slot_id=int(slot_id),
                    target_name=str(target_name or "dynamic_object"),
                    model_type="Slot4DDeformationModel",
                    trainable_params=int(self.slot_4d_deformation_trainer.trainable_params()),
                    enabled=bool(metrics.get("enabled", False)),
                    trainable=bool(metrics.get("trainable", False)),
                )
            if hasattr(self, "log_slot_4d_deformation_train"):
                self.log_slot_4d_deformation_train(**metrics)

            self._maybe_log_slot_4d_deformation_success()
        except Exception as e:
            if not hasattr(self, "_slot_4d_deformation_warned"):
                print(f"[slot_4d] deformation update failed: {e}")
                self._slot_4d_deformation_warned = True

    def _maybe_log_slot_4d_deformation_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_4d_deformation_success_logged", False)):
                return
            if not hasattr(self, "slot_4d_deformation_trainer"):
                return

            s = self.slot_4d_deformation_trainer.summary()
            slot0 = dict(s.get("slot_0", {}) or {})
            slot1 = dict(s.get("slot_1", {}) or {})

            ok0 = (
                str(slot0.get("target_name", "")) == "tetrahedron"
                and bool(slot0.get("valid", False))
                and int(slot0.get("updates", 0) or 0) > 0
            )
            ok1 = (
                str(slot1.get("target_name", "")) == "cube"
                and bool(slot1.get("valid", False))
                and int(slot1.get("updates", 0) or 0) > 0
            )
            if not (ok0 and ok1):
                return

            if hasattr(self, "log_success_slot_4d_deformation_step3b"):
                self.log_success_slot_4d_deformation_step3b(
                    slot_0_target=str(slot0.get("target_name", "tetrahedron")),
                    slot_0_deformation_updates=int(slot0.get("updates", 0) or 0),
                    slot_0_deformation_loss=float(slot0.get("loss", 0.0) or 0.0),
                    slot_0_motion_norm=float(slot0.get("motion_norm", 0.0) or 0.0),
                    slot_0_sample_count=int(slot0.get("sample_count", 0) or 0),
                    slot_1_target=str(slot1.get("target_name", "cube")),
                    slot_1_deformation_updates=int(slot1.get("updates", 0) or 0),
                    slot_1_deformation_loss=float(slot1.get("loss", 0.0) or 0.0),
                    slot_1_motion_norm=float(slot1.get("motion_norm", 0.0) or 0.0),
                    slot_1_sample_count=int(slot1.get("sample_count", 0) or 0),
                    trainable_params=int(self.slot_4d_deformation_trainer.trainable_params()),
                )
                self._slot_4d_deformation_success_logged = True
        except Exception:
            pass



    # -------------------------------------------------------------------------
    # Step 3C: deformation-aware 4D playback preview
    # -------------------------------------------------------------------------
    def _ensure_slot_4d_playback(self) -> None:
        if hasattr(self, "slot_4d_playback_renderer"):
            return
        from src.modules.m01_object_imagery.slot_4d_playback import Slot4DPlaybackRenderer

        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_4d_playback_renderer = Slot4DPlaybackRenderer(
            period_steps=int(getattr(cfg_obj, "slot_4d_playback_period_steps", 120)),
            strength=float(getattr(cfg_obj, "slot_4d_playback_strength", 1.0)),
        )
        self._slot_4d_playback_latest_metrics = {}
        self._slot_4d_playback_success_logged = False
        print("[slot_4d] Step 3C deformation-aware playback initialized")

    def _slot_4d_playback_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if not bool(getattr(cfg_obj, "slot_4d_playback_enabled", True)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor") or not hasattr(self, "slot_4d_deformation_trainer"):
                return
            self._ensure_slot_4d_playback()

            live_step = int(getattr(self, "live_step", getattr(self, "global_step", 0)))
            metrics = self.slot_4d_playback_renderer.render_slot(
                slot_id=int(slot_id),
                target_name=str(target_name or "dynamic_object"),
                live_step=live_step,
                gaussian_reconstructor=self.slot_gaussian_reconstructor,
                deformation_trainer=self.slot_4d_deformation_trainer,
            )
            self._slot_4d_playback_latest_metrics = dict(metrics)

            if hasattr(self, "log_slot_4d_playback_frame"):
                self.log_slot_4d_playback_frame(**metrics)
            if hasattr(self, "log_slot_4d_deformed_render"):
                self.log_slot_4d_deformed_render(**metrics)

            self._maybe_log_slot_4d_playback_success()
        except Exception as e:
            if not hasattr(self, "_slot_4d_playback_warned"):
                print(f"[slot_4d] playback update failed: {e}")
                self._slot_4d_playback_warned = True

    def _maybe_log_slot_4d_playback_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_4d_playback_success_logged", False)):
                return
            if not hasattr(self, "slot_4d_playback_renderer"):
                return

            s = self.slot_4d_playback_renderer.summary()
            slot0 = dict(s.get("slot_0", {}) or {})
            slot1 = dict(s.get("slot_1", {}) or {})

            ok0 = (
                str(slot0.get("target_name", "")) == "tetrahedron"
                and bool(slot0.get("render_valid", False))
                and bool(slot0.get("deformation_used", False))
                and int(slot0.get("frame_count", 0) or 0) > 0
            )
            ok1 = (
                str(slot1.get("target_name", "")) == "cube"
                and bool(slot1.get("render_valid", False))
                and bool(slot1.get("deformation_used", False))
                and int(slot1.get("frame_count", 0) or 0) > 0
            )
            if not (ok0 and ok1):
                return

            if hasattr(self, "log_success_slot_4d_playback_step3c"):
                self.log_success_slot_4d_playback_step3c(
                    slot_0_target=str(slot0.get("target_name", "tetrahedron")),
                    slot_0_playback_frames=int(slot0.get("frame_count", 0) or 0),
                    slot_0_playback_phase=float(slot0.get("playback_phase", 0.0) or 0.0),
                    slot_0_pred_delta_norm=float(slot0.get("pred_delta_norm", 0.0) or 0.0),
                    slot_0_backend=str(slot0.get("backend", "unknown")),
                    slot_1_target=str(slot1.get("target_name", "cube")),
                    slot_1_playback_frames=int(slot1.get("frame_count", 0) or 0),
                    slot_1_playback_phase=float(slot1.get("playback_phase", 0.0) or 0.0),
                    slot_1_pred_delta_norm=float(slot1.get("pred_delta_norm", 0.0) or 0.0),
                    slot_1_backend=str(slot1.get("backend", "unknown")),
                )
                self._slot_4d_playback_success_logged = True
        except Exception:
            pass



    # -------------------------------------------------------------------------
    # Step 3F: export per-slot raw/deformed Gaussian point clouds for Open3D
    # -------------------------------------------------------------------------
    def _ensure_slot_4d_open3d_exporter(self) -> None:
        if hasattr(self, "slot_4d_open3d_exporter"):
            return
        from src.modules.m01_object_imagery.slot_4d_open3d_export import Slot4DOpen3DExporter
        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_4d_open3d_exporter = Slot4DOpen3DExporter(
            export_path=str(getattr(cfg_obj, "slot_4d_open3d_export_path", "./checkpoint/slot_viewer/slot_4d_open3d_latest.npz")),
            sample_points=int(getattr(cfg_obj, "slot_4d_open3d_sample_points", 4096)),
            min_interval_sec=float(getattr(cfg_obj, "slot_4d_open3d_min_interval_sec", 0.05)),
        )
        self._slot_4d_open3d_latest_metrics = {}
        self._slot_4d_open3d_success_logged = False
        print("[slot_4d] Step 3F Open3D export initialized")

    def _slot_4d_open3d_export_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if not bool(getattr(cfg_obj, "slot_4d_open3d_export_enabled", True)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return
            self._ensure_slot_4d_open3d_exporter()
            result = self.slot_4d_open3d_exporter.export(
                gaussian_reconstructor=self.slot_gaussian_reconstructor,
                deformation_trainer=getattr(self, "slot_4d_deformation_trainer", None),
                playback_renderer=getattr(self, "slot_4d_playback_renderer", None),
            )
            self._slot_4d_open3d_latest_metrics = dict(result)
            if hasattr(self, "log_slot_4d_open3d_export"):
                self.log_slot_4d_open3d_export(slot_id=int(slot_id), target_name=str(target_name or "dynamic_object"), **result)
            self._maybe_log_slot_4d_open3d_success()
        except Exception as e:
            if not hasattr(self, "_slot_4d_open3d_export_warned"):
                print(f"[slot_4d] Open3D export failed: {e}")
                self._slot_4d_open3d_export_warned = True

    def _maybe_log_slot_4d_open3d_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_4d_open3d_success_logged", False)) or not hasattr(self, "slot_4d_open3d_exporter"):
                return
            m = getattr(self.slot_4d_open3d_exporter, "last_metrics", {}) or {}
            m0, m1 = m.get(0), m.get(1)
            if m0 is None or m1 is None:
                return
            if int(getattr(m0, "raw_points", 0) or 0) <= 0 or int(getattr(m1, "raw_points", 0) or 0) <= 0:
                return
            if hasattr(self, "log_success_slot_4d_open3d_viewer_step3f"):
                self.log_success_slot_4d_open3d_viewer_step3f(
                    export_path=str(getattr(m0, "export_path", "")),
                    slot_0_target=str(getattr(m0, "target_name", "tetrahedron")),
                    slot_0_raw_points=int(getattr(m0, "raw_points", 0) or 0),
                    slot_0_deformed_points=int(getattr(m0, "deformed_points", 0) or 0),
                    slot_0_deformation_used=bool(getattr(m0, "deformation_used", False)),
                    slot_1_target=str(getattr(m1, "target_name", "cube")),
                    slot_1_raw_points=int(getattr(m1, "raw_points", 0) or 0),
                    slot_1_deformed_points=int(getattr(m1, "deformed_points", 0) or 0),
                    slot_1_deformation_used=bool(getattr(m1, "deformation_used", False)),
                )
                self._slot_4d_open3d_success_logged = True
        except Exception:
            pass


    def _ensure_slot_4d_jsonrpc_streamer(self) -> None:
        if hasattr(self, "slot_4d_jsonrpc_streamer") and bool(getattr(self.slot_4d_jsonrpc_streamer, "started", False)):
            return
        from src.modules.m02_event_dream_replay.slot_4d_jsonrpc_stream import Slot4DJsonRpcStreamer
        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_4d_jsonrpc_streamer = Slot4DJsonRpcStreamer(host=str(getattr(cfg_obj, "slot_4d_jsonrpc_host", "127.0.0.1")), port=int(getattr(cfg_obj, "slot_4d_jsonrpc_port", 8771)), sample_points=int(getattr(cfg_obj, "slot_4d_jsonrpc_sample_points", 4096)))
        started = self.slot_4d_jsonrpc_streamer.start()
        self._slot_4d_jsonrpc_latest_metrics = {}
        self._slot_4d_jsonrpc_success_logged = False
        if started:
            try:
                from src.apps.runner_thread_affinity import apply_thread_affinity

                apply_thread_affinity(
                    self.cfg,
                    "slot_4d_jsonrpc",
                    getattr(self.slot_4d_jsonrpc_streamer, "thread", None),
                    label="Slot4D JSON-RPC",
                )
            except Exception as e:
                print(f"[slot_4d] JSON-RPC affinity skipped: {e}")
            print(f"[slot_4d] JSON-RPC stream started at {self.slot_4d_jsonrpc_streamer.host}:{self.slot_4d_jsonrpc_streamer.port}")

    def start_slot_4d_jsonrpc_streamer_if_enabled(self) -> None:
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if bool(getattr(cfg_obj, "slot_4d_jsonrpc_enabled", True)):
                self._ensure_slot_4d_jsonrpc_streamer()
        except Exception as e:
            if not hasattr(self, "_slot_4d_jsonrpc_start_warned"):
                print(f"[slot_4d] JSON-RPC early start failed: {e}")
                self._slot_4d_jsonrpc_start_warned = True

    def shutdown_slot_4d_jsonrpc_streamer(self) -> None:
        try:
            streamer = getattr(self, "slot_4d_jsonrpc_streamer", None)
            if streamer is not None and hasattr(streamer, "shutdown"):
                streamer.shutdown(timeout=1.0)
        except Exception as e:
            print(f"[slot_4d] JSON-RPC shutdown failed: {e}")

    def _slot_4d_jsonrpc_stream_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if not bool(getattr(cfg_obj, "slot_4d_jsonrpc_enabled", True)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return
            self._ensure_slot_4d_jsonrpc_streamer()
            live_step = int(getattr(self, "live_step", getattr(self, "global_step", 0)))
            metrics = self.slot_4d_jsonrpc_streamer.publish(self.slot_gaussian_reconstructor, deformation_trainer=getattr(self, "slot_4d_deformation_trainer", None), playback_renderer=getattr(self, "slot_4d_playback_renderer", None), live_step=live_step)
            self._slot_4d_jsonrpc_latest_metrics = dict(metrics)
            if hasattr(self, "log_slot_4d_jsonrpc_stream"):
                self.log_slot_4d_jsonrpc_stream(slot_id=int(slot_id), target_name=str(target_name or "dynamic_object"), **metrics)
            self._maybe_log_slot_4d_jsonrpc_success()
        except Exception as e:
            if not hasattr(self, "_slot_4d_jsonrpc_warned"):
                print(f"[slot_4d] JSON-RPC stream failed: {e}")
                self._slot_4d_jsonrpc_warned = True

    def tick_slot_4d_jsonrpc_streamer(self) -> None:
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if not bool(getattr(cfg_obj, "slot_4d_jsonrpc_enabled", True)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return
            now = time.time()
            last = float(getattr(self, "_slot_4d_jsonrpc_last_tick_unix", 0.0) or 0.0)
            if (now - last) < 0.05:
                return
            self._slot_4d_jsonrpc_last_tick_unix = now
            self._ensure_slot_4d_jsonrpc_streamer()
            live_step = int(getattr(self, "live_step", getattr(self, "global_step", 0)))
            metrics = self.slot_4d_jsonrpc_streamer.publish(
                self.slot_gaussian_reconstructor,
                deformation_trainer=getattr(self, "slot_4d_deformation_trainer", None),
                playback_renderer=getattr(self, "slot_4d_playback_renderer", None),
                live_step=live_step,
            )
            self._slot_4d_jsonrpc_latest_metrics = dict(metrics)
        except Exception as e:
            if not hasattr(self, "_slot_4d_jsonrpc_tick_warned"):
                print(f"[slot_4d] JSON-RPC live tick failed: {e}")
                self._slot_4d_jsonrpc_tick_warned = True

    def _maybe_log_slot_4d_jsonrpc_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_4d_jsonrpc_success_logged", False)):
                return
            m = getattr(self, "_slot_4d_jsonrpc_latest_metrics", {}) or {}
            if int(m.get("slot_0_points", 0) or 0) <= 0 or int(m.get("slot_1_points", 0) or 0) <= 0:
                return
            if hasattr(self, "log_success_slot_4d_jsonrpc_stream_step3h"):
                self.log_success_slot_4d_jsonrpc_stream_step3h(host=str(m.get("host", "127.0.0.1")), port=int(m.get("port", 8771)), slot_0_points=int(m.get("slot_0_points", 0) or 0), slot_1_points=int(m.get("slot_1_points", 0) or 0), slot_0_deformed_points=int(m.get("slot_0_deformed_points", 0) or 0), slot_1_deformed_points=int(m.get("slot_1_deformed_points", 0) or 0))
                self._slot_4d_jsonrpc_success_logged = True
        except Exception:
            pass


    # -------------------------------------------------------------------------
    # Step 4D/4E: persistent inner-object memory and recall diagnostics
    # -------------------------------------------------------------------------
    def _ensure_slot_object_memory_manager(self) -> None:
        if hasattr(self, "slot_object_memory_manager"):
            return
        from src.modules.m01_object_imagery.slot_object_memory import SlotObjectMemoryManager
        cfg_obj = getattr(self.cfg, "object_image", None)
        self.slot_object_memory_manager = SlotObjectMemoryManager(
            root=str(getattr(cfg_obj, "slot_object_memory_root", "./checkpoint/slot_memory")),
            match_threshold=float(getattr(cfg_obj, "slot_object_memory_match_threshold", 0.30)),
        )
        self._slot_object_memory_latest_metrics = {}
        self._slot_object_memory_success_logged = False
        print("[slot_memory] Step 4D/4E persistent object memory initialized")

    def _slot_object_memory_step(self, slot_id: int, target_name: str, source: str) -> None:
        if "dynamic" not in str(source):
            return
        try:
            cfg_obj = getattr(self.cfg, "object_image", None)
            if not bool(getattr(cfg_obj, "slot_object_memory_enabled", True)):
                return
            if not hasattr(self, "slot_gaussian_reconstructor"):
                return
            self._ensure_slot_object_memory_manager()
            result = self.slot_object_memory_manager.save_from_runtime(
                slot_id=int(slot_id),
                target_name=str(target_name or "dynamic_object"),
                gaussian_reconstructor=self.slot_gaussian_reconstructor,
                deformation_trainer=getattr(self, "slot_4d_deformation_trainer", None),
                playback_renderer=getattr(self, "slot_4d_playback_renderer", None),
            )
            self._slot_object_memory_latest_metrics[int(slot_id)] = dict(result)
            if hasattr(self, "log_slot_object_memory_step4"):
                self.log_slot_object_memory_step4(**result)
            self._maybe_log_slot_object_memory_success()
        except Exception as e:
            if not hasattr(self, "_slot_object_memory_warned"):
                print(f"[slot_memory] save/recall failed: {e}")
                self._slot_object_memory_warned = True

    def _maybe_log_slot_object_memory_success(self) -> None:
        try:
            if bool(getattr(self, "_slot_object_memory_success_logged", False)):
                return
            m = getattr(self, "_slot_object_memory_latest_metrics", {}) or {}
            m0 = m.get(0)
            m1 = m.get(1)
            if not (isinstance(m0, dict) and isinstance(m1, dict)):
                return
            if not (bool(m0.get("written", False)) and bool(m1.get("written", False))):
                return
            if hasattr(self, "log_success_slot_object_memory_step4"):
                self.log_success_slot_object_memory_step4(
                    slot_0_target=str(m0.get("target_name", "tetrahedron")),
                    slot_0_raw_points=int(m0.get("raw_points", 0) or 0),
                    slot_0_deformed_points=int(m0.get("deformed_points", 0) or 0),
                    slot_0_path=str(m0.get("path", "")),
                    slot_0_recall_matched=bool((m0.get("best_match", {}) or {}).get("matched", False)),
                    slot_1_target=str(m1.get("target_name", "cube")),
                    slot_1_raw_points=int(m1.get("raw_points", 0) or 0),
                    slot_1_deformed_points=int(m1.get("deformed_points", 0) or 0),
                    slot_1_path=str(m1.get("path", "")),
                    slot_1_recall_matched=bool((m1.get("best_match", {}) or {}).get("matched", False)),
                )
                self._slot_object_memory_success_logged = True
        except Exception:
            pass


    def _attach_long_dynamic_debug_tensors(self, decoded: dict, ref_tensor: torch.Tensor) -> dict:
        try:
            dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
            if not torch.is_tensor(ref_tensor):
                return decoded
            def put(name: str, value, default: float = 0.0):
                try:
                    if value is None:
                        value = default
                    decoded[name] = torch.tensor([[float(value)]], device=ref_tensor.device, dtype=ref_tensor.dtype)
                except Exception:
                    pass
            put("long_dynamic_ready", 1.0 if bool(dbg.get("dynamic_ready", False)) else 0.0)
            put("long_dynamic_slot_update_allowed", 1.0 if bool(dbg.get("slot_update_allowed", False)) else 0.0)
            put("dynamic_score", dbg.get("dynamic_score", 0.0))
            put("scene_novelty", dbg.get("scene_novelty", 0.0))
            put("interaction", dbg.get("interaction", 0.0))
            ldm_stats = getattr(self, "latest_long_dynamic_memory_stats", {}) or {}
            ldm_loss = float(ldm_stats.get("loss", 0.0) or 0.0)
            ldm_recon = float(ldm_stats.get("recon", 0.0) or 0.0)
            put("long_dynamic_loss", ldm_loss)
            put("long_dynamic_recon", ldm_recon)
            put("long_dynamic_loss_x1e6", ldm_loss * 1.0e6)
            put("long_dynamic_recon_x1e5", ldm_recon * 1.0e5)
            put("long_dynamic_train_updates", ldm_stats.get("updates", 0.0))
            dyn_conf = float(dbg.get("long_dynamic_confidence", 0.0) or 0.0)
            dyn_ready = bool(dbg.get("dynamic_ready", False))
            dyn_write = bool(dbg.get("slot_update_allowed", False))
            put("long_dynamic_confidence", dyn_conf)
            put("slot_confidence_raw", decoded.get("confidence", 0.0))
            try:
                dyn_motion_ok = bool(float(dbg.get("long_dynamic_motion_ok", 0.0) or 0.0) > 0.5)
            except Exception:
                dyn_motion_ok = False
            dyn_effective = dyn_conf if (dyn_ready and dyn_write and dyn_motion_ok) else 0.0
            live_formed = bool(dyn_effective > 0.0)
            if live_formed:
                self._long_dynamic_formed_slot_latch = {
                    "dynamic_object_confidence_raw": dyn_conf,
                    "dynamic_object_confidence": dyn_effective,
                    "object_formed_confidence": dyn_effective,
                    "long_dynamic_confidence": dyn_conf,
                    "long_dynamic_ready_streak": float(dbg.get("long_dynamic_ready_streak", 0.0) or 0.0),
                    "long_dynamic_steps": float(dbg.get("long_dynamic_steps", 0.0) or 0.0),
                    "long_dynamic_active_steps": float(dbg.get("long_dynamic_active_steps", 0.0) or 0.0),
                    "long_dynamic_dz": float(dbg.get("long_dynamic_dz", 0.0) or 0.0),
                    "long_dynamic_depth_motion": float(dbg.get("long_dynamic_depth_motion", 0.0) or 0.0),
                    "long_dynamic_z_static_norm": float(dbg.get("long_dynamic_z_static_norm", 0.0) or 0.0),
                    "long_dynamic_z_dynamic_norm": float(dbg.get("long_dynamic_z_dynamic_norm", 0.0) or 0.0),
                }
            formed_latch = getattr(self, "_long_dynamic_formed_slot_latch", {}) or {}
            persisted = bool(formed_latch) and not live_formed
            if persisted:
                put("long_dynamic_confidence", formed_latch.get("long_dynamic_confidence", dyn_conf))
            put("dynamic_object_confidence_raw", formed_latch.get("dynamic_object_confidence_raw", dyn_conf) if persisted else dyn_conf)
            put("dynamic_object_confidence", formed_latch.get("dynamic_object_confidence", dyn_effective) if persisted else dyn_effective)
            put("object_formed_confidence", formed_latch.get("object_formed_confidence", dyn_effective) if persisted else dyn_effective)
            put("object_formed_ready", 1.0 if (live_formed or bool(formed_latch)) else 0.0)
            put("long_dynamic_slot_persisted", 1.0 if persisted else 0.0)
            put("long_dynamic_ready_streak", formed_latch.get("long_dynamic_ready_streak", 0.0) if persisted else dbg.get("long_dynamic_ready_streak", 0.0))
            put("long_dynamic_steps", formed_latch.get("long_dynamic_steps", 0.0) if persisted else dbg.get("long_dynamic_steps", 0.0))
            put("long_dynamic_active_steps", formed_latch.get("long_dynamic_active_steps", 0.0) if persisted else dbg.get("long_dynamic_active_steps", 0.0))
            put("long_dynamic_dz", formed_latch.get("long_dynamic_dz", 0.0) if persisted else dbg.get("long_dynamic_dz", 0.0))
            put("long_dynamic_depth_motion", formed_latch.get("long_dynamic_depth_motion", 0.0) if persisted else dbg.get("long_dynamic_depth_motion", 0.0))
            put("long_dynamic_depth_motion_threshold", dbg.get("long_dynamic_depth_motion_threshold", 0.0))
            put("long_dynamic_dz_threshold", dbg.get("long_dynamic_dz_threshold", 0.0))
            put("long_dynamic_motion_ok", dbg.get("long_dynamic_motion_ok", 0.0))
            put("long_dynamic_z_static_norm", formed_latch.get("long_dynamic_z_static_norm", 0.0) if persisted else dbg.get("long_dynamic_z_static_norm", 0.0))
            put("long_dynamic_z_dynamic_norm", formed_latch.get("long_dynamic_z_dynamic_norm", 0.0) if persisted else dbg.get("long_dynamic_z_dynamic_norm", 0.0))
            recon = getattr(self, "_slot_reconstruction_latest_metrics", {}) or {}
            put("slot_recon_point_count", float(recon.get("points_total", 0.0) or 0.0))
            put("slot_recon_frame_count", float(recon.get("frame_count", 0.0) or 0.0))
            put("slot_recon_points_added", float(recon.get("points_added", 0.0) or 0.0))
            g = getattr(self, "_slot_gaussian_latest_metrics", {}) or {}
            put("slot_gaussian_count", float(g.get("gaussian_count", 0.0) or 0.0))
            put("slot_gaussian_updates", float(g.get("updates", 0.0) or 0.0))
            put("slot_gaussian_rgb_loss", float(g.get("rgb_loss", 0.0) or 0.0))
            put("slot_gaussian_depth_loss", float(g.get("depth_loss", 0.0) or 0.0))
            put("slot_gaussian_total_loss", float(g.get("total_loss", 0.0) or 0.0))
            put("slot_gaussian_preview_fps", float(g.get("preview_fps", 0.0) or 0.0))
            put("slot_gaussian_backend_is_cuda", float(1.0 if str(g.get("backend", "")) == "cuda_3dgs" else 0.0))
            put("slot_gaussian_fallback_used", float(1.0 if bool(g.get("fallback_used", False)) else 0.0))
            t4d = getattr(self, "_slot_4d_latest_metrics", {}) or {}
            put("slot_4d_timeline_frames", float(t4d.get("frame_count", 0.0) or 0.0))
            put("slot_4d_temporal_span", float(t4d.get("temporal_span", 0.0) or 0.0))
            put("slot_4d_motion_norm", float(t4d.get("motion_norm", 0.0) or 0.0))
            put("slot_4d_gaussian_count", float(t4d.get("gaussian_count", 0.0) or 0.0))
            d4d = getattr(self, "_slot_4d_deformation_latest_metrics", {}) or {}
            put("slot_4d_deformation_updates", float(d4d.get("updates", 0.0) or 0.0))
            put("slot_4d_deformation_loss", float(d4d.get("loss", 0.0) or 0.0))
            put("slot_4d_deformation_pred_delta_norm", float(d4d.get("pred_delta_norm", 0.0) or 0.0))
            put("slot_4d_deformation_sample_count", float(d4d.get("sample_count", 0.0) or 0.0))
            p4d = getattr(self, "_slot_4d_playback_latest_metrics", {}) or {}
            put("slot_4d_playback_frames", float(p4d.get("frame_count", 0.0) or 0.0))
            put("slot_4d_playback_phase", float(p4d.get("playback_phase", 0.0) or 0.0))
            put("slot_4d_playback_pred_delta_norm", float(p4d.get("pred_delta_norm", 0.0) or 0.0))
            put("slot_4d_playback_render_valid", float(1.0 if bool(p4d.get("render_valid", False)) else 0.0))
            put("slot_4d_playback_deformation_used", float(1.0 if bool(p4d.get("deformation_used", False)) else 0.0))
            put("slot_4d_playback_preview_fps", float(p4d.get("preview_fps", 0.0) or 0.0))
            put("slot_4d_playback_backend_is_cuda", float(1.0 if str(p4d.get("backend", "")) == "cuda_3dgs" else 0.0))
            preview_shapes = {}
            def put_tensor_preview(name: str, value):
                try:
                    if value is None or not torch.is_tensor(value):
                        return False
                    t = value.detach()
                    if t.ndim == 2:
                        t = t.unsqueeze(-1)
                    if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
                        t = t.permute(2, 0, 1).unsqueeze(0)
                    elif t.ndim == 3 and t.shape[0] in (1, 3, 4):
                        t = t.unsqueeze(0)
                    elif t.ndim != 4:
                        return False
                    out = t.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
                    decoded[name] = out
                    preview_shapes[name] = tuple(int(x) for x in out.shape)
                    return True
                except Exception:
                    return False
            try:
                pslot = int(p4d.get("slot_id", -1))
                pviews = getattr(getattr(self, "slot_4d_playback_renderer", None), "last_preview", {}) or {}
                pview = pviews.get(pslot) if pslot >= 0 else None
                if isinstance(pview, dict):
                    put_tensor_preview("slot_4d_playback_rgb", pview.get("rgb"))
                    put_tensor_preview("slot_4d_playback_depth", pview.get("depth"))
                    put_tensor_preview("slot_4d_playback_alpha", pview.get("alpha"))
                if hasattr(self, "_tetra_diag_write"):
                    self._tetra_diag_write(
                        "slot_4d_preview_tensor_bridge",
                        slot_id=pslot,
                        target_name=str(p4d.get("target_name", "unknown")),
                        has_rgb=decoded.get("slot_4d_playback_rgb") is not None,
                        has_depth=decoded.get("slot_4d_playback_depth") is not None,
                        has_alpha=decoded.get("slot_4d_playback_alpha") is not None,
                        rgb_shape=preview_shapes.get("slot_4d_playback_rgb", ()),
                        depth_shape=preview_shapes.get("slot_4d_playback_depth", ()),
                        alpha_shape=preview_shapes.get("slot_4d_playback_alpha", ()),
                        render_valid=bool(p4d.get("render_valid", False)),
                        deformation_used=bool(p4d.get("deformation_used", False)),
                    )
            except Exception:
                pass
            g = getattr(self, "_slot_gaussian_latest_metrics", {}) or {}
            put("slot_gaussian_count", float(g.get("gaussian_count", 0.0) or 0.0))
            put("slot_gaussian_updates", float(g.get("updates", 0.0) or 0.0))
            put("slot_gaussian_rgb_loss", float(g.get("rgb_loss", 0.0) or 0.0))
            put("slot_gaussian_depth_loss", float(g.get("depth_loss", 0.0) or 0.0))
            put("slot_gaussian_total_loss", float(g.get("total_loss", 0.0) or 0.0))
        except Exception:
            pass
        return decoded

    # -------------------------------------------------------------------------
    # Level-5 neural event decoder
    # -------------------------------------------------------------------------
    def _ensure_neural_event_decoder(self) -> None:
        if hasattr(self, "neural_event_decoder") and self.neural_event_decoder is not None:
            return
        cfg_event = getattr(self.cfg, "event_memory", None)
        self.neural_event_decoder = NeuralEventDecoder(NeuralEventDecoderConfig(
            enabled=bool(getattr(cfg_event, "neural_event_decoder_enabled", True)),
            latent_dim=int(getattr(self.cfg.object_image, "latent_dim", 128)),
            event_code_dim=8,
            role_dim=16,
            hidden_dim=int(getattr(cfg_event, "neural_event_decoder_hidden_dim", 256)),
            loss_weight=float(getattr(cfg_event, "neural_event_decoder_loss_weight", 0.05)),
            max_delta=float(getattr(cfg_event, "neural_event_decoder_max_delta", 0.35)),
        )).to(self.device)

        # Add this module to optimizer if runtime supports dynamic optimizer groups.
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                existing = set()
                for group in self.optimizer.param_groups:
                    for p in group.get("params", []):
                        existing.add(id(p))
                params = [p for p in self.neural_event_decoder.parameters() if id(p) not in existing]
                if params:
                    self.optimizer.add_param_group({"params": params})
                    print(f"[event_decoder] added neural_event_decoder params to optimizer: {sum(p.numel() for p in params):,}")
        except Exception as e:
            if not hasattr(self, "_neural_event_decoder_opt_warned"):
                print(f"[event_decoder] optimizer add skipped: {e}")
                self._neural_event_decoder_opt_warned = True

    def compute_neural_event_decoder_loss(self) -> dict:
        """
        Train Level-5 decoder from latest event memory.

        Target is deterministic:
            z_before + event code + sentence roles -> z_after

        This is a self-supervised latent dynamics loss.
        """
        try:
            cfg_event = getattr(self.cfg, "event_memory", None)
            if not bool(getattr(cfg_event, "neural_event_decoder_enabled", True)):
                return {}
            if not hasattr(self, "event_latent_memory") or self.event_latent_memory is None:
                return {}
            events = list(getattr(self.event_latent_memory, "events", []) or [])
            if not events:
                return {}
            self._ensure_neural_event_decoder()
            event = events[-1]
            losses = self.neural_event_decoder.loss_from_event(event, device=self.device, dtype=torch.float32)
            return losses if isinstance(losses, dict) else {}
        except Exception as e:
            if not hasattr(self, "_neural_event_decoder_loss_warned"):
                print(f"[event_decoder] loss failed: {e}")
                self._neural_event_decoder_loss_warned = True
            return {}


    def _decode_event_scenario_if_needed(self, obj: dict, dream_mode: bool) -> dict:
        """
        Level 4: sentence/episode memory -> scenario latent -> 2D/3D decode.

        When full sleep is active, this can override the static dream slot z
        with a replayed z trajectory from EventSentenceMemory.
        """
        try:
            cfg_event = getattr(self.cfg, "event_memory", None)
            if not bool(getattr(cfg_event, "use_scenario_decoder", True)):
                return obj
            if dream_mode and not bool(getattr(cfg_event, "scenario_decode_in_sleep", True)):
                return obj
            if not hasattr(self, "event_latent_memory") or self.event_latent_memory is None:
                return obj
            scenario_decoder = getattr(self.event_latent_memory, "scenario_decoder", None)
            sentence_memory = getattr(self.event_latent_memory, "sentence_memory", None)
            if scenario_decoder is None or sentence_memory is None:
                return obj

            z_ref = obj.get("z_obj")
            if not torch.is_tensor(z_ref):
                return obj

            sc = scenario_decoder.decode_next(
                sentence_memory,
                device=z_ref.device,
                dtype=z_ref.dtype,
            )
            if not sc or not sc.get("scenario_active"):
                return obj

            z_scenario = sc.get("scenario_z")
            if not torch.is_tensor(z_scenario):
                return obj

            # Optional Level-5 neural decoder:
            # use event sentence/code to predict the next z instead of directly
            # replaying stored z. Falls back to deterministic scenario_z.
            try:
                cfg_event = getattr(self.cfg, "event_memory", None)
                if bool(getattr(cfg_event, "neural_event_decoder_enabled", True)):
                    self._ensure_neural_event_decoder()
                    ev_refs = sc.get("event_refs", None)
                    # decode_next returns only current z, but build_sequence keeps
                    # event refs; use cursor if available, else latest event.
                    source_event = None
                    if isinstance(ev_refs, list) and len(ev_refs) > 0:
                        ci = int(sc.get("scenario_cursor", 0))
                        ci = max(0, min(ci, len(ev_refs) - 1))
                        source_event = ev_refs[ci]
                    if source_event is None and hasattr(self, "event_latent_memory"):
                        source_event = getattr(self.event_latent_memory, "last_event", None)
                    if isinstance(source_event, dict):
                        pred = self.neural_event_decoder.decode_event(source_event, device=z_scenario.device, dtype=z_scenario.dtype)
                        if pred and torch.is_tensor(pred.get("neural_pred_z_after")):
                            z_scenario = pred["neural_pred_z_after"]
                            obj["neural_event_decoder_active"] = torch.tensor([[1.0]], device=z_scenario.device, dtype=z_scenario.dtype)
            except Exception:
                pass

            extra = {k: v for k, v in obj.items() if k != "z_obj"}
            decoded = self.inner_object_system.decode_z(z_scenario, extra)

            # Keep memory/slot fields from current object, but replace decoded image
            # with scenario replay. This is still a deterministic decoder stage.
            for k, v in obj.items():
                if k not in decoded:
                    decoded[k] = v

            decoded["z_obj"] = z_scenario
            decoded["scenario_active"] = torch.tensor([[1.0]], device=z_scenario.device, dtype=z_scenario.dtype)
            decoded["scenario_episode_id"] = torch.tensor([[float(sc.get("scenario_episode_id", 0))]], device=z_scenario.device, dtype=z_scenario.dtype)
            decoded["scenario_cursor"] = torch.tensor([[float(sc.get("scenario_cursor", 0))]], device=z_scenario.device, dtype=z_scenario.dtype)
            decoded["scenario_sequence_len"] = torch.tensor([[float(sc.get("scenario_sequence_len", 0))]], device=z_scenario.device, dtype=z_scenario.dtype)
            decoded["scenario_sentence"] = sc.get("scenario_sentence", "")
            decoded["scenario_summary"] = sc.get("scenario_summary", "")
            decoded["scenario_roles"] = sc.get("scenario_roles", {})
            return decoded
        except Exception as e:
            if not hasattr(self, "_event_scenario_decoder_warned"):
                print(f"[event_scenario_decoder] decode failed: {e}")
                self._event_scenario_decoder_warned = True
            return obj


    def compute_inner_object_image(self, obs: dict, out: dict):
        if not self.cfg.object_image.enabled:
            return None

        # During train_once this must allow gradients, otherwise ObjectImaginationHead
        # receives no learning signal from object_decoder_loss.
        grad_enabled = bool(torch.is_grad_enabled() and self.inner_object_system.training)

        with torch.set_grad_enabled(grad_enabled):
            vision = self.build_inner_object_vision_proposals(obs)
            tactile = obs.get("tactile", torch.zeros(1, self.cfg.tactile_dim, device=self.device))

            if not self._input_sensor_enabled("contact"):
                tactile = torch.zeros_like(tactile.float())

            tactile = pad_or_trim(tactile.float(), self.cfg.object_image.tactile_dim)

            body = out.get("embodied_targets", self.prev_embodied_action).float()
            hand = out.get("hand_ctrl", self.prev_hand_motor).float()
            leg = out.get("leg_ctrl", self.prev_leg_motor).float() if hasattr(self, "prev_leg_motor") else torch.zeros(1, self.cfg.leg_control.leg_motor_dim, device=self.device)

            body = pad_or_trim(body, self.cfg.embodied_dim)
            hand = pad_or_trim(hand, self.cfg.hand_motor_dim)
            leg = pad_or_trim(leg, self.cfg.leg_control.leg_motor_dim)

            # IMU/body sensor gate: prevent body/proprio context from bypassing
            # imu_sensor_enabled inside the object-fusion path.
            if not self._input_sensor_enabled("imu"):
                body = torch.zeros_like(body)
                hand = torch.zeros_like(hand)
                leg = torch.zeros_like(leg)

            prev_state = {
                "z_obj": self.inner_object_state.get("z_obj", torch.zeros(1, self.cfg.object_image.latent_dim, device=self.device)).detach(),
                "confidence": self.inner_object_state.get("confidence", torch.zeros(1, 1, device=self.device)).detach(),
                "z_obj_slots": self.inner_object_state.get("z_obj_slots", None),
                "confidence_slots": self.inner_object_state.get("confidence_slots", None),
                "memory_stability_slots": self.inner_object_state.get("memory_stability_slots", None),
                "memory_stability": self.inner_object_state.get("memory_stability", None),
                "dream_activation_slots": self.inner_object_state.get("dream_activation_slots", None),
                "dream_activation": self.inner_object_state.get("dream_activation", None),
                "slot_update_strength": self.inner_object_state.get("slot_update_strength", None),
                "slot_age": self.inner_object_state.get("slot_age", None),
                "active_slot_index": self.inner_object_state.get("active_slot_index", None),
                "dream_tick": self.inner_object_state.get("dream_tick", None),
            }
            prev_state = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in prev_state.items() if v is not None}

            dream_mode = self.is_full_sleep_mode() if hasattr(self, "is_full_sleep_mode") else (
                not bool(self.video_sensor_enabled)
                and not bool(self.contact_sensor_enabled)
                and not bool(self.imu_sensor_enabled)
            )

            if dream_mode:
                requested_slot = getattr(getattr(self, "inner_object_viz", None), "requested_dream_slot_index", None)
                if requested_slot is not None:
                    try:
                        n_slots = int(getattr(self.cfg.object_image, "num_slots", 10))
                        requested_slot = max(0, min(int(requested_slot), n_slots - 1))
                        prev_state["active_slot_index"] = torch.tensor([[requested_slot]], device=self.device, dtype=torch.long)
                        self.inner_object_state["active_slot_index"] = prev_state["active_slot_index"].detach()
                    except Exception as e:
                        print(f"[inner_object][dream_slot_key] failed: {e}")
            else:
                # Leaving full sleep immediately cancels dream-only UI state.
                try:
                    if getattr(self, "inner_object_viz", None) is not None:
                        self.inner_object_viz.requested_dream_slot_index = None
                except Exception:
                    pass

            obj = self._run_progressive_inner_object_system(
                prev_state,
                vision,
                tactile,
                body,
                hand,
                leg,
                dream_mode=dream_mode,
            )

            # Explicit code ladder debug: z_static -> z_dynamic -> scenario_z.
            obj = self.annotate_static_dynamic_codes(obj, vision=vision)

            if not dream_mode:
                try:
                    ref = obj.get("confidence", obj.get("z_obj"))
                    if torch.is_tensor(ref):
                        b = int(ref.shape[0]) if ref.ndim > 0 else 1
                        obj["sleep_dream_mode"] = torch.zeros(b, 1, device=ref.device, dtype=ref.dtype)
                        obj["dream_empty_mode"] = torch.zeros(b, 1, device=ref.device, dtype=ref.dtype)
                        obj["dream_latent_delta"] = torch.zeros(b, 1, device=ref.device, dtype=ref.dtype)
                except Exception:
                    pass

            # Event latent memory: convert slot transitions into compact
            # "code sentences" before detaching runtime memory.
            obj = self._update_event_latent_memory(prev_state, obj, obs, out, dream_mode)

            # Semantic identity layer: SLOT_N address -> OBJ_NNN dynamic passport.
            obj = self.update_dynamic_object_passport(obj, obs=obs, out=out, dream_mode=dream_mode)

            # Reproduce passport into first-order inner world, and in sleep decode to second-order view.
            obj = self.reproduce_dynamic_object_from_passport(obj, dream_mode=dream_mode)

            obj = self.annotate_static_dynamic_codes(obj, vision=vision)  # after passport

            # Heatmap viewer for z_static, z_dynamic and scenario_z.
            obj = self.update_static_dynamic_code_visualizer(obj)

            # Read the last key observed by the dedicated OpenCV GUI thread.
            try:
                self.last_highgui_key = int(pump_highgui_events())
            except Exception as e:
                if not hasattr(self, "_highgui_event_pump_warned"):
                    print(f"[highgui_event_pump] failed: {e}")
                    self._highgui_event_pump_warned = True

            # Debug: live z_obj vs passport replay_z, first-order and second-order comparison.
            obj = self.update_passport_debug_visualizer(obj)

            # Level 4 decoder: in sleep/dream this can replay an event episode
            # through ObjectImaginationHead2D/Object3DHead as a scenario.
            obj = self._decode_event_scenario_if_needed(obj, dream_mode)

            # First-order coded-world thinking over scenario_z candidates.
            obj = self.update_inner_scenario_mind(obj)

            # Decode selected inner scenario into an action intention.
            obj = self.update_inner_action_decoder(obj, out=out)

            # Evaluate whether previous inner scenario matched current latent outcome.
            obj = self.update_inner_outcome_evaluator(obj)

            # Convert outcome feedback into trust for possible safe policy blending.
            obj = self.update_inner_trust_gate(obj, out=out)

            # Trace final real-action path after trust-gated blending.
            obj = self.update_inner_real_action_trace(obj, out=out)

            # Runtime memory should not keep graph history.
            self.inner_object_state = {
                "z_obj": obj["z_obj"].detach(),
                "confidence": obj["confidence"].detach(),
                "z_obj_slots": obj.get("z_obj_slots", None).detach() if torch.is_tensor(obj.get("z_obj_slots", None)) else None,
                "confidence_slots": obj.get("confidence_slots", None).detach() if torch.is_tensor(obj.get("confidence_slots", None)) else None,
                "memory_stability_slots": obj.get("memory_stability_slots", None).detach() if torch.is_tensor(obj.get("memory_stability_slots", None)) else None,
                "memory_stability": obj.get("memory_stability", None).detach() if torch.is_tensor(obj.get("memory_stability", None)) else None,
                "dream_activation_slots": obj.get("dream_activation_slots", None).detach() if torch.is_tensor(obj.get("dream_activation_slots", None)) else None,
                "dream_activation": obj.get("dream_activation", None).detach() if torch.is_tensor(obj.get("dream_activation", None)) else None,
                "slot_update_strength": obj.get("slot_update_strength", None).detach() if torch.is_tensor(obj.get("slot_update_strength", None)) else None,
                "slot_age": obj.get("slot_age", None).detach() if torch.is_tensor(obj.get("slot_age", None)) else None,
                "active_slot_index": obj.get("active_slot_index", None).detach() if torch.is_tensor(obj.get("active_slot_index", None)) else None,
                "dream_tick": obj.get("dream_tick", None).detach() if torch.is_tensor(obj.get("dream_tick", None)) else None,
            }
            self.inner_object_state = {k: v for k, v in self.inner_object_state.items() if v is not None}
            out["inner_object"] = obj
            if hasattr(self, "log_tetra_object_diagnostics"):
                self.log_tetra_object_diagnostics(obj)
            return obj


    def update_inner_object_window(self, obs: dict, out: dict):
        if not self.cfg.object_image.enabled:
            try:
                self.inner_object_viz.close()
            except Exception:
                pass
            try:
                if hasattr(self, "event_code_viz") and self.event_code_viz is not None:
                    self.event_code_viz.close()
            except Exception:
                pass
            return

        object_window_requested = bool(getattr(self, "show_inner_object_window", False))
        event_code_requested = self._event_code_visualizer_enabled()
        if not object_window_requested:
            try:
                self.inner_object_viz.close()
            except Exception:
                pass

        if not object_window_requested and not event_code_requested:
            return

        object_due = object_window_requested and self.global_step % max(1, self.cfg.object_image.show_every_steps) == 0
        event_due = event_code_requested and self.global_step % max(1, int(getattr(self.cfg.event_code_visualizer, "show_every_steps", 1))) == 0
        if not object_due and not event_due:
            return

        obj = out.get("inner_object")
        if obj is None:
            obj = self.compute_inner_object_image(obs, out)
        if obj is None:
            return

        if object_due:
            tactile_values = []
            try:
                n_tactile = int(getattr(self.cfg.object_image, "tactile_dim", self.cfg.tactile_dim))
                if not self._input_sensor_enabled("contact"):
                    tactile_values = [0.0] * n_tactile
                else:
                    arr = obs.get("tactile").detach().cpu().numpy().reshape(-1).astype(np.float32)
                    padded = np.zeros(n_tactile, dtype=np.float32)
                    n = min(n_tactile, int(arr.size))
                    if n > 0:
                        padded[:n] = arr[:n]
                    tactile_values = padded.tolist()
            except Exception:
                pass

            obs_for_viz = self._safe_obs_for_inner_object_viz(obs)
            self.inner_object_viz.draw(obj, obs=obs_for_viz, tactile_values=tactile_values)
            if hasattr(self, "log_tetra_inner_object_window_state"):
                self.log_tetra_inner_object_window_state(obj, visible=True, reason="object_image_window_drawn")

        # Second-level semantic/event-code visualizer.
        if event_due:
            self.update_event_code_visualizer_window(obj)


    def maybe_capture_inner_object_snapshot(self, obj: dict):
        if obj is None:
            return
        try:
            conf = float(obj["confidence"][0, 0].detach().cpu().item())
            if conf < float(self.cfg.object_image_open3d.snapshot_conf_threshold):
                return
            if (self.global_step - self.last_inner_object_snapshot_step) < int(self.cfg.object_image_open3d.min_steps_between_snapshots):
                return

            snap = {}
            for k in [
                "point_cloud",
                "point_conf",
                "voxel_occ",
                "color_rgb",
                "confidence",
                "debug_imit_fallback_shape",
                "debug_imit_source",
                "debug_imit_shape_kind",
            ]:
                if k in obj:
                    v = obj[k]
                    if hasattr(v, "detach"):
                        snap[k] = v.detach().cpu().clone()
                    else:
                        snap[k] = v
            self.inner_object_slot_snapshots.insert(0, snap)
            self.inner_object_slot_snapshots = self.inner_object_slot_snapshots[: int(self.cfg.object_image_open3d.max_slots)]
            self.last_inner_object_snapshot_step = int(self.global_step)
        except Exception as e:
            if not hasattr(self, "_inner_object_snapshot_warned"):
                print(f"[inner_object_open3d] snapshot capture failed: {e}")
                self._inner_object_snapshot_warned = True


    def export_inner_object_model(self, fmt: str = "ply"):
        try:
            path = self.inner_object_open3d_viz.export_current(fmt)
            if path:
                print(f"[inner_object_open3d] exported current model -> {path}")
            else:
                print("[inner_object_open3d] nothing to export yet")
            return path
        except Exception as e:
            print(f"[inner_object_open3d] export failed: {e}")
            return None


    def update_inner_object_open3d_window(self, out: dict):
        if not self.cfg.object_image_open3d.enabled or not self.show_inner_object_open3d_window:
            try:
                self.inner_object_open3d_viz.close()
            except Exception:
                pass
            return

        if self.global_step % max(1, self.cfg.object_image_open3d.update_every_steps) != 0:
            return

        obj = out.get("inner_object")
        if obj is None:
            return

        self.maybe_capture_inner_object_snapshot(obj)
        try:
            self.inner_object_open3d_viz.update(obj, slot_snapshots=self.inner_object_slot_snapshots)
        except Exception as e:
            if not hasattr(self, "_inner_object_open3d_warned"):
                print(f"[inner_object_open3d] update failed: {e}")
                self._inner_object_open3d_warned = True


    def make_object_decoder_targets(self, obs: dict, inner: dict) -> dict:
        """
        Build self-supervised targets for ObjectImaginationHead.

        Current minimal target:
            RGB/depth = current left camera frame, downsampled to decoder size.
            mask      = pseudo object mask from depth edges + RGB contrast, optionally
                        boosted by geometry visibility when pose/object_state exists.

        This is not perfect segmentation yet, but it gives the decoder a real
        reconstruction target instead of only weak regularization.
        """
        pred_rgb = inner.get("rgb")
        pred_depth = inner.get("depth")
        if pred_rgb is None or pred_depth is None:
            return {}

        size = pred_rgb.shape[-2:]
        left = obs.get("left")
        depth = obs.get("depth")
        if left is None:
            return {}

        target_rgb = left.float()
        if target_rgb.ndim == 3:
            target_rgb = target_rgb.unsqueeze(0)
        target_rgb = target_rgb[:, :3]
        target_rgb = F.interpolate(target_rgb, size=size, mode="bilinear", align_corners=False)
        target_rgb = target_rgb.clamp(0.0, 1.0)

        if depth is None:
            target_depth = target_rgb.mean(dim=1, keepdim=True)
        else:
            target_depth = depth.float()
            if target_depth.ndim == 3:
                target_depth = target_depth.unsqueeze(0)
            if target_depth.shape[1] != 1:
                target_depth = target_depth[:, :1]
            target_depth = F.interpolate(target_depth, size=size, mode="bilinear", align_corners=False)
            d_min = target_depth.amin(dim=(-2, -1), keepdim=True)
            d_max = target_depth.amax(dim=(-2, -1), keepdim=True)
            target_depth = (target_depth - d_min) / (d_max - d_min + 1e-6)
            target_depth = target_depth.clamp(0.0, 1.0)

        # Pseudo mask: object-like areas are regions with depth change and visual contrast.
        gray = target_rgb.mean(dim=1, keepdim=True)
        gray_centered = torch.abs(gray - gray.mean(dim=(-2, -1), keepdim=True))
        gray_score = gray_centered / (gray_centered.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        dx = torch.abs(target_depth[:, :, :, 1:] - target_depth[:, :, :, :-1])
        dy = torch.abs(target_depth[:, :, 1:, :] - target_depth[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = dx + dy
        edge_score = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        mask_soft = torch.clamp(0.55 * edge_score + 0.45 * gray_score, 0.0, 1.0)

        # If object geometry says the object is in front of camera, avoid an all-zero pseudo mask.
        geometry_visible = self.estimate_object_visible_torch(obs)
        if geometry_visible is not None:
            mask_soft = torch.where(
                geometry_visible.view(-1, 1, 1, 1) > 0.5,
                torch.clamp(mask_soft + 0.15, 0.0, 1.0),
                mask_soft * 0.35,
            )

        target_mask = (mask_soft > 0.28).float()
        # Keep soft edges; BCE receives soft-ish target.
        target_mask = torch.clamp(0.70 * target_mask + 0.30 * mask_soft, 0.0, 1.0)

        target_objectness = torch.clamp(target_mask.mean(dim=(-2, -1), keepdim=True) * 4.0, 0.0, 1.0)
        if geometry_visible is not None:
            target_objectness = torch.maximum(target_objectness, geometry_visible.view(-1, 1, 1, 1) * 0.75)

        return {
            "target_rgb": target_rgb.detach(),
            "target_depth": target_depth.detach(),
            "target_mask": target_mask.detach(),
            "target_objectness": target_objectness.detach().view(-1, 1),
        }


    def estimate_object_visible_torch(self, obs: dict):
        """
        Geometry visibility teacher: 1 when any object_state point is roughly in
        front of the camera pose. Returns tensor [B] or None.
        """
        try:
            # Geometry visibility is a teacher derived from pose/object_state.
            # It must not bypass sensor gates:
            #   video OFF -> no visual object supervision
            #   imu OFF   -> pose/body orientation is not a sensed input
            if not self._input_sensor_enabled("video") or not self._input_sensor_enabled("imu"):
                return None

            pose = obs.get("pose")
            object_state = obs.get("object_state")
            if pose is None or object_state is None:
                return None
            pose = pose.float()
            object_state = object_state.float()
            if pose.ndim == 1:
                pose = pose.unsqueeze(0)
            if object_state.ndim == 1:
                object_state = object_state.unsqueeze(0)

            cam_pos = pose[:, :3]
            pts = object_state.reshape(object_state.shape[0], -1, 3)
            # Lightweight approximation: use +X world-facing component from camera pose.
            # This is intentionally a teacher heuristic, not the model state.
            q = pose[:, 3:7]
            # quat w,x,y,z -> forward local +X
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            fx = 1 - 2 * (y * y + z * z)
            fy = 2 * (x * y + w * z)
            fz = 2 * (x * z - w * y)
            forward = torch.stack([fx, fy, fz], dim=-1)
            forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-6)

            vec = pts - cam_pos[:, None, :]
            dist = vec.norm(dim=-1).clamp_min(1e-6)
            direction = vec / dist[..., None]
            cos = (direction * forward[:, None, :]).sum(dim=-1)
            best_cos = cos.max(dim=-1).values
            best_dist = dist.gather(1, cos.argmax(dim=-1, keepdim=True)).squeeze(1)

            visible = ((best_cos > 0.58) & (best_dist < 12.0)).float()
            return visible
        except Exception:
            return None


    def compute_object_decoder_loss(self, inner: dict, obs: dict):
        if not isinstance(inner, dict):
            return None, {}
        if not bool(getattr(self.cfg.object_image, "decoder_loss_enabled", True)):
            return None, {}
        # Never train decoder against black/zeroed camera frames.
        # Video OFF may be blind mode or full dream mode, but either way
        # reconstruction targets are not valid visual supervision.
        if not bool(getattr(self, "video_sensor_enabled", True)) and bool(getattr(self.cfg.object_image, "sleep_skip_decoder_loss", True)):
            return None, {"object_decoder_skipped": 1.0, "reason": "video_sensor_disabled"}

        targets = self.make_object_decoder_targets(obs, inner)
        if not targets:
            return None, {}

        pred_rgb = inner.get("rgb")
        pred_depth = inner.get("depth")
        pred_mask = inner.get("mask")
        pred_conf = inner.get("confidence")
        if pred_rgb is None or pred_depth is None or pred_mask is None:
            return None, {}

        target_rgb = targets["target_rgb"].to(pred_rgb.device)
        target_depth = targets["target_depth"].to(pred_depth.device)
        target_mask = targets["target_mask"].to(pred_mask.device)
        target_objectness = targets["target_objectness"].to(pred_rgb.device)

        rgb_loss = F.smooth_l1_loss(pred_rgb, target_rgb)
        depth_loss = F.smooth_l1_loss(pred_depth, target_depth)
        mask_loss = F.binary_cross_entropy(pred_mask.clamp(1e-4, 1.0 - 1e-4), target_mask)

        conf_loss = torch.tensor(0.0, device=pred_rgb.device)
        if pred_conf is not None:
            conf_loss = F.mse_loss(pred_conf.float(), target_objectness.float())

        total = (
            self.cfg.object_image.decoder_rgb_weight * rgb_loss
            + self.cfg.object_image.decoder_depth_weight * depth_loss
            + self.cfg.object_image.decoder_mask_weight * mask_loss
            + self.cfg.object_image.decoder_conf_weight * conf_loss
        )
        total = self.cfg.object_image.decoder_loss_weight * total

        stats = {
            "object_decoder_loss": float(total.detach().cpu().item()),
            "object_rgb_loss": float(rgb_loss.detach().cpu().item()),
            "object_depth_loss": float(depth_loss.detach().cpu().item()),
            "object_mask_loss": float(mask_loss.detach().cpu().item()),
            "object_conf_loss": float(conf_loss.detach().cpu().item()),
            "object_target_mask_mean": float(target_mask.detach().mean().cpu().item()),
            "object_target_objectness": float(target_objectness.detach().mean().cpu().item()),
        }
        return total, stats
