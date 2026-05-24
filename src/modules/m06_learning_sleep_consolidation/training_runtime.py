from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
import time

class TrainingRuntimeMixin:
    def _is_full_sleep_mode_for_training(self) -> bool:
        """
        Full sleep = all external sensors are off.
        In this state we may run dream visualization, but we must not train:
        no visual/contact/imu supervision is valid and replay would contain
        synthetic/zeroed observations.
        """
        if hasattr(self, "is_full_sleep_mode"):
            return bool(self.is_full_sleep_mode())
        return (
            not bool(getattr(self, "video_sensor_enabled", True))
            and not bool(getattr(self, "contact_sensor_enabled", True))
            and not bool(getattr(self, "imu_sensor_enabled", True))
        )

    def compute_step_loss(self, out, obs):
        # keep all parent losses, add tiny stabilization terms for object imagery
        loss = super().compute_step_loss(out, obs)

        imagery = out.get("object_imagery")
        if imagery is not None:
            rgb_mean = imagery["rgb"].mean()
            alpha_mean = imagery["alpha"].mean()
            depth_mean = imagery["depth"].mean()
            conf_mean = imagery["object_confidence"].mean()

            # light regularization only, not a hard supervision target
            loss = loss + 0.001 * rgb_mean + 0.001 * depth_mean + 0.0005 * alpha_mean - 0.0005 * conf_mean

        obj3d = out.get("inner_object")
        if obj3d is not None:
            loss = loss + 0.0002 * obj3d["point_cloud"].abs().mean() + 0.0002 * obj3d["voxel_occ"].mean()
        inner = out.get("inner_object")
        if isinstance(inner, dict) and getattr(self.cfg.object_image, "decoder_loss_enabled", True):
            obj_loss, obj_stats = self.compute_object_decoder_loss(inner, obs)
            if obj_loss is not None:
                loss = loss + obj_loss
                try:
                    self.latest_object_decoder_stats = obj_stats
                except Exception:
                    pass

        
        # Vestibular balance shaping.
        # This teaches the action head to counter roll/pitch motion using the
        # stereo IMU signal. It is a differentiable loss on the model output,
        # with the vestibular observation as teacher signal.
        vest = obs.get("vestibular")
        if vest is not None and self.cfg.vestibular.enabled:
            v = vest.float()
            gyro_common = v[:, 12:15]
            gyro_diff = v[:, 15:18]
            accel_diff = v[:, 21:24]

            emb = out.get("embodied_targets")
            if emb is not None and emb.shape[-1] >= 6:
                # embodied order: vx, vy, vz, yaw_rate, pitch_rate, roll_rate
                # To damp pitch/roll, target angular commands oppose measured angular velocity.
                target_pitch = torch.clamp(-gyro_common[:, 1:2], -1.0, 1.0)
                target_roll = torch.clamp(-gyro_common[:, 0:1], -1.0, 1.0)
                pred_pitch = emb[:, 4:5]
                pred_roll = emb[:, 5:6]
                counter_loss = torch.mean((pred_pitch - target_pitch) ** 2 + (pred_roll - target_roll) ** 2)

                # Differential channel should be minimized for stable stereo balance.
                diff_loss = torch.mean(gyro_diff ** 2) + 0.25 * torch.mean(accel_diff ** 2)

                loss = loss + self.cfg.vestibular.balance_loss_weight * (counter_loss + 0.1 * diff_loss)



        # SelfCore consistency loss: train the self layer to predict next self-state
        # and keep agency/ownership/continuity non-degenerate.
        sc = out.get("self_core")
        if isinstance(sc, dict) and self.cfg.self_core.enabled:
            pred_err = sc.get("self_prediction_error")
            agency = sc.get("agency_score")
            ownership = sc.get("body_ownership_score")
            continuity = sc.get("self_continuity_score")
            if pred_err is not None:
                loss = loss + self.cfg.self_core.loss_weight * pred_err.mean()
            if agency is not None and ownership is not None and continuity is not None:
                # Soft anti-collapse: do not force maximum selfhood, but avoid all-zero scores.
                loss = loss + 0.002 * ((0.5 - agency).clamp_min(0).mean())
                loss = loss + 0.002 * ((0.5 - ownership).clamp_min(0).mean())
                loss = loss + 0.002 * ((0.5 - continuity).clamp_min(0).mean())



        # LongDynamicObjectMemory self-supervised stabilizing loss.
        # This does not teach class labels; it makes the long-memory module trainable
        # and encourages a stable dynamic representation that is not silent.
        try:
            ldm = getattr(self, "_last_long_dynamic_training_tensors", None)
            long_dynamic_train_enabled = bool(getattr(getattr(self, "module_training_gate", None), "flags", {}).get("long_dynamic_memory", True))
            if not long_dynamic_train_enabled:
                self.latest_long_dynamic_memory_stats = {
                    "enabled": False,
                    "loss": 0.0,
                    "loss_ema": float(getattr(self, "_long_dynamic_memory_loss_ema", 0.0) or 0.0),
                    "reward_proxy": 0.0,
                    "recon": 0.0,
                    "smooth": 0.0,
                    "note": "disabled_by_module_checkbox",
                }
                ldm = None

            if isinstance(ldm, dict) and bool(getattr(self.cfg.object_image, "long_dynamic_memory_loss_enabled", True)):
                z_static = ldm.get("z_static")
                z_dynamic = ldm.get("z_dynamic")
                confidence = ldm.get("confidence")
                if torch.is_tensor(z_static) and torch.is_tensor(z_dynamic):
                    # Weak reconstruction/consistency teacher:
                    # z_dynamic must carry enough information to explain the current static summary,
                    # while temporal gating still prevents static frames from writing slots.
                    recon = F.mse_loss(z_dynamic.float(), z_static.detach().float())

                    smooth = torch.tensor(0.0, device=z_dynamic.device)
                    prev = getattr(self, "_prev_long_dynamic_z_for_loss", None)
                    if torch.is_tensor(prev) and prev.shape == z_dynamic.shape:
                        smooth = F.mse_loss(z_dynamic.float(), prev.detach().float())
                    self._prev_long_dynamic_z_for_loss = z_dynamic.detach()

                    conf_reg = torch.tensor(0.0, device=z_dynamic.device)
                    if torch.is_tensor(confidence):
                        conf_reg = ((confidence.float() - 0.5) ** 2).mean()

                    w = float(getattr(self.cfg.object_image, "long_dynamic_memory_loss_weight", 0.01))
                    loss_ldm = w * (recon + 0.15 * smooth + 0.01 * conf_reg)
                    loss = loss + loss_ldm
                    self._long_dynamic_memory_train_updates = int(getattr(self, "_long_dynamic_memory_train_updates", 0)) + 1

                    self.latest_long_dynamic_memory_stats = {
                        "updates": int(getattr(self, "_long_dynamic_memory_train_updates", 0)),
                        "loss": float(loss_ldm.detach().cpu().item()),
                        "recon": float(recon.detach().cpu().item()),
                        "smooth": float(smooth.detach().cpu().item()),
                        "confidence_reg": float(conf_reg.detach().cpu().item()),
                        "z_static_norm": float(z_static.detach().float().norm(dim=-1).mean().cpu().item()),
                        "z_dynamic_norm": float(z_dynamic.detach().float().norm(dim=-1).mean().cpu().item()),
                    }
        except Exception as e:
            if not hasattr(self, "_long_dynamic_loss_warned"):
                print(f"[long_dynamic_memory] loss skipped: {e}")
                self._long_dynamic_loss_warned = True

        if out.get("symbolic_report") is not None:
            with torch.no_grad():
                target_ids = self.speech_teacher.target_ids(obs, out, device=out["workspace_out"].device)
            speech_loss = self.speech_teacher.report_loss(out["symbolic_report"], target_ids)
            loss = loss + self.cfg.inner_speech_loss_weight * speech_loss

        return loss


    def train_once(self):
        try:
            if self._is_full_sleep_mode_for_training():
                self.last_train_reason = "sleep_mode_training_disabled"
                self.last_train_error = ""
                if hasattr(self, "log_tetra_optimizer_step"):
                    self.log_tetra_optimizer_step(False, self.last_train_reason)
                return None
            if not bool(getattr(self.cfg.train, "enabled", False)):
                self.last_train_reason = "cfg.train.enabled=false"
                if hasattr(self, "log_tetra_optimizer_step"):
                    self.log_tetra_optimizer_step(False, self.last_train_reason)
                return None
            if not bool(getattr(self, "training_enabled", False)):
                self.last_train_reason = "runtime training_enabled=false"
                if hasattr(self, "log_tetra_optimizer_step"):
                    self.log_tetra_optimizer_step(False, self.last_train_reason)
                return None
            if len(self.replay) < self.cfg.replay.min_ready:
                self.last_train_reason = f"replay_not_ready {len(self.replay)}/{self.cfg.replay.min_ready}"
                return None

            batch = self.replay.sample(self.cfg.replay.batch_size, recent_bias=self.cfg.replay.recent_bias)
            if not batch:
                self.last_train_reason = "replay_sample_empty"
                return None

            trainable = []
            if hasattr(self, "module_training_gate"):
                trainable = [p for _n, p in self.module_training_gate.trainable_named_parameters()]
            else:
                trainable = [p for p in self.model.parameters() if getattr(p, "requires_grad", False)]

            if not trainable:
                self.last_train_reason = "no_trainable_parameters"
                return None

            self.model.train()
            if hasattr(self, "inner_object_system"):
                self.inner_object_system.train()
            if hasattr(self, "self_core"):
                self.self_core.train()
            if hasattr(self, "long_dynamic_object_memory") and self.long_dynamic_object_memory is not None:
                self.long_dynamic_object_memory.train()
            if hasattr(self, "leg_control_head") and self.leg_control_head is not None:
                self.leg_control_head.train()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for sample in batch:
                obs = {k: v.to(self.device) for k, v in sample.items()}
                obs = self.gate_observation_for_sleep(obs)
                local_state = self.model.initial_state(batch_size=1, device=self.device)

                out = self.model.step(
                    left=obs["left"],
                    right=obs["right"],
                    pose=obs["pose"],
                    body_state=obs["body_state"],
                    state=local_state,
                    tactile=obs["tactile"],
                    hand_motor=obs["hand_motor"],
                    embodied_action=obs["embodied_action"],
                    depth=obs["depth"],
                    object_state=obs["object_state"],
                    action_override=obs["action_id"],
                    write_memory=False,
                )

                # Recompute auxiliary trainable modules during training pass.
                try:
                    out["leg_ctrl"] = self.compute_leg_control(out)
                except Exception:
                    pass
                try:
                    out["inner_object"] = self.compute_inner_object_image(obs, out)
                except Exception:
                    pass
                try:
                    out["self_core"] = self.compute_self_core(obs, out)
                except Exception:
                    pass

                total_loss = total_loss + self.compute_step_loss(out, obs)

            total_loss = total_loss / len(batch)
            if not torch.isfinite(total_loss):
                self.last_train_reason = f"non_finite_loss {float(total_loss.detach().cpu().item())}"
                return None

            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable, self.cfg.train.gradient_clip)
            self.optimizer.step()

            # Checkpoint saving is intentionally NOT done here.
            # Single owner for last.pt is life_step() via maybe_save_checkpoint(owner="life").
            # This avoids train thread vs life thread write races.
            self.train_steps += 1
            self.last_train_loss = float(total_loss.detach().cpu().item())
            self.last_train_reason = "trained"
            if hasattr(self, "log_tetra_optimizer_step"):
                self.log_tetra_optimizer_step(True, self.last_train_reason)
            q = self.quality.update(self.last_train_loss)
            self.write_module_debug_status()
            return q
        except Exception as e:
            self.last_train_error = repr(e)
            self.last_train_reason = f"train_error: {e}"
            if getattr(self, "global_step", 0) % 50 == 0:
                print(f"[train_once] error: {e}")
            return None


    def train_loop(self):
        """
        Parallel online training loop.

        This thread must be separate from life_step:
            main thread: life_step() / MuJoCo / visualizers / pult
            train thread: train_once() consumes replay when ready

        Full sleep disables optimizer updates, but the life/dream loop continues.
        """
        while not bool(getattr(self, "shutdown", False)):
            try:
                if self._is_full_sleep_mode_for_training():
                    self.last_train_reason = "sleep_mode_training_disabled"
                    if hasattr(self, "log_tetra_optimizer_step"):
                        step = int(getattr(self, "global_step", 0))
                        if int(getattr(self, "_tetra_diag_last_optimizer_sleep_skip_step", -1)) != step:
                            self._tetra_diag_last_optimizer_sleep_skip_step = step
                            self.log_tetra_optimizer_step(False, self.last_train_reason)
                    time.sleep(float(getattr(self.cfg.train, "train_sleep_sec", 0.01)))
                    continue

                if bool(getattr(self.cfg.train, "enabled", False)) and bool(getattr(self, "training_enabled", False)):
                    self.train_once()
                else:
                    if not bool(getattr(self.cfg.train, "enabled", False)):
                        self.last_train_reason = "cfg.train.enabled=false"
                    elif not bool(getattr(self, "training_enabled", False)):
                        self.last_train_reason = "runtime training_enabled=false"

                time.sleep(float(getattr(self.cfg.train, "train_sleep_sec", 0.01)))
            except Exception as e:
                self.last_train_error = repr(e)
                self.last_train_reason = f"train_loop_error: {e}"
                print(f"[train_loop] error: {e}")
                time.sleep(0.25)

