from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import re
from typing import Any

import torch


class TetraDynamicSlotDiagnosticMixin:
    def _tetra_diag_config(self):
        return getattr(getattr(self, "cfg", None), "tetra_dynamic_slot_diagnostic", None)

    def _tetra_diag_enabled(self) -> bool:
        cfg = self._tetra_diag_config()
        return bool(getattr(cfg, "enabled", False))

    def _init_tetra_dynamic_slot_diagnostic_log(self, reset: bool = False) -> None:
        if not self._tetra_diag_enabled():
            return
        cfg = self._tetra_diag_config()
        runtime_cfg = getattr(getattr(self, "cfg", None), "runtime", None)
        out_dir = Path(str(getattr(runtime_cfg, "out_dir", ".")))
        file_name = str(getattr(cfg, "file_name", "tetra_dynamic_slot_diagnostic.log") or "tetra_dynamic_slot_diagnostic.log")
        path = out_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tetra_dynamic_slot_diag_path = path
        self._tetra_diag_live_step = int(getattr(self, "global_step", 0))
        self._tetra_diag_success_logged_slots = set()
        if reset:
            path.write_text("", encoding="utf-8")

    def _tetra_diag_path(self) -> Path:
        path = getattr(self, "_tetra_dynamic_slot_diag_path", None)
        if path is None:
            self._init_tetra_dynamic_slot_diagnostic_log(reset=False)
            path = getattr(self, "_tetra_dynamic_slot_diag_path")
        return path

    def _tetra_diag_effective_training(self) -> bool:
        full_sleep = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        train_cfg = getattr(getattr(self, "cfg", None), "train", None)
        return bool(
            getattr(self, "training_enabled", False)
            and bool(getattr(train_cfg, "enabled", False))
            and not full_sleep
        )

    def _tetra_diag_runtime_mode(self) -> str:
        if hasattr(self, "is_full_sleep_mode") and self.is_full_sleep_mode():
            return "dream"
        if self._tetra_diag_effective_training():
            return "train"
        return "awake"

    def _tetra_diag_scenario_status(self) -> dict:
        status = getattr(self, "_fly_to_cube_palpate_status", None)
        return status if isinstance(status, dict) else {}

    def _tetra_diag_target_name(self) -> str:
        name = str(self._tetra_diag_scenario_status().get("gaze_target", "") or "").strip().lower()
        if name in ("tetra", "tetrahedron"):
            return "tetrahedron"
        if name == "cube":
            return "cube"
        return name or "unknown"

    def _tetra_diag_target_slot(self, target_name: str | None = None) -> int:
        target = str(target_name or self._tetra_diag_target_name()).strip().lower()
        try:
            slot_by_target = getattr(self, "_dynamic_object_slot_by_target", {}) or {}
            if target in slot_by_target:
                return int(slot_by_target[target])
        except Exception:
            pass
        return 0

    def _tetra_diag_tensor_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().float().reshape(-1)[0].cpu().item())
            return float(value)
        except Exception:
            return float(default)

    def _tetra_diag_tensor_norm(self, value: Any, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                return float(value.detach().float().norm(dim=-1).mean().cpu().item())
        except Exception:
            pass
        return float(default)

    def _tetra_diag_slot_scalar(self, value: Any, slot_id: int, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                tensor = value.detach().float()
                if tensor.ndim >= 3:
                    if tensor.shape[1] <= int(slot_id):
                        return float(default)
                    return float(tensor[:, int(slot_id), :].mean().cpu().item())
                if tensor.ndim == 2:
                    if tensor.shape[1] > int(slot_id):
                        return float(tensor[:, int(slot_id)].mean().cpu().item())
                    if tensor.shape[0] > int(slot_id):
                        return float(tensor[int(slot_id), :].mean().cpu().item())
                if tensor.ndim == 1 and tensor.shape[0] > int(slot_id):
                    return float(tensor[int(slot_id)].cpu().item())
                if tensor.numel() == 1:
                    return float(tensor.reshape(-1)[0].cpu().item())
        except Exception:
            pass
        return float(default)

    def _tetra_diag_slot_norm(self, value: Any, slot_id: int, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                tensor = value.detach().float()
                if tensor.ndim >= 3 and tensor.shape[1] > int(slot_id):
                    return float(tensor[:, int(slot_id), :].norm(dim=-1).mean().cpu().item())
                if tensor.ndim == 2 and tensor.shape[0] > int(slot_id):
                    return float(tensor[int(slot_id), :].norm().cpu().item())
        except Exception:
            pass
        return float(default)

    def _tetra_diag_active_slot_id(self, obj: dict | None = None) -> int:
        if isinstance(obj, dict):
            idx = obj.get("active_slot_index")
            try:
                if torch.is_tensor(idx):
                    return int(idx.detach().reshape(-1)[0].cpu().item())
                if idx is not None:
                    return int(idx)
            except Exception:
                pass
        return 0

    def _tetra_diag_format_value(self, value: Any) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                return "nan"
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return "(" + ",".join(self._tetra_diag_format_value(v) for v in value) + ")"
        text = str(value)
        text = re.sub(r"\s+", "_", text.strip())
        return text if text else "none"

    def _tetra_diag_write(self, event: str, **fields: Any) -> None:
        if not self._tetra_diag_enabled():
            return
        try:
            status = self._tetra_diag_scenario_status()
            timestamp = datetime.now().isoformat(timespec="milliseconds")
            global_step = int(getattr(self, "global_step", 0))
            live_step = int(getattr(self, "_tetra_diag_live_step", global_step))
            phase = str(status.get("phase", fields.pop("phase", "")) or "none")
            mode = str(fields.pop("mode", self._tetra_diag_runtime_mode()) or "none")
            prefix = {
                "step": global_step,
                "live_step": live_step,
                "phase": phase,
                "mode": mode,
                "event": event,
            }
            merged = {**prefix, **fields}
            body = " ".join(f"{k}={self._tetra_diag_format_value(v)}" for k, v in merged.items())
            with self._tetra_diag_path().open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {body}\n")
        except Exception:
            pass

    def log_tetra_runner_started(self) -> None:
        if not self._tetra_diag_enabled():
            return
        cfg = self._tetra_diag_config()
        self._init_tetra_dynamic_slot_diagnostic_log(reset=bool(getattr(cfg, "reset_on_start", True)))
        self._tetra_diag_write(
            "runner_started",
            effective_training=self._tetra_diag_effective_training(),
            last_train_reason=str(getattr(self, "last_train_reason", "")),
        )

    def log_tetra_live_step_started(self) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_live_step = int(getattr(self, "global_step", 0))
        self._tetra_diag_write(
            "live_step_started",
            effective_training=self._tetra_diag_effective_training(),
            last_train_reason=str(getattr(self, "last_train_reason", "")),
        )

    def log_tetra_life_tick_diagnostics(self) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "live_step_tick",
            effective_training=self._tetra_diag_effective_training(),
            last_train_reason=str(getattr(self, "last_train_reason", "")),
        )
        self.log_tetra_sensor_state()
        self.log_tetra_scenario_state()
        self.log_tetra_post_success_monitor_tick()

    def log_tetra_sensor_state(self) -> None:
        if not self._tetra_diag_enabled():
            return
        full_sleep = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        self._tetra_diag_write(
            "sensor_state",
            video_enabled=bool(getattr(self, "video_sensor_enabled", True)),
            contact_enabled=bool(getattr(self, "contact_sensor_enabled", True)),
            imu_enabled=bool(getattr(self, "imu_sensor_enabled", True)),
            full_sleep=full_sleep,
            dream_mode=full_sleep,
            training_disabled_by_sleep=full_sleep,
            optimizer_step=0 if full_sleep else "",
        )

    def log_tetra_scenario_state(self) -> None:
        if not self._tetra_diag_enabled():
            return
        status = self._tetra_diag_scenario_status()
        tetra_pos = (
            float(status.get("tetra_x", 0.0) or 0.0),
            float(status.get("tetra_y", 0.0) or 0.0),
            float(status.get("tetra_z", 0.0) or 0.0),
        )
        tetra_quat = (
            float(status.get("tetra_qw", 0.0) or 0.0),
            float(status.get("tetra_qx", 0.0) or 0.0),
            float(status.get("tetra_qy", 0.0) or 0.0),
            float(status.get("tetra_qz", 0.0) or 0.0),
        )
        self._tetra_diag_write(
            "scenario_state",
            scenario_name=str(status.get("scenario", "")),
            target_name=str(status.get("gaze_target", "")),
            tetra_visible=bool(status.get("active", False)) and bool(status.get("tetra_pose_ok", False)),
            tetra_pos=tetra_pos,
            tetra_quat=tetra_quat,
            tetra_spin_rad_per_step=float(status.get("tetra_spin_rad_per_step", 0.0) or 0.0),
            tetra_face_steps=int(status.get("tetra_face_steps", 0) or 0),
        )
        if str(status.get("gaze_target", "")) == "cube":
            cube_pos = (
                float(status.get("cube_x", 0.0) or 0.0),
                float(status.get("cube_y", 0.0) or 0.0),
                float(status.get("cube_z", 0.0) or 0.0),
            )
            cube_quat = (
                float(status.get("cube_qw", 0.0) or 0.0),
                float(status.get("cube_qx", 0.0) or 0.0),
                float(status.get("cube_qy", 0.0) or 0.0),
                float(status.get("cube_qz", 0.0) or 0.0),
            )
            self._tetra_diag_write(
                "cube_scenario_state",
                scenario_name=str(status.get("scenario", "")),
                phase=str(status.get("phase", "")),
                target_name="cube",
                cube_visible=bool(status.get("active", False)) and bool(status.get("cube_pose_ok", False)),
                cube_pos=cube_pos,
                cube_quat=cube_quat,
                cube_angle=float(status.get("cube_angle", 0.0) or 0.0),
                cube_spin_rad_per_step=float(status.get("cube_spin_rad_per_step", 0.0) or 0.0),
                cube_tumble_rad_per_step=float(status.get("cube_tumble_rad_per_step", 0.0) or 0.0),
                cube_flight_rad_per_step=float(status.get("cube_flight_rad_per_step", 0.0) or 0.0),
            )

    def _tetra_diag_tetra_angle(self) -> tuple[float, float]:
        status = self._tetra_diag_scenario_status()
        angle = float(status.get("tetra_angle", 0.0) or 0.0)
        prev = getattr(self, "_tetra_diag_prev_tetra_angle", None)
        delta = 0.0 if prev is None else float(angle - float(prev))
        self._tetra_diag_prev_tetra_angle = angle
        return angle, delta

    def _tetra_diag_cube_motion(self) -> tuple[float, float, float]:
        status = self._tetra_diag_scenario_status()
        angle = float(status.get("cube_angle", 0.0) or 0.0)
        tumble = float(status.get("cube_tumble_angle", 0.0) or 0.0)
        prev_angle = getattr(self, "_tetra_diag_prev_cube_angle", None)
        prev_tumble = getattr(self, "_tetra_diag_prev_cube_tumble", None)
        angle_delta = 0.0 if prev_angle is None else float(angle - float(prev_angle))
        tumble_delta = 0.0 if prev_tumble is None else float(tumble - float(prev_tumble))
        self._tetra_diag_prev_cube_angle = angle
        self._tetra_diag_prev_cube_tumble = tumble
        return angle, angle_delta, tumble_delta

    def log_tetra_object_diagnostics(self, obj: dict | None) -> None:
        if not self._tetra_diag_enabled():
            return
        if not isinstance(obj, dict):
            return
        self.log_inner_object_slot_metrics(obj)
        dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
        ready = bool(dbg.get("dynamic_ready", False))
        write = bool(dbg.get("slot_update_allowed", False))
        dyn_conf = float(dbg.get("long_dynamic_confidence", 0.0) or 0.0)
        motion_ok = bool(float(dbg.get("long_dynamic_motion_ok", 0.0) or 0.0) > 0.5)
        formed_conf = dyn_conf if (ready and write and motion_ok) else 0.0
        temporal = float(dbg.get("dynamic_score", 0.0) or 0.0)
        depth_motion = float(dbg.get("long_dynamic_depth_motion", 0.0) or 0.0)
        streak = float(dbg.get("long_dynamic_ready_streak", 0.0) or 0.0)
        z_static_norm = float(dbg.get("long_dynamic_z_static_norm", 0.0) or 0.0)
        z_dynamic_norm = float(dbg.get("long_dynamic_z_dynamic_norm", 0.0) or 0.0)
        slot_conf = self._tetra_diag_tensor_float(obj.get("confidence"), 0.0)

        dynamic_event = bool(dbg.get("dynamic_input_active", False)) or ready or write
        common_motion_fields = {
            "raw_dynamic_input": bool(dbg.get("dynamic_input_raw_active", False)),
            "target_motion_allowed": bool(dbg.get("dynamic_target_motion_allowed", True)),
            "target_motion_reason": str(dbg.get("dynamic_target_motion_reason", "unknown")),
        }
        if dynamic_event:
            target_name = self._tetra_diag_target_name()
            if target_name == "cube":
                angle, angle_delta, tumble_delta = self._tetra_diag_cube_motion()
                self._tetra_diag_write(
                    "cube_dynamic_input_check",
                    cube_angle=angle,
                    cube_angle_delta=angle_delta,
                    cube_tumble_delta=tumble_delta,
                    temporal_motion_score=temporal,
                    depth_motion_score=depth_motion,
                    streak=streak,
                    dyn_eff=formed_conf,
                    formed_conf=formed_conf,
                    ready=ready,
                    write=write,
                    z_dynamic_norm=z_dynamic_norm,
                    z_static_norm=z_static_norm,
                    **common_motion_fields,
                )
                self._tetra_diag_cube_checks = {
                    **dict(getattr(self, "_tetra_diag_cube_checks", {}) or {}),
                    "dynamic_input": bool((abs(angle_delta) > 0.0 or abs(tumble_delta) > 0.0) and temporal > 0.0 and depth_motion > 0.0),
                    "dynamic_alive": bool((abs(angle_delta) > 0.0 or abs(tumble_delta) > 0.0) and temporal > 0.0),
                    "streak": streak,
                    "dyn_eff": formed_conf,
                    "formed_conf": formed_conf,
                    "z_dynamic_norm": z_dynamic_norm,
                }
            else:
                angle, angle_delta = self._tetra_diag_tetra_angle()
                self._tetra_diag_write(
                    "dynamic_input_check",
                    tetra_angle=angle,
                    tetra_angle_delta=angle_delta,
                    temporal_motion_score=temporal,
                    depth_motion_score=depth_motion,
                    streak=streak,
                    dyn_eff=formed_conf,
                    formed_conf=formed_conf,
                    ready=ready,
                    write=write,
                    z_dynamic_norm=z_dynamic_norm,
                    z_static_norm=z_static_norm,
                    **common_motion_fields,
                )
        else:
            self._tetra_diag_write(
                "static_input_check",
                z_static_norm=z_static_norm,
                slot_conf_raw=slot_conf,
                temporal_motion_score=temporal,
                depth_motion_score=depth_motion,
                streak=streak,
                dyn_eff=formed_conf,
                formed_conf=formed_conf,
                ready=ready,
                write=write,
                **common_motion_fields,
            )

        self.log_tetra_long_dynamic_memory(obj, ready=ready, write=write, formed_conf=formed_conf)
        self.log_tetra_slot_identity_check(obj)

    def log_inner_object_slot_metrics(self, obj: dict | None) -> None:
        if not self._tetra_diag_enabled():
            return
        if not isinstance(obj, dict):
            return
        state = getattr(self, "inner_object_state", {}) or {}
        z_slots = obj.get("z_obj_slots", state.get("z_obj_slots") if isinstance(state, dict) else None)
        conf_slots = obj.get("confidence_slots", state.get("confidence_slots") if isinstance(state, dict) else None)
        mem_slots = obj.get("memory_stability_slots", state.get("memory_stability_slots") if isinstance(state, dict) else None)
        dream_slots = obj.get("dream_activation_slots", state.get("dream_activation_slots") if isinstance(state, dict) else None)
        update_slots = obj.get("slot_update_strength", state.get("slot_update_strength") if isinstance(state, dict) else None)
        age_slots = obj.get("slot_age", state.get("slot_age") if isinstance(state, dict) else None)
        active_slot = self._tetra_diag_active_slot_id(obj)
        selected_slot = active_slot
        try:
            requested = getattr(getattr(self, "inner_object_viz", None), "requested_dream_slot_index", None)
            if requested is not None:
                selected_slot = int(requested)
        except Exception:
            pass
        try:
            n_slots = int(z_slots.shape[1]) if torch.is_tensor(z_slots) and z_slots.ndim >= 3 else int(getattr(getattr(self, "cfg", None).object_image, "num_slots", 10))
        except Exception:
            n_slots = 10
        full_sleep = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        optimizer_delta = int(getattr(self, "train_steps", 0)) - int(getattr(self, "_tetra_diag_prev_train_steps", getattr(self, "train_steps", 0)))
        self._tetra_diag_prev_train_steps = int(getattr(self, "train_steps", 0))
        for slot_id in range(max(1, min(n_slots, 10))):
            c = self._tetra_diag_slot_scalar(conf_slots, slot_id, 0.0)
            m = self._tetra_diag_slot_scalar(mem_slots, slot_id, c)
            d = self._tetra_diag_slot_scalar(dream_slots, slot_id, 0.0)
            u = self._tetra_diag_slot_scalar(update_slots, slot_id, 0.0)
            z_norm = self._tetra_diag_slot_norm(z_slots, slot_id, 0.0)
            if z_norm <= 0.0 and m <= 0.0 and c <= 0.0 and d <= 0.0 and u <= 0.0:
                continue
            self._tetra_diag_write(
                "slot_metrics",
                slot_id=slot_id,
                active_slot=active_slot,
                selected_slot=selected_slot,
                confidence_slots=c,
                C=c,
                memory_stability_slots=m,
                M=m,
                dream_activation_slots=d,
                D=d,
                slot_update_strength=u,
                U=u,
                sleep_dream_mode=self._tetra_diag_tensor_float(obj.get("sleep_dream_mode"), 0.0),
                dream_empty_mode=self._tetra_diag_tensor_float(obj.get("dream_empty_mode"), 0.0),
                dream_tick=self._tetra_diag_tensor_float(obj.get("dream_tick"), 0.0),
                z_obj_norm=z_norm,
                active_slot_age=self._tetra_diag_slot_scalar(age_slots, slot_id, 0.0),
                video_enabled=bool(getattr(self, "video_sensor_enabled", True)),
                contact_enabled=bool(getattr(self, "contact_sensor_enabled", True)),
                imu_enabled=bool(getattr(self, "imu_sensor_enabled", True)),
                full_sleep=full_sleep,
                optimizer_step_delta=optimizer_delta,
                object_visible=bool((self._tetra_diag_scenario_status() or {}).get("active", False)),
            )

    def log_tetra_long_dynamic_memory(self, obj: dict | None, *, ready: bool, write: bool, formed_conf: float) -> None:
        if not self._tetra_diag_enabled():
            return
        stats = getattr(self, "latest_long_dynamic_memory_stats", {}) or {}
        dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
        ldm = getattr(self, "long_dynamic_object_memory", None)
        train_flag = bool(getattr(getattr(self, "module_training_gate", None), "flags", {}).get("long_dynamic_memory", False))
        self._tetra_diag_write(
            "cube_long_dynamic_memory" if self._tetra_diag_target_name() == "cube" else "long_dynamic_memory",
            enabled=bool(ldm is not None),
            trainable=bool(train_flag and ldm is not None and any(p.requires_grad for p in ldm.parameters())),
            target=self._tetra_diag_target_name(),
            slot_candidate=self._tetra_diag_target_slot(),
            loss=float(stats.get("loss", 0.0) or 0.0),
            loss_ema=float(stats.get("loss_ema", 0.0) or 0.0),
            reward_proxy=float(stats.get("reward_proxy", 0.0) or 0.0),
            recon=float(stats.get("recon", 0.0) or 0.0),
            z_static_norm=float(stats.get("z_static_norm", dbg.get("long_dynamic_z_static_norm", 0.0)) or 0.0),
            z_dynamic_norm=float(stats.get("z_dynamic_norm", dbg.get("long_dynamic_z_dynamic_norm", 0.0)) or 0.0),
            ready=ready,
            write=write,
            formed_conf=float(formed_conf),
            slot_id=self._tetra_diag_active_slot_id(obj),
            optimizer_update=0 if not train_flag else "",
        )

    def _tetra_diag_slot_values(self, slot_id: int) -> tuple[float, float]:
        state = getattr(self, "inner_object_state", {}) or {}
        try:
            conf = state.get("confidence_slots")
            z_slots = state.get("z_obj_slots")
            c = 0.0
            z = 0.0
            if torch.is_tensor(conf) and conf.ndim >= 3 and conf.shape[1] > slot_id:
                c = float(conf[:, slot_id, :].detach().float().mean().cpu().item())
            if torch.is_tensor(z_slots) and z_slots.ndim >= 3 and z_slots.shape[1] > slot_id:
                z = float(z_slots[:, slot_id, :].detach().float().norm(dim=-1).mean().cpu().item())
            return c, z
        except Exception:
            return 0.0, 0.0

    def log_dynamic_object_slot_policy(
        self,
        *,
        target_name: str,
        is_new_object: bool,
        matched_existing_slot,
        allocated_slot: int,
        next_free_slot: int,
        protected_slots,
        overwrite_allowed: bool,
        decision_reason: str,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "new_dynamic_object_slot_policy",
            target_name=str(target_name),
            is_new_object=bool(is_new_object),
            matched_existing_slot=matched_existing_slot,
            allocated_slot=int(allocated_slot),
            next_free_slot=int(next_free_slot),
            protected_slots=",".join(str(int(s)) for s in protected_slots) if protected_slots else "none",
            overwrite_allowed=bool(overwrite_allowed),
            decision_reason=str(decision_reason),
        )

    def log_slot_protection_state(
        self,
        *,
        slot_id: int,
        target_name: str,
        formed_conf: float,
        z_dynamic_norm: float,
        protected: bool,
        reason: str,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_protection_state",
            slot_id=int(slot_id),
            target_name=str(target_name),
            formed_conf=float(formed_conf),
            z_dynamic_norm=float(z_dynamic_norm),
            protected=bool(protected),
            reason=str(reason),
        )


    def log_slot_observation_buffer(self, *, slot_id: int, target_name: str, frame_count: int, depth_valid: bool, live_step: int) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write("slot_observation_buffer", slot_id=int(slot_id), target_name=str(target_name), frame_count=int(frame_count), depth_valid=bool(depth_valid), live_step=int(live_step))

    def log_slot_pointcloud_reconstruction(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write("slot_pointcloud_reconstruction", slot_id=int(metrics.get("slot_id", -1)), target_name=str(metrics.get("target_name", "unknown")), depth_valid=bool(metrics.get("depth_valid", False)), points_added=int(metrics.get("points_added", 0) or 0), points_total=int(metrics.get("points_total", 0) or 0), frame_count=int(metrics.get("frame_count", 0) or 0), formed_conf=float(metrics.get("formed_conf", 0.0) or 0.0), z_dynamic_norm=float(metrics.get("z_dynamic_norm", 0.0) or 0.0))
        slot_metrics = dict(getattr(self, "_tetra_diag_slot_pointcloud_metrics", {}) or {})
        sid = int(metrics.get("slot_id", -1))
        slot_metrics[sid] = {
            "target_name": str(metrics.get("target_name", "unknown")),
            "points_total": int(metrics.get("points_total", 0) or 0),
            "frame_count": int(metrics.get("frame_count", 0) or 0),
        }
        self._tetra_diag_slot_pointcloud_metrics = slot_metrics
        self._tetra_diag_maybe_log_slot_observation_pointcloud_success()

    def _tetra_diag_maybe_log_slot_observation_pointcloud_success(self) -> None:
        if not self._tetra_diag_enabled():
            return
        if bool(getattr(self, "_tetra_diag_slot_observation_pointcloud_success_logged", False)):
            return
        slot_metrics = dict(getattr(self, "_tetra_diag_slot_pointcloud_metrics", {}) or {})
        slot0 = dict(slot_metrics.get(0, {}) or {})
        slot1 = dict(slot_metrics.get(1, {}) or {})
        identity = dict(getattr(self, "_tetra_diag_slot_identity", {}) or {})
        slot0_target = str(slot0.get("target_name", identity.get("slot_0_target", "unknown")))
        slot1_target = str(slot1.get("target_name", identity.get("slot_1_target", "unknown")))
        slot0_frames = int(slot0.get("frame_count", 0) or 0)
        slot1_frames = int(slot1.get("frame_count", 0) or 0)
        slot0_points = int(slot0.get("points_total", 0) or 0)
        slot1_points = int(slot1.get("points_total", 0) or 0)
        slot0_overwritten = bool(identity.get("slot_0_overwritten", slot0_target != "tetrahedron"))
        slot1_allocated = bool(identity.get("slot_1_allocated", slot1_target == "cube"))
        if not (
            slot0_target == "tetrahedron"
            and slot1_target == "cube"
            and slot0_frames > 0
            and slot1_frames > 0
            and slot0_points > 0
            and slot1_points > 0
            and not slot0_overwritten
            and slot1_allocated
        ):
            return
        self._tetra_diag_slot_observation_pointcloud_success_logged = True
        self._tetra_diag_write(
            "SUCCESS_SLOT_OBSERVATION_POINTCLOUD_STEP1",
            slot_0_target=slot0_target,
            slot_0_frame_count=slot0_frames,
            slot_0_points_total=slot0_points,
            slot_1_target=slot1_target,
            slot_1_frame_count=slot1_frames,
            slot_1_points_total=slot1_points,
            slot_0_overwritten=slot0_overwritten,
            slot_1_allocated=slot1_allocated,
            reason="rgbd_observation_buffers_and_pointclouds_separate",
        )


    def log_slot_gaussian_init(
        self,
        *,
        slot_id: int,
        target_name: str,
        gaussian_count: int,
        source_points: int,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_gaussian_init",
            slot_id=int(slot_id),
            target_name=str(target_name),
            gaussian_count=int(gaussian_count),
            source_points=int(source_points),
        )

    def log_slot_gaussian_train(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_gaussian_train",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            initialized=bool(metrics.get("initialized", False)),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            rgb_loss=float(metrics.get("rgb_loss", 0.0) or 0.0),
            depth_loss=float(metrics.get("depth_loss", 0.0) or 0.0),
            total_loss=float(metrics.get("total_loss", 0.0) or 0.0),
            render_valid=bool(metrics.get("render_valid", False)),
        )

    def log_slot_gaussian_render(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_gaussian_render",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            render_valid=bool(metrics.get("render_valid", False)),
            rgb_loss=float(metrics.get("rgb_loss", 0.0) or 0.0),
            depth_loss=float(metrics.get("depth_loss", 0.0) or 0.0),
        )

    def log_success_slot_gaussian_step2(
        self,
        *,
        slot_0_target: str,
        slot_0_gaussian_count: int,
        slot_0_recon_loss: float,
        slot_0_updates: int,
        slot_1_target: str,
        slot_1_gaussian_count: int,
        slot_1_recon_loss: float,
        slot_1_updates: int,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_SLOT_GAUSSIAN_3DGS_STEP2",
            slot_0_target=str(slot_0_target),
            slot_0_gaussian_count=int(slot_0_gaussian_count),
            slot_0_recon_loss=float(slot_0_recon_loss),
            slot_0_updates=int(slot_0_updates),
            slot_1_target=str(slot_1_target),
            slot_1_gaussian_count=int(slot_1_gaussian_count),
            slot_1_recon_loss=float(slot_1_recon_loss),
            slot_1_updates=int(slot_1_updates),
            slot_0_overwritten=0,
            slot_1_allocated=1,
            reason="per_slot_low_res_gaussian_training_active",
        )


    def log_slot_gaussian_train(self, **metrics) -> None:
        if not self._tetra_diag_enabled(): return
        self._tetra_diag_write(
            "slot_gaussian_train", slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            initialized=bool(metrics.get("initialized", False)),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            rgb_loss=float(metrics.get("rgb_loss", 0.0) or 0.0),
            depth_loss=float(metrics.get("depth_loss", 0.0) or 0.0),
            total_loss=float(metrics.get("total_loss", 0.0) or 0.0),
            render_valid=bool(metrics.get("render_valid", False)),
            backend=str(metrics.get("backend", metrics.get("active_backend", "torch_lowres"))))

    def log_slot_gaussian_render(self, **metrics) -> None:
        if not self._tetra_diag_enabled(): return
        self._tetra_diag_write(
            "slot_gaussian_render", slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            render_valid=bool(metrics.get("render_valid", False)),
            rgb_loss=float(metrics.get("rgb_loss", 0.0) or 0.0),
            depth_loss=float(metrics.get("depth_loss", 0.0) or 0.0),
            backend=str(metrics.get("backend", metrics.get("active_backend", "torch_lowres"))))

    def log_slot_gaussian_cuda_backend(self, **metrics) -> None:
        if not self._tetra_diag_enabled(): return
        self._tetra_diag_write(
            "slot_gaussian_cuda_backend",
            requested_backend=str(metrics.get("requested_backend", "auto")),
            active_backend=str(metrics.get("backend", metrics.get("active_backend", "torch_lowres"))),
            cuda_available=bool(metrics.get("cuda_available", False)),
            rasterizer_available=bool(metrics.get("rasterizer_available", False)),
            fallback_used=bool(metrics.get("fallback_used", False)),
            fps=float(metrics.get("preview_fps", metrics.get("fps", 0.0)) or 0.0),
            import_error=str(metrics.get("import_error", ""))[:180])

    def log_slot_gaussian_preview_frame(self, **metrics) -> None:
        if not self._tetra_diag_enabled(): return
        self._tetra_diag_write(
            "slot_gaussian_preview_frame", slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            backend=str(metrics.get("backend", metrics.get("active_backend", "torch_lowres"))),
            render_valid=bool(metrics.get("render_valid", False)),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            preview_fps=float(metrics.get("preview_fps", metrics.get("fps", 0.0)) or 0.0),
            fallback_used=bool(metrics.get("fallback_used", False)))

    def log_success_slot_gaussian_cuda_step2b(
        self, *, slot_0_target: str, slot_0_gaussian_count: int, slot_0_recon_loss: float,
        slot_0_updates: int, slot_0_backend: str, slot_0_preview_fps: float,
        slot_1_target: str, slot_1_gaussian_count: int, slot_1_recon_loss: float,
        slot_1_updates: int, slot_1_backend: str, slot_1_preview_fps: float, fallback_used: bool) -> None:
        if not self._tetra_diag_enabled(): return
        self._tetra_diag_write(
            "SUCCESS_SLOT_GAUSSIAN_CUDA_PREVIEW_STEP2B",
            slot_0_target=str(slot_0_target), slot_0_gaussian_count=int(slot_0_gaussian_count),
            slot_0_recon_loss=float(slot_0_recon_loss), slot_0_updates=int(slot_0_updates),
            slot_0_backend=str(slot_0_backend), slot_0_preview_fps=float(slot_0_preview_fps),
            slot_1_target=str(slot_1_target), slot_1_gaussian_count=int(slot_1_gaussian_count),
            slot_1_recon_loss=float(slot_1_recon_loss), slot_1_updates=int(slot_1_updates),
            slot_1_backend=str(slot_1_backend), slot_1_preview_fps=float(slot_1_preview_fps),
            fallback_used=bool(fallback_used), slot_0_overwritten=0, slot_1_allocated=1,
            reason="configurable_gaussian_renderer_backend_with_preview")


    def log_slot_4d_frame(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_frame",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            live_step=int(metrics.get("live_step", 0) or 0),
            frame_count=int(metrics.get("frame_count", 0) or 0),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            recon_loss=float(metrics.get("recon_loss", 0.0) or 0.0),
            formed_conf=float(metrics.get("formed_conf", 0.0) or 0.0),
            z_dynamic_norm=float(metrics.get("z_dynamic_norm", 0.0) or 0.0),
            motion_norm=float(metrics.get("motion_norm", 0.0) or 0.0),
            backend=str(metrics.get("backend", "unknown")),
            valid=bool(metrics.get("valid", False)),
        )

    def log_slot_4d_timeline(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_timeline",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            timeline_frames=int(metrics.get("frame_count", 0) or 0),
            temporal_span=int(metrics.get("temporal_span", 0) or 0),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            updates=int(metrics.get("updates", 0) or 0),
            motion_norm=float(metrics.get("motion_norm", 0.0) or 0.0),
            mean_delta_x=float(metrics.get("mean_delta_x", 0.0) or 0.0),
            mean_delta_y=float(metrics.get("mean_delta_y", 0.0) or 0.0),
            mean_delta_z=float(metrics.get("mean_delta_z", 0.0) or 0.0),
            backend=str(metrics.get("backend", "unknown")),
            valid=bool(metrics.get("valid", False)),
        )

    def log_success_slot_4d_timeline_step3a(
        self,
        *,
        slot_0_target: str,
        slot_0_timeline_frames: int,
        slot_0_gaussian_count: int,
        slot_0_temporal_span: int,
        slot_1_target: str,
        slot_1_timeline_frames: int,
        slot_1_gaussian_count: int,
        slot_1_temporal_span: int,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_SLOT_4D_TIMELINE_STEP3A",
            slot_0_target=str(slot_0_target),
            slot_0_timeline_frames=int(slot_0_timeline_frames),
            slot_0_gaussian_count=int(slot_0_gaussian_count),
            slot_0_temporal_span=int(slot_0_temporal_span),
            slot_1_target=str(slot_1_target),
            slot_1_timeline_frames=int(slot_1_timeline_frames),
            slot_1_gaussian_count=int(slot_1_gaussian_count),
            slot_1_temporal_span=int(slot_1_temporal_span),
            slot_0_overwritten=0,
            slot_1_allocated=1,
            reason="per_slot_gaussian_timeline_ready_for_4d_deformation",
        )


    def log_slot_4d_deformation_model(
        self,
        *,
        slot_id: int,
        target_name: str,
        model_type: str,
        trainable_params: int,
        enabled: bool,
        trainable: bool,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_deformation_model",
            slot_id=int(slot_id),
            target_name=str(target_name),
            model_type=str(model_type),
            trainable_params=int(trainable_params),
            enabled=bool(enabled),
            trainable=bool(trainable),
        )

    def log_slot_4d_deformation_train(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_deformation_train",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            enabled=bool(metrics.get("enabled", False)),
            trainable=bool(metrics.get("trainable", False)),
            valid=bool(metrics.get("valid", False)),
            updates=int(metrics.get("updates", 0) or 0),
            loss=float(metrics.get("loss", 0.0) or 0.0),
            motion_norm=float(metrics.get("motion_norm", 0.0) or 0.0),
            pred_delta_norm=float(metrics.get("pred_delta_norm", 0.0) or 0.0),
            sample_count=int(metrics.get("sample_count", 0) or 0),
            temporal_dt=float(metrics.get("temporal_dt", 0.0) or 0.0),
            model_type=str(metrics.get("model_type", "Slot4DDeformationModel")),
        )

    def log_success_slot_4d_deformation_step3b(
        self,
        *,
        slot_0_target: str,
        slot_0_deformation_updates: int,
        slot_0_deformation_loss: float,
        slot_0_motion_norm: float,
        slot_0_sample_count: int,
        slot_1_target: str,
        slot_1_deformation_updates: int,
        slot_1_deformation_loss: float,
        slot_1_motion_norm: float,
        slot_1_sample_count: int,
        trainable_params: int,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_SLOT_4D_DEFORMATION_STEP3B",
            slot_0_target=str(slot_0_target),
            slot_0_deformation_updates=int(slot_0_deformation_updates),
            slot_0_deformation_loss=float(slot_0_deformation_loss),
            slot_0_motion_norm=float(slot_0_motion_norm),
            slot_0_sample_count=int(slot_0_sample_count),
            slot_1_target=str(slot_1_target),
            slot_1_deformation_updates=int(slot_1_deformation_updates),
            slot_1_deformation_loss=float(slot_1_deformation_loss),
            slot_1_motion_norm=float(slot_1_motion_norm),
            slot_1_sample_count=int(slot_1_sample_count),
            trainable_params=int(trainable_params),
            slot_0_overwritten=0,
            slot_1_allocated=1,
            reason="per_slot_4d_deformation_field_training_active",
        )


    def log_slot_4d_playback_frame(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_playback_frame",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            enabled=bool(metrics.get("enabled", False)),
            valid=bool(metrics.get("valid", False)),
            render_valid=bool(metrics.get("render_valid", False)),
            deformation_used=bool(metrics.get("deformation_used", False)),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
            playback_phase=float(metrics.get("playback_phase", 0.0) or 0.0),
            pred_delta_norm=float(metrics.get("pred_delta_norm", 0.0) or 0.0),
            backend=str(metrics.get("backend", "unknown")),
            preview_fps=float(metrics.get("preview_fps", 0.0) or 0.0),
            frame_count=int(metrics.get("frame_count", 0) or 0),
        )

    def log_slot_4d_deformed_render(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_deformed_render",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            render_valid=bool(metrics.get("render_valid", False)),
            deformation_used=bool(metrics.get("deformation_used", False)),
            playback_phase=float(metrics.get("playback_phase", 0.0) or 0.0),
            pred_delta_norm=float(metrics.get("pred_delta_norm", 0.0) or 0.0),
            backend=str(metrics.get("backend", "unknown")),
            preview_fps=float(metrics.get("preview_fps", 0.0) or 0.0),
            gaussian_count=int(metrics.get("gaussian_count", 0) or 0),
        )

    def log_success_slot_4d_playback_step3c(
        self,
        *,
        slot_0_target: str,
        slot_0_playback_frames: int,
        slot_0_playback_phase: float,
        slot_0_pred_delta_norm: float,
        slot_0_backend: str,
        slot_1_target: str,
        slot_1_playback_frames: int,
        slot_1_playback_phase: float,
        slot_1_pred_delta_norm: float,
        slot_1_backend: str,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_SLOT_4D_PLAYBACK_STEP3C",
            slot_0_target=str(slot_0_target),
            slot_0_playback_frames=int(slot_0_playback_frames),
            slot_0_playback_phase=float(slot_0_playback_phase),
            slot_0_pred_delta_norm=float(slot_0_pred_delta_norm),
            slot_0_backend=str(slot_0_backend),
            slot_1_target=str(slot_1_target),
            slot_1_playback_frames=int(slot_1_playback_frames),
            slot_1_playback_phase=float(slot_1_playback_phase),
            slot_1_pred_delta_norm=float(slot_1_pred_delta_norm),
            slot_1_backend=str(slot_1_backend),
            slot_0_overwritten=0,
            slot_1_allocated=1,
            reason="deformation_aware_4d_playback_preview_active",
        )


    def log_slot_4d_open3d_export(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "slot_4d_open3d_export",
            slot_id=int(metrics.get("slot_id", -1)), target_name=str(metrics.get("target_name", "unknown")),
            export_path=str(metrics.get("export_path", "")), written=bool(metrics.get("written", False)), reason=str(metrics.get("reason", "")),
            slot_0_raw_points=int(metrics.get("slot_0_raw_points", 0) or 0), slot_1_raw_points=int(metrics.get("slot_1_raw_points", 0) or 0),
            slot_0_deformed_points=int(metrics.get("slot_0_deformed_points", 0) or 0), slot_1_deformed_points=int(metrics.get("slot_1_deformed_points", 0) or 0),
        )

    def log_success_slot_4d_open3d_viewer_step3f(self, *, export_path: str, slot_0_target: str, slot_0_raw_points: int, slot_0_deformed_points: int, slot_0_deformation_used: bool, slot_1_target: str, slot_1_raw_points: int, slot_1_deformed_points: int, slot_1_deformation_used: bool) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_SLOT_4D_OPEN3D_VIEWER_STEP3F",
            export_path=str(export_path), slot_0_target=str(slot_0_target), slot_0_raw_points=int(slot_0_raw_points), slot_0_deformed_points=int(slot_0_deformed_points), slot_0_deformation_used=bool(slot_0_deformation_used),
            slot_1_target=str(slot_1_target), slot_1_raw_points=int(slot_1_raw_points), slot_1_deformed_points=int(slot_1_deformed_points), slot_1_deformation_used=bool(slot_1_deformation_used),
            viewer_command="python -m src.modules.m01_object_imagery.open3d_slot_viewer --path " + str(export_path), reason="separate_open3d_slot_viewer_export_ready",
        )

    def log_slot_4d_jsonrpc_stream(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write("slot_4d_jsonrpc_stream", slot_id=int(metrics.get("slot_id", -1)), target_name=str(metrics.get("target_name", "unknown")), host=str(metrics.get("host", "127.0.0.1")), port=int(metrics.get("port", 8771)), started=bool(metrics.get("started", False)), updated=bool(metrics.get("updated", False)), reason=str(metrics.get("reason", "")), slot_0_points=int(metrics.get("slot_0_points", 0) or 0), slot_1_points=int(metrics.get("slot_1_points", 0) or 0), slot_0_deformed_points=int(metrics.get("slot_0_deformed_points", 0) or 0), slot_1_deformed_points=int(metrics.get("slot_1_deformed_points", 0) or 0))

    def log_success_slot_4d_jsonrpc_stream_step3h(self, *, host: str, port: int, slot_0_points: int, slot_1_points: int, slot_0_deformed_points: int, slot_1_deformed_points: int) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write("SUCCESS_SLOT_4D_JSONRPC_STREAM_STEP3H", host=str(host), port=int(port), slot_0_points=int(slot_0_points), slot_1_points=int(slot_1_points), slot_0_deformed_points=int(slot_0_deformed_points), slot_1_deformed_points=int(slot_1_deformed_points), viewer_command=f"python -m src.modules.m01_object_imagery.open3d_slot_viewer_rpc --host {host} --port {port} --slot both --mode deformed", reason="jsonrpc_stream_ready_for_inner_object_open3d")

    def log_success_open3d_rpc_window_output_verified(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_OPEN3D_RPC_WINDOW_OUTPUT_VERIFIED",
            slot_0_target=str(metrics.get("slot_0_target", "tetrahedron")),
            slot_0_raw_points=int(metrics.get("slot_0_raw_points", 0) or 0),
            slot_0_deformed_points=int(metrics.get("slot_0_deformed_points", 0) or 0),
            slot_0_rpc_points=int(metrics.get("slot_0_rpc_points", 0) or 0),
            slot_0_xyz_shape=str(metrics.get("slot_0_xyz_shape", "")),
            slot_0_color_shape=str(metrics.get("slot_0_color_shape", "")),
            slot_0_deformation_used=bool(metrics.get("slot_0_deformation_used", False)),
            slot_1_target=str(metrics.get("slot_1_target", "cube")),
            slot_1_raw_points=int(metrics.get("slot_1_raw_points", 0) or 0),
            slot_1_deformed_points=int(metrics.get("slot_1_deformed_points", 0) or 0),
            slot_1_rpc_points=int(metrics.get("slot_1_rpc_points", 0) or 0),
            slot_1_xyz_shape=str(metrics.get("slot_1_xyz_shape", "")),
            slot_1_color_shape=str(metrics.get("slot_1_color_shape", "")),
            slot_1_deformation_used=bool(metrics.get("slot_1_deformation_used", False)),
            viewer_mode=str(metrics.get("viewer_mode", "deformed")),
            viewer_slot=str(metrics.get("viewer_slot", "both")),
            jsonrpc_host=str(metrics.get("jsonrpc_host", "127.0.0.1")),
            jsonrpc_port=int(metrics.get("jsonrpc_port", 8771)),
            window_launch_attempted=int(metrics.get("window_launch_attempted", 0) or 0),
            window_expected_visible=int(metrics.get("window_expected_visible", 0) or 0),
        )

    def log_slot_object_memory_step4(self, **metrics) -> None:
        if not self._tetra_diag_enabled():
            return
        best = metrics.get("best_match", {}) or {}
        self._tetra_diag_write(
            "slot_object_memory_step4",
            slot_id=int(metrics.get("slot_id", -1)),
            target_name=str(metrics.get("target_name", "unknown")),
            written=bool(metrics.get("written", False)),
            reason=str(metrics.get("reason", "")),
            path=str(metrics.get("path", "")),
            raw_points=int(metrics.get("raw_points", 0) or 0),
            deformed_points=int(metrics.get("deformed_points", 0) or 0),
            deformation_used=bool(metrics.get("deformation_used", False)),
            recall_matched=bool(best.get("matched", False)),
            recall_distance=float(best.get("distance", 999999.0) if best else 999999.0),
            recall_path=str(best.get("path", "") if best else ""),
        )

    def log_success_slot_object_memory_step4(
        self,
        *,
        slot_0_target: str,
        slot_0_raw_points: int,
        slot_0_deformed_points: int,
        slot_0_path: str,
        slot_0_recall_matched: bool,
        slot_1_target: str,
        slot_1_raw_points: int,
        slot_1_deformed_points: int,
        slot_1_path: str,
        slot_1_recall_matched: bool,
    ) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "SUCCESS_SLOT_OBJECT_MEMORY_STEP4D4E",
            slot_0_target=str(slot_0_target),
            slot_0_raw_points=int(slot_0_raw_points),
            slot_0_deformed_points=int(slot_0_deformed_points),
            slot_0_path=str(slot_0_path),
            slot_0_recall_matched=bool(slot_0_recall_matched),
            slot_1_target=str(slot_1_target),
            slot_1_raw_points=int(slot_1_raw_points),
            slot_1_deformed_points=int(slot_1_deformed_points),
            slot_1_path=str(slot_1_path),
            slot_1_recall_matched=bool(slot_1_recall_matched),
            reason="persistent_inner_object_memory_written",
        )

    def log_tetra_slot_identity_check(self, obj: dict | None = None) -> None:
        if not self._tetra_diag_enabled():
            return
        targets = dict(getattr(self, "_tetra_diag_slot_targets", {}) or {})
        slot0_conf, slot0_z = self._tetra_diag_slot_values(0)
        slot1_conf, slot1_z = self._tetra_diag_slot_values(1)
        slot0_target = str(targets.get(0, "unknown"))
        slot1_target = str(targets.get(1, "unknown"))
        slot0_overwritten = bool(slot0_target == "cube")
        slot1_allocated = bool(slot1_target == "cube" and slot1_conf > 0.0 and slot1_z > 0.0)
        self._tetra_diag_write(
            "slot_identity_check",
            slot_0_target=slot0_target,
            slot_0_formed_conf=slot0_conf,
            slot_0_z_dynamic_norm=slot0_z,
            slot_1_target=slot1_target,
            slot_1_formed_conf=slot1_conf,
            slot_1_z_dynamic_norm=slot1_z,
            slot_0_overwritten=slot0_overwritten,
            slot_1_allocated=slot1_allocated,
        )
        self._tetra_diag_slot_identity = {
            "slot_0_target": slot0_target,
            "slot_0_formed_conf": slot0_conf,
            "slot_0_z_dynamic_norm": slot0_z,
            "slot_1_target": slot1_target,
            "slot_1_formed_conf": slot1_conf,
            "slot_1_z_dynamic_norm": slot1_z,
            "slot_0_overwritten": slot0_overwritten,
            "slot_1_allocated": slot1_allocated,
        }
        self._tetra_diag_maybe_log_tetra_success()
        self._tetra_diag_maybe_log_cube_success()

    def log_tetra_slot_write(self, slot_id: int, source: str, z_source: Any, slot: dict) -> None:
        if not self._tetra_diag_enabled():
            return
        dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
        formed_conf = float(dbg.get("long_dynamic_confidence", 0.0) or 0.0) if bool(dbg.get("dynamic_ready", False)) else 0.0
        target_name = self._tetra_diag_target_name()
        if "dynamic" in str(source) and target_name in ("tetrahedron", "cube"):
            targets = dict(getattr(self, "_tetra_diag_slot_targets", {}) or {})
            if targets.get(int(slot_id)) != target_name:
                if target_name == "cube" and targets.get(0) == "tetrahedron" and int(slot_id) == 1:
                    self._tetra_diag_write(
                        "attention_switch",
                        from_target="tetrahedron",
                        to_target="cube",
                        from_slot=0,
                        candidate_slot=1,
                        reason="cube_dynamic_target_selected",
                    )
                    self._tetra_diag_cube_checks = {
                        **dict(getattr(self, "_tetra_diag_cube_checks", {}) or {}),
                        "attention_switch": True,
                    }
                targets[int(slot_id)] = target_name
                self._tetra_diag_slot_targets = targets
        event_name = "cube_slot_write" if target_name == "cube" else "slot_write"
        previous_slot0_target = str((getattr(self, "_tetra_diag_slot_targets", {}) or {}).get(0, "unknown"))
        previous_slot0_conf, _previous_slot0_z = self._tetra_diag_slot_values(0)
        self._tetra_diag_write(
            event_name,
            slot_id=int(slot_id),
            source=str(source),
            z_source=str("z_dynamic_object" if "dynamic" in str(source) else "z_static"),
            z_static_norm=float(dbg.get("long_dynamic_z_static_norm", 0.0) or 0.0),
            z_dynamic_norm=self._tetra_diag_tensor_norm(z_source, float(dbg.get("long_dynamic_z_dynamic_norm", 0.0) or 0.0)),
            slot_conf=self._tetra_diag_tensor_float(slot.get("confidence"), 0.0) if isinstance(slot, dict) else 0.0,
            formed_conf=formed_conf,
            previous_slot_0_target=previous_slot0_target,
            previous_slot_0_conf=previous_slot0_conf,
        )
        if event_name == "cube_slot_write":
            self._tetra_diag_cube_checks = {
                **dict(getattr(self, "_tetra_diag_cube_checks", {}) or {}),
                "slot_write": bool(int(slot_id) == 1 and "dynamic" in str(source)),
            }
        self.log_tetra_slot_identity_check(slot if isinstance(slot, dict) else None)

    def _tetra_diag_maybe_log_tetra_success(self) -> None:
        if not self._tetra_diag_enabled():
            return
        if bool(getattr(self, "_tetra_diag_tetra_success_logged", False)):
            return
        if self._tetra_diag_target_name() != "tetrahedron":
            return
        identity = dict(getattr(self, "_tetra_diag_slot_identity", {}) or {})
        dbg = getattr(self, "_inner_object_dynamic_debug", {}) or {}
        z_dynamic_norm = float(dbg.get("long_dynamic_z_dynamic_norm", identity.get("slot_0_z_dynamic_norm", 0.0)) or 0.0)
        formed_conf = float(dbg.get("long_dynamic_confidence", identity.get("slot_0_formed_conf", 0.0)) or 0.0)
        ok = (
            identity.get("slot_0_target") == "tetrahedron"
            and not bool(identity.get("slot_0_overwritten", True))
            and float(identity.get("slot_0_formed_conf", 0.0) or 0.0) > 0.0
            and float(identity.get("slot_0_z_dynamic_norm", 0.0) or 0.0) > 0.0
            and z_dynamic_norm > 0.0
            and bool(dbg.get("dynamic_ready", False))
            and bool(dbg.get("slot_update_allowed", False))
        )
        if not ok:
            return
        self._tetra_diag_tetra_success_logged = True
        self._tetra_diag_write(
            "SUCCESS_DYNAMIC_TETRA_SLOT_FORMED",
            slot_id=0,
            target="tetrahedron",
            streak=float(dbg.get("long_dynamic_ready_streak", 0.0) or 0.0),
            dyn_eff=formed_conf,
            formed_conf=formed_conf,
            z_dynamic_norm=z_dynamic_norm,
            reason="slot0_tetra_dynamic_write_verified",
            must_verify="all_ok",
        )

    def _tetra_diag_maybe_log_cube_success(self) -> None:
        if not self._tetra_diag_enabled():
            return
        if bool(getattr(self, "_tetra_diag_cube_success_logged", False)):
            return
        checks = dict(getattr(self, "_tetra_diag_cube_checks", {}) or {})
        identity = dict(getattr(self, "_tetra_diag_slot_identity", {}) or {})
        ok = (
            bool(checks.get("attention_switch", False))
            and bool(checks.get("dynamic_input", False))
            and bool(checks.get("slot_write", False))
            and identity.get("slot_0_target") == "tetrahedron"
            and identity.get("slot_1_target") == "cube"
            and bool(identity.get("slot_1_allocated", False))
            and not bool(identity.get("slot_0_overwritten", True))
        )
        if not ok:
            return
        self._tetra_diag_cube_success_logged = True
        self._tetra_diag_write(
            "SUCCESS_DYNAMIC_CUBE_SLOT1_FORMED",
            slot_id=1,
            previous_slot_id=0,
            previous_target="tetrahedron",
            target="cube",
            streak=float(checks.get("streak", 0.0) or 0.0),
            dyn_eff=float(checks.get("dyn_eff", 0.0) or 0.0),
            formed_conf=float(checks.get("formed_conf", 0.0) or 0.0),
            z_dynamic_norm=float(checks.get("z_dynamic_norm", 0.0) or 0.0),
            reason="slot_identity_and_cube_dynamic_write_verified",
            must_verify="all_ok",
        )

    def log_tetra_inner_object_window_state(self, obj: dict | None = None, *, visible: bool, reason: str) -> None:
        if not self._tetra_diag_enabled():
            return
        identity = dict(getattr(self, "_tetra_diag_slot_identity", {}) or {})
        active_slot = self._tetra_diag_active_slot_id(obj)
        window_name = str(getattr(getattr(getattr(self, "cfg", None), "object_image", None), "window_name", "inner_object_visualizer"))
        self._tetra_diag_write(
            "inner_object_window_state",
            visible=bool(visible),
            window_name=window_name,
            active_slot=active_slot,
            slot_0_target=str(identity.get("slot_0_target", "unknown")),
            slot_1_target=str(identity.get("slot_1_target", "unknown")),
            reason=str(reason),
        )
        self._tetra_diag_cube_checks = {
            **dict(getattr(self, "_tetra_diag_cube_checks", {}) or {}),
            "inner_object_window_visible": bool(visible),
        }

    def log_tetra_post_success_monitor_tick(self) -> None:
        if not self._tetra_diag_enabled():
            return
        if not bool(getattr(self, "_tetra_diag_cube_success_logged", False)):
            return
        step = int(getattr(self, "global_step", 0))
        every = 10
        last = int(getattr(self, "_tetra_diag_last_monitor_step", -every))
        if step - last < every:
            return
        self._tetra_diag_last_monitor_step = step
        identity = dict(getattr(self, "_tetra_diag_slot_identity", {}) or {})
        checks = dict(getattr(self, "_tetra_diag_cube_checks", {}) or {})
        visible = bool(getattr(self, "show_inner_object_window", False))
        cube_alive = bool(checks.get("dynamic_alive", checks.get("dynamic_input", False)))
        if not cube_alive:
            cube_alive = (
                identity.get("slot_1_target") == "cube"
                and float(checks.get("formed_conf", 0.0) or 0.0) > 0.0
                and float(checks.get("z_dynamic_norm", 0.0) or 0.0) > 0.0
            )
        slot0_ok = identity.get("slot_0_target") == "tetrahedron" and not bool(identity.get("slot_0_overwritten", True))
        slot1_ok = identity.get("slot_1_target") == "cube" and bool(identity.get("slot_1_allocated", False))
        window_ok = visible and bool(checks.get("inner_object_window_visible", False))
        flags = dict(getattr(getattr(self, "module_training_gate", None), "flags", {}) or {})
        counts = getattr(getattr(self, "module_training_gate", None), "count_trainable", lambda: {})()
        training_enabled = self._tetra_diag_effective_training()
        updated = [
            name for name in ("world_model", "object_imagery", "long_dynamic_memory")
            if bool(training_enabled) and bool(flags.get(name, False)) and int(counts.get(name, 0) or 0) > 0
        ]
        long_dynamic_trainable = bool(
            flags.get("long_dynamic_memory", False)
            and int(counts.get("long_dynamic_memory", 0) or 0) > 0
        )
        self._tetra_diag_write(
            "post_success_train_tick",
            training_enabled=training_enabled,
            updated_modules=",".join(updated) if updated else "none",
            slot_0_target=str(identity.get("slot_0_target", "unknown")),
            slot_0_formed_conf=float(identity.get("slot_0_formed_conf", 0.0) or 0.0),
            slot_0_z_dynamic_norm=float(identity.get("slot_0_z_dynamic_norm", 0.0) or 0.0),
            slot_1_target=str(identity.get("slot_1_target", "unknown")),
            slot_1_formed_conf=float(identity.get("slot_1_formed_conf", 0.0) or 0.0),
            slot_1_z_dynamic_norm=float(identity.get("slot_1_z_dynamic_norm", 0.0) or 0.0),
            slot_0_overwritten=bool(identity.get("slot_0_overwritten", True)),
            slot_1_visible=slot1_ok,
            inner_object_window_visible=window_ok,
            cube_dynamic_alive=cube_alive,
            long_dynamic_memory_trainable=long_dynamic_trainable,
            reason="slot0_tetra_slot1_cube_monitor",
        )
        if not (slot0_ok and slot1_ok and window_ok and cube_alive):
            self._tetra_diag_write(
                "POST_SUCCESS_MONITOR_FAILED",
                failed_condition="post_success_stability",
                expected="slot0_tetra_slot1_cube_window_visible_cube_dynamic_alive",
                actual=f"slot0_ok={int(slot0_ok)},slot1_ok={int(slot1_ok)},window_ok={int(window_ok)},cube_alive={int(cube_alive)}",
                planned_fix="continue_runtime_recheck_and_repair_if_persistent",
            )

    def log_tetra_optimizer_step(self, optimizer_step: bool, reason: str | None = None) -> None:
        if not self._tetra_diag_enabled():
            return
        flags = dict(getattr(getattr(self, "module_training_gate", None), "flags", {}) or {})
        counts = getattr(getattr(self, "module_training_gate", None), "count_trainable", lambda: {})()
        updated = [
            name for name in ("world_model", "object_imagery", "long_dynamic_memory")
            if bool(optimizer_step) and bool(flags.get(name, False)) and int(counts.get(name, 0) or 0) > 0
        ]
        self._tetra_diag_write(
            "optimizer_step",
            optimizer_step=bool(optimizer_step),
            effective_training=self._tetra_diag_effective_training(),
            last_train_reason=str(reason or getattr(self, "last_train_reason", "")),
            **{
                "module_training.world_model": bool(flags.get("world_model", False)),
                "module_training.object_imagery": bool(flags.get("object_imagery", False)),
                "module_training.long_dynamic_memory": bool(flags.get("long_dynamic_memory", False)),
                "trainable_counts.world_model": int(counts.get("world_model", 0) or 0),
                "trainable_counts.object_imagery": int(counts.get("object_imagery", 0) or 0),
                "trainable_counts.long_dynamic_memory": int(counts.get("long_dynamic_memory", 0) or 0),
                "updated_modules": ",".join(updated) if updated else "none",
            },
        )

    def log_tetra_diagnosis_failed(self, failed_stage: str, expected: str, actual: str, suspected_file: str, planned_fix: str) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "DIAGNOSIS_FAILED",
            failed_stage=failed_stage,
            expected=expected,
            actual=actual,
            suspected_file=suspected_file,
            planned_fix=planned_fix,
        )

    def log_tetra_repair_applied(self, repair_stage: str, reason: str, changed_area: str) -> None:
        if not self._tetra_diag_enabled():
            return
        self._tetra_diag_write(
            "repair_applied",
            repair_stage=str(repair_stage),
            reason=str(reason),
            changed_area=str(changed_area),
        )
