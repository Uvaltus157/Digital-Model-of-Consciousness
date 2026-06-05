from __future__ import annotations

import numpy as np
import torch


class SleepSensorsMixin:
    def apply_startup_state(self, force: bool = False) -> bool:
        """
        Apply initial sensor mode from config exactly once.

        Config location:
            cfg.sleep_sensors.startup_state

        Supported values:
            "config" / "manual"  -> use explicit video/contact/imu booleans
            "active" / "awake"   -> video/contact/imu ON
            "sleep" / "dream"    -> video/contact/imu OFF
            "blind"              -> video OFF, contact/imu ON
            "body_only"          -> video/contact OFF, imu ON

        This is only a startup initializer. Runtime IPC/pult changes are not
        overwritten after startup_state has been applied.
        """
        if bool(getattr(self, "_startup_state_applied", False)) and not bool(force):
            return False

        cfg_sleep = getattr(self.cfg, "sleep_sensors", None)
        state = str(getattr(cfg_sleep, "startup_state", "config")).lower().strip()
        old = self.input_sensors_enabled_dict_no_startup_apply()
        old_sleep = (not old.get("video", True) and not old.get("contact", True) and not old.get("imu", True))

        print("SleepSensorsMixin: ------------------------> ", state)

        if state in ("", "config", "manual", "from_config"):
            # Keep explicit cfg.sleep_sensors.video/contact/imu values already
            # copied by runner.__init__.
            pass
        elif state in ("active", "awake", "run", "life"):
            self.video_sensor_enabled = True
            self.contact_sensor_enabled = True
            self.imu_sensor_enabled = True
        elif state in ("sleep", "dream", "dreaming", "full_sleep"):
            self.video_sensor_enabled = False
            self.contact_sensor_enabled = False
            self.imu_sensor_enabled = False
        elif state in ("blind", "eyes_closed"):
            self.video_sensor_enabled = False
            self.contact_sensor_enabled = True
            self.imu_sensor_enabled = True
        elif state in ("body_only", "imu_only"):
            self.video_sensor_enabled = False
            self.contact_sensor_enabled = False
            self.imu_sensor_enabled = True
        else:
            print(f"[startup_state] unknown value {state!r}; using explicit sleep_sensors booleans")

        self._startup_state_applied = True
        new = self.input_sensors_enabled_dict_no_startup_apply()
        new_sleep = (not new.get("video", True) and not new.get("contact", True) and not new.get("imu", True))
        changed = old != new
        if changed and (not old_sleep) and new_sleep:
            self._zero_prev_motor_state_on_sleep_entry()

        print(
            "[startup_state] "
            f"state={state or 'config'} "
            f"video={'ON' if new['video'] else 'OFF'} "
            f"contact={'ON' if new['contact'] else 'OFF'} "
            f"imu={'ON' if new['imu'] else 'OFF'} "
            f"sensor_state={self.sensor_state_label_no_startup_apply()}"
        )

        return changed

    def input_sensors_enabled_dict_no_startup_apply(self) -> dict:
        return {
            "video": bool(getattr(self, "video_sensor_enabled", True)),
            "contact": bool(getattr(self, "contact_sensor_enabled", True)),
            "imu": bool(getattr(self, "imu_sensor_enabled", True)),
        }

    def sensor_state_label_no_startup_apply(self) -> str:
        enabled = self.input_sensors_enabled_dict_no_startup_apply()
        if not enabled["video"] and not enabled["contact"] and not enabled["imu"]:
            return "sleep"
        if all(bool(v) for v in enabled.values()):
            return "awake"
        parts = [k for k, v in enabled.items() if not bool(v)]
        return "partial_cut:" + ",".join(parts)

    def is_full_sleep_mode(self) -> bool:
        """
        Full sleep only when ALL external input channels are cut.
        If any sensor is enabled again, especially IMU, the system is awake
        or partially deprived, but not dreaming.
        """
        self.apply_startup_state()
        return (
            not bool(getattr(self, "video_sensor_enabled", True))
            and not bool(getattr(self, "contact_sensor_enabled", True))
            and not bool(getattr(self, "imu_sensor_enabled", True))
        )

    def sensor_state_label(self) -> str:
        self.apply_startup_state()
        if self.is_full_sleep_mode():
            return "sleep"
        enabled = self.input_sensors_enabled_dict_no_startup_apply()
        if all(bool(v) for v in enabled.values()):
            return "awake"
        parts = [k for k, v in enabled.items() if not bool(v)]
        return "partial_cut:" + ",".join(parts)

    def input_sensors_enabled_dict(self) -> dict:
        self.apply_startup_state()
        return self.input_sensors_enabled_dict_no_startup_apply()


    def sleep_sensor_mask_dict(self) -> dict:
        enabled = self.input_sensors_enabled_dict()
        return {k: not bool(v) for k, v in enabled.items()}


    def _zero_prev_motor_state_on_sleep_entry(self) -> None:
        """
        Clear one-step awake motor tail when entering full sleep/replay mode.

        life_runtime reads prev_embodied_action / prev_hand_motor at the very
        beginning of the next step. Without this reset, the first sleep frame
        can still execute the last awake command before sleep_motor_guard runs.
        """
        zeroed = []
        norms = {}
        for attr in ("prev_embodied_action", "prev_hand_motor"):
            value = getattr(self, attr, None)
            if not torch.is_tensor(value):
                continue
            try:
                norms[attr] = float(value.detach().float().norm().cpu().item())
            except Exception:
                norms[attr] = 0.0
            setattr(self, attr, torch.zeros_like(value))
            zeroed.append(attr)

        self._sleep_replay_prev_motor_reset = {
            "reason": "entered_full_sleep",
            "zeroed": list(zeroed),
            "norms": dict(norms),
        }
        if zeroed:
            print(
                "[sleep_replay] zeroed previous motor tail on sleep entry: "
                + ", ".join(f"{k}={norms.get(k, 0.0):.4f}" for k in zeroed)
            )


    def apply_sleep_sensor_state(self, state: dict) -> bool:
        """
        Accepts sensor-gate IPC from pyqt_module_debug_ipc_status.py.

        Meaning:
            video/contact/imu enabled=True  -> sensor passes through
            enabled=False                  -> sensor is zeroed, sleep imitation

        Supported fields:
            input_sensors_enabled: {video, contact, imu}
            sleep_sensor_mask: {video, contact, imu}  # True means cut
            video_sensor_enabled/contact_sensor_enabled/imu_sensor_enabled
            sleep_video_cut/sleep_contact_cut/sleep_imu_cut
        """
        if not isinstance(state, dict):
            return False

        # Startup state is only an initializer. Once external control arrives,
        # do not let startup_state overwrite it later.
        self.apply_startup_state()

        old = self.input_sensors_enabled_dict_no_startup_apply()
        old_sleep = (
            not old.get("video", True)
            and not old.get("contact", True)
            and not old.get("imu", True)
        )

        enabled = state.get("input_sensors_enabled")
        if isinstance(enabled, dict):
            if "video" in enabled:
                self.video_sensor_enabled = bool(enabled["video"])
            if "contact" in enabled:
                self.contact_sensor_enabled = bool(enabled["contact"])
            if "imu" in enabled:
                self.imu_sensor_enabled = bool(enabled["imu"])

        mask = state.get("sleep_sensor_mask")
        if isinstance(mask, dict):
            if "video" in mask:
                self.video_sensor_enabled = not bool(mask["video"])
            if "contact" in mask:
                self.contact_sensor_enabled = not bool(mask["contact"])
            if "imu" in mask:
                self.imu_sensor_enabled = not bool(mask["imu"])

        if "video_sensor_enabled" in state:
            self.video_sensor_enabled = bool(state["video_sensor_enabled"])
        if "contact_sensor_enabled" in state:
            self.contact_sensor_enabled = bool(state["contact_sensor_enabled"])
        if "imu_sensor_enabled" in state:
            self.imu_sensor_enabled = bool(state["imu_sensor_enabled"])

        if "sleep_video_cut" in state:
            self.video_sensor_enabled = not bool(state["sleep_video_cut"])
        if "sleep_contact_cut" in state:
            self.contact_sensor_enabled = not bool(state["sleep_contact_cut"])
        if "sleep_imu_cut" in state:
            self.imu_sensor_enabled = not bool(state["sleep_imu_cut"])

        new = self.input_sensors_enabled_dict_no_startup_apply()
        new_sleep = (
            not new.get("video", True)
            and not new.get("contact", True)
            and not new.get("imu", True)
        )
        changed = old != new
        if changed and (not old_sleep) and new_sleep:
            self._zero_prev_motor_state_on_sleep_entry()
        if changed:
            print(
                "[ipc][sleep_sensors] "
                f"video={'ON' if new['video'] else 'OFF'} "
                f"contact={'ON' if new['contact'] else 'OFF'} "
                f"imu={'ON' if new['imu'] else 'OFF'} "
                f"state={self.sensor_state_label_no_startup_apply()}"
            )
            self.write_module_debug_status()
        return changed


    def _zero_sensor_like(self, value):
        if value is None:
            return None
        try:
            return torch.zeros_like(value)
        except Exception:
            pass
        try:
            return np.zeros_like(value)
        except Exception:
            pass
        try:
            return value * 0
        except Exception:
            return 0


    def _zero_obs_keys(self, obs: dict, keys) -> None:
        for key in keys:
            if key in obs:
                obs[key] = self._zero_sensor_like(obs[key])


    def gate_observation_for_sleep(self, obs: dict) -> dict:
        """
        Zero selected input sensors while preserving tensor shapes.

        Central rule:
            after this function, disabled MuJoCo sensor channels must NOT reach
            the world model, object model, replay, or visualizers.

        Therefore we do not only zero raw camera/contact tensors; we also zero
        helper/teacher fields that are derived from MuJoCo state and could
        otherwise leak hidden information:
            video OFF -> left/right/depth/object_state = 0
            contact OFF -> tactile/contact/... = 0
            imu OFF -> pose/body_state/vestibular/... = 0
        """
        if not isinstance(obs, dict):
            return obs
        if not bool(getattr(self.cfg.sleep_sensors, "enabled", True)):
            return obs

        self.apply_startup_state()

        video_on = bool(getattr(self, "video_sensor_enabled", True))
        contact_on = bool(getattr(self, "contact_sensor_enabled", True))
        imu_on = bool(getattr(self, "imu_sensor_enabled", True))

        if not video_on:
            # Visual channel and geometry-derived visual teacher state.
            # object_state is MuJoCo-derived scene/object info; when eyes are
            # closed it must not act as hidden vision.
            self._zero_obs_keys(obs, ("left", "right", "depth", "object_state"))

        if not contact_on:
            # All tactile/contact channels.
            self._zero_obs_keys(obs, ("tactile", "contact", "contacts", "contact_sensors"))

        if not imu_on:
            # Pose/body/vestibular channels. body_state is MuJoCo-derived and
            # should not leak into the model when IMU/body sensing is off.
            self._zero_obs_keys(
                obs,
                (
                    "pose",
                    "body_state",
                    "vestibular",
                    "imu",
                    "gyro",
                    "accel",
                    "balance_reward",
                ),
            )

        # Explicit masks for downstream modules/debug. Do not rely on zero values alone.
        obs["input_sensors_enabled"] = self.input_sensors_enabled_dict_no_startup_apply()
        obs["sleep_sensor_mask"] = self.sleep_sensor_mask_dict()
        obs["sensor_gate_applied"] = True
        return obs
