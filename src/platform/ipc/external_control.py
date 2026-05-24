from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.platform.gui.opencv_gui_thread import close_cv2_window


class ExternalControlMixin:

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

    def _external_control_path(self):
        return Path(self.cfg.external_control.state_file)


    def _write_initial_external_control_flags(self):
        if not self.cfg.external_control.enabled:
            return
        path = self._external_control_path()
        if path.exists():
            return
        data = {
            "mujoco_next_run": bool(self.cfg.viewer.allow_mujoco_window),
            "inner_world": bool(self.show_inner_world_window),
            "cameras": bool(self.show_camera_preview_window),
            "depth": bool(self.cfg.camera_preview.show_depth),
            "actions": bool(self.show_action_outputs_window),
            "manual_actions": bool(self.show_manual_action_override_window),
            "object_image": bool(self.show_inner_object_window),
            "event_code_visualizer": bool(getattr(self, "show_event_code_visualizer_window", False)),
            "object_image_open3d": bool(self.show_inner_object_open3d_window),
            "training": bool(self.training_enabled),
            "input_sensors_enabled": self.input_sensors_enabled_dict() if hasattr(self, "input_sensors_enabled_dict") else {"video": True, "contact": True, "imu": True},
            "sleep_sensor_mask": self.sleep_sensor_mask_dict() if hasattr(self, "sleep_sensor_mask_dict") else {"video": False, "contact": False, "imu": False},
            "latent_semantic": bool(getattr(self, "show_latent_semantic_window", False)),
            "static_dynamic_code": bool(getattr(self, "show_static_dynamic_code_window", False)),
            "close_aux_counter": 0,
            "stop": False,
            "updated_at": time.time(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[external_control] cannot write initial flags: {e}")


    def poll_external_control_flags(self):
        if not self.cfg.external_control.enabled:
            return
        if self.global_step % max(1, self.cfg.external_control.poll_every_steps) != 0:
            return

        path = self._external_control_path()
        if not path.exists():
            self._write_initial_external_control_flags()
            return

        try:
            mtime = path.stat().st_mtime
            if mtime == self.external_control_last_mtime:
                return
            self.external_control_last_mtime = mtime

            data = json.loads(path.read_text(encoding="utf-8"))

            # Sensor sleep/awake controls may come from the JSON pult too.
            # This keeps video/contact/imu flags synchronized with IPC.
            try:
                self.apply_sleep_sensor_state(data)
            except Exception as e:
                print(f"[external_control] sleep sensor state failed: {e}")

            self.cfg.viewer.allow_mujoco_window = bool(data.get("mujoco_next_run", self.cfg.viewer.allow_mujoco_window))
            self.show_inner_world_window = bool(data.get("inner_world", self.show_inner_world_window))
            self.show_camera_preview_window = bool(data.get("cameras", self.show_camera_preview_window))
            # _external_control_camera_preview_close_when_off
            if not bool(self.show_camera_preview_window):
                try:
                    close_cv2_window(self.cfg.camera_preview.window_name)
                except Exception:
                    pass
            self.cfg.camera_preview.show_depth = bool(data.get("depth", self.cfg.camera_preview.show_depth))
            self.show_action_outputs_window = bool(data.get("actions", self.show_action_outputs_window))
            self.show_module_debug_window = bool(data.get("module_debug", self.show_module_debug_window))
            if isinstance(data.get("module_training", None), dict):
                self.set_module_training_flags(data["module_training"])
            #self.show_manual_action_override_window = bool(data.get("manual_actions", self.show_manual_action_override_window))
            self.show_inner_object_window = bool(data.get("object_image", self.show_inner_object_window))
            self.show_event_code_visualizer_window = bool(data.get("event_code_visualizer", getattr(self, "show_event_code_visualizer_window", False)))
            self.show_inner_object_open3d_window = bool(data.get("object_image_open3d", self.show_inner_object_open3d_window))
            self.training_enabled = bool(data.get("training", self.training_enabled))
            self.show_latent_semantic_window = bool(data.get("latent_semantic", self.show_latent_semantic_window))

            # _external_control_close_off_toggles
            if not bool(self.show_inner_world_window):
                self._close_inner_world_gui_windows()
            if not bool(self.show_latent_semantic_window):
                self._close_latent_semantic_gui_window()
            if not bool(getattr(self, "show_event_code_visualizer_window", False)):
                try:
                    if hasattr(self, "event_code_viz") and self.event_code_viz is not None:
                        self.event_code_viz.close()
                except Exception:
                    pass
                try:
                    close_cv2_window(self.cfg.event_code_visualizer.window_name)
                except Exception:
                    pass

            self.show_static_dynamic_code_window = bool(data.get("static_dynamic_code", getattr(self, "show_static_dynamic_code_window", False)))
            if not self.show_static_dynamic_code_window:
                try:
                    self.static_dynamic_code_viz.close()
                except Exception:
                    pass

            close_counter = int(data.get("close_aux_counter", 0))
            if close_counter != self.external_control_last_close_counter:
                self.external_control_last_close_counter = close_counter
                self.show_inner_world_window = False
                self.show_camera_preview_window = False
                self.show_latent_semantic_window = False
                self.show_event_code_visualizer_window = False
                # _external_control_close_aux_gui_thread
                self._close_inner_world_gui_windows()
                self._close_latent_semantic_gui_window()
                try:
                    close_cv2_window(self.cfg.camera_preview.window_name)
                    close_cv2_window(self._action_window_name())
                    self._action_window_created = False
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
                try:
                    if hasattr(self, "event_code_viz") and self.event_code_viz is not None:
                        self.event_code_viz.close()
                except Exception:
                    pass
                try:
                    close_cv2_window(self.cfg.event_code_visualizer.window_name)
                except Exception:
                    pass

            if bool(data.get("stop", False)):
                self.shutdown = True

        except Exception as e:
            print(f"[external_control] read failed: {e}")
