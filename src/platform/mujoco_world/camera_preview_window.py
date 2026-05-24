from __future__ import annotations

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, get_cv2_last_key

from src.platform.scene_builder.realistic_hand_mjcf import hand_sensor_names


class CameraPreviewMixin:
    def _tensor_rgb_to_bgr_u8(self, x: torch.Tensor) -> np.ndarray:
        arr = x.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)


    def _tensor_depth_to_bgr_u8(
        self,
        x: torch.Tensor,
        focus_depth: float | None = None,
        focus_half_range: float | None = None,
        zero_is_valid: bool = False,
        range_suffix: str = "m",
        draw_range_label: bool = False,
    ) -> np.ndarray:
        arr = x.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        arr = arr.astype(np.float32)
        valid = np.isfinite(arr)
        if not bool(zero_is_valid):
            valid = valid & (arr > 1e-6)
        if arr.size == 0 or not bool(valid.any()):
            h = int(arr.shape[-2]) if arr.ndim >= 2 else 1
            w = int(arr.shape[-1]) if arr.ndim >= 2 else 1
            return np.zeros((h, w, 3), dtype=np.uint8)

        values = arr[valid]
        if focus_depth is not None and np.isfinite(float(focus_depth)):
            half = float(focus_half_range) if focus_half_range is not None else 0.85
            half = max(0.10, half)
            lo = max(0.0, float(focus_depth) - half)
            hi = float(focus_depth) + half
        else:
            lo = float(np.percentile(values, 2.0))
            hi = float(np.percentile(values, 98.0))
            if hi <= lo + 1e-6:
                lo = float(values.min())
                hi = float(values.max())

        clipped = np.clip(arr, lo, hi)
        norm = (clipped - lo) / max(hi - lo, 1e-6)
        norm[~valid] = 1.0
        # Nearer surfaces become brighter/warmer, making the inspected object
        # stand out from floor/background depth.
        u8 = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)

        edges = cv2.Canny(u8, 35, 110)
        color[edges > 0] = (255, 255, 255)

        if draw_range_label:
            suffix = str(range_suffix or "")
            text = f"depth {lo:.2f}-{hi:.2f}{suffix}"
            cv2.rectangle(color, (4, color.shape[0] - 24), (180, color.shape[0] - 4), (10, 10, 10), -1)
            cv2.putText(color, text, (10, color.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        return cv2.flip(color, 0)


    def _floating_depth_focus(self) -> tuple[float | None, float | None, str]:
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
            half_range = float(scenario.get("depth_focus_half_range", 0.85))
            return focus_depth, half_range, str(scenario.get("gaze_target", ""))
        except Exception:
            return None, None, ""


    def _init_sensor_preview_metadata(self):
        self.preview_left_hand_sensor_names = hand_sensor_names("left")
        self.preview_right_hand_sensor_names = hand_sensor_names("right")
        self.preview_left_foot_sensor_names = [
            "touch_left_toe_front_inner_tip",
            "touch_left_toe_front_mid_tip",
            "touch_left_toe_front_outer_tip",
            "touch_left_toe_rear_tip",
        ]
        self.preview_right_foot_sensor_names = [
            "touch_right_toe_front_inner_tip",
            "touch_right_toe_front_mid_tip",
            "touch_right_toe_front_outer_tip",
            "touch_right_toe_rear_tip",
        ]

    def _tactile_slice(self, tactile_arr: np.ndarray | None, start: int, count: int) -> list[float]:
        if tactile_arr is None or tactile_arr.size <= start:
            return [0.0] * int(count)
        out = np.zeros(int(count), dtype=np.float32)
        end = min(int(start) + int(count), int(tactile_arr.size))
        n = max(0, end - int(start))
        if n > 0:
            out[:n] = tactile_arr[int(start):end]
        return out.tolist()


    def _sensor_preview_label(self, name: str) -> str:
        return (
            str(name)
            .replace("touch_", "")
            .replace("_touch", "")
            .replace("left_", "L:")
            .replace("right_", "R:")
            .replace("_front_", ":f_")
            .replace("_inner", "in")
            .replace("_outer", "out")
            .replace("_mid", "mid")
            .replace("_rear", "rear")
            .replace("_thumb", "th")
            .replace("_index", "ix")
            .replace("_middle", "mid")
            .replace("_ring", "ring")
            .replace("_little", "lit")
            .replace("_palm", "palm")
            .replace("_mcp", "mcp")
            .replace("_pip", "pip")
            .replace("_dip", "dip")
            .replace("_tip", "tip")
        )


    def _draw_sensor_bar_panel(self, title: str, names, values, width: int, height: int, color=(80, 220, 120)) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (14, 18, 26)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 90, 110), 1)
        cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)

        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        if vals.size == 0:
            return panel

        vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
        vals = np.clip(vals, 0.0, None)
        vmax = max(float(vals.max()), 1e-6)

        left = 10
        right = width - 10
        top = 34
        bottom = height - 30
        n = len(vals)
        slot_w = max(8, int((right - left) / max(1, n)))
        bar_w = max(4, int(slot_w * 0.65))

        for i, (name, value) in enumerate(zip(names, vals)):
            x0 = left + i * slot_w + max(0, (slot_w - bar_w) // 2)
            x1 = min(x0 + bar_w, right - 1)
            h = int((bottom - top) * float(value / vmax))
            y0 = bottom - h
            cv2.rectangle(panel, (x0, y0), (x1, bottom), color, -1)
            cv2.rectangle(panel, (x0, top), (x1, bottom), (60, 70, 90), 1)

            cv2.putText(
                panel,
                f"{float(value):.2f}",
                (x0 - 2, max(top + 10, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            label = self._sensor_preview_label(name)[:10]
            cv2.putText(
                panel,
                label,
                (x0 - 2, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (180, 200, 220),
                1,
                cv2.LINE_AA,
            )

        return panel


    def _draw_imu_panel(self, title: str, labels, values, width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (12, 16, 24)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 90, 110), 1)
        cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 1, cv2.LINE_AA)

        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if vals.size == 0:
            return panel

        vmax = max(float(np.max(np.abs(vals))), 1e-6)
        left = 12
        right = width - 12
        top = 38
        bottom = height - 28
        mid = (top + bottom) // 2
        cv2.line(panel, (left, mid), (right, mid), (90, 100, 120), 1, cv2.LINE_AA)

        n = len(vals)
        slot_w = max(18, int((right - left) / max(1, n)))
        bar_w = max(8, int(slot_w * 0.58))
        scale_h = max(1, (bottom - top) // 2 - 4)

        for i, (lab, value) in enumerate(zip(labels, vals)):
            x0 = left + i * slot_w + max(0, (slot_w - bar_w) // 2)
            x1 = min(x0 + bar_w, right - 1)
            v = float(np.clip(value / vmax, -1.0, 1.0))
            h = int(abs(v) * scale_h)
            if v >= 0:
                y0, y1 = mid - h, mid
                color = (80, 220, 120)
            else:
                y0, y1 = mid, mid + h
                color = (90, 140, 255)

            cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(panel, (x0, top), (x1, bottom), (55, 65, 85), 1)
            cv2.putText(panel, f"{value:+.2f}", (x0 - 4, max(35, y0 - 4 if v >= 0 else y1 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(panel, str(lab)[:8], (x0 - 4, height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (185, 205, 225), 1, cv2.LINE_AA)
        return panel


    def _extract_imu_panels(self, obs):
        vest = obs.get("vestibular", None)
        if vest is None:
            v = np.zeros(24, dtype=np.float32)
        else:
            try:
                v = vest.detach().cpu().numpy().reshape(-1).astype(np.float32)
            except Exception:
                v = np.zeros(24, dtype=np.float32)

        if v.size < 24:
            v2 = np.zeros(24, dtype=np.float32)
            v2[:min(24, v.size)] = v[:min(24, v.size)]
            v = v2

        left = np.concatenate([v[0:3], v[3:6]])       # left gyro + accel
        right = np.concatenate([v[6:9], v[9:12]])     # right gyro + accel
        common = np.concatenate([v[12:15], v[18:21]]) # gyro common + accel common
        diff = np.concatenate([v[15:18], v[21:24]])   # gyro diff + accel diff
        labels = ["gx", "gy", "gz", "ax", "ay", "az"]
        return labels, left, right, common, diff


    def update_camera_preview_window(self, obs):
        cfg = self.cfg.camera_preview

        if not getattr(self, "camera_preview_armed", False) or not self.show_camera_preview_window:
            # Sensors window is independent from Actions window.
            # Do NOT close action outputs here; otherwise standalone Actions
            # gets destroyed/recreated every life_step.
            return
        if self.global_step % max(1, cfg.show_every_steps) != 0:
            return

        left = self._tensor_rgb_to_bgr_u8(obs["left"])
        right = self._tensor_rgb_to_bgr_u8(obs["right"])
        panels = [left, right]
        labels = ["left", "right"]

        if "depth" in obs:
            focus_applied = bool(obs.get("depth_focus_applied", False)) if isinstance(obs, dict) else False
            if focus_applied:
                panels.append(self._tensor_depth_to_bgr_u8(obs["depth"], zero_is_valid=True, range_suffix=""))
                labels.append("depth_input:focus")
            else:
                panels.append(self._tensor_depth_to_bgr_u8(obs["depth"]))
                labels.append("depth_input:raw")

        top_h = max(p.shape[0] for p in panels)
        resized = []
        for p in panels:
            if p.shape[0] != top_h:
                p = cv2.resize(p, (int(p.shape[1] * top_h / p.shape[0]), top_h), interpolation=cv2.INTER_AREA)
            resized.append(p)

        top_row = np.concatenate(resized, axis=1)
        x = 4
        for lab, p in zip(labels, resized):
            label_w = max(100, min(p.shape[1] - 8, 10 + len(str(lab)) * 8))
            cv2.rectangle(top_row, (x, 4), (x + label_w, 24), (20, 20, 20), -1)
            cv2.putText(top_row, lab, (x + 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
            x += p.shape[1]

        tactile = obs.get("tactile", None)
        tactile_arr = None
        if tactile is not None:
            tactile_arr = tactile.detach().cpu().numpy().reshape(-1).astype(np.float32)

        left_hand_vals = self._tactile_slice(tactile_arr, 0, len(self.preview_left_hand_sensor_names))
        right_hand_vals = self._tactile_slice(tactile_arr, 21, len(self.preview_right_hand_sensor_names))
        left_foot_vals = self._tactile_slice(tactile_arr, 42, len(self.preview_left_foot_sensor_names))
        right_foot_vals = self._tactile_slice(tactile_arr, 46, len(self.preview_right_foot_sensor_names))

        panel_w = max(360, top_row.shape[1] // 2)
        hand_h = 220
        foot_h = 160

        lh = self._draw_sensor_bar_panel("left hand contacts", self.preview_left_hand_sensor_names, left_hand_vals, panel_w, hand_h, color=(70, 220, 120))
        rh = self._draw_sensor_bar_panel("right hand contacts", self.preview_right_hand_sensor_names, right_hand_vals, panel_w, hand_h, color=(70, 170, 255))
        lf = self._draw_sensor_bar_panel("left foot contacts", self.preview_left_foot_sensor_names, left_foot_vals, panel_w, foot_h, color=(255, 180, 70))
        rf = self._draw_sensor_bar_panel("right foot contacts", self.preview_right_foot_sensor_names, right_foot_vals, panel_w, foot_h, color=(255, 110, 90))

        tactile_row1 = np.concatenate([lh, rh], axis=1)
        tactile_row2 = np.concatenate([lf, rf], axis=1)
        tactile_grid = np.concatenate([tactile_row1, tactile_row2], axis=0)

        imu_labels, imu_left, imu_right, imu_common, imu_diff = self._extract_imu_panels(obs)
        imu_h = 150
        imu_panel_w = max(280, top_row.shape[1] // 4)
        imu_l = self._draw_imu_panel("left IMU gyro+accel", imu_labels, imu_left, imu_panel_w, imu_h)
        imu_r = self._draw_imu_panel("right IMU gyro+accel", imu_labels, imu_right, imu_panel_w, imu_h)
        imu_c = self._draw_imu_panel("IMU common", imu_labels, imu_common, imu_panel_w, imu_h)
        imu_d = self._draw_imu_panel("IMU diff / anti-phase", imu_labels, imu_diff, imu_panel_w, imu_h)
        imu_grid = np.concatenate([imu_l, imu_r, imu_c, imu_d], axis=1)

        total_w = max(top_row.shape[1], tactile_grid.shape[1], imu_grid.shape[1])
        if top_row.shape[1] != total_w:
            top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, total_w - top_row.shape[1], cv2.BORDER_CONSTANT, value=(10, 10, 10))
        if tactile_grid.shape[1] != total_w:
            tactile_grid = cv2.copyMakeBorder(tactile_grid, 0, 0, 0, total_w - tactile_grid.shape[1], cv2.BORDER_CONSTANT, value=(10, 10, 10))
        if imu_grid.shape[1] != total_w:
            imu_grid = cv2.copyMakeBorder(imu_grid, 0, 0, 0, total_w - imu_grid.shape[1], cv2.BORDER_CONSTANT, value=(10, 10, 10))

        header = np.zeros((34, total_w, 3), dtype=np.uint8)
        header[:] = (8, 12, 18)
        cv2.putText(header, "input sensors visualizer: stereo + tactile contacts + left/right IMU", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)

        frame = np.concatenate([header, top_row, tactile_grid, imu_grid], axis=0)

        if abs(cfg.scale - 1.0) > 1e-6:
            frame = cv2.resize(frame, (int(frame.shape[1] * cfg.scale), int(frame.shape[0] * cfg.scale)), interpolation=cv2.INTER_AREA)

        submit_cv2_frame(cfg.window_name, frame, frame.shape[1], frame.shape[0])
        key = int(get_cv2_last_key())
        if key in (27, ord('q')):
            self.shutdown = True
