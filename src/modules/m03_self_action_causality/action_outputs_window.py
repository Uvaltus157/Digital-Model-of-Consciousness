from __future__ import annotations

import cv2
import numpy as np
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, get_cv2_last_key


class ActionOutputsMixin:
    def _action_window_name(self) -> str:
        return self.cfg.action_outputs.window_name


    def _draw_signed_bar_panel(self, title: str, labels, values, width: int, height: int,
                               color_pos=(80, 220, 120), color_neg=(90, 140, 255)) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (14, 18, 26)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 90, 110), 1)
        cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)

        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        if vals.size == 0:
            return panel

        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        vals = np.clip(vals, -1.0, 1.0)

        left = 10
        right = width - 10
        top = 34
        bottom = height - 24
        mid = (top + bottom) // 2
        cv2.line(panel, (left, mid), (right, mid), (95, 95, 120), 1, cv2.LINE_AA)

        n = len(vals)
        slot_w = max(8, int((right - left) / max(1, n)))
        bar_w = max(4, int(slot_w * 0.65))

        for i, (lab, v) in enumerate(zip(labels, vals)):
            x0 = left + i * slot_w + max(0, (slot_w - bar_w) // 2)
            x1 = min(x0 + bar_w, right - 1)
            amp = int((bottom - top) * 0.45 * abs(float(v)))
            if v >= 0:
                y0, y1 = mid - amp, mid
                color = color_pos
                text_y = max(top + 12, y0 - 4)
            else:
                y0, y1 = mid, min(bottom, mid + amp)
                color = color_neg
                text_y = min(bottom - 2, y1 + 12)
            cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(panel, (x0, top), (x1, bottom), (60, 70, 90), 1)
            cv2.putText(panel, f"{float(v):.2f}", (x0 - 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(panel, str(lab)[:8], (x0 - 2, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 200, 220), 1, cv2.LINE_AA)

        return panel


    def close_action_outputs_window(self):
        close_cv2_window(self._action_window_name())
        self._action_window_created = False


    def update_action_output_window(self):
        cfg = self.cfg.action_outputs

        if not cfg.enabled or not self.show_action_outputs_window:
            self.close_action_outputs_window()
            return

        if self.global_step % max(1, cfg.show_every_steps) != 0:
            return

        # Window creation/resize is owned by src/platform/gui/opencv_gui_thread.py.
        if not getattr(self, "_action_window_created", False):
            self._action_window_created = True

        latest = getattr(self, "latest_out", None)
        if isinstance(latest, dict) and "embodied_targets" in latest:
            emb = latest["embodied_targets"][0].detach().cpu().numpy().astype(np.float32)
        else:
            emb = self.prev_embodied_action[0].detach().cpu().numpy().astype(np.float32) if hasattr(self, "prev_embodied_action") else np.zeros((12,), dtype=np.float32)

        if isinstance(latest, dict) and "hand_ctrl" in latest:
            hand = latest["hand_ctrl"][0].detach().cpu().numpy().astype(np.float32)
        else:
            hand = self.prev_hand_motor[0].detach().cpu().numpy().astype(np.float32) if hasattr(self, "prev_hand_motor") else np.zeros((44,), dtype=np.float32)

        if isinstance(latest, dict) and "leg_ctrl" in latest:
            leg = latest["leg_ctrl"][0].detach().cpu().numpy().astype(np.float32)
        else:
            leg = self.prev_leg_motor[0].detach().cpu().numpy().astype(np.float32) if hasattr(self, "prev_leg_motor") else np.zeros((18,), dtype=np.float32)

        body_labels = ["vx", "vy", "vz", "yaw", "pitch", "roll"]
        head_labels = ["head_yaw", "head_pitch", "head_roll"]
        arm_labels = ["Lsy", "Lsp", "Lel", "Rsy", "Rsp", "Rel"]
        hand_labels_22_l = [
            "Lpr", "Lpp",
            "Lth_y", "Lth_m", "Lth_p", "Lth_d",
            "Lin_y", "Lin_m", "Lin_p", "Lin_d",
            "Lmi_y", "Lmi_m", "Lmi_p", "Lmi_d",
            "Lri_y", "Lri_m", "Lri_p", "Lri_d",
            "Lli_y", "Lli_m", "Lli_p", "Lli_d",
        ]
        hand_labels_22_r = [
            "Rpr", "Rpp",
            "Rth_y", "Rth_m", "Rth_p", "Rth_d",
            "Rin_y", "Rin_m", "Rin_p", "Rin_d",
            "Rmi_y", "Rmi_m", "Rmi_p", "Rmi_d",
            "Rri_y", "Rri_m", "Rri_p", "Rri_d",
            "Rli_y", "Rli_m", "Rli_p", "Rli_d",
        ]
        if len(hand) == 44:
            lh_labels = hand_labels_22_l
            rh_labels = hand_labels_22_r
        else:
            lh_labels = [f"L{i}" for i in range(len(hand) // 2)]
            rh_labels = [f"R{i}" for i in range(len(hand) - len(lh_labels))]
        ll_labels = ["Lhy", "Lhp", "Lkn", "Lap", "Lar", "Lti", "Ltm", "Lto", "Ltr"]
        rl_labels = ["Rhy", "Rhp", "Rkn", "Rap", "Rar", "Rti", "Rtm", "Rto", "Rtr"]

        # Layout for agent_head branch:
        # 0:5 = body vx/vy/vz/yaw/pitch, 5:10 = old arm outputs,
        # 11 = body roll, 12:15 = head yaw/pitch/roll.
        body_vals = np.zeros((6,), dtype=np.float32)
        if emb.shape[0] >= 5:
            body_vals[:5] = emb[:5]
        body_vals[5] = emb[11] if emb.shape[0] > 11 else (emb[5] if emb.shape[0] > 5 else 0.0)
        arm_vals = emb[5:11] if emb.shape[0] >= 11 else np.zeros((6,), dtype=np.float32)
        head_vals = emb[12:15] if emb.shape[0] >= 15 else np.zeros((3,), dtype=np.float32)
        split_h = len(hand) // 2
        lh_vals = hand[:split_h]
        rh_vals = hand[split_h:]
        ll_vals = leg[:9]
        rl_vals = leg[9:18]

        w = max(int(cfg.width), 1780)
        half_w = w // 2
        header_h = 86
        row1_h = 170
        row2_h = 270
        row3_h = max(160, int(cfg.height) - header_h - row1_h - row2_h)

        info = np.zeros((header_h, w, 3), dtype=np.uint8)
        info[:] = (8, 12, 18)
        cv2.putText(info, "action outputs visualizer", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 1, cv2.LINE_AA)
        try:
            action_id = int(self.latest_out["action_ids"].item()) if hasattr(self, "latest_out") and self.latest_out is not None and "action_ids" in self.latest_out else int(self.state["prev_action_ids"].item())
        except Exception:
            action_id = -1
        cv2.putText(info, f"discrete action id: {action_id}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            info,
            f"norms: body={float(np.linalg.norm(body_vals)):.3f} head={float(np.linalg.norm(head_vals)):.3f} hand={float(np.linalg.norm(hand)):.3f} leg={float(np.linalg.norm(leg)):.3f}",
            (300, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 240, 200),
            1,
            cv2.LINE_AA,
        )
        if not isinstance(getattr(self, "latest_out", None), dict):
            cv2.putText(info, "waiting for first neural step...", (300, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 120), 1, cv2.LINE_AA)

        if len(hand) == 44:
            cv2.putText(
                info,
                "hand layout: 44 = palms + 10 mcp_yaw/spread + 30 curl joints",
                (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 230, 130),
                1,
                cv2.LINE_AA,
            )

        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        if isinstance(scenario, dict) and scenario.get("active", False):
            phase = str(scenario.get("phase", ""))
            body = float(scenario.get("body_dist", 0.0))
            ring = float(scenario.get("ring_err", 0.0))
            reach = float(scenario.get("reach_distance", 0.0))
            hand = float(scenario.get("hand_dist", 0.0))
            touch = float(scenario.get("tactile_sum", 0.0))
            zerr = float(scenario.get("z_err", 0.0))
            sign = float(scenario.get("arm_side_sign", 0.0))
            cv2.putText(
                info,
                f"ADAPT: {phase}  body={body:.2f} ring={ring:.2f} reach={reach:.2f} hand={hand:.2f} touch={touch:.3f} zerr={zerr:.2f} armSign={sign:.0f}",
                (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 230, 130),
                1,
                cv2.LINE_AA,
            )

        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        if isinstance(scenario, dict) and scenario.get("active", False):
            phase = str(scenario.get("phase", ""))
            body = float(scenario.get("body_dist", 0.0))
            ring = float(scenario.get("ring_err", 0.0))
            reach = float(scenario.get("reach_distance", 0.0))
            hand = float(scenario.get("hand_dist", 0.0))
            touch = float(scenario.get("tactile_sum", 0.0))
            cv2.putText(
                info,
                f"ADAPT: {phase} body={body:.2f} ring={ring:.2f} reach={reach:.2f} hand={hand:.2f} touch={touch:.3f}",
                (720, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 220, 120),
                1,
                cv2.LINE_AA,
            )

        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        if isinstance(scenario, dict) and scenario.get("active", False):
            phase = str(scenario.get("phase", ""))
            dist = float(scenario.get("body_dist", 0.0))
            ring = float(scenario.get("ring_err", 0.0))
            hand = float(scenario.get("hand_dist", 0.0))
            touch = float(scenario.get("tactile_sum", 0.0))
            cv2.putText(
                info,
                f"IMIT elbow: {phase} body={dist:.2f} ring={ring:.2f} hand={hand:.2f} touch={touch:.3f}",
                (760, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 220, 120),
                1,
                cv2.LINE_AA,
            )

        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        if isinstance(scenario, dict) and scenario.get("active", False):
            phase = str(scenario.get("phase", ""))
            dist = float(scenario.get("body_dist", 0.0))
            hand = float(scenario.get("hand_dist", 0.0))
            touch = float(scenario.get("tactile_sum", 0.0))
            hold = int(scenario.get("contact_hold_steps", 0))
            cv2.putText(
                info,
                f"IMIT fly_to_cube: {phase} dist={dist:.2f} hand={hand:.2f} touch={touch:.3f} hold={hold}",
                (760, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 220, 120),
                1,
                cv2.LINE_AA,
            )

        third_w = w // 3
        p1 = self._draw_signed_bar_panel("body / rig actions", body_labels, body_vals, third_w, row1_h)
        p2 = self._draw_signed_bar_panel("agent head actions", head_labels, head_vals, third_w, row1_h, color_pos=(120, 230, 255), color_neg=(255, 150, 220))
        p2b = self._draw_signed_bar_panel("shoulder / elbow actions", arm_labels, arm_vals, w - 2 * third_w, row1_h)
        row1 = np.concatenate([p1, p2, p2b], axis=1)

        p3 = self._draw_signed_bar_panel("left hand actions", lh_labels, lh_vals, half_w, row2_h, color_pos=(70, 220, 120), color_neg=(90, 150, 255))
        p4 = self._draw_signed_bar_panel("right hand actions", rh_labels, rh_vals, w - half_w, row2_h, color_pos=(70, 170, 255), color_neg=(255, 140, 120))
        row2 = np.concatenate([p3, p4], axis=1)

        p5 = self._draw_signed_bar_panel("left leg actions", ll_labels, ll_vals, half_w, row3_h, color_pos=(255, 180, 70), color_neg=(255, 120, 70))
        p6 = self._draw_signed_bar_panel("right leg actions", rl_labels, rl_vals, w - half_w, row3_h, color_pos=(255, 110, 90), color_neg=(210, 140, 255))
        row3 = np.concatenate([p5, p6], axis=1)

        frame = np.concatenate([info, row1, row2, row3], axis=0)

        if abs(cfg.scale - 1.0) > 1e-6:
            frame = cv2.resize(frame, (int(frame.shape[1] * cfg.scale), int(frame.shape[0] * cfg.scale)), interpolation=cv2.INTER_AREA)

        submit_cv2_frame(self._action_window_name(), frame, frame.shape[1], frame.shape[0])
        key = int(get_cv2_last_key())
        if key in (27, ord('q')):
            self.show_action_outputs_window = False
            self.close_action_outputs_window()
