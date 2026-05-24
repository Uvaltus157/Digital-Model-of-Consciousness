from __future__ import annotations

"""
src/modules/m15_counterfactual_imagination_planning/small_cube_arm_reach_servo.py

V7 over V5/V6:
    Fixes "кружит вокруг кубика, рука не тянется".

Reason:
    V5/V6 mostly used body servo + almost static arm pose. If the body yaw/xy
    controller keeps orbiting the cube, palm distance may not shrink and the
    hand never really reaches.

V7 adds an explicit adaptive reach controller:
    - body yaw still tracks the cube
    - body xy is damped near the cube
    - arm extension is a state variable that increases while servo_dist is not
      improving
    - shoulder/elbow/palm controls are driven by palm->cube error
    - fingers stay open until near surface, then close while reach servo remains active
"""

import numpy as np

from src.modules.m15_counterfactual_imagination_planning.small_cube_grasp_rotate_finger_gait import (
    ARM_DIM,
    BODY_DIM,
    HAND_DIM,
    LEG_DIM,
    Hand44,
    ServoState,
    _agent,
    _approach,
    _cfg,
    _cmd,
    _contacts,
    _gaze_level,
    _servo_error,
    _small_cube_pose,
)


def _ensure_reach_state(sc) -> None:
    if not hasattr(sc, "_reach_extension"):
        sc._reach_extension = 0.0
    if not hasattr(sc, "_reach_side"):
        sc._reach_side = 0.0
    if not hasattr(sc, "_reach_height"):
        sc._reach_height = 0.0
    if not hasattr(sc, "_reach_last_dist"):
        sc._reach_last_dist = 999.0
    if not hasattr(sc, "_reach_no_improve"):
        sc._reach_no_improve = 0


def _world_to_local(vec, yaw):
    import math
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    return (
        cy * float(vec[0]) - sy * float(vec[1]),
        sy * float(vec[0]) + cy * float(vec[1]),
        float(vec[2]),
    )


def _adaptive_reach_servo(sc, err_world, yaw, dist, phase_gain=1.0):
    """
    Convert palm/tip -> cube surface error into BOTH body micro-motion and
    an adaptive arm reach state.

    Key point:
        If distance is not shrinking, extension increases.
        This makes the hand reach even when the body controller only circles.
    """
    _ensure_reach_state(sc)

    err_world = np.asarray(err_world, dtype=np.float64).reshape(3)
    lx, ly, lz = _world_to_local(err_world, yaw)

    last = float(getattr(sc, "_reach_last_dist", 999.0))
    improved = float(dist) < last - _cfg(sc, "small_cube_reach_improve_eps", 0.003)
    sc._reach_last_dist = float(dist)

    if improved:
        sc._reach_no_improve = 0
    else:
        sc._reach_no_improve = int(getattr(sc, "_reach_no_improve", 0)) + 1

    # Explicit extension drive.
    # lx is local forward error; if hand is far from cube, reach further.
    ext = float(getattr(sc, "_reach_extension", 0.0))
    ext += _cfg(sc, "small_cube_reach_forward_gain", 0.36) * float(lx) * float(phase_gain)

    # If distance stalls, push the arm farther regardless of body orbiting.
    if sc._reach_no_improve >= int(_cfg(sc, "small_cube_reach_stall_steps", 6)):
        ext += _cfg(sc, "small_cube_reach_stall_extend", 0.018) * float(phase_gain)

    # If penetrating/contacting, stop extending aggressively.
    if float(dist) < _cfg(sc, "small_cube_reach_near_dist", 0.030):
        ext -= _cfg(sc, "small_cube_reach_near_relax", 0.006)

    ext = float(np.clip(ext, 0.0, _cfg(sc, "small_cube_reach_extension_max", 1.0)))
    sc._reach_extension = ext

    side = float(getattr(sc, "_reach_side", 0.0))
    side += _cfg(sc, "small_cube_reach_side_gain", 0.18) * float(ly) * float(phase_gain)
    side = float(np.clip(side, -0.35, 0.35))
    sc._reach_side = side

    height = float(getattr(sc, "_reach_height", 0.0))
    height += _cfg(sc, "small_cube_reach_height_gain", 0.20) * float(lz) * float(phase_gain)
    height = float(np.clip(height, -0.28, 0.28))
    sc._reach_height = height

    # Body micro servo: damped, so it does not keep circling forever.
    body = np.zeros(BODY_DIM, dtype=np.float32)
    body[0] = float(np.clip(lx * _cfg(sc, "small_cube_reach_body_forward_gain", 0.18), -0.10, 0.14))
    body[1] = float(np.clip(ly * _cfg(sc, "small_cube_reach_body_side_gain", 0.10), -0.08, 0.08))
    body[2] = float(np.clip(lz * _cfg(sc, "small_cube_reach_body_z_gain", 0.18), -0.10, 0.10))

    # Arm mapping. More extension -> shoulder/elbow go forward.
    base_yaw = _cfg(sc, "small_cube_reach_arm_yaw_abs", 0.05)
    base_pitch = _cfg(sc, "small_cube_reach_arm_base_pitch", -0.55)
    base_elbow = _cfg(sc, "small_cube_reach_arm_base_elbow", -0.45)

    shoulder_pitch = base_pitch - _cfg(sc, "small_cube_reach_shoulder_extend", 0.52) * ext + height
    elbow = base_elbow - _cfg(sc, "small_cube_reach_elbow_extend", 0.62) * ext + 0.35 * height

    # Clamp to full extension range.
    shoulder_pitch = float(np.clip(shoulder_pitch, -1.0, 0.25))
    elbow = float(np.clip(elbow, -1.0, 0.20))

    left_yaw = -base_yaw + side
    right_yaw = base_yaw + side

    arm = np.asarray(
        [left_yaw, shoulder_pitch, elbow, right_yaw, shoulder_pitch, elbow],
        dtype=np.float32,
    )
    arm = np.clip(arm, -1.0, 1.0)

    diag = {
        "reach_extension": float(ext),
        "reach_side": float(side),
        "reach_height": float(height),
        "reach_no_improve": int(sc._reach_no_improve),
        "reach_lx": float(lx),
        "reach_ly": float(ly),
        "reach_lz": float(lz),
        "arm_l_shoulder": float(arm[1]),
        "arm_l_elbow": float(arm[2]),
        "arm_r_shoulder": float(arm[4]),
        "arm_r_elbow": float(arm[5]),
    }
    return body, arm, diag


def _contact_seen(sc, contact):
    return (
        float(contact.get("sum", 0.0)) >= _cfg(sc, "small_cube_adaptive_contact_sum", 0.020)
        or float(contact.get("count", 0.0)) >= _cfg(sc, "small_cube_adaptive_contact_count", 2.0)
    )


def _near_surface(sc, dist, signed):
    return (
        float(dist) <= _cfg(sc, "small_cube_open_to_close_dist", 0.035)
        or float(dist) <= _cfg(sc, "small_cube_adaptive_surface_dist_goal", 0.018)
        or float(signed) <= _cfg(sc, "small_cube_adaptive_surface_signed_goal", 0.010)
    )


def _confirmed_grasp(sc, dist, contact):
    return (
        _contact_seen(sc, contact)
        and int(getattr(sc, "_stable_contact_steps", 0)) >= int(_cfg(sc, "small_cube_chase_confirm_contact_steps", 16))
        and float(getattr(sc, "_grasp_strength", 0.0)) >= _cfg(sc, "small_cube_chase_confirm_strength", 0.72)
        and float(dist) <= _cfg(sc, "small_cube_chase_confirm_dist", 0.060)
    )


def install_small_cube_arm_reach_servo(cls) -> None:
    if getattr(cls, "_small_cube_arm_reach_servo_installed", False):
        return

    original_start = getattr(cls, "start", None)
    original_stop = getattr(cls, "stop", None)

    def start(self):
        if callable(original_start):
            try:
                original_start(self)
            except Exception:
                pass

        self.active = True
        self.phase = "approach"
        self.started_step = int(getattr(self.owner, "global_step", 0))
        self._phase_started_step = self.started_step
        self._palm_servo_state = ServoState.make(self.started_step)

        self._grasp_strength = 0.0
        self._stable_contact_steps = 0
        self._lost_contact_steps = 0
        self._reacquire_count = 0

        self._reach_extension = 0.0
        self._reach_side = 0.0
        self._reach_height = 0.0
        self._reach_last_dist = 999.0
        self._reach_no_improve = 0

        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_leg_action = np.zeros(LEG_DIM, dtype=np.float32)

        try:
            self.owner.show_action_outputs_window = True
        except Exception:
            pass

        print("[small_cube_arm_reach_v7] started: adaptive arm reach enabled")

    def step(self):
        if not getattr(self, "active", False):
            return None

        now = int(getattr(self.owner, "global_step", 0))
        age = now - int(getattr(self, "started_step", now))

        if age > int(_cfg(self, "small_cube_adaptive_timeout_steps", 5000)):
            if callable(original_stop):
                original_stop(self, "timeout")
            else:
                self.stop("timeout")
            return None

        pose = _small_cube_pose(self)
        cube = pose[0] if pose is not None else np.zeros(3, dtype=np.float64)
        agent, yaw, quat = _agent(self)

        servo = _servo_error(self)
        err = np.asarray(servo.get("err", np.zeros(3)), dtype=np.float64).reshape(3)
        dist = float(servo.get("dist", 999.0))
        signed = float(servo.get("signed", 999.0))

        contact = _contacts(self)
        contact_ok = _contact_seen(self, contact)
        near = _near_surface(self, dist, signed)
        phase = str(getattr(self, "phase", "approach"))

        if contact_ok and dist <= _cfg(self, "small_cube_chase_contact_dist_gate", 0.085):
            self._stable_contact_steps = int(getattr(self, "_stable_contact_steps", 0)) + 1
            self._lost_contact_steps = 0
        else:
            self._stable_contact_steps = max(0, int(getattr(self, "_stable_contact_steps", 0)) - 1)
            self._lost_contact_steps = int(getattr(self, "_lost_contact_steps", 0)) + 1

        lost = (
            dist > _cfg(self, "small_cube_chase_reacquire_dist", 0.095)
            and int(getattr(self, "_lost_contact_steps", 0)) > int(_cfg(self, "small_cube_chase_reacquire_after_steps", 8))
        )

        diag = {}
        approach_err = dist

        if phase == "approach":
            body, approach_err = _approach(self, agent, yaw, cube)

            # Do not keep arms dead during approach: already prepare/reach forward.
            _ensure_reach_state(self)
            prep_gain = _cfg(self, "small_cube_reach_approach_prep_gain", 0.35)
            self._reach_extension = max(float(getattr(self, "_reach_extension", 0.0)), prep_gain)
            _b2, arm, diag = _adaptive_reach_servo(self, err, yaw, dist, phase_gain=0.20)

            hand = Hand44()
            hand.open_capture()

            if approach_err < _cfg(self, "small_cube_approach_tol", 0.075):
                self.phase = "palm_servo_open"
                self._phase_started_step = now
                self._palm_servo_state = ServoState.make(now)

        elif phase == "palm_servo_open":
            body, arm, diag = _adaptive_reach_servo(self, err, yaw, dist, phase_gain=_cfg(self, "small_cube_chase_open_servo_gain", 1.15))
            hand = Hand44()
            hand.open_capture()

            # Transition only when the reach servo is close, not just because body arrived.
            if near:
                self.phase = "capture_close"
                self._phase_started_step = now
                self._grasp_strength = max(0.0, float(getattr(self, "_grasp_strength", 0.0)))

        elif phase == "capture_close":
            body, arm, diag = _adaptive_reach_servo(self, err, yaw, dist, phase_gain=_cfg(self, "small_cube_chase_close_servo_gain", 1.05))

            if not near and not contact_ok:
                # Stay open/reopen while still chasing.
                self._grasp_strength = max(
                    0.0,
                    float(getattr(self, "_grasp_strength", 0.0)) - _cfg(self, "small_cube_chase_reopen_rate", 0.010),
                )
            else:
                self._grasp_strength = min(
                    1.0,
                    float(getattr(self, "_grasp_strength", 0.0)) + _cfg(self, "small_cube_adaptive_close_rate", 0.022),
                )

            hand = Hand44()
            hand.close_capture(self._grasp_strength)

            if _confirmed_grasp(self, dist, contact):
                self.phase = "hold"
                self._phase_started_step = now

        elif phase == "hold":
            body, arm, diag = _adaptive_reach_servo(self, err, yaw, dist, phase_gain=_cfg(self, "small_cube_chase_hold_servo_gain", 0.55))
            hand = Hand44()
            hand.close_capture(1.0)

            if lost:
                self.phase = "capture_close"
                self._phase_started_step = now
                self._reacquire_count = int(getattr(self, "_reacquire_count", 0)) + 1
            elif now - int(getattr(self, "_phase_started_step", now)) > int(_cfg(self, "small_cube_adaptive_hold_steps", 40)):
                self.phase = "rotate"
                self._phase_started_step = now

        else:
            if lost or not _confirmed_grasp(self, dist, contact):
                self.phase = "capture_close"
                self._phase_started_step = now
                self._reacquire_count = int(getattr(self, "_reacquire_count", 0)) + 1
                body, arm, diag = _adaptive_reach_servo(self, err, yaw, dist, phase_gain=_cfg(self, "small_cube_chase_close_servo_gain", 1.05))
                hand = Hand44()
                hand.close_capture(float(getattr(self, "_grasp_strength", 0.75)))
            else:
                self.phase = "rotate"
                body, arm, diag = _adaptive_reach_servo(self, err, yaw, dist, phase_gain=_cfg(self, "small_cube_chase_rotate_servo_gain", 0.25))
                body[:3] *= 0.35
                hand = Hand44()
                contact_norm = float(np.clip(float(contact.get("sum", 0.0)) / 0.08, 0.0, 1.0))
                hand.rotate_gait(now - int(getattr(self, "_phase_started_step", now)), contact=contact_norm)

        gaze = _gaze_level(self, body, agent, yaw, quat, cube)

        status = {
            "active": True,
            "controller": "FlyToSmallCubeGraspRotateScenario",
            "mode": "small_cube_arm_reach_servo",
            "phase": str(getattr(self, "phase", phase)),
            "age": int(age),
            "approach_err": float(approach_err),
            "servo_dist": float(dist),
            "servo_signed": float(signed),
            "servo_site": str(servo.get("site", "")),
            "servo_cube": str(servo.get("cube", "")),
            "servo_err_x": float(err[0]),
            "servo_err_y": float(err[1]),
            "servo_err_z": float(err[2]),
            "contact_sum": float(contact.get("sum", 0.0)),
            "contact_count": float(contact.get("count", 0.0)),
            "contact_ok": bool(contact_ok),
            "near_surface": bool(near),
            "confirmed_grasp": bool(_confirmed_grasp(self, dist, contact)),
            "lost": bool(lost),
            "grasp_strength": float(getattr(self, "_grasp_strength", 0.0)),
            "stable_contact_steps": int(getattr(self, "_stable_contact_steps", 0)),
            "lost_contact_steps": int(getattr(self, "_lost_contact_steps", 0)),
            "reacquire_count": int(getattr(self, "_reacquire_count", 0)),
            "hand_dim": HAND_DIM,
        }
        status.update(gaze)
        status.update(diag)
        self.status = status

        return _cmd(self, body, arm, hand.v, status)

    cls.start = start
    cls.step = step
    cls._small_cube_arm_reach_servo_installed = True
    cls._small_cube_chase_until_grasp_v6_installed = True
    cls._small_cube_adaptive_servo_v5_installed = True

    print("[small_cube_arm_reach_v7] installed on FlyToSmallCubeGraspRotateScenario")
