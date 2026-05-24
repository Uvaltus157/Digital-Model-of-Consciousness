from __future__ import annotations

"""
src/modules/m15_counterfactual_imagination_planning/small_cube_adaptive_chase.py

V6 add-on over V5:
    chase the small cube until it is really grasped.

Problem fixed:
    if the cube touches the palm, V5 may treat it as "done/ready" and stop
    searching. V6 keeps the palm/tip servo active during close/hold/rotate and
    reacquires the cube whenever contact is not stable.

Main rule:
    palm contact is NOT enough.
    The scenario keeps chasing until:
        grasp_strength is high
        and contact is stable for several steps
        and servo distance is still near the cube.

It still uses the V5 helper functions:
    _servo_error(), _servo_cmd(), _contacts(), _gaze_level(), Hand44, ...
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
    _servo_cmd,
    _servo_error,
    _small_cube_pose,
)


def _contact_seen(sc, contact: dict) -> bool:
    return (
        float(contact.get("sum", 0.0)) >= _cfg(sc, "small_cube_adaptive_contact_sum", 0.020)
        or float(contact.get("count", 0.0)) >= _cfg(sc, "small_cube_adaptive_contact_count", 2.0)
    )


def _near_surface(sc, dist: float, signed: float) -> bool:
    return (
        float(dist) <= _cfg(sc, "small_cube_open_to_close_dist", 0.035)
        or float(dist) <= _cfg(sc, "small_cube_adaptive_surface_dist_goal", 0.018)
        or float(signed) <= _cfg(sc, "small_cube_adaptive_surface_signed_goal", 0.010)
    )


def _confirmed_grasp(sc, dist: float, contact: dict) -> bool:
    stable = int(getattr(sc, "_stable_contact_steps", 0))
    strength = float(getattr(sc, "_grasp_strength", 0.0))
    return (
        _contact_seen(sc, contact)
        and stable >= int(_cfg(sc, "small_cube_chase_confirm_contact_steps", 16))
        and strength >= _cfg(sc, "small_cube_chase_confirm_strength", 0.72)
        and float(dist) <= _cfg(sc, "small_cube_chase_confirm_dist", 0.060)
    )


def install_small_cube_chase_until_grasp(cls) -> None:
    if getattr(cls, "_small_cube_chase_until_grasp_v6_installed", False):
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

        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_leg_action = np.zeros(LEG_DIM, dtype=np.float32)

        try:
            self.owner.show_action_outputs_window = True
        except Exception:
            pass

        print("[small_cube_chase_v6] started: chase until real grasp")

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

        # Track contact stability separately. Palm touch alone may appear and
        # disappear, so do not stop chasing from a single contact.
        if contact_ok and dist <= _cfg(self, "small_cube_chase_contact_dist_gate", 0.085):
            self._stable_contact_steps = int(getattr(self, "_stable_contact_steps", 0)) + 1
            self._lost_contact_steps = 0
        else:
            self._stable_contact_steps = max(0, int(getattr(self, "_stable_contact_steps", 0)) - 1)
            self._lost_contact_steps = int(getattr(self, "_lost_contact_steps", 0)) + 1

        # Reacquire rule: if cube slips or the servo distance grows, go back
        # to active close/chase instead of freezing.
        lost = (
            dist > _cfg(self, "small_cube_chase_reacquire_dist", 0.095)
            and int(getattr(self, "_lost_contact_steps", 0)) > int(_cfg(self, "small_cube_chase_reacquire_after_steps", 8))
        )

        diag = {}
        approach_err = dist

        if phase == "approach":
            body, approach_err = _approach(self, agent, yaw, cube)
            arm = np.zeros(ARM_DIM, dtype=np.float32)
            hand = Hand44()
            hand.open_capture()

            if approach_err < _cfg(self, "small_cube_approach_tol", 0.075):
                self.phase = "palm_servo_open"
                self._phase_started_step = now
                self._palm_servo_state = ServoState.make(now)

        elif phase == "palm_servo_open":
            # Fingers fully open. Chase the cube with nearest palm/tip site.
            body, arm, diag = _servo_cmd(self, err, yaw, _cfg(self, "small_cube_chase_open_servo_gain", 1.15))
            hand = Hand44()
            hand.open_capture()

            if near:
                self.phase = "capture_close"
                self._phase_started_step = now
                # keep previous strength if reacquiring, but never below 0
                self._grasp_strength = max(0.0, float(getattr(self, "_grasp_strength", 0.0)))

        elif phase == "capture_close":
            # This is the main adaptive part: while fingers close, the servo
            # keeps chasing the cube. Touching the palm does not stop the chase.
            body, arm, diag = _servo_cmd(self, err, yaw, _cfg(self, "small_cube_chase_close_servo_gain", 1.05))

            if not near and not contact_ok:
                # Keep fingers open enough while we are still chasing.
                self._grasp_strength = max(0.0, float(getattr(self, "_grasp_strength", 0.0)) - _cfg(self, "small_cube_chase_reopen_rate", 0.010))
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
            # Still servo; do not freeze after first contact.
            body, arm, diag = _servo_cmd(self, err, yaw, _cfg(self, "small_cube_chase_hold_servo_gain", 0.55))
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
            # Rotate only while grasp is stable. If cube slips, chase again.
            if lost or not _confirmed_grasp(self, dist, contact):
                self.phase = "capture_close"
                self._phase_started_step = now
                self._reacquire_count = int(getattr(self, "_reacquire_count", 0)) + 1
                body, arm, diag = _servo_cmd(self, err, yaw, _cfg(self, "small_cube_chase_close_servo_gain", 1.05))
                hand = Hand44()
                hand.close_capture(float(getattr(self, "_grasp_strength", 0.75)))
            else:
                self.phase = "rotate"
                body, arm, diag = _servo_cmd(self, err, yaw, _cfg(self, "small_cube_chase_rotate_servo_gain", 0.25))
                body[:3] *= 0.35
                hand = Hand44()
                contact_norm = float(np.clip(float(contact.get("sum", 0.0)) / 0.08, 0.0, 1.0))
                hand.rotate_gait(now - int(getattr(self, "_phase_started_step", now)), contact=contact_norm)

        gaze = _gaze_level(self, body, agent, yaw, quat, cube)

        status = {
            "active": True,
            "controller": "FlyToSmallCubeGraspRotateScenario",
            "mode": "small_cube_chase_until_grasp_v6",
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
    cls._small_cube_chase_until_grasp_v6_installed = True
    cls._small_cube_adaptive_servo_v5_installed = True

    print("[small_cube_chase_v6] installed on FlyToSmallCubeGraspRotateScenario")
