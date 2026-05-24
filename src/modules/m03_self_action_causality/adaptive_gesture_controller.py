from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


HAND_DIM = 44
ARM_DIM = 6
BODY_DIM = 9
LEG_DIM = 18

FINGERS = ("thumb", "index", "middle", "ring", "little")
SIDES = ("left", "right")


@dataclass
class GestureCommand:
    body: np.ndarray
    arm: np.ndarray
    hand: np.ndarray
    leg: np.ndarray
    status: Dict


def _vec(value, dim: int, fill: float = 0.0) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros(0, dtype=np.float32)

    if arr.size == dim:
        return arr.copy()

    out = np.full(dim, fill, dtype=np.float32)
    n = min(dim, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


class Hand44:
    """
    44 hand controls.

    Per hand:
        palm_roll, palm_pitch + 5 * (mcp_yaw, mcp, pip, dip)

    Both hands:
        left 22 + right 22 = 44
    """

    def __init__(self, values: Optional[np.ndarray] = None):
        self.v = _vec(values, HAND_DIM, fill=0.5)

    @staticmethod
    def side_base(side: str) -> int:
        if side == "left":
            return 0
        if side == "right":
            return 22
        raise KeyError(side)

    @staticmethod
    def finger_index(finger: str) -> int:
        return FINGERS.index(finger)

    def palm_roll(self, side: str, value: float) -> None:
        self.v[self.side_base(side) + 0] = float(value)

    def palm_pitch(self, side: str, value: float) -> None:
        self.v[self.side_base(side) + 1] = float(value)

    def finger_base(self, side: str, finger: str) -> int:
        return self.side_base(side) + 2 + 4 * self.finger_index(finger)

    def mcp_yaw(self, side: str, finger: str, value: float) -> None:
        self.v[self.finger_base(side, finger) + 0] = float(value)

    def finger(self, side: str, finger: str, mcp: float, pip: float, dip: float) -> None:
        b = self.finger_base(side, finger)
        self.v[b + 1] = float(mcp)
        self.v[b + 2] = float(pip)
        self.v[b + 3] = float(dip)

    def all_fingers(self, value: float) -> None:
        for side in SIDES:
            for finger in FINGERS:
                self.finger(side, finger, value, value, value)

    def all_yaw_neutral(self) -> None:
        for side in SIDES:
            for finger in FINGERS:
                self.mcp_yaw(side, finger, 0.5)

    def open_palms(self) -> None:
        self.v[:] = 0.5
        for side in SIDES:
            self.palm_roll(side, 0.5)
            self.palm_pitch(side, 0.5)
            for finger in FINGERS:
                self.mcp_yaw(side, finger, 0.5)
                self.finger(side, finger, 0.05, 0.05, 0.05)

    def splay_coplanar(self) -> None:
        """
        True spread: uses real xxx_mcp_yaw joints.
        Almost one plane: curl is tiny and almost equal.
        """
        yaw_left = {"thumb": 0.08, "index": 0.25, "middle": 0.50, "ring": 0.75, "little": 0.92}
        yaw_right = {"thumb": 0.92, "index": 0.75, "middle": 0.50, "ring": 0.25, "little": 0.08}

        self.v[:] = 0.5
        self.palm_roll("left", 0.40)
        self.palm_pitch("left", 0.50)
        self.palm_roll("right", 0.60)
        self.palm_pitch("right", 0.50)

        for side, yaw_map in (("left", yaw_left), ("right", yaw_right)):
            for finger in FINGERS:
                self.mcp_yaw(side, finger, yaw_map[finger])

            self.finger(side, "thumb", 0.08, 0.03, 0.02)
            for finger in ("index", "middle", "ring", "little"):
                self.finger(side, finger, 0.015, 0.010, 0.010)

    def soft_curl(self) -> None:
        self.all_yaw_neutral()
        for side in SIDES:
            self.palm_roll(side, 0.5)
            self.palm_pitch(side, 0.55)
        self.all_fingers(0.55)

    def strong_fist(self) -> None:
        self.all_yaw_neutral()
        for side in SIDES:
            self.palm_roll(side, 0.5)
            self.palm_pitch(side, 0.62)
        self.all_fingers(0.95)

    def pinch(self, side: str) -> None:
        for finger in FINGERS:
            self.mcp_yaw(side, finger, 0.5)
            self.finger(side, finger, 0.08, 0.08, 0.08)
        self.finger(side, "thumb", 0.82, 0.82, 0.82)
        self.finger(side, "index", 0.78, 0.78, 0.78)
        self.finger(side, "middle", 0.20, 0.20, 0.20)
        self.finger(side, "ring", 0.12, 0.12, 0.12)
        self.finger(side, "little", 0.10, 0.10, 0.10)
        self.palm_pitch(side, 0.55)

    def point_index(self, side: str) -> None:
        for finger in FINGERS:
            self.finger(side, finger, 0.85, 0.85, 0.85)
            self.mcp_yaw(side, finger, 0.5)
        self.finger(side, "index", 0.05, 0.05, 0.05)
        self.finger(side, "thumb", 0.35, 0.35, 0.35)

    def peace(self, side: str) -> None:
        for finger in FINGERS:
            self.finger(side, finger, 0.85, 0.85, 0.85)
            self.mcp_yaw(side, finger, 0.5)
        self.finger(side, "index", 0.05, 0.05, 0.05)
        self.finger(side, "middle", 0.05, 0.05, 0.05)
        self.finger(side, "thumb", 0.45, 0.45, 0.45)
        self.mcp_yaw(side, "index", 0.34 if side == "left" else 0.66)
        self.mcp_yaw(side, "middle", 0.66 if side == "left" else 0.34)

    def thumbs_up_both(self) -> None:
        self.all_yaw_neutral()
        for side in SIDES:
            for finger in FINGERS:
                self.finger(side, finger, 0.85, 0.85, 0.85)
            self.finger(side, "thumb", 0.05, 0.05, 0.05)
            self.palm_pitch(side, 0.35)

    def cup_palms(self) -> None:
        self.all_yaw_neutral()
        for side in SIDES:
            self.palm_roll(side, 0.5)
            self.palm_pitch(side, 0.62)
            self.finger(side, "thumb", 0.48, 0.48, 0.48)
            self.finger(side, "index", 0.55, 0.55, 0.55)
            self.finger(side, "middle", 0.60, 0.60, 0.60)
            self.finger(side, "ring", 0.62, 0.62, 0.62)
            self.finger(side, "little", 0.65, 0.65, 0.65)

    def power_grasp(self) -> None:
        self.all_yaw_neutral()
        for side in SIDES:
            self.palm_pitch(side, 0.62)
            self.finger(side, "thumb", 0.72, 0.72, 0.72)
            self.finger(side, "index", 0.88, 0.88, 0.88)
            self.finger(side, "middle", 0.92, 0.92, 0.92)
            self.finger(side, "ring", 0.92, 0.92, 0.92)
            self.finger(side, "little", 0.90, 0.90, 0.90)


class AdaptiveGestureController:
    """
    High-level gesture controller.

    PyQt only sends action commands:
        gesture_splay_fingers, gesture_open_palms, ...

    This controller converts those commands to manual body/arm/hand/leg actions.
    """

    def __init__(self, owner):
        self.owner = owner
        self.active_command: Optional[str] = None
        self.status: Dict = {}

    def start(self, command: str) -> None:
        self.active_command = str(command)
        self.apply(command)
        print(f"[adaptive_gesture] applied {command}")

    def stop(self, reason: str = "stopped") -> None:
        self.active_command = None
        self.status = {"active": False, "reason": reason}

    def update(self) -> None:
        return

    def _current_body(self) -> np.ndarray:
        return _vec(getattr(self.owner, "_ipc_manual_body_action", None), BODY_DIM, fill=0.0)

    def _current_arm(self) -> np.ndarray:
        return _vec(getattr(self.owner, "_ipc_manual_arm_action", None), ARM_DIM, fill=0.0)

    def _current_hand(self) -> np.ndarray:
        return _vec(getattr(self.owner, "_ipc_manual_hand_action", None), HAND_DIM, fill=0.5)

    def _current_leg(self) -> np.ndarray:
        return _vec(getattr(self.owner, "_ipc_manual_leg_action", None), LEG_DIM, fill=0.0)

    def build(self, command: str) -> GestureCommand:
        body = self._current_body()
        arm = self._current_arm()
        leg = self._current_leg()
        hand = Hand44(self._current_hand())

        command = str(command)

        if command == "gesture_neutral_hands":
            hand.v[:] = 0.5
        elif command == "gesture_open_palms":
            hand.open_palms()
        elif command == "gesture_splay_fingers":
            hand.splay_coplanar()
        elif command == "gesture_soft_curl":
            hand.soft_curl()
        elif command == "gesture_strong_fist":
            hand.strong_fist()
        elif command == "gesture_pinch_left":
            hand.pinch("left")
        elif command == "gesture_pinch_right":
            hand.pinch("right")
        elif command == "gesture_point_index_left":
            hand.point_index("left")
        elif command == "gesture_point_index_right":
            hand.point_index("right")
        elif command == "gesture_peace_left":
            hand.peace("left")
        elif command == "gesture_peace_right":
            hand.peace("right")
        elif command == "gesture_thumbs_up_both":
            hand.thumbs_up_both()
        elif command == "gesture_cup_palms":
            hand.cup_palms()
        elif command == "gesture_precision_grasp":
            hand.pinch("left")
            hand.pinch("right")
        elif command == "gesture_power_grasp":
            hand.power_grasp()
        elif command == "gesture_reach_forward":
            arm[:] = np.asarray([-0.15, 0.45, -0.25, 0.15, 0.45, -0.25], dtype=np.float32)
            hand.open_palms()
        elif command == "gesture_arms_open":
            arm[:] = np.asarray([-0.85, 0.15, -0.35, 0.85, 0.15, -0.35], dtype=np.float32)
        elif command == "gesture_hands_to_center":
            arm[:] = np.asarray([0.35, 0.25, 0.35, -0.35, 0.25, 0.35], dtype=np.float32)
            hand.cup_palms()
        elif command == "gesture_touch_object_pose":
            arm[:] = np.asarray([0.0, 0.0, 0.0, -0.15, 0.55, -0.10], dtype=np.float32)
            hand.point_index("right")
            for finger in FINGERS:
                hand.finger("left", finger, 0.45, 0.45, 0.45)
        else:
            raise KeyError(f"unknown adaptive gesture command: {command}")

        return GestureCommand(
            body=body.astype(np.float32),
            arm=arm.astype(np.float32),
            hand=np.clip(hand.v, 0.0, 1.0).astype(np.float32),
            leg=leg.astype(np.float32),
            status={
                "active": True,
                "controller": "AdaptiveGestureController",
                "command": command,
                "hand_dim": HAND_DIM,
                "arm_dim": ARM_DIM,
            },
        )

    def apply(self, command: str) -> None:
        cmd = self.build(command)

        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_body_action = cmd.body
        self.owner._ipc_manual_arm_action = cmd.arm
        self.owner._ipc_manual_hand_action = cmd.hand
        self.owner._ipc_manual_leg_action = cmd.leg

        self.owner._adaptive_gesture_status = dict(cmd.status)
        self.status = dict(cmd.status)

        try:
            self.owner.show_action_outputs_window = True
        except Exception:
            pass


def ensure_adaptive_gesture_controller(owner) -> AdaptiveGestureController:
    ctrl = getattr(owner, "adaptive_gesture_controller", None)
    if ctrl is None:
        ctrl = AdaptiveGestureController(owner)
        owner.adaptive_gesture_controller = ctrl
    return ctrl


def is_adaptive_gesture_command(action: str) -> bool:
    return str(action).startswith("gesture_")
