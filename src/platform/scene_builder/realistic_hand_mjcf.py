from __future__ import annotations

"""
realistic_hand_mjcf.py

MJCF generator for realistic 5-finger hands.

One hand:
- palm: 2 DOF, roll + pitch
- 5 fingers
- 4 joints per finger: MCP yaw, MCP, PIP, DIP
- simple capsule phalanges
- touch sensors on palm and finger joints/tips
- position actuators for all hand joints

Per hand: 17 DOF. Both hands: 44 DOF.
"""

FINGERS = ["thumb", "index", "middle", "ring", "little"]


def _finger_base_position(side: str, finger: str) -> tuple[float, float, float]:
    if finger == "thumb":
        y = -0.070 if side == "left" else 0.070
        return 0.035, y, -0.010
    if finger == "index":
        if side == "left":
          return 0.060, -0.040, -0.035
        else:
          return 0.045, 0.050, -0.030   
    if finger == "middle":
        if side == "left":
          return 0.065, -0.010, -0.038
        else:
          return 0.058, 0.020, -0.036  
    if finger == "ring":
        if side == "left":
          return 0.058, 0.020, -0.036
        else:
          return 0.065, -0.010, -0.038         
    if finger == "little":
        if side == "left":
          return 0.045, 0.050, -0.030
        else:
          return 0.060, -0.040, -0.035  
    raise KeyError(finger)


def _finger_lengths(finger: str) -> tuple[float, float, float]:
    if finger == "thumb":
        return 0.070, 0.055, 0.045
    if finger == "middle":
        return 0.085, 0.065, 0.050
    if finger == "index":
        return 0.080, 0.060, 0.045
    if finger == "ring":
        return 0.076, 0.058, 0.043
    if finger == "little":
        return 0.062, 0.050, 0.038
    raise KeyError(finger)


def make_finger_xml(side: str, finger: str, rgba: str) -> str:
    bx, by, bz = _finger_base_position(side, finger)
    l1, l2, l3 = _finger_lengths(finger)
    radius = 0.020 if finger == "thumb" else 0.018

    if finger == "thumb":
        ydir = -0.030 if side == "left" else 0.030
        seg1 = f"{l1:.3f} {ydir:.3f} {-l1 * 0.55:.3f}"
        seg2 = f"{l2:.3f} {ydir * 0.5:.3f} {-l2 * 0.55:.3f}"
        seg3 = f"{l3:.3f} 0 {-l3 * 0.50:.3f}"
        p2 = f"{l1:.3f} {ydir:.3f} {-l1 * 0.55:.3f}"
        p3 = f"{l2:.3f} {ydir * 0.5:.3f} {-l2 * 0.55:.3f}"
        p4 = f"{l3:.3f} 0 {-l3 * 0.50:.3f}"
    else:
        seg1 = f"{l1:.3f} 0 {-l1 * 0.35:.3f}"
        seg2 = f"{l2:.3f} 0 {-l2 * 0.35:.3f}"
        seg3 = f"{l3:.3f} 0 {-l3 * 0.35:.3f}"
        p2 = f"{l1:.3f} 0 {-l1 * 0.35:.3f}"
        p3 = f"{l2:.3f} 0 {-l2 * 0.35:.3f}"
        p4 = f"{l3:.3f} 0 {-l3 * 0.35:.3f}"

    return f'''
            <body name="{side}_{finger}_mcp_body" pos="{bx:.3f} {by:.3f} {bz:.3f}">
              <joint name="{side}_{finger}_mcp_yaw" type="hinge" axis="0 0 1" range="-28 28" damping="0.45" armature="0.002"/>
              <joint name="{side}_{finger}_mcp" type="hinge" axis="0 1 0" range="-20 90" damping="0.55" armature="0.003"/>
              <geom name="{side}_{finger}_proximal" type="capsule" fromto="0 0 0 {seg1}" size="{radius:.3f}" mass="0.025" rgba="{rgba}" contype="1" conaffinity="1"/>
              <site name="{side}_{finger}_mcp_site" pos="0 0 0" size="0.010" rgba="{rgba}"/>
              <body name="{side}_{finger}_pip_body" pos="{p2}">
                <joint name="{side}_{finger}_pip" type="hinge" axis="0 1 0" range="0 100" damping="0.45" armature="0.002"/>
                <geom name="{side}_{finger}_middle" type="capsule" fromto="0 0 0 {seg2}" size="{radius * 0.88:.3f}" mass="0.020" rgba="{rgba}" contype="1" conaffinity="1"/>
                <site name="{side}_{finger}_pip_site" pos="0 0 0" size="0.010" rgba="{rgba}"/>
                <body name="{side}_{finger}_dip_body" pos="{p3}">
                  <joint name="{side}_{finger}_dip" type="hinge" axis="0 1 0" range="0 80" damping="0.35" armature="0.001"/>
                  <geom name="{side}_{finger}_distal" type="capsule" fromto="0 0 0 {seg3}" size="{radius * 0.78:.3f}" mass="0.015" rgba="{rgba}" contype="1" conaffinity="1"/>
                  <site name="{side}_{finger}_dip_site" pos="0 0 0" size="0.010" rgba="{rgba}"/>
                  <site name="{side}_{finger}_tip_site" pos="{p4}" size="0.014" rgba="{rgba}"/>
                </body>
              </body>
            </body>'''


def make_hand_body_xml(side: str, rgba: str, palm_pos: str = "0 0 -0.40") -> str:
    fingers_xml = "\n".join(make_finger_xml(side, f, rgba) for f in FINGERS)
    return f'''
          <body name="{side}_hand_base" pos="{palm_pos}">
            <joint name="{side}_palm_roll" type="hinge" axis="1 0 0" range="-35 35" damping="0.7" armature="0.006"/>
            <joint name="{side}_palm_pitch" type="hinge" axis="0 1 0" range="-35 35" damping="0.7" armature="0.006"/>
            <geom name="{side}_palm" type="box" pos="0 0 -0.015" size="0.095 0.060 0.025" mass="0.18" rgba="{rgba}" contype="1" conaffinity="1"/>
            <site name="{side}_hand_site" pos="0.045 0 -0.015" size="0.025" rgba="{rgba}"/>
            <site name="{side}_palm_center_site" pos="0.030 0 -0.040" size="0.015" rgba="{rgba}"/>
{fingers_xml}
          </body>'''


def make_hand_actuators_xml(side: str) -> str:
    items = [
        f'    <position name="act_{side}_palm_roll" joint="{side}_palm_roll" kp="80" ctrlrange="-0.610865 0.610865" forcerange="-50 50"/>',
        f'    <position name="act_{side}_palm_pitch" joint="{side}_palm_pitch" kp="80" ctrlrange="-0.610865 0.610865" forcerange="-50 50"/>',
    ]
    for finger in FINGERS:
        items.extend([
            f'    <position name="act_{side}_{finger}_mcp_yaw" joint="{side}_{finger}_mcp_yaw" kp="35" ctrlrange="-0.488692 0.488692" forcerange="-18 18"/>',
            f'    <position name="act_{side}_{finger}_mcp" joint="{side}_{finger}_mcp" kp="45" ctrlrange="-0.349066 1.5708" forcerange="-25 25"/>',
            f'    <position name="act_{side}_{finger}_pip" joint="{side}_{finger}_pip" kp="38" ctrlrange="0 1.74533" forcerange="-22 22"/>',
            f'    <position name="act_{side}_{finger}_dip" joint="{side}_{finger}_dip" kp="30" ctrlrange="0 1.39626" forcerange="-18 18"/>',
        ])
    return "\n".join(items)


def make_hand_touch_sensors_xml(side: str) -> str:
    items = [f'    <touch name="{side}_palm_touch" site="{side}_palm_center_site"/>']
    for finger in FINGERS:
        items.extend([
            f'    <touch name="{side}_{finger}_mcp_touch" site="{side}_{finger}_mcp_site"/>',
            f'    <touch name="{side}_{finger}_pip_touch" site="{side}_{finger}_pip_site"/>',
            f'    <touch name="{side}_{finger}_dip_touch" site="{side}_{finger}_dip_site"/>',
            f'    <touch name="{side}_{finger}_tip_touch" site="{side}_{finger}_tip_site"/>',
        ])
    return "\n".join(items)


def make_both_hands_actuators_xml() -> str:
    return make_hand_actuators_xml("left") + "\n" + make_hand_actuators_xml("right")


def make_both_hands_sensors_xml() -> str:
    return make_hand_touch_sensors_xml("left") + "\n" + make_hand_touch_sensors_xml("right")


def hand_control_names(side: str) -> list[str]:
    names = [f"{side}_palm_roll", f"{side}_palm_pitch"]
    for finger in FINGERS:
        names.extend([
            f"{side}_{finger}_mcp_yaw",
            f"{side}_{finger}_mcp",
            f"{side}_{finger}_pip",
            f"{side}_{finger}_dip",
        ])
    return names


def both_hand_control_names() -> list[str]:
    return hand_control_names("left") + hand_control_names("right")


def hand_sensor_names(side: str) -> list[str]:
    names = [f"{side}_palm_touch"]
    for finger in FINGERS:
        names.extend([f"{side}_{finger}_mcp_touch", f"{side}_{finger}_pip_touch", f"{side}_{finger}_dip_touch", f"{side}_{finger}_tip_touch"])
    return names


def both_hand_sensor_names() -> list[str]:
    return hand_sensor_names("left") + hand_sensor_names("right")
