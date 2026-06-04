from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

HAND_DIM = 44
ARM_DIM = 6
BODY_DIM = 9
LEG_DIM = 18
FINGERS = ("thumb", "index", "middle", "ring", "little")
SIDES = ("left", "right")


def _cfg(sc, key: str, default: float) -> float:
    for name in ("_cfg_float", "_cfg"):
        try:
            v = getattr(sc, name)(key, default)
            if v is not None:
                return float(v)
        except Exception:
            pass
    cfg = getattr(getattr(sc, "owner", None), "cfg", None)
    try:
        from omegaconf import OmegaConf
        for path in (f"adaptive_scenario_controller.{key}", key):
            v = OmegaConf.select(cfg, path, default=None)
            if v is not None:
                return float(v)
    except Exception:
        pass
    try:
        sec = cfg.get("adaptive_scenario_controller") if isinstance(cfg, dict) else getattr(cfg, "adaptive_scenario_controller", None)
        if sec is not None:
            if isinstance(sec, dict) and key in sec:
                return float(sec[key])
            return float(getattr(sec, key))
    except Exception:
        pass
    return float(default)


def _clip01(x): return float(np.clip(float(x), 0.0, 1.0))
def _wrap_pi(x): return float((float(x) + math.pi) % (2 * math.pi) - math.pi)


def _q_to_rpy(q):
    qw, qx, qy, qz = [float(x) for x in q]
    roll = math.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    s = 2*(qw*qy-qz*qx)
    pitch = math.copysign(math.pi/2, s) if abs(s) >= 1 else math.asin(s)
    yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return roll, pitch, yaw


@dataclass
class ServoState:
    integ: np.ndarray
    prev: np.ndarray
    prev_step: int

    @classmethod
    def make(cls, step: int):
        return cls(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), int(step))


class Hand44:
    def __init__(self):
        self.v = np.full(HAND_DIM, 0.5, dtype=np.float32)

    def base(self, side): return 0 if side == "left" else 22
    def fb(self, side, finger): return self.base(side) + 2 + 4 * FINGERS.index(finger)
    def palm_roll(self, side, v): self.v[self.base(side) + 0] = _clip01(v)
    def palm_pitch(self, side, v): self.v[self.base(side) + 1] = _clip01(v)
    def yaw(self, side, finger, v): self.v[self.fb(side, finger) + 0] = _clip01(v)

    def curl(self, side, finger, mcp, pip=None, dip=None):
        pip = mcp if pip is None else pip
        dip = pip if dip is None else dip
        b = self.fb(side, finger)
        self.v[b+1] = _clip01(mcp)
        self.v[b+2] = _clip01(pip)
        self.v[b+3] = _clip01(dip)

    def open_capture(self):
        self.v[:] = 0.5
        self.palm_roll("left", 0.32); self.palm_pitch("left", 0.62)
        self.palm_roll("right", 0.68); self.palm_pitch("right", 0.62)
        yl = {"thumb":0.10, "index":0.30, "middle":0.48, "ring":0.66, "little":0.86}
        yr = {"thumb":0.90, "index":0.70, "middle":0.52, "ring":0.34, "little":0.14}
        for side, yaws in (("left", yl), ("right", yr)):
            for f in FINGERS:
                self.yaw(side, f, yaws[f])
                self.curl(side, f, 0.0, 0.0, 0.0)
            self.curl(side, "thumb", 0.04, 0.01, 0.0)

    def close_capture(self, s):
        s = float(np.clip(s, 0, 1))
        self.open_capture()
        for side in SIDES:
            self.curl(side, "thumb", 0.10+0.72*s, 0.04+0.62*s, 0.02+0.52*s)
            self.curl(side, "index", 0.04+0.78*s, 0.02+0.66*s, 0.01+0.56*s)
            self.curl(side, "middle",0.04+0.78*s, 0.02+0.66*s, 0.01+0.56*s)
            self.curl(side, "ring",  0.03+0.62*s, 0.02+0.54*s, 0.01+0.46*s)
            self.curl(side, "little",0.03+0.58*s, 0.02+0.48*s, 0.01+0.42*s)

    def rotate_gait(self, step, contact=1.0):
        self.close_capture(0.84 + 0.12 * float(np.clip(contact, 0, 1)))
        phase = (int(step) % 96) / 96.0
        groups = [("left", ("thumb","index")), ("right", ("thumb","index")),
                  ("left", ("middle","ring","little")), ("right", ("middle","ring","little"))]
        side, fingers = groups[int(phase * 4) % 4]
        local = (phase * 4) % 1.0
        release = max(0.0, 1.0 - abs(local - 0.35) / 0.35)
        slide = math.sin(2 * math.pi * local)
        sign = -1.0 if side == "left" else 1.0
        for f in fingers:
            b = self.fb(side, f)
            self.v[b+0] = _clip01(float(self.v[b+0]) + 0.09 * sign * slide)
            self.v[b+1] = _clip01(float(self.v[b+1]) - 0.08 * release)
            self.v[b+2] = _clip01(float(self.v[b+2]) - 0.06 * release)


def _agent(sc):
    owner = sc.owner
    ctrl = getattr(owner, "dynamic_agent_rig_controller", None)
    data = getattr(getattr(owner, "world", None), "data", None)
    if ctrl is not None and hasattr(ctrl, "qpos_adr") and data is not None:
        try:
            qpos = data.qpos[ctrl.qpos_adr:ctrl.qpos_adr+7]
            xyz = np.asarray(qpos[:3], dtype=np.float64).copy()
            quat = np.asarray(qpos[3:7], dtype=np.float64).copy()
            _, _, yaw = _q_to_rpy(quat)
            return xyz, yaw, quat
        except Exception:
            pass
    xyz = np.asarray(getattr(owner.world, "cam_pos", np.zeros(3)), dtype=np.float64).reshape(3)
    yaw = float(np.deg2rad(float(getattr(owner.world, "yaw_deg", 0.0))))
    return xyz, yaw, None


def _small_cube_pose(sc):
    w = getattr(sc.owner, "world", None); model = getattr(w, "model", None); data = getattr(w, "data", None)
    if model is None or data is None: return None
    try:
        import mujoco
        for name in ("obj_box_small", "small_cube", "small_box", "obj_small_cube", "obj_small_box", "target_small_cube", "obj_box", "box", "cube"):
            gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if gid >= 0:
                c = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3).copy()
                xm = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3,3).copy()
                half = np.asarray(model.geom_size[gid, :3], dtype=np.float64).reshape(3).copy()
                return c, xm, half, name
    except Exception:
        pass
    return None


def _site_positions(sc, names):
    w = getattr(sc.owner, "world", None); model = getattr(w, "model", None); data = getattr(w, "data", None)
    if model is None or data is None: return []
    out = []
    try:
        import mujoco
        for name in names:
            sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
            if sid >= 0:
                out.append((name, np.asarray(data.site_xpos[sid], dtype=np.float64).reshape(3).copy()))
    except Exception:
        pass
    return out


def _palm_tip_sites(sc):
    palms = ("left_palm_center_site","right_palm_center_site","left_hand_site","right_hand_site","left_palm_site","right_palm_site")
    tips = tuple(f"{s}_{f}_tip_site" for s in SIDES for f in FINGERS)
    return _site_positions(sc, palms + tips)


def _point_box(point, center, xmat, half):
    local = np.asarray(xmat).T @ (np.asarray(point) - center)
    q = np.abs(local) - half
    outside = np.maximum(q, 0.0)
    dist = float(np.linalg.norm(outside))
    signed = dist if dist > 1e-9 else float(np.max(q))
    closest = center + np.asarray(xmat) @ np.clip(local, -half, half)
    return dist, signed, closest


def _servo_error(sc):
    pose = _small_cube_pose(sc)
    if pose is None:
        return {"ok": False, "dist":999.0, "signed":999.0, "err":np.zeros(3), "site":"", "cube":""}
    center, xmat, half, cname = pose
    sites = _palm_tip_sites(sc)
    if not sites:
        return {"ok": False, "dist":999.0, "signed":999.0, "err":np.zeros(3), "site":"", "cube":cname}
    best = None
    for sname, pos in sites:
        dist, signed, closest = _point_box(pos, center, xmat, half)
        if best is None or signed < best[0]:
            best = (signed, dist, closest, sname, pos)
    signed, dist, closest, sname, pos = best
    err = np.asarray(closest - pos, dtype=np.float64)
    if signed < 0: err *= 0.1
    return {"ok": True, "dist":float(dist), "signed":float(signed), "err":err, "site":sname, "cube":cname}


def _contacts(sc):
    w = getattr(sc.owner, "world", None); model = getattr(w, "model", None); data = getattr(w, "data", None)
    if model is None or data is None: return {"sum":0.0, "count":0.0}
    vals = []
    try:
        import mujoco
        hand = ("hand","finger","fingertip","tip","palm","thumb","index","middle","ring","little")
        cube = ("obj_box_small","small","box","cube")
        for i in range(int(getattr(data, "ncon", 0))):
            con = data.contact[i]
            n1 = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom1)) or "").lower()
            n2 = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom2)) or "").lower()
            ok = (any(k in n1 for k in hand) and any(k in n2 for k in cube)) or (any(k in n2 for k in hand) and any(k in n1 for k in cube))
            if ok: vals.append(0.02)
    except Exception:
        pass
    return {"sum": float(sum(vals)), "count": float(len(vals))}


def _world_to_local(vec, yaw):
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    return cy*vec[0] - sy*vec[1], sy*vec[0] + cy*vec[1]


def _servo_cmd(sc, err, yaw, gain=1.0):
    step = int(getattr(sc.owner, "global_step", 0))
    st = getattr(sc, "_palm_servo_state", None)
    if st is None:
        st = ServoState.make(step); sc._palm_servo_state = st
    dt = max(1e-4, (step - st.prev_step if step > st.prev_step else 1) * _cfg(sc, "small_cube_servo_dt_per_step", 0.02))
    err = np.asarray(err, dtype=np.float64).reshape(3)
    st.integ = np.clip(st.integ + err * dt, -_cfg(sc,"small_cube_servo_integral_limit",0.25), _cfg(sc,"small_cube_servo_integral_limit",0.25))
    der = (err - st.prev) / dt
    out = (_cfg(sc,"small_cube_servo_kp",2.2)*err + _cfg(sc,"small_cube_servo_ki",0.04)*st.integ + _cfg(sc,"small_cube_servo_kd",0.28)*der) * gain
    out = np.clip(out, -_cfg(sc,"small_cube_servo_output_limit",0.45), _cfg(sc,"small_cube_servo_output_limit",0.45))
    st.prev = err.copy(); st.prev_step = step
    lx, ly = _world_to_local(out, yaw)
    body = np.zeros(BODY_DIM, dtype=np.float32)
    body[0] = float(np.clip(lx * _cfg(sc,"small_cube_servo_body_xy_gain",0.55), -0.22, 0.22))
    body[1] = float(np.clip(ly * _cfg(sc,"small_cube_servo_body_xy_gain",0.55), -0.22, 0.22))
    body[2] = float(np.clip(out[2] * _cfg(sc,"small_cube_servo_body_z_gain",0.55), -0.16, 0.16))
    arm = np.zeros(ARM_DIM, dtype=np.float32)
    pitch = _cfg(sc,"small_cube_servo_arm_base_pitch",-0.90) + np.clip(lx*0.55 + out[2]*0.35, -0.18, 0.18)
    elbow = _cfg(sc,"small_cube_servo_arm_base_elbow",-0.98) + 0.5*np.clip(lx*0.55 + out[2]*0.35, -0.18, 0.18)
    yaw_abs = _cfg(sc,"small_cube_servo_arm_yaw_abs",0.06)
    yd = np.clip(ly * _cfg(sc,"small_cube_servo_arm_side_gain",0.40), -0.12, 0.12)
    arm[:] = np.asarray([-yaw_abs+yd, pitch, elbow, yaw_abs+yd, pitch, elbow], dtype=np.float32)
    return body, np.clip(arm, -1, 1), {"servo_cmd_x":float(out[0]), "servo_cmd_y":float(out[1]), "servo_cmd_z":float(out[2])}


def _gaze_level(sc, body, agent, yaw, quat, cube):
    diff = cube - agent
    yaw_err = _wrap_pi(math.atan2(float(diff[1]), float(diff[0])) - yaw)
    pitch = math.atan2(float(diff[2]) + _cfg(sc,"small_cube_gaze_target_z_offset",0.03), float(np.linalg.norm(diff[:2])) + 1e-6)
    body[3] = float(np.clip(_cfg(sc,"small_cube_gaze_body_yaw_sign",1)*_cfg(sc,"small_cube_gaze_body_yaw_kp",1.0)*yaw_err, -0.85, 0.85))
    body[6] = float(np.clip(_cfg(sc,"small_cube_gaze_head_yaw_sign",1)*_cfg(sc,"small_cube_gaze_head_yaw_kp",1.35)*yaw_err, -1, 1))
    body[7] = float(np.clip(_cfg(sc,"small_cube_gaze_head_pitch_sign",-1)*_cfg(sc,"small_cube_gaze_head_pitch_kp",1.25)*pitch, -1, 1))
    body[8] = 0
    roll_err = pitch_err = 0.0
    if quat is not None:
        roll_err, pitch_err, _ = _q_to_rpy(quat)
    body[4] = float(np.clip(_cfg(sc,"small_cube_level_pitch_sign",1)*_cfg(sc,"small_cube_level_kp",1.1)*pitch_err, -0.55, 0.55))
    body[5] = float(np.clip(_cfg(sc,"small_cube_level_roll_sign",1)*_cfg(sc,"small_cube_level_kp",1.1)*roll_err, -0.55, 0.55))
    return {"yaw_err":abs(float(yaw_err)), "gaze_pitch_err":abs(float(pitch)), "level_pitch_err":float(pitch_err), "level_roll_err":float(roll_err)}


def _approach(sc, agent, yaw, cube):
    standoff = _cfg(sc,"small_cube_approach_standoff",0.44)
    diff = cube[:2] - agent[:2]
    dist = float(np.linalg.norm(diff) + 1e-9)
    target = cube[:2] - diff / dist * standoff
    err_xy = target - agent[:2]
    lx, ly = _world_to_local(np.asarray([err_xy[0], err_xy[1], 0.0]), yaw)
    body = np.zeros(BODY_DIM, dtype=np.float32)
    body[0] = float(np.clip(lx*_cfg(sc,"small_cube_approach_speed",0.62), -0.5, 0.5))
    body[1] = float(np.clip(ly*_cfg(sc,"small_cube_approach_speed",0.62), -0.5, 0.5))
    body[2] = float(np.clip((cube[2]+_cfg(sc,"small_cube_approach_hover_above_cube",0.70)-agent[2])*0.85, -0.35, 0.35))
    return body, float(np.linalg.norm(err_xy))


def _cmd(sc, body, arm, hand, status):
    ScenarioCommand = getattr(sys.modules.get(sc.__class__.__module__), "ScenarioCommand")
    return ScenarioCommand(body=body.astype(np.float32), arm=arm.astype(np.float32), hand=np.clip(hand,0,1).astype(np.float32), leg=np.zeros(LEG_DIM,dtype=np.float32), status=status)


def install_small_cube_grasp_rotate_gait(cls):
    if getattr(cls, "_small_cube_adaptive_servo_v5_installed", False): return
    orig_start = getattr(cls, "start", None)
    orig_stop = getattr(cls, "stop", None)

    def start(self):
        if callable(orig_start):
            try: orig_start(self)
            except Exception: pass
        self.active = True; self.phase = "approach"
        self.started_step = int(getattr(self.owner, "global_step", 0))
        self._phase_started_step = self.started_step
        self._palm_servo_state = ServoState.make(self.started_step)
        self._grasp_strength = 0.0; self._stable_contact_steps = 0
        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_leg_action = np.zeros(LEG_DIM, dtype=np.float32)
        print("[small_cube_adaptive_servo_v5] started")

    def step(self):
        if not getattr(self, "active", False): return None
        now = int(getattr(self.owner, "global_step", 0)); age = now - int(getattr(self, "started_step", now))
        if age > int(_cfg(self, "small_cube_adaptive_timeout_steps", 5000)):
            if callable(orig_stop): orig_stop(self, "timeout")
            else: self.stop("timeout")
            return None
        pose = _small_cube_pose(self)
        cube = pose[0] if pose is not None else np.zeros(3, dtype=np.float64)
        agent, yaw, quat = _agent(self)
        phase = str(getattr(self, "phase", "approach"))
        se = _servo_error(self); contact = _contacts(self)
        dist = float(se["dist"]); signed = float(se["signed"]); err = np.asarray(se["err"], dtype=np.float64)
        ready = contact["sum"] >= _cfg(self,"small_cube_adaptive_contact_sum",0.02) or contact["count"] >= _cfg(self,"small_cube_adaptive_contact_count",2) or dist <= _cfg(self,"small_cube_adaptive_surface_dist_goal",0.018) or signed <= _cfg(self,"small_cube_adaptive_surface_signed_goal",0.010)

        if phase == "approach":
            body, approach_err = _approach(self, agent, yaw, cube); arm = np.zeros(ARM_DIM, dtype=np.float32); h = Hand44(); h.open_capture()
            if approach_err < _cfg(self, "small_cube_approach_tol", 0.075):
                self.phase = "palm_servo_open"; self._phase_started_step = now; self._palm_servo_state = ServoState.make(now)
        elif phase == "palm_servo_open":
            body, arm, diag = _servo_cmd(self, err, yaw, 1.0); approach_err = dist; h = Hand44(); h.open_capture()
            if ready or dist < _cfg(self, "small_cube_open_to_close_dist", 0.035):
                self.phase = "capture_close"; self._phase_started_step = now; self._grasp_strength = 0.0
        elif phase == "capture_close":
            body, arm, diag = _servo_cmd(self, err, yaw, 0.65); approach_err = dist
            self._grasp_strength = float(np.clip(self._grasp_strength + _cfg(self,"small_cube_adaptive_close_rate",0.022), 0, 1))
            h = Hand44(); h.close_capture(self._grasp_strength)
            self._stable_contact_steps = max(0, int(getattr(self, "_stable_contact_steps", 0)) + (1 if ready else -1))
            if self._stable_contact_steps >= int(_cfg(self,"small_cube_adaptive_hold_contact_steps",10)) or self._grasp_strength > _cfg(self,"small_cube_adaptive_force_hold_strength",0.97):
                self.phase = "hold"; self._phase_started_step = now
        elif phase == "hold":
            body, arm, diag = _servo_cmd(self, err, yaw, 0.25); approach_err = dist; h = Hand44(); h.close_capture(1.0)
            if now - int(getattr(self, "_phase_started_step", now)) > int(_cfg(self,"small_cube_adaptive_hold_steps",40)):
                self.phase = "rotate"; self._phase_started_step = now
        else:
            self.phase = "rotate"; body, arm, diag = _servo_cmd(self, err, yaw, 0.12); body[:3] *= 0.25; approach_err = dist
            h = Hand44(); h.rotate_gait(now - int(getattr(self, "_phase_started_step", now)), contact=np.clip(contact["sum"]/0.08,0,1))
        gl = _gaze_level(self, body, agent, yaw, quat, cube)
        status = {"active":True, "controller":"FlyToSmallCubeGraspRotateScenario", "mode":"small_cube_adaptive_palm_servo_v5", "phase":str(getattr(self,"phase",phase)), "age":int(age), "approach_err":float(approach_err), "servo_dist":dist, "servo_signed":signed, "servo_site":str(se["site"]), "servo_cube":str(se["cube"]), "servo_err_x":float(err[0]), "servo_err_y":float(err[1]), "servo_err_z":float(err[2]), "contact_sum":float(contact["sum"]), "contact_count":float(contact["count"]), "grasp_strength":float(getattr(self,"_grasp_strength",0.0)), "stable_contact_steps":int(getattr(self,"_stable_contact_steps",0)), "hand_dim":HAND_DIM}
        status.update(gl)
        if "diag" in locals(): status.update(diag)
        self.status = status
        return _cmd(self, body, arm, h.v, status)

    cls.start = start; cls.step = step
    cls._small_cube_adaptive_servo_v5_installed = True
    cls._small_cube_gait_installed = True
    print("[small_cube_adaptive_servo_v5] installed on FlyToSmallCubeGraspRotateScenario")
