from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from src.shared.unified_conscious_utils import ArmActuatorConfig, simple_arm_ik_local
except Exception:
    ArmActuatorConfig = None
    simple_arm_ik_local = None


@dataclass
class ScenarioCommand:
    body: np.ndarray
    arm: np.ndarray
    hand: np.ndarray
    leg: np.ndarray
    status: Dict


_MISSING = object()


def _config_value(container: Any, key: str, default: Any = _MISSING) -> Any:
    if container is None:
        return default
    try:
        if isinstance(container, dict) and key in container:
            return container[key]
    except Exception:
        pass
    try:
        if key in container:
            return container[key]
    except Exception:
        pass
    try:
        if hasattr(container, key):
            return getattr(container, key)
    except Exception:
        pass
    try:
        return container.get(key, default)
    except Exception:
        return default


def _adaptive_section_value(section: Any, key: str) -> Any:
    if section is None:
        print(f"\033[31m ######## error section - adaptive_scenario_controller missing section key={key}; using fallback/default \033[0m")
        return _MISSING
    val = _config_value(section, key)
    if val is _MISSING:
        print(f"\033[31m ######## error section - adaptive_scenario_controller missing key={key}; using fallback/default \033[0m")
    return val


def _cfg_float_value(val: Any, key: str, source: str) -> Any:
    try:
        return float(val)
    except Exception as e:
        print(
            f"\033[31m ######## error section - adaptive_scenario_controller invalid float "
            f"source={source} key={key} value={val!r} err={e} \033[0m"
        )
        return _MISSING


class BaseScenario:
    name: str = "base"

    def __init__(self, owner):
        self.owner = owner
        self.started_step = int(getattr(owner, "global_step", 0))
        self.active = False
        self.phase = "idle"
        self.status: Dict = {}

    def start(self) -> None:
        self.started_step = int(getattr(self.owner, "global_step", 0))
        self.active = True

    def stop(self, reason: str = "stopped") -> None:
        self.active = False
        self.phase = "done"
        self.status.update(active=False, phase="done", reason=str(reason))

    def step(self) -> Optional[ScenarioCommand]:
        raise NotImplementedError


class FlyToCubePalpateScenario(BaseScenario):
    """
    Adaptive scenario:

        approach -> align -> reach -> palpate

    The body/rig stops at a safe standoff ring around the cube and hovers high
    enough that legs do not touch ground. Hands are then extended using an
    adaptive hand/arm PID loop until fingertip contact is detected from tactile
    observations, then fingers keep probing continuously.
    """

    name = "fly_to_cube_palpate"

    def __init__(self, owner):
        super().__init__(owner)
        self.phase = "approach"
        self.contact_hold_steps = 0
        self.align_hold_steps = 0
        self.no_contact_reach_steps = 0

        self.adaptive_reach_distance = self._cfg("fly_to_cube_arm_reach_distance", 1.35)
        self.timeout_steps = int(self._cfg("fly_to_cube_timeout_steps", 2400))

        # Vertical hover PID state.
        self._z_pid_integral = 0.0
        self._z_pid_prev_err = 0.0
        self._z_pid_prev_step = int(getattr(owner, "global_step", 0))
        self._z_pid_vz = 0.0

        # Hand/arm contact PID state.
        self._hand_pid_integral = 0.0
        self._hand_pid_prev_err = 0.0
        self._hand_pid_prev_step = int(getattr(owner, "global_step", 0))
        self._hand_pid_drive = 0.0
        self._hand_extension = 0.0
        self._finger_curl_bias = 0.0

        # Online distance minimizer state. The scenario should not prescribe a
        # pose; it should adjust joints only by whether fingertip-to-cube-face
        # distance improves.
        self._reach_arm_action = np.zeros(6, dtype=np.float32)
        self._reach_palm_pitch = 0.65
        self._reach_left_palm_roll = 0.48
        self._reach_right_palm_roll = 0.52
        self._reach_finger_base = 0.06
        self._reach_axis = 0
        self._reach_dir = -1.0
        self._reach_last_dist = 999.0
        self._reach_best_dist = 999.0
        self._reach_stall_steps = 0
        self._reach_updates = 0

    def _cfg(self, key: str, default: float) -> float:
        """
        Preferred config layout:
            adaptive_scenario_controller:
              fly_to_cube_...: value

        Backward-compatible fallback:
            fly_to_cube_...: value
        """
        cfg = getattr(self.owner, "cfg", None)

        try:
            section = getattr(cfg, "adaptive_scenario_controller", None)
            section_val = _adaptive_section_value(section, key)
            if section_val is not _MISSING:
                val = _cfg_float_value(section_val, key, "section")
                if val is _MISSING:
                    raise ValueError(f"invalid adaptive_scenario_controller.{key}")
                #print("\033[32m section - adaptive_scenario_controller \033[0m", key , val)    
                return val
        except Exception:
            pass

        try:
            root_val = _config_value(cfg, key, default)
            val = _cfg_float_value(root_val, key, "root/default")
            if val is not _MISSING:
                return val
        except Exception:
            pass
        return float(default)

    def _cfg_float(self, key: str, default: float) -> float:
        return self._cfg(key, default)

    def start(self) -> None:
        super().start()
        self.phase = "approach"
        self.contact_hold_steps = 0
        self.align_hold_steps = 0
        self.no_contact_reach_steps = 0
        self.adaptive_reach_distance = self._cfg("fly_to_cube_arm_reach_distance", 1.35)

        self._z_pid_integral = 0.0
        self._z_pid_prev_err = 0.0
        self._z_pid_prev_step = int(getattr(self.owner, "global_step", 0))
        self._z_pid_vz = 0.0

        self._hand_pid_integral = 0.0
        self._hand_pid_prev_err = 0.0
        self._hand_pid_prev_step = int(getattr(self.owner, "global_step", 0))
        self._hand_pid_drive = 0.0
        self._hand_extension = 0.0
        self._finger_curl_bias = 0.0

        self._reset_reach_distance_minimizer()

        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_leg_action = np.zeros(18, dtype=np.float32)

        print("[adaptive_scenario] fly_to_cube_palpate started with hand PID")

    def _agent_xyz_yaw(self) -> Tuple[np.ndarray, float]:
        owner = self.owner
        ctrl = getattr(owner, "dynamic_agent_rig_controller", None)
        if ctrl is not None and hasattr(ctrl, "qpos_adr"):
            try:
                qpos = owner.world.data.qpos[ctrl.qpos_adr:ctrl.qpos_adr + 7]
                xyz = np.asarray(qpos[:3], dtype=np.float64).copy()
                qw, qx, qy, qz = np.asarray(qpos[3:7], dtype=np.float64)
                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                return xyz, float(np.arctan2(siny_cosp, cosy_cosp))
            except Exception:
                pass

        xyz = np.asarray(getattr(owner.world, "cam_pos", np.zeros(3)), dtype=np.float64).copy()
        return xyz, float(np.deg2rad(float(getattr(owner.world, "yaw_deg", 0.0))))

    def _cube_xyz(self) -> np.ndarray:
        owner = self.owner
        for key in ("box", "cube", "obj_box_body"):
            try:
                if hasattr(owner.world, "get_object_pos"):
                    arr = np.asarray(owner.world.get_object_pos(key), dtype=np.float64).reshape(3)
                    if np.isfinite(arr).all():
                        return arr
            except Exception:
                pass
        return np.zeros(3, dtype=np.float64)

    def _as_np_vector(self, value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.isfinite(arr).all():
                return arr
        except Exception:
            return None
        return None


    def _mujoco_name(self, obj_type, obj_id: int) -> str:
        try:
            import mujoco
            name = mujoco.mj_id2name(self.owner.world.model, obj_type, int(obj_id))
            return str(name or "")
        except Exception:
            return ""


    def _mujoco_contact_force_scalar(self, contact_index: int) -> float:
        """
        Read a MuJoCo contact force magnitude directly from mjData/contact.

        This bypasses obs and therefore is not affected by sleep-mode sensor
        masking.
        """
        try:
            import mujoco
            data = self.owner.world.data
            model = self.owner.world.model
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, int(contact_index), wrench)
            return float(np.linalg.norm(wrench[:3]))
        except Exception:
            return 0.0


    def _mujoco_tactile_vec(self) -> tuple[np.ndarray, str]:
        """
        Tactile/contact source for the scenario.

        Important:
            Do NOT read owner.latest_obs here.
            Do NOT read masked obs tactile here.

        Reason:
            in sleep simulation mode, obs sensors may be deliberately zeroed.
            The scenario still needs real MuJoCo contacts to know whether
            fingers touched the cube.

        Sources:
            1. mjData.sensordata for MuJoCo sensors with names containing
               touch/contact/tactile/finger/palm/thumb/index/middle/ring/little.
            2. mjData.contact contacts between hand/finger/palm geoms and object
               geoms, using mj_contactForce when available.
            3. world.latest_tactile only as a final MuJoCo-side fallback.
        """
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)

        values = []

        # 1) Direct MuJoCo sensor data.
        try:
            sensordata = np.asarray(data.sensordata, dtype=np.float32).reshape(-1)
            selected = []

            n_sens = int(getattr(model, "nsensor", 0))
            sensor_adr = getattr(model, "sensor_adr", None)
            sensor_dim = getattr(model, "sensor_dim", None)

            keywords = (
                "finger", "fingertip",
                "palm", "thumb", "index", "middle", "ring", "little",
                "left_hand", "right_hand", "hand",
            )

            if n_sens > 0 and sensor_adr is not None and sensor_dim is not None:
                try:
                    import mujoco
                    for sid in range(n_sens):
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or ""
                        lname = str(name).lower()
                        if any(k in lname for k in keywords):
                            adr = int(sensor_adr[sid])
                            dim = int(sensor_dim[sid])
                            if dim > 0:
                                selected.append(sensordata[adr:adr + dim])
                except Exception:
                    selected = []

            if selected:
                arr = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in selected])
                if arr.size > 0:
                    values.append(arr)

            # If sensor names are unavailable but sensordata exists, keep it as
            # direct MuJoCo data. This is still better than obs because it is not
            # sleep-masked by the perception pipeline.
            elif sensordata.size > 0:
                values.append(sensordata)
        except Exception:
            pass

        # 2) Direct MuJoCo contacts.
        try:
            ncon = int(getattr(data, "ncon", 0))
            contact_vals = []
            hand_keys = (
                "hand", "finger", "fingertip", "tip",
                "palm", "thumb", "index", "middle", "ring", "little",
            )
            obj_keys = ("box", "cube", "object", "sphere", "cylinder", "target")

            try:
                import mujoco
                obj_geom = mujoco.mjtObj.mjOBJ_GEOM
            except Exception:
                obj_geom = None

            for ci in range(ncon):
                con = data.contact[ci]
                g1 = int(con.geom1)
                g2 = int(con.geom2)

                if obj_geom is not None:
                    n1 = self._mujoco_name(obj_geom, g1).lower()
                    n2 = self._mujoco_name(obj_geom, g2).lower()
                else:
                    n1 = n2 = ""

                hand_obj = (
                    (any(k in n1 for k in hand_keys) and any(k in n2 for k in obj_keys))
                    or
                    (any(k in n2 for k in hand_keys) and any(k in n1 for k in obj_keys))
                )

                if hand_obj:
                    force = self._mujoco_contact_force_scalar(ci)
                    floor = self._cfg("fly_to_cube_contact_force_floor", 0.020)
                    scale = max(self._cfg("fly_to_cube_contact_force_scale", 1.0), 1e-6)
                    contact_vals.append(max(force / scale, floor))

            if contact_vals:
                values.append(np.asarray(contact_vals, dtype=np.float32))
        except Exception:
            pass

        # 3) Final world-level fallback. This is not obs, but depending on world
        # implementation it may already be postprocessed. Keep it last.
        try:
            raw = getattr(world, "latest_mujoco_tactile", None)
            if raw is None:
                raw = getattr(world, "latest_raw_tactile", None)
            if raw is None:
                raw = getattr(world, "latest_tactile", None)
            if raw is not None:
                arr = np.asarray(raw, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    values.append(arr)
        except Exception:
            pass

        if values:
            vec = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values])
            return vec, "mujoco.direct"

        return np.zeros(1, dtype=np.float32), "mujoco.none"


    def _tactile_vec_from_obs(self) -> tuple[np.ndarray, str]:
        """
        Compatibility wrapper.

        Despite the old name, this now reads only direct MuJoCo contact/tactile
        sources, not obs. This prevents sleep-mode masked sensors from hiding
        real fingertip contact from the scenario controller.
        """
        return self._mujoco_tactile_vec()

    def _tactile_features(self) -> Dict[str, float | str]:
        return self._mujoco_hand_cube_contact_features()

    def _palm_site_positions(self, side: Optional[str] = None) -> list[tuple[str, np.ndarray]]:
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return []

        if side in ("left", "right"):
            names = (f"{side}_palm_center_site", f"{side}_hand_site")
        else:
            names = ("left_palm_center_site", "right_palm_center_site", "left_hand_site", "right_hand_site")
        out: list[tuple[str, np.ndarray]] = []
        try:
            import mujoco
            for name in names:
                site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
                if site_id >= 0:
                    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy()
                    out.append((name, pos))
        except Exception:
            pass
        return out

    def _palm_face_distance_to_cube(self, side: Optional[str] = None) -> Dict[str, float | str]:
        pose = self._cube_box_geom_pose()
        palms = self._palm_site_positions(side=side)
        if pose is None or not palms:
            return {
                "distance": 999.0,
                "signed_distance": 999.0,
                "closest_palm": "",
                "closest_point_x": 0.0,
                "closest_point_y": 0.0,
                "closest_point_z": 0.0,
                "source": "mujoco.geom.missing",
            }

        center, xmat, half, cube_name = pose
        best_name = ""
        best_distance = 999.0
        best_signed = 999.0
        best_point = np.zeros(3, dtype=np.float64)
        for name, pos in palms:
            dist, signed, closest = self._point_to_box_surface(pos, center, xmat, half)
            if dist < best_distance:
                best_name = name
                best_distance = dist
                best_signed = signed
                best_point = closest
        return {
            "distance": float(max(best_distance, 0.0)),
            "signed_distance": float(best_signed),
            "closest_palm": best_name,
            "closest_point_x": float(best_point[0]),
            "closest_point_y": float(best_point[1]),
            "closest_point_z": float(best_point[2]),
            "source": f"mujoco.geom.palm.{cube_name}",
        }

    def _mujoco_hand_cube_contact_features(self) -> Dict[str, float | str]:
        """
        Direct MuJoCo contact source for hand/finger/palm touching the cube.

        This is the scenario's contact truth. It deliberately reads mjData.contact
        instead of obs, so sleep-mode sensor masking cannot hide real contacts.
        """
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return {"sum": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "source": "mujoco.contact.none"}

        hand_keys = (
            "hand", "finger", "fingertip", "palm",
            "thumb", "index", "middle", "ring", "little",
        )
        cube_keys = ("obj_box", "box", "cube")
        values = []

        try:
            import mujoco
            obj_geom = mujoco.mjtObj.mjOBJ_GEOM
        except Exception:
            obj_geom = None

        try:
            ncon = int(getattr(data, "ncon", 0))
            floor = self._cfg("fly_to_cube_contact_force_floor", 0.020)
            scale = max(self._cfg("fly_to_cube_contact_force_scale", 1.0), 1e-6)

            for ci in range(ncon):
                con = data.contact[ci]
                g1 = int(con.geom1)
                g2 = int(con.geom2)

                if obj_geom is not None:
                    n1 = self._mujoco_name(obj_geom, g1).lower()
                    n2 = self._mujoco_name(obj_geom, g2).lower()
                else:
                    n1 = n2 = ""

                hand_cube = (
                    (any(k in n1 for k in hand_keys) and any(k in n2 for k in cube_keys))
                    or
                    (any(k in n2 for k in hand_keys) and any(k in n1 for k in cube_keys))
                )
                if not hand_cube:
                    continue

                force = self._mujoco_contact_force_scalar(ci)
                values.append(max(force / scale, floor))
        except Exception:
            values = []

        if not values:
            return {"sum": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "source": "mujoco.contact.none"}

        arr = np.asarray(values, dtype=np.float32)
        return {
            "sum": float(arr.sum()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "count": int(arr.size),
            "source": "mujoco.contact.hand_cube",
        }

    def _mujoco_id_by_name(self, obj_type, names: tuple[str, ...]) -> int:
        try:
            import mujoco
            for name in names:
                obj_id = int(mujoco.mj_name2id(self.owner.world.model, obj_type, str(name)))
                if obj_id >= 0:
                    return obj_id
        except Exception:
            pass
        return -1

    def _cube_box_geom_pose(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return None

        try:
            import mujoco
            geom_id = self._mujoco_id_by_name(mujoco.mjtObj.mjOBJ_GEOM, ("obj_box", "box", "cube"))
            if geom_id < 0:
                return None
            center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3).copy()
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy()
            half = np.asarray(model.geom_size[geom_id, :3], dtype=np.float64).reshape(3).copy()
            name = self._mujoco_name(mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "obj_box"
            return center, xmat, half, name
        except Exception:
            return None

    def _fingertip_site_positions(self) -> list[tuple[str, np.ndarray]]:
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return []

        names = tuple(
            f"{side}_{finger}_tip_site"
            for side in ("left", "right")
            for finger in ("thumb", "index", "middle", "ring", "little")
        )
        out: list[tuple[str, np.ndarray]] = []

        try:
            import mujoco
            for name in names:
                site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
                if site_id >= 0:
                    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy()
                    out.append((name, pos))
        except Exception:
            pass

        if out:
            return out

        site_ids = getattr(world, "site_ids", {}) or {}
        for name in names:
            for key in (name, name.replace("_site", "")):
                try:
                    if key in site_ids:
                        site_id = int(site_ids[key])
                        pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy()
                        out.append((name, pos))
                        break
                except Exception:
                    pass
        return out

    def _point_to_box_surface(self, point: np.ndarray, center: np.ndarray, xmat: np.ndarray, half: np.ndarray) -> tuple[float, float, np.ndarray]:
        local = np.asarray(xmat, dtype=np.float64).T @ (np.asarray(point, dtype=np.float64) - center)
        q = np.abs(local) - half
        outside = np.maximum(q, 0.0)
        unsigned = float(np.linalg.norm(outside))
        if unsigned > 1e-9:
            signed = unsigned
        else:
            signed = float(np.max(q))

        closest_local = np.clip(local, -half, half)
        closest_world = center + np.asarray(xmat, dtype=np.float64) @ closest_local
        return float(max(unsigned, 0.0)), signed, closest_world

    def _fingertip_face_distance_to_cube(self) -> Dict[str, float | str]:
        """
        Distance from the closest fingertip site to the actual MuJoCo cube box.

        Positive signed distance means separation from the cube surface, zero is
        touching the surface, and negative means the fingertip site is inside the
        box volume after contact/penetration.
        """
        pose = self._cube_box_geom_pose()
        tips = self._fingertip_site_positions()
        if pose is None or not tips:
            return {
                "distance": 999.0,
                "signed_distance": 999.0,
                "closest_tip": "",
                "closest_point_x": 0.0,
                "closest_point_y": 0.0,
                "closest_point_z": 0.0,
                "source": "mujoco.geom.missing",
            }

        center, xmat, half, cube_name = pose
        best_name = ""
        best_distance = 999.0
        best_signed = 999.0
        best_point = np.zeros(3, dtype=np.float64)

        for name, pos in tips:
            distance, signed, closest = self._point_to_box_surface(pos, center, xmat, half)
            if signed < best_signed:
                best_name = name
                best_distance = distance
                best_signed = signed
                best_point = closest

        return {
            "distance": float(max(best_distance, 0.0)),
            "signed_distance": float(best_signed),
            "closest_tip": best_name,
            "closest_point_x": float(best_point[0]),
            "closest_point_y": float(best_point[1]),
            "closest_point_z": float(best_point[2]),
            "source": f"mujoco.geom.{cube_name}",
        }

    def _hand_distance_to_cube(self, cube: np.ndarray) -> float:
        owner = self.owner
        tip_dist = self._fingertip_face_distance_to_cube()
        try:
            distance = float(tip_dist.get("distance", 999.0))
            if np.isfinite(distance) and distance < 999.0:
                return distance
        except Exception:
            pass

        try:
            left = np.asarray(owner.world.data.site_xpos[owner.world.site_ids["left_hand"]], dtype=np.float64)
            right = np.asarray(owner.world.data.site_xpos[owner.world.site_ids["right_hand"]], dtype=np.float64)
            return float(min(np.linalg.norm(left - cube), np.linalg.norm(right - cube)))
        except Exception:
            return 999.0

    def _desired_hover_z(self, cube: np.ndarray) -> float:
        owner = self.owner
        min_world_z = float(getattr(owner.world, "min_flight_z", 0.30))

        # Higher by default so bird legs do not touch the ground.
        leg_clearance = self._cfg("fly_to_cube_leg_clearance", 0.90)
        hover_above_cube = self._cfg("fly_to_cube_hover_above_cube", 1.02)
        max_hover = self._cfg("fly_to_cube_max_hover_z", 1.80)

        return float(np.clip(float(cube[2]) + hover_above_cube, min_world_z + leg_clearance, max_hover))

    def _vertical_pid(self, desired_z: float, current_z: float) -> Tuple[float, float]:
        step_i = int(getattr(self.owner, "global_step", 0))
        prev_step = int(getattr(self, "_z_pid_prev_step", step_i))
        dt_steps = max(1, step_i - prev_step)
        dt = max(1e-3, float(dt_steps) * self._cfg("fly_to_cube_pid_dt_per_step", 0.02))

        err = float(desired_z - current_z)

        kp = self._cfg("fly_to_cube_z_pid_kp", 2.70)
        ki = self._cfg("fly_to_cube_z_pid_ki", 0.20)
        kd = self._cfg("fly_to_cube_z_pid_kd", 0.48)

        integ = float(getattr(self, "_z_pid_integral", 0.0)) + err * dt
        integ_limit = self._cfg("fly_to_cube_z_pid_integral_limit", 1.20)
        integ = float(np.clip(integ, -integ_limit, integ_limit))

        prev_err = float(getattr(self, "_z_pid_prev_err", 0.0))
        derr = (err - prev_err) / dt

        raw = kp * err + ki * integ + kd * derr

        max_vz = self._cfg("fly_to_cube_z_pid_max_vz", 0.85)
        vz_cmd = float(np.clip(raw, -max_vz, max_vz))

        max_step_z = self._cfg("fly_to_cube_z_pid_max_direct_step", 0.095)
        direct_step_z = float(np.clip(0.42 * raw * dt, -max_step_z, max_step_z))

        self._z_pid_integral = integ
        self._z_pid_prev_err = err
        self._z_pid_prev_step = step_i
        self._z_pid_vz = vz_cmd
        return vz_cmd, direct_step_z

    def _reset_reach_distance_minimizer(self) -> None:
        sign = self._cfg("fly_to_cube_arm_side_sign", -1.0)
        yaw_abs = self._cfg("fly_to_cube_reach_shoulder_yaw_abs", 0.04)
        shoulder = self._cfg("fly_to_cube_reach_shoulder_pitch", -0.70)
        elbow = self._cfg("fly_to_cube_reach_elbow", -0.85)
        self._reach_arm_action = np.asarray(
            [sign * yaw_abs, shoulder, elbow, -sign * yaw_abs, shoulder, elbow],
            dtype=np.float32,
        )
        self._reach_palm_pitch = self._cfg("fly_to_cube_reach_palm_pitch", 0.65)
        self._reach_left_palm_roll = self._cfg("fly_to_cube_reach_left_palm_roll", 0.48)
        self._reach_right_palm_roll = self._cfg("fly_to_cube_reach_right_palm_roll", 0.52)
        self._reach_finger_base = self._cfg("fly_to_cube_reach_finger_base", 0.06)
        self._reach_axis = 0
        self._reach_dir = -1.0
        self._reach_last_dist = 999.0
        self._reach_best_dist = 999.0
        self._reach_stall_steps = 0
        self._reach_updates = 0

    def _reach_axes(self) -> tuple[str, ...]:
        return (
            "shoulder_pitch",
            "shoulder_yaw",
            "palm_roll",
            "finger_base",
        )

    def _apply_reach_axis_delta(self, axis_name: str, delta: float) -> None:
        arm = np.asarray(getattr(self, "_reach_arm_action", np.zeros(6, dtype=np.float32)), dtype=np.float32).copy()
        sign = self._cfg("fly_to_cube_arm_side_sign", -1.0)

        if axis_name == "shoulder_pitch":
            arm[1] = float(np.clip(arm[1] + delta, -1.0, 0.35))
            arm[4] = float(np.clip(arm[4] + delta, -1.0, 0.35))
        elif axis_name == "elbow":
            arm[2] = float(np.clip(arm[2] + delta, -1.0, -0.35))
            arm[5] = float(np.clip(arm[5] + delta, -1.0, -0.35))
        elif axis_name == "shoulder_yaw":
            yaw_abs = float(np.clip(abs(float(arm[0])) + delta, 0.0, 0.55))
            arm[0] = float(sign * yaw_abs)
            arm[3] = float(-sign * yaw_abs)
        elif axis_name == "palm_pitch":
            lo = self._cfg("fly_to_cube_reach_palm_pitch_min", 0.58)
            hi = self._cfg("fly_to_cube_reach_palm_pitch_max", 0.72)
            self._reach_palm_pitch = float(np.clip(float(self._reach_palm_pitch) + delta, lo, hi))
        elif axis_name == "palm_roll":
            spread = float(np.clip((float(self._reach_right_palm_roll) - float(self._reach_left_palm_roll)) * 0.5 + delta, -0.22, 0.22))
            self._reach_left_palm_roll = float(np.clip(0.50 - spread, 0.20, 0.80))
            self._reach_right_palm_roll = float(np.clip(0.50 + spread, 0.20, 0.80))
        elif axis_name == "finger_base":
            self._reach_finger_base = float(np.clip(float(self._reach_finger_base) + delta, 0.00, 0.38))

        self._reach_arm_action = np.clip(arm, -1.0, 1.0).astype(np.float32)

    def _update_reach_distance_minimizer(self, hand_dist: float, contact_active: bool) -> Dict[str, float | str | bool]:
        """
        Online coordinate descent over arm/wrist/finger commands.

        It does not assume a fixed pose or a trusted kinematic model. The only
        objective is the MuJoCo-measured fingertip-to-cube-face distance. If the
        previous command reduced distance, keep moving along that control axis;
        otherwise reverse direction and then try the next axis.
        """
        dist = float(hand_dist)
        axes = self._reach_axes()
        axis_i = int(getattr(self, "_reach_axis", 0)) % len(axes)
        axis_name = axes[axis_i]

        if not np.isfinite(dist):
            return {
                "axis": axis_name,
                "dir": float(getattr(self, "_reach_dir", -1.0)),
                "improved": False,
                "enabled": False,
            }

        goal = self._cfg("fly_to_cube_fingertip_face_goal", 0.025)
        eps = self._cfg("fly_to_cube_distance_improve_epsilon", 0.003)
        step = self._cfg("fly_to_cube_distance_minimizer_step", 0.045)

        last = float(getattr(self, "_reach_last_dist", 999.0))
        improved = dist < (last - eps)
        if dist < float(getattr(self, "_reach_best_dist", 999.0)):
            self._reach_best_dist = dist

        if contact_active:
            self._reach_stall_steps = 0
            self._reach_last_dist = dist
            self._reach_updates += 1
            return {
                "axis": axis_name,
                "dir": float(getattr(self, "_reach_dir", -1.0)),
                "improved": bool(improved),
                "enabled": False,
            }

        if dist > goal:
            if last < 900.0:
                if improved:
                    self._reach_stall_steps = 0
                else:
                    self._reach_stall_steps += 1
                    self._reach_dir *= -1.0
                    if self._reach_stall_steps >= int(self._cfg("fly_to_cube_distance_minimizer_stall_steps", 3)):
                        self._reach_axis = (axis_i + 1) % len(axes)
                        self._reach_stall_steps = 0
                        axis_i = int(self._reach_axis)
                        axis_name = axes[axis_i]

            self._apply_reach_axis_delta(axis_name, float(self._reach_dir) * step)
            self._reach_updates += 1

        self._reach_last_dist = dist
        return {
            "axis": axis_name,
            "dir": float(getattr(self, "_reach_dir", -1.0)),
            "improved": bool(improved),
            "enabled": bool(dist > goal and not contact_active),
        }

    def _hand_contact_pid(self, tactile_sum: float, hand_dist: float) -> Dict[str, float]:
        """
        PID-like contact controller.

        Goal:
            extend arms/fingers while the closest fingertip-to-cube-face gap is
            still open; after direct MuJoCo contact appears, keep a light target
            contact and continue probing.

        It controls two internal states:
            _hand_extension    -> added to arm extension
            _finger_curl_bias  -> added to finger curl / probing
        """
        step_i = int(getattr(self.owner, "global_step", 0))
        prev_step = int(getattr(self, "_hand_pid_prev_step", step_i))
        dt_steps = max(1, step_i - prev_step)
        dt = max(1e-3, float(dt_steps) * self._cfg("fly_to_cube_pid_dt_per_step", 0.02))

        target_touch = self._cfg("fly_to_cube_hand_target_touch_sum", 0.055)
        touch_scale = max(self._cfg("fly_to_cube_hand_touch_scale", 0.10), 1e-6)
        touch_norm = float(np.clip(tactile_sum / touch_scale, 0.0, 2.0))
        target_norm = float(np.clip(target_touch / touch_scale, 0.0, 1.5))

        err = target_norm - touch_norm

        kp = self._cfg("fly_to_cube_hand_pid_kp", 0.65)
        ki = self._cfg("fly_to_cube_hand_pid_ki", 0.08)
        kd = self._cfg("fly_to_cube_hand_pid_kd", 0.18)

        integ = float(getattr(self, "_hand_pid_integral", 0.0)) + err * dt
        integ_limit = self._cfg("fly_to_cube_hand_pid_integral_limit", 1.0)
        integ = float(np.clip(integ, -integ_limit, integ_limit))

        prev_err = float(getattr(self, "_hand_pid_prev_err", 0.0))
        derr = (err - prev_err) / dt

        drive = kp * err + ki * integ + kd * derr

        # If fingertip sites are still separated from the cube face and no direct
        # MuJoCo hand-cube contact exists, keep extending even if tactile is quiet.
        hand_goal = self._cfg("fly_to_cube_fingertip_face_goal", 0.025)
        near_gap = max(self._cfg("fly_to_cube_fingertip_face_near", 0.060), 1e-6)
        no_contact = tactile_sum < self._cfg("fly_to_cube_contact_enter_sum", 0.015)
        if no_contact and hand_dist > hand_goal:
            gap_gain = float(np.clip((hand_dist - hand_goal) / near_gap, 0.15, 2.0))
            drive += self._cfg("fly_to_cube_hand_no_contact_boost", 0.28) * gap_gain

        ext = float(getattr(self, "_hand_extension", 0.0))
        curl = float(getattr(self, "_finger_curl_bias", 0.0))

        ext_rate = self._cfg("fly_to_cube_hand_extension_rate", 0.020)
        curl_rate = self._cfg("fly_to_cube_finger_curl_rate", 0.014)

        ext = float(np.clip(ext + drive * ext_rate, 0.0, 1.0))
        curl = float(np.clip(curl + drive * curl_rate, 0.0, 1.0))

        # When contact is too strong, release a little; when it is light,
        # keep small probing.
        release_threshold = self._cfg("fly_to_cube_hand_release_touch_sum", 0.14)
        if tactile_sum > release_threshold:
            ext = max(0.0, ext - 0.012)
            curl = max(0.0, curl - 0.020)

        self._hand_pid_integral = integ
        self._hand_pid_prev_err = err
        self._hand_pid_prev_step = step_i
        self._hand_pid_drive = float(drive)
        self._hand_extension = ext
        self._finger_curl_bias = curl

        return {
            "touch_norm": float(touch_norm),
            "target_norm": float(target_norm),
            "err": float(err),
            "drive": float(drive),
            "extension": float(ext),
            "curl": float(curl),
        }

    def _joint_raw_from_rad(self, value: float, lo: float, hi: float) -> float:
        return float(np.clip(2.0 * (float(value) - lo) / max(hi - lo, 1e-9) - 1.0, -1.0, 1.0))

    def _arm_raw_from_ik(self, target_world: np.ndarray, yaw: float) -> Optional[np.ndarray]:
        """
        Convert a world-space reach target into the normalized arm vector expected
        by MujocoLiveWorldV57: L yaw/pitch/elbow + R yaw/pitch/elbow in [-1, 1].
        """
        if simple_arm_ik_local is None or ArmActuatorConfig is None:
            return None

        owner = self.owner
        world = getattr(owner, "world", None)
        data = getattr(world, "data", None)
        if data is None:
            return None

        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rows = []
        for site_name in ("left_arm_mount_site", "right_arm_mount_site"):
            shoulder = None
            try:
                import mujoco
                sid = int(mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, site_name))
                if sid >= 0:
                    shoulder = np.asarray(data.site_xpos[sid], dtype=np.float64).reshape(3).copy()
            except Exception:
                shoulder = None

            if shoulder is None:
                try:
                    sid = int((getattr(world, "site_ids", {}) or {})[site_name])
                    shoulder = np.asarray(data.site_xpos[sid], dtype=np.float64).reshape(3).copy()
                except Exception:
                    return None

            delta = np.asarray(target_world, dtype=np.float64).reshape(3) - shoulder
            local = np.asarray(
                [
                    c * float(delta[0]) + s * float(delta[1]),
                    -s * float(delta[0]) + c * float(delta[1]),
                    float(delta[2]),
                ],
                dtype=np.float64,
            )
            rows.append(simple_arm_ik_local(local, ArmActuatorConfig()))

        raw = np.asarray(
            [
                self._joint_raw_from_rad(rows[0][0], -1.48353, 1.48353),
                self._joint_raw_from_rad(rows[0][1], -1.22173, 1.22173),
                self._joint_raw_from_rad(rows[0][2], 0.0872665, 2.53073),
                self._joint_raw_from_rad(rows[1][0], -1.48353, 1.48353),
                self._joint_raw_from_rad(rows[1][1], -1.22173, 1.22173),
                self._joint_raw_from_rad(rows[1][2], 0.0872665, 2.53073),
            ],
            dtype=np.float32,
        )
        return np.clip(raw, -1.0, 1.0).astype(np.float32)

    def _cube_reach_target_world(self, agent: np.ndarray) -> Optional[np.ndarray]:
        pose = self._cube_box_geom_pose()
        if pose is None:
            return None

        center, xmat, half, _name = pose
        agent = np.asarray(agent, dtype=np.float64).reshape(3)

        local_agent = xmat.T @ (agent - center)
        # Choose the side face in the horizontal plane. If Z participates here,
        # a hovering agent targets the cube top and the arms simply drop down.
        axis = int(np.argmax(np.abs(local_agent[:2])))
        sign = 1.0 if local_agent[axis] >= 0.0 else -1.0

        target_local = np.zeros(3, dtype=np.float64)
        target_local[axis] = sign * (half[axis] + self._cfg("fly_to_cube_reach_surface_offset", 0.015))

        # Aim near the face center, but keep it reachable from the current hover.
        desired_z = float(center[2] + self._cfg("fly_to_cube_reach_target_z_above_center", 0.28))
        z_local = float((xmat.T @ (np.asarray([center[0], center[1], desired_z], dtype=np.float64) - center))[2])
        target_local[2] = float(np.clip(z_local, -half[2] * 0.55, half[2] * 0.90))

        return center + xmat @ target_local

    def _arm_action(self, phase: str, extension: float = 0.0, pulse: float = 0.0, target_world: Optional[np.ndarray] = None, yaw: float = 0.0) -> np.ndarray:
        """
        Stronger arm lift.

        Previous controller mostly waited for hand PID extension, so arms could
        stay down. This version raises shoulders by phase first, then adds PID
        extension.
        """
        sign = self._cfg("fly_to_cube_arm_side_sign", -1.0)
        ext = float(np.clip(extension, 0.0, 1.0))

        if phase in ("reach", "palpate"):
            arr = np.asarray(getattr(self, "_reach_arm_action", np.zeros(6, dtype=np.float32)), dtype=np.float32).copy()
            if phase == "palpate":
                arr[1] = float(np.clip(arr[1] + 0.020 * np.sin(pulse * 0.7), -1.0, 1.0))
                arr[2] = float(np.clip(arr[2] + 0.025 * np.sin(pulse * 1.1), -1.0, 1.0))
                arr[4] = float(np.clip(arr[4] + 0.020 * np.sin(pulse * 0.7 + 0.3), -1.0, 1.0))
                arr[5] = float(np.clip(arr[5] + 0.025 * np.sin(pulse * 1.1 + 0.3), -1.0, 1.0))
            return np.clip(arr, -1.0, 1.0).astype(np.float32)

        if phase == "approach":
            yaw_abs, shoulder, elbow = 0.05, 0.30, -0.12
        elif phase == "align":
            yaw_abs, shoulder, elbow = 0.08, -0.20, 0.35
        else:
            yaw_abs, shoulder, elbow = 0.10, -0.20, 0.35

        if phase in ("reach", "palpate"):
            shoulder += -0.16 * ext
            elbow += -0.10 * ext
        else:
            shoulder += 0.18 * ext
            elbow += -0.18 * ext

        if phase == "palpate":
            shoulder += 0.035 * np.sin(pulse * 0.7)
            elbow += 0.050 * np.sin(pulse * 1.1)

        arr = np.asarray(
            [sign * yaw_abs, shoulder, elbow, -sign * yaw_abs, shoulder, elbow],
            dtype=np.float32,
        )
        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def _hand_action(self, phase: str, extension: float = 0.0, curl: float = 0.0, pulse: float = 0.0) -> np.ndarray:
        """
        Hand control vector with new mcp_yaw joints.

        New order, 44 dims:
            per hand = palm_roll, palm_pitch + 5 * (mcp_yaw, mcp, pip, dip)
        """
        ext = float(np.clip(extension, 0.0, 1.0))
        curl = float(np.clip(curl, 0.0, 1.0))

        if phase == "approach":
            bases, palm_pitch, amp = [0.03, 0.04, 0.04, 0.04, 0.03], 0.45, 0.0
        elif phase == "align":
            bases, palm_pitch, amp = [0.04, 0.05, 0.05, 0.05, 0.04], 0.50, 0.0
        elif phase == "reach":
            base = float(getattr(self, "_reach_finger_base", self._cfg("fly_to_cube_reach_finger_base", 0.06)))
            bases = [base * 0.80, base, base * 1.10, base, base * 0.85]
            palm_pitch = float(getattr(self, "_reach_palm_pitch", self._cfg("fly_to_cube_reach_palm_pitch", 0.65)))
            amp = 0.015
        else:
            base = max(float(getattr(self, "_reach_finger_base", 0.06)), self._cfg("fly_to_cube_palpate_finger_base", 0.16))
            bases = [base * 0.75, base * 1.05, base * 1.20, base * 1.05, base * 0.85]
            palm_pitch = float(getattr(self, "_reach_palm_pitch", self._cfg("fly_to_cube_palpate_palm_pitch", 0.63)))
            amp = 0.07

        yaw_left = [0.12, 0.30, 0.50, 0.70, 0.88]
        yaw_right = [0.88, 0.70, 0.50, 0.30, 0.12]

        vals = []
        for side_idx, yaw_values in enumerate((yaw_left, yaw_right)):
            if phase in ("reach", "palpate"):
                left_roll = float(getattr(self, "_reach_left_palm_roll", self._cfg("fly_to_cube_reach_left_palm_roll", 0.48)))
                right_roll = float(getattr(self, "_reach_right_palm_roll", self._cfg("fly_to_cube_reach_right_palm_roll", 0.52)))
                palm_roll = left_roll if side_idx == 0 else right_roll
            else:
                palm_roll = 0.38 if side_idx == 0 else 0.62
            vals.extend([palm_roll, float(np.clip(palm_pitch, 0.0, 1.0))])

            for i, base in enumerate(bases):
                probe = amp * np.sin(pulse + i * 0.75 + side_idx * 0.35)
                mcp_yaw = float(np.clip(yaw_values[i], 0.0, 1.0))
                v = float(np.clip(base + 0.14 * ext + 0.18 * curl + probe, 0.0, 0.95))
                vals.extend([mcp_yaw, v, v, v])

        return np.asarray(vals, dtype=np.float32)

    def _direct_nudge_to_standoff(self, target_xyz: np.ndarray, desired_yaw: float, dist_to_cube: float, keepout: float) -> None:
        owner = self.owner
        if not hasattr(owner, "world") or not hasattr(owner.world, "cam_pos"):
            return

        try:
            cur = np.asarray(owner.world.cam_pos, dtype=np.float64)
            err = np.asarray(target_xyz, dtype=np.float64) - cur

            max_step_xy = self._cfg("fly_to_cube_max_direct_xy_step", 0.045)
            max_step_z = self._cfg("fly_to_cube_max_direct_z_step", 0.095)

            if dist_to_cube < keepout:
                max_step_xy = min(max_step_xy, 0.020)

            step = np.zeros(3, dtype=np.float64)
            step[:2] = np.clip(err[:2] * 0.16, -max_step_xy, max_step_xy)

            _vz_cmd, pid_step_z = self._vertical_pid(float(target_xyz[2]), float(cur[2]))
            step[2] = float(np.clip(pid_step_z, -max_step_z, max_step_z))

            owner.world.cam_pos[:] = cur + step

            cur_yaw = float(np.deg2rad(float(getattr(owner.world, "yaw_deg", 0.0))))
            yaw_err = float((desired_yaw - cur_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            owner.world.yaw_deg = float(getattr(owner.world, "yaw_deg", 0.0)) + float(np.rad2deg(np.clip(yaw_err * 0.18, -0.07, 0.07)))

            # Level body and keep a configurable downward gaze.
            if hasattr(owner.world, "roll_deg"):
                owner.world.roll_deg = float(0.84 * float(getattr(owner.world, "roll_deg", 0.0)))
            if hasattr(owner.world, "pitch_deg"):
                target_pitch_deg = self._cfg("fly_to_cube_direct_body_pitch_deg", -5.0)
                owner.world.pitch_deg = float(0.84 * float(getattr(owner.world, "pitch_deg", 0.0)) + 0.16 * target_pitch_deg)

            if hasattr(owner.world, "_clamp_flight_zone"):
                owner.world._clamp_flight_zone()
            if hasattr(owner.world, "_update_rig_pose"):
                owner.world._update_rig_pose()
        except Exception as e:
            if not hasattr(owner, "_adaptive_scenario_direct_warned"):
                print(f"[adaptive_scenario] direct standoff nudge failed: {e}")
                owner._adaptive_scenario_direct_warned = True

    def step(self) -> Optional[ScenarioCommand]:
        owner = self.owner
        if not self.active:
            return None

        step_i = int(getattr(owner, "global_step", 0))
        if step_i - self.started_step > self.timeout_steps:
            # Timeout only exits the script; normal user controls can restart it.
            self.stop("timeout")
            return ScenarioCommand(
                body=np.zeros(9, dtype=np.float32),
                arm=self._arm_action("align"),
                hand=self._hand_action("align"),
                leg=np.zeros(18, dtype=np.float32),
                status=dict(self.status),
            )

        agent, yaw = self._agent_xyz_yaw()
        cube = self._cube_xyz()

        delta = cube - agent
        dist_xy = float(np.linalg.norm(delta[:2]))
        if dist_xy > 1e-6:
            to_cube_xy = delta[:2] / dist_xy
        else:
            to_cube_xy = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

        desired_yaw = float(np.arctan2(to_cube_xy[1], to_cube_xy[0]))
        desired_z = self._desired_hover_z(cube)

        keepout = self._cfg("fly_to_cube_body_keepout", 0.78)
        keepout_buffer = self._cfg("fly_to_cube_body_keepout_buffer", 0.04)
        min_reach_distance = max(keepout + keepout_buffer, self._cfg("fly_to_cube_min_arm_reach_distance", 0.82))
        max_reach_distance = self._cfg("fly_to_cube_max_arm_reach_distance", 1.55)

        tactile = self._tactile_features()
        tactile_sum = float(tactile["sum"])
        tactile_max = float(tactile["max"])
        contact_count = int(tactile.get("count", 0))
        contact_active = contact_count > 0 and tactile_sum >= self._cfg("fly_to_cube_contact_enter_sum", 0.015)
        palm_sum = float(tactile.get("palm_sum", tactile_sum))
        palm_max = float(tactile.get("palm_max", tactile_max))
        palm_count = int(tactile.get("palm_count", contact_count))
        finger_sum = float(tactile.get("finger_sum", tactile_sum))
        finger_max = float(tactile.get("finger_max", tactile_max))
        finger_count = int(tactile.get("finger_count", contact_count))
        palm_contact_active = palm_count > 0 and palm_sum >= self._cfg("fly_to_cube_contact_enter_sum", 0.015)
        finger_contact_active = finger_count > 0 and finger_sum >= self._cfg("fly_to_cube_contact_enter_sum", 0.015)
        fingertip_face = self._fingertip_face_distance_to_cube()
        palm_face = self._palm_face_distance_to_cube()
        try:
            hand_dist = float(fingertip_face.get("distance", 999.0))
            if not np.isfinite(hand_dist):
                hand_dist = self._hand_distance_to_cube(cube)
        except Exception:
            hand_dist = self._hand_distance_to_cube(cube)
        fingertip_dist = hand_dist
        fingertip_face_signed = float(fingertip_face.get("signed_distance", hand_dist))
        try:
            palm_dist = float(palm_face.get("distance", hand_dist))
            if not np.isfinite(palm_dist):
                palm_dist = hand_dist
        except Exception:
            palm_dist = hand_dist
        palm_face_signed = float(palm_face.get("signed_distance", palm_dist))

        # Reach/palpate should be solved by arms/fingers. Body standoff stays
        # fixed by default so the controller does not substitute whole-body
        # approach for failed arm extension.
        if self.phase in ("reach", "palpate"):
            no_contact = not contact_active
            face_goal = self._cfg("fly_to_cube_fingertip_face_goal", 0.025)
            if no_contact and hand_dist > 0.002:
                self.no_contact_reach_steps += 1
            else:
                self.no_contact_reach_steps = 0

            body_adapt = bool(self._cfg("fly_to_cube_body_adapt_to_fingertip_gap", 0.0))
            start_steps = int(self._cfg("fly_to_cube_reach_distance_adapt_start_steps", 20))
            if body_adapt and self.no_contact_reach_steps > start_steps:
                near_gap = max(self._cfg("fly_to_cube_fingertip_face_near", 0.060), 1e-6)
                gap_gain = float(np.clip(max(hand_dist - face_goal, 0.0) / near_gap, 0.25, 2.5))
                adapt_rate = self._cfg("fly_to_cube_reach_distance_adapt_rate", 0.0040)
                self.adaptive_reach_distance = max(min_reach_distance, self.adaptive_reach_distance - adapt_rate * gap_gain)
        else:
            self.no_contact_reach_steps = 0

        self.adaptive_reach_distance = float(np.clip(self.adaptive_reach_distance, min_reach_distance, max_reach_distance))

        target_xy = cube[:2] - to_cube_xy * self.adaptive_reach_distance
        target_xyz = np.asarray([target_xy[0], target_xy[1], desired_z], dtype=np.float64)

        ring_err = float(np.linalg.norm(target_xy - agent[:2]))
        z_err = float(desired_z - float(agent[2]))
        yaw_err = float((desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)

        # Phase transitions.
        # reach starts only after body is aligned vertically; palpate starts after
        # arms extend enough OR tactile appears. Then palpation continues.
        if self.phase == "approach" and ring_err < 0.22:
            self.phase = "align"

        if self.phase == "align":
            align_z_tol = self._cfg("fly_to_cube_align_z_tolerance", 0.16)
            if ring_err < 0.20 and abs(z_err) < align_z_tol and abs(yaw_err) < 0.24:
                self.align_hold_steps += 1
            else:
                self.align_hold_steps = 0
            if self.align_hold_steps > int(self._cfg("fly_to_cube_align_hold_steps", 8)):
                self.phase = "reach"
                self._reach_last_dist = 999.0
                self._reach_best_dist = min(float(getattr(self, "_reach_best_dist", 999.0)), hand_dist)

        if self.phase == "reach":
            if (
                contact_active
                or hand_dist < self._cfg("fly_to_cube_fingertip_face_near", 0.060)
            ):
                self.phase = "palpate"

        if contact_active:
            self.contact_hold_steps += 1
        else:
            self.contact_hold_steps = 0

        if self.phase in ("reach", "palpate"):
            minimizer = self._update_reach_distance_minimizer(hand_dist=hand_dist, contact_active=contact_active)
        else:
            minimizer = {
                "axis": "",
                "dir": 0.0,
                "improved": False,
                "enabled": False,
            }

        # Body stays on the standoff ring; direct nudge never targets cube center.
        self._direct_nudge_to_standoff(target_xyz, desired_yaw, dist_xy, keepout)

        # Re-read pose after direct nudge.
        agent, yaw = self._agent_xyz_yaw()
        target_delta = target_xyz - agent
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        local_target_x = c * float(target_delta[0]) + s * float(target_delta[1])
        local_target_y = -s * float(target_delta[0]) + c * float(target_delta[1])

        z_err = float(desired_z - float(agent[2]))
        yaw_err = float((desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
        pid_vz, _pid_direct_step = self._vertical_pid(desired_z, float(agent[2]))

        if self.phase == "approach":
            vx = float(np.clip(local_target_x * 1.45, -0.55, 0.55))
            vy = float(np.clip(local_target_y * 1.45, -0.42, 0.42))
            vz = float(pid_vz)
            yaw_cmd = float(np.clip(yaw_err * 0.50, -0.30, 0.30))
        elif self.phase == "align":
            vx = float(np.clip(local_target_x * 0.65, -0.18, 0.18))
            vy = float(np.clip(local_target_y * 0.65, -0.16, 0.16))
            vz = float(np.clip(pid_vz, -0.34, 0.34))
            yaw_cmd = float(np.clip(yaw_err * 0.40, -0.22, 0.22))
        else:
            # Reach/palpate: keep body nearly still; motion comes from arms/fingers.
            vx = float(np.clip(local_target_x * 0.32, -0.08, 0.08))
            vy = float(np.clip(local_target_y * 0.32, -0.08, 0.08))
            vz = float(np.clip(pid_vz, -0.24, 0.24))
            yaw_cmd = float(np.clip(yaw_err * 0.25, -0.14, 0.14))

        # If body got too close, back away.
        if dist_xy < keepout:
            vx = -0.28
            vy = 0.0

        # Hand/arm PID runs only once body is aligned.
        if self.phase in ("reach", "palpate"):
            hand_pid = self._hand_contact_pid(tactile_sum=tactile_sum, hand_dist=hand_dist)
        else:
            hand_pid = {
                "touch_norm": 0.0,
                "target_norm": 0.0,
                "err": 0.0,
                "drive": 0.0,
                "extension": 0.0,
                "curl": 0.0,
            }
            # Slowly reset if user restarts.
            self._hand_extension = max(0.0, float(getattr(self, "_hand_extension", 0.0)) - 0.02)
            self._finger_curl_bias = max(0.0, float(getattr(self, "_finger_curl_bias", 0.0)) - 0.02)

        t = float(step_i - self.started_step)
        pulse = t * self._cfg("fly_to_cube_palpate_pulse_speed", 0.30)
        reach_target_world = self._cube_reach_target_world(agent) if self.phase in ("reach", "palpate") else None

        arm = self._arm_action(
            self.phase,
            extension=float(hand_pid["extension"]),
            pulse=pulse,
            target_world=reach_target_world,
            yaw=yaw,
        )
        hand = self._hand_action(self.phase, extension=float(hand_pid["extension"]), curl=float(hand_pid["curl"]), pulse=pulse)

        body = np.asarray(
            [vx, vy, vz, yaw_cmd, -0.06, 0.0, 0.0, 0.24, 0.0],
            dtype=np.float32,
        )
        leg = np.zeros(18, dtype=np.float32)

        status = {
            "active": True,
            "scenario": self.name,
            "phase": self.phase,
            "body_dist": float(dist_xy),
            "ring_err": float(np.linalg.norm(target_xyz[:2] - agent[:2])),
            "reach_distance": float(self.adaptive_reach_distance),
            "arm_side_sign": float(self._cfg("fly_to_cube_arm_side_sign", -1.0)),
            "keepout": float(keepout),
            "hand_dist": float(hand_dist),
            "palm_face_dist": float(palm_dist),
            "palm_face_signed_dist": float(palm_face_signed),
            "palm_face_source": str(palm_face.get("source", "")),
            "closest_palm": str(palm_face.get("closest_palm", "")),
            "fingertip_face_dist": float(fingertip_dist),
            "fingertip_face_signed_dist": float(fingertip_face_signed),
            "fingertip_face_source": str(fingertip_face.get("source", "")),
            "closest_fingertip": str(fingertip_face.get("closest_tip", "")),
            "reach_target_x": float(reach_target_world[0]) if reach_target_world is not None else 0.0,
            "reach_target_y": float(reach_target_world[1]) if reach_target_world is not None else 0.0,
            "reach_target_z": float(reach_target_world[2]) if reach_target_world is not None else 0.0,
            "arm_cmd_l_yaw": float(arm[0]),
            "arm_cmd_l_pitch": float(arm[1]),
            "arm_cmd_l_elbow": float(arm[2]),
            "arm_cmd_r_yaw": float(arm[3]),
            "arm_cmd_r_pitch": float(arm[4]),
            "arm_cmd_r_elbow": float(arm[5]),
            "distance_minimizer_axis": str(minimizer.get("axis", "")),
            "distance_minimizer_dir": float(minimizer.get("dir", 0.0)),
            "distance_minimizer_improved": bool(minimizer.get("improved", False)),
            "distance_minimizer_enabled": bool(minimizer.get("enabled", False)),
            "distance_minimizer_best": float(getattr(self, "_reach_best_dist", 999.0)),
            "reach_palm_pitch": float(getattr(self, "_reach_palm_pitch", 0.0)),
            "reach_finger_base": float(getattr(self, "_reach_finger_base", 0.0)),
            "tactile_sum": float(tactile_sum),
            "tactile_max": float(tactile_max),
            "palm_touch_sum": float(palm_sum),
            "palm_touch_max": float(palm_max),
            "palm_contact_count": int(palm_count),
            "palm_contact_active": bool(palm_contact_active),
            "palm_contact_hold_steps": int(getattr(self, "palm_contact_hold_steps", 0)),
            "finger_touch_sum": float(finger_sum),
            "finger_touch_max": float(finger_max),
            "finger_contact_count": int(finger_count),
            "finger_contact_active": bool(finger_contact_active),
            "finger_contact_hold_steps": int(getattr(self, "finger_contact_hold_steps", 0)),
            "grasp_lost_contact_steps": int(getattr(self, "grasp_lost_contact_steps", 0)),
            "contact_count": int(contact_count),
            "contact_active": bool(contact_active),
            "tactile_source": str(tactile["source"]),
            "contact_hold_steps": int(self.contact_hold_steps),
            "desired_z": float(desired_z),
            "agent_z": float(agent[2]),
            "z_err": float(z_err),
            "z_pid_vz": float(getattr(self, "_z_pid_vz", 0.0)),
            "hand_pid_drive": float(hand_pid["drive"]),
            "hand_extension": float(hand_pid["extension"]),
            "finger_curl": float(hand_pid["curl"]),
            "hand_touch_norm": float(hand_pid["touch_norm"]),
            "yaw_err": float(yaw_err),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "yaw_cmd": float(yaw_cmd),
        }
        self.status = status

        if step_i % 20 == 0:
            print(
                "\033[32m [adaptive_scenario] fly_to_cube_palpate \033[0m"
                f"phase={self.phase} body={dist_xy:.3f} ring={status['ring_err']:.3f} "
                f"tip_face={hand_dist:.3f}/{fingertip_face_signed:.3f} touch={tactile_sum:.3f}/{tactile_max:.3f} "
                f"contact={int(contact_active)} count={contact_count} no_contact_steps={self.no_contact_reach_steps} "
                f"reach={self.adaptive_reach_distance:.3f} arm=({arm[0]:.2f},{arm[1]:.2f},{arm[2]:.2f}) "
                f"min={minimizer.get('axis','')}:{float(minimizer.get('dir',0.0)):.0f} "
                f"ext={status['hand_extension']:.2f} curl={status['finger_curl']:.2f} "
                f"z={agent[2]:.3f}->{desired_z:.3f} vz={vz:.3f} src={status['tactile_source']}"
            )

        return ScenarioCommand(body=body, arm=arm, hand=hand, leg=leg, status=status)


class FlyToSmallCubeGraspRotateScenario(FlyToCubePalpateScenario):
    """
    Adaptive scenario for the small cube resting on the large cube:

        approach -> align -> reach -> grasp -> lift -> rotate

    It reuses the fingertip-to-box distance minimizer from the palpation
    scenario, but all geometry/contact truth is scoped to obj_box_small.
    """

    name = "fly_to_small_cube_grasp_rotate"

    def __init__(self, owner):
        super().__init__(owner)
        self.phase_started_step = int(getattr(owner, "global_step", 0))
        self.grasp_hold_steps = 0
        self.lift_started_z = 0.0
        self.ground_toe_contact_hold_steps = 0
        self._ground_leg_crouch = 0.0
        self._ground_leg_hard_contact = False
        self._ground_leg_contact_ramp = 0.0
        self._ground_toe_support_ratio = 0.0
        self._ground_leg_balance_pitch = 0.0
        self._ground_leg_balance_roll = 0.0
        self._ground_leg_reach_lean = 0.0
        self._ground_leg_com_pitch = 0.0
        self._ground_ankle_pitch_bias = 0.0
        self.palm_contact_hold_steps = 0
        self.finger_contact_hold_steps = 0
        self.grasp_lost_contact_steps = 0

    def _cfg(self, key: str, default: float) -> float:
        if key.startswith("fly_to_cube_"):
            small_key = "fly_to_small_cube_" + key[len("fly_to_cube_"):]
            cfg = getattr(self.owner, "cfg", None)

            try:
                section = getattr(cfg, "adaptive_scenario_controller", None)
                section_val = _adaptive_section_value(section, small_key)
                if section_val is not _MISSING:
                    val = _cfg_float_value(section_val, small_key, "section")
                    if val is not _MISSING:
                        return val
            except Exception:
                pass

            try:
                cfg_val = _config_value(cfg, small_key)
                if cfg_val is not _MISSING:
                    val = _cfg_float_value(cfg_val, small_key, "root")
                    if val is not _MISSING:
                        return val
            except Exception:
                pass

        return super()._cfg(key, default)

    def start(self) -> None:
        super().start()
        self._set_phase("approach")
        self.grasp_hold_steps = 0
        self.lift_started_z = 0.0
        self.ground_toe_contact_hold_steps = 0
        self._ground_leg_crouch = 0.0
        self._ground_leg_hard_contact = False
        self._ground_leg_contact_ramp = 0.0
        self._ground_toe_support_ratio = 0.0
        self._ground_leg_balance_pitch = 0.0
        self._ground_leg_balance_roll = 0.0
        self._ground_leg_reach_lean = 0.0
        self._ground_leg_com_pitch = 0.0
        self._ground_ankle_pitch_bias = 0.0
        self.palm_contact_hold_steps = 0
        self.finger_contact_hold_steps = 0
        self.grasp_lost_contact_steps = 0
        self.adaptive_reach_distance = self._cfg("fly_to_cube_arm_reach_distance", 0.76)
        print("[adaptive_scenario] fly_to_small_cube_grasp_rotate started")

    def _set_phase(self, phase: str) -> None:
        if self.phase != phase:
            self.phase = phase
            self.phase_started_step = int(getattr(self.owner, "global_step", 0))

    def _phase_age(self) -> int:
        return max(0, int(getattr(self.owner, "global_step", 0)) - int(getattr(self, "phase_started_step", 0)))

    def _cube_xyz(self) -> np.ndarray:
        pose = self._cube_box_geom_pose()
        if pose is not None:
            return np.asarray(pose[0], dtype=np.float64).reshape(3).copy()

        owner = self.owner
        try:
            if hasattr(owner.world, "get_object_pos"):
                arr = np.asarray(owner.world.get_object_pos("obj_box_small"), dtype=np.float64).reshape(3)
                if np.isfinite(arr).all():
                    return arr
        except Exception:
            pass

        return super()._cube_xyz()

    def _cube_box_geom_pose(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return None

        try:
            import mujoco
            geom_id = self._mujoco_id_by_name(mujoco.mjtObj.mjOBJ_GEOM, ("obj_box_small",))
            if geom_id < 0:
                return None
            center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3).copy()
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy()
            half = np.asarray(model.geom_size[geom_id, :3], dtype=np.float64).reshape(3).copy()
            name = self._mujoco_name(mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "obj_box_small"
            return center, xmat, half, name
        except Exception:
            return None

    def _mujoco_hand_cube_contact_features(self) -> Dict[str, float | str]:
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return {"sum": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "source": "mujoco.contact.none"}

        hand_keys = ("hand", "finger", "fingertip", "tip", "palm", "thumb", "index", "middle", "ring", "little")
        palm_keys = ("palm",)
        finger_keys = ("thumb", "index", "middle", "ring", "little", "finger", "tip", "proximal", "middle", "distal")
        object_names = {"obj_box_small"}
        values = []
        palm_values = []
        finger_values = []
        left_palm_values = []
        right_palm_values = []
        left_finger_values = []
        right_finger_values = []

        try:
            import mujoco
            obj_geom = mujoco.mjtObj.mjOBJ_GEOM
        except Exception:
            obj_geom = None

        try:
            ncon = int(getattr(data, "ncon", 0))
            floor = self._cfg("fly_to_cube_contact_force_floor", 0.020)
            scale = max(self._cfg("fly_to_cube_contact_force_scale", 1.0), 1e-6)

            for ci in range(ncon):
                con = data.contact[ci]
                g1 = int(con.geom1)
                g2 = int(con.geom2)

                if obj_geom is not None:
                    n1 = self._mujoco_name(obj_geom, g1).lower()
                    n2 = self._mujoco_name(obj_geom, g2).lower()
                else:
                    n1 = n2 = ""

                hand_small_cube = (
                    (any(k in n1 for k in hand_keys) and n2 in object_names)
                    or
                    (any(k in n2 for k in hand_keys) and n1 in object_names)
                )
                if not hand_small_cube:
                    continue

                force = self._mujoco_contact_force_scalar(ci)
                value = max(force / scale, floor)
                values.append(value)
                hand_name = n1 if any(k in n1 for k in hand_keys) else n2
                if any(k in hand_name for k in palm_keys):
                    palm_values.append(value)
                    if "left_" in hand_name:
                        left_palm_values.append(value)
                    elif "right_" in hand_name:
                        right_palm_values.append(value)
                elif any(k in hand_name for k in finger_keys):
                    finger_values.append(value)
                    if "left_" in hand_name:
                        left_finger_values.append(value)
                    elif "right_" in hand_name:
                        right_finger_values.append(value)
        except Exception:
            values = []

        if not values:
            return {
                "sum": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "count": 0,
                "palm_sum": 0.0,
                "palm_max": 0.0,
                "palm_count": 0,
                "left_palm_sum": 0.0,
                "left_palm_max": 0.0,
                "left_palm_count": 0,
                "right_palm_sum": 0.0,
                "right_palm_max": 0.0,
                "right_palm_count": 0,
                "finger_sum": 0.0,
                "finger_max": 0.0,
                "finger_count": 0,
                "left_finger_sum": 0.0,
                "left_finger_max": 0.0,
                "left_finger_count": 0,
                "right_finger_sum": 0.0,
                "right_finger_max": 0.0,
                "right_finger_count": 0,
                "source": "mujoco.contact.none",
            }

        arr = np.asarray(values, dtype=np.float32)
        palm_arr = np.asarray(palm_values, dtype=np.float32) if palm_values else np.zeros(0, dtype=np.float32)
        finger_arr = np.asarray(finger_values, dtype=np.float32) if finger_values else np.zeros(0, dtype=np.float32)
        left_palm_arr = np.asarray(left_palm_values, dtype=np.float32) if left_palm_values else np.zeros(0, dtype=np.float32)
        right_palm_arr = np.asarray(right_palm_values, dtype=np.float32) if right_palm_values else np.zeros(0, dtype=np.float32)
        left_finger_arr = np.asarray(left_finger_values, dtype=np.float32) if left_finger_values else np.zeros(0, dtype=np.float32)
        right_finger_arr = np.asarray(right_finger_values, dtype=np.float32) if right_finger_values else np.zeros(0, dtype=np.float32)
        return {
            "sum": float(arr.sum()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "count": int(arr.size),
            "palm_sum": float(palm_arr.sum()) if palm_arr.size else 0.0,
            "palm_max": float(palm_arr.max()) if palm_arr.size else 0.0,
            "palm_count": int(palm_arr.size),
            "left_palm_sum": float(left_palm_arr.sum()) if left_palm_arr.size else 0.0,
            "left_palm_max": float(left_palm_arr.max()) if left_palm_arr.size else 0.0,
            "left_palm_count": int(left_palm_arr.size),
            "right_palm_sum": float(right_palm_arr.sum()) if right_palm_arr.size else 0.0,
            "right_palm_max": float(right_palm_arr.max()) if right_palm_arr.size else 0.0,
            "right_palm_count": int(right_palm_arr.size),
            "finger_sum": float(finger_arr.sum()) if finger_arr.size else 0.0,
            "finger_max": float(finger_arr.max()) if finger_arr.size else 0.0,
            "finger_count": int(finger_arr.size),
            "left_finger_sum": float(left_finger_arr.sum()) if left_finger_arr.size else 0.0,
            "left_finger_max": float(left_finger_arr.max()) if left_finger_arr.size else 0.0,
            "left_finger_count": int(left_finger_arr.size),
            "right_finger_sum": float(right_finger_arr.sum()) if right_finger_arr.size else 0.0,
            "right_finger_max": float(right_finger_arr.max()) if right_finger_arr.size else 0.0,
            "right_finger_count": int(right_finger_arr.size),
            "source": "mujoco.contact.hand_obj_box_small",
        }

    def _agent_roll_pitch(self) -> Tuple[float, float]:
        owner = self.owner
        ctrl = getattr(owner, "dynamic_agent_rig_controller", None)
        if ctrl is not None and hasattr(ctrl, "qpos_adr"):
            try:
                qpos = owner.world.data.qpos[ctrl.qpos_adr:ctrl.qpos_adr + 7]
                qw, qx, qy, qz = np.asarray(qpos[3:7], dtype=np.float64)
                sinr_cosp = 2.0 * (qw * qx + qy * qz)
                cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                roll = float(np.arctan2(sinr_cosp, cosr_cosp))
                sinp = 2.0 * (qw * qy - qz * qx)
                pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
                return roll, pitch
            except Exception:
                pass
        return 0.0, 0.0

    def _head_track_target_action(self, agent: np.ndarray, yaw: float, target: np.ndarray) -> tuple[float, float, float]:
        delta = np.asarray(target, dtype=np.float64).reshape(3) - np.asarray(agent, dtype=np.float64).reshape(3)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        local_x = c * float(delta[0]) + s * float(delta[1])
        local_y = -s * float(delta[0]) + c * float(delta[1])
        local_z = float(delta[2])
        horiz = max(float(np.hypot(local_x, local_y)), 1e-6)

        yaw_angle = float(np.arctan2(local_y, local_x))
        pitch_angle = float(np.arctan2(local_z, horiz))
        head_yaw = float(np.clip(yaw_angle * self._cfg("fly_to_cube_head_track_yaw_gain", 0.85), -1.0, 1.0))
        head_pitch = float(np.clip(-pitch_angle * self._cfg("fly_to_cube_head_track_pitch_gain", 1.15), -1.0, 1.0))
        head_roll = 0.0
        return head_yaw, head_pitch, head_roll

    def _toe_support_key(self, geom_name: str) -> str:
        name = str(geom_name).lower()
        side = "left" if "left_" in name else "right" if "right_" in name else ""
        if not side:
            return ""
        if "toe_front_inner" in name:
            return f"{side}_front_inner"
        if "toe_front_mid" in name:
            return f"{side}_front_mid"
        if "toe_front_outer" in name:
            return f"{side}_front_outer"
        if "toe_rear" in name:
            return f"{side}_rear"
        return ""

    def _mujoco_toe_ground_contact_features(self) -> Dict[str, float | str]:
        owner = self.owner
        world = getattr(owner, "world", None)
        model = getattr(world, "model", None)
        data = getattr(world, "data", None)
        if model is None or data is None:
            return {"sum": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "source": "mujoco.contact.none"}

        support_keys = (
            "left_front_inner", "left_front_mid", "left_front_outer", "left_rear",
            "right_front_inner", "right_front_mid", "right_front_outer", "right_rear",
        )
        ground_names = {"ground"}
        values = []
        by_key = {key: 0.0 for key in support_keys}
        count_by_key = {key: 0 for key in support_keys}

        try:
            import mujoco
            obj_geom = mujoco.mjtObj.mjOBJ_GEOM
        except Exception:
            obj_geom = None

        try:
            ncon = int(getattr(data, "ncon", 0))
            scale = max(self._cfg("fly_to_cube_ground_toe_contact_force_scale", 1.0), 1e-6)

            for ci in range(ncon):
                con = data.contact[ci]
                g1 = int(con.geom1)
                g2 = int(con.geom2)

                if obj_geom is not None:
                    n1 = self._mujoco_name(obj_geom, g1).lower()
                    n2 = self._mujoco_name(obj_geom, g2).lower()
                else:
                    n1 = n2 = ""

                key = ""
                if n2 in ground_names:
                    key = self._toe_support_key(n1)
                elif n1 in ground_names:
                    key = self._toe_support_key(n2)
                if not key:
                    continue

                force = self._mujoco_contact_force_scalar(ci) / scale
                values.append(force)
                by_key[key] += float(force)
                count_by_key[key] += 1
        except Exception:
            values = []

        if not values:
            empty = {"sum": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "source": "mujoco.contact.none"}
            for key in support_keys:
                empty[f"{key}_force"] = 0.0
                empty[f"{key}_count"] = 0
            empty.update(left_force=0.0, right_force=0.0, front_force=0.0, rear_force=0.0, support_ratio=0.0)
            return empty

        arr = np.asarray(values, dtype=np.float32)
        support_min = max(self._cfg("fly_to_cube_ground_toe_support_min_force", 0.018), 1e-6)
        support_hits = sum(1 for key in support_keys if by_key[key] >= support_min)
        left_force = sum(by_key[key] for key in support_keys if key.startswith("left_"))
        right_force = sum(by_key[key] for key in support_keys if key.startswith("right_"))
        front_force = sum(by_key[key] for key in support_keys if "_front_" in key)
        rear_force = by_key["left_rear"] + by_key["right_rear"]

        out = {
            "sum": float(arr.sum()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "count": int(arr.size),
            "source": "mujoco.contact.toe_ground",
            "left_force": float(left_force),
            "right_force": float(right_force),
            "front_force": float(front_force),
            "rear_force": float(rear_force),
            "support_ratio": float(support_hits / max(1, len(support_keys))),
        }
        for key in support_keys:
            out[f"{key}_force"] = float(by_key[key])
            out[f"{key}_count"] = int(count_by_key[key])
        return out

    def _desired_hover_z(self, cube: np.ndarray) -> float:
        owner = self.owner
        min_world_z = float(getattr(owner.world, "min_flight_z", 0.30))
        if float(cube[2]) < self._cfg("fly_to_cube_ground_cube_z_threshold", 0.35):
            leg_clearance = self._cfg("fly_to_cube_ground_leg_clearance", 0.42)
            hover_above_cube = self._cfg("fly_to_cube_ground_hover_above_cube", 0.64)
        else:
            leg_clearance = self._cfg("fly_to_cube_leg_clearance", 0.74)
            hover_above_cube = self._cfg("fly_to_cube_hover_above_cube", 0.30)
        max_hover = self._cfg("fly_to_cube_max_hover_z", 1.90)
        return float(np.clip(float(cube[2]) + hover_above_cube, min_world_z + leg_clearance, max_hover))

    def _reset_reach_distance_minimizer(self) -> None:
        sign = self._cfg("fly_to_cube_arm_side_sign", -1.0)
        right_yaw = self._cfg("fly_to_cube_reach_shoulder_yaw_abs", 0.04)
        right_shoulder = self._cfg("fly_to_cube_reach_shoulder_pitch", -0.86)
        right_elbow = self._cfg("fly_to_cube_reach_elbow", -0.98)
        left_yaw = self._cfg("fly_to_cube_left_nearby_shoulder_yaw_abs", 0.11)
        left_shoulder = self._cfg("fly_to_cube_left_nearby_shoulder_pitch", -0.28)
        left_elbow = self._cfg("fly_to_cube_left_nearby_elbow", -0.24)
        self._reach_arm_action = np.asarray(
            [sign * left_yaw, left_shoulder, left_elbow, -sign * right_yaw, right_shoulder, right_elbow],
            dtype=np.float32,
        )
        self._reach_palm_pitch = self._cfg("fly_to_cube_reach_palm_pitch", 0.64)
        self._reach_left_palm_roll = self._cfg("fly_to_cube_left_nearby_palm_roll", 0.46)
        self._reach_right_palm_roll = self._cfg("fly_to_cube_reach_right_palm_roll", 0.53)
        self._reach_finger_base = self._cfg("fly_to_cube_reach_finger_base", 0.05)
        self._reach_axis = 0
        self._reach_dir = -1.0
        self._reach_last_dist = 999.0
        self._reach_best_dist = 999.0
        self._reach_stall_steps = 0
        self._reach_updates = 0

    def _reach_axes(self) -> tuple[str, ...]:
        return (
            "right_shoulder_pitch",
            "right_elbow",
            "right_shoulder_yaw",
            "right_palm_roll",
            "palm_pitch",
        )

    def _apply_reach_axis_delta(self, axis_name: str, delta: float) -> None:
        arm = np.asarray(getattr(self, "_reach_arm_action", np.zeros(6, dtype=np.float32)), dtype=np.float32).copy()
        sign = self._cfg("fly_to_cube_arm_side_sign", -1.0)

        if axis_name == "right_shoulder_pitch":
            arm[4] = float(np.clip(arm[4] + delta, -1.0, 0.35))
        elif axis_name == "right_elbow":
            arm[5] = float(np.clip(arm[5] + delta, -1.0, -0.20))
        elif axis_name == "right_shoulder_yaw":
            yaw_abs = float(np.clip(abs(float(arm[3])) + delta, 0.0, 0.60))
            arm[3] = float(-sign * yaw_abs)
        elif axis_name == "right_palm_roll":
            self._reach_right_palm_roll = float(np.clip(float(self._reach_right_palm_roll) + delta, 0.20, 0.84))
        elif axis_name == "palm_pitch":
            lo = self._cfg("fly_to_cube_reach_palm_pitch_min", 0.54)
            hi = self._cfg("fly_to_cube_reach_palm_pitch_max", 0.76)
            self._reach_palm_pitch = float(np.clip(float(self._reach_palm_pitch) + delta, lo, hi))

        self._reach_arm_action = np.clip(arm, -1.0, 1.0).astype(np.float32)

    def _small_cube_arm_action(self, phase: str, extension: float, pulse: float, yaw: float, target_world: Optional[np.ndarray]) -> np.ndarray:
        base_phase = "palpate" if phase in ("grasp", "lift", "rotate") else phase
        if phase in ("align", "pre_reach"):
            base_phase = "reach"
            extension = max(float(extension), self._cfg("fly_to_cube_pre_reach_extension", 0.55))
        arm = self._arm_action(base_phase, extension=extension, pulse=pulse, target_world=target_world, yaw=yaw)

        if target_world is not None and phase in ("align", "reach", "grasp", "lift"):
            ik = self._arm_raw_from_ik(target_world, yaw)
            if ik is not None and ik.size >= 6:
                gain = self._cfg("fly_to_cube_right_palm_ik_gain", 0.62)
                if phase == "align":
                    gain *= self._cfg("fly_to_cube_right_palm_ik_align_scale", 0.70)
                elif phase in ("grasp", "lift"):
                    gain *= self._cfg("fly_to_cube_right_palm_ik_grasp_scale", 0.45)
                arm[3:6] = (1.0 - gain) * arm[3:6] + gain * ik[3:6]
                if phase in ("align", "reach"):
                    reach_arm = np.asarray(getattr(self, "_reach_arm_action", arm), dtype=np.float32).copy()
                    reach_arm[3:6] = arm[3:6]
                    self._reach_arm_action = np.clip(reach_arm, -1.0, 1.0).astype(np.float32)

        if phase in ("align", "pre_reach"):
            prep = float(np.clip(self._phase_age() / max(1.0, self._cfg("fly_to_cube_pre_reach_ramp_steps", 35.0)), 0.0, 1.0))
            rest = np.asarray(
                [
                    self._cfg("fly_to_cube_arm_side_sign", -1.0) * self._cfg("fly_to_cube_left_nearby_shoulder_yaw_abs", 0.11),
                    self._cfg("fly_to_cube_left_nearby_shoulder_pitch", -0.28),
                    self._cfg("fly_to_cube_left_nearby_elbow", -0.24),
                    -self._cfg("fly_to_cube_arm_side_sign", -1.0) * 0.05,
                    -0.18,
                    -0.20,
                ],
                dtype=np.float32,
            )
            arm = (1.0 - prep) * rest + prep * arm

        if phase == "rotate":
            roll = float(np.sin(pulse * self._cfg("fly_to_cube_rotate_speed", 0.62)))
            twist = self._cfg("fly_to_cube_rotate_arm_twist", 0.075)
            lift = self._cfg("fly_to_cube_rotate_arm_lift", 0.035)
            arm[3] = float(np.clip(arm[3] - twist * roll, -1.0, 1.0))
            arm[4] = float(np.clip(arm[4] - lift * np.cos(pulse * 0.55), -1.0, 1.0))

        arm[0] = float(self._cfg("fly_to_cube_arm_side_sign", -1.0) * self._cfg("fly_to_cube_left_nearby_shoulder_yaw_abs", 0.11))
        arm[1] = float(self._cfg("fly_to_cube_left_nearby_shoulder_pitch", -0.28))
        arm[2] = float(self._cfg("fly_to_cube_left_nearby_elbow", -0.24))
        return np.clip(arm, -1.0, 1.0).astype(np.float32)

    def _open_small_cube_fingers(self, hand: np.ndarray, side: str = "both") -> np.ndarray:
        hand = np.asarray(hand, dtype=np.float32).copy()
        open_mcp = self._cfg("fly_to_cube_pre_palm_contact_finger_mcp", 0.015)
        open_pip = self._cfg("fly_to_cube_pre_palm_contact_finger_pip", 0.0)
        open_dip = self._cfg("fly_to_cube_pre_palm_contact_finger_dip", 0.0)
        bases = (0, 22)
        if side == "left":
            bases = (0,)
        elif side == "right":
            bases = (22,)
        for base in bases:
            for i in range(5):
                j = base + 2 + 4 * i
                hand[j + 1] = open_mcp
                hand[j + 2] = open_pip
                hand[j + 3] = open_dip
        return hand

    def _small_cube_hand_action(
        self,
        phase: str,
        extension: float,
        curl: float,
        pulse: float,
        palm_contact_active: bool = False,
    ) -> np.ndarray:
        base_phase = "palpate" if phase in ("grasp", "lift", "rotate") else phase
        hand = self._hand_action(base_phase, extension=extension, curl=curl, pulse=pulse).astype(np.float32)
        hand = self._open_small_cube_fingers(hand, side="left")

        if phase in ("align", "reach") and not palm_contact_active:
            hand = self._open_small_cube_fingers(hand, side="right")

        if phase in ("grasp", "lift", "rotate"):
            close_target = self._cfg("fly_to_cube_grasp_finger_base", 0.58)
            rate = self._cfg("fly_to_cube_grasp_close_rate", 0.018)
            self._reach_finger_base = float(np.clip(float(self._reach_finger_base) + rate, 0.0, close_target))
            hand = self._hand_action("palpate", extension=1.0, curl=1.0, pulse=pulse).astype(np.float32)
            hand = self._open_small_cube_fingers(hand, side="left")

        if phase == "rotate":
            wave = float(np.sin(pulse * self._cfg("fly_to_cube_rotate_speed", 0.62)))
            counter = float(np.cos(pulse * self._cfg("fly_to_cube_rotate_speed", 0.62)))
            palm_roll_amp = self._cfg("fly_to_cube_rotate_palm_roll_amp", 0.12)
            palm_pitch_amp = self._cfg("fly_to_cube_rotate_palm_pitch_amp", 0.045)
            hand[22] = float(np.clip(0.52 - palm_roll_amp * wave, 0.20, 0.80))
            hand[23] = float(np.clip(float(self._reach_palm_pitch) - palm_pitch_amp * counter, 0.48, 0.78))
            hand = self._open_small_cube_fingers(hand, side="left")

        hand[0] = float(self._cfg("fly_to_cube_left_nearby_palm_roll", 0.46))
        hand[1] = float(self._cfg("fly_to_cube_left_nearby_palm_pitch", 0.54))

        return np.clip(hand, 0.0, 1.0).astype(np.float32)

    def _small_cube_standing_leg_action(self) -> np.ndarray:
        """Neutral standing support; the small-cube pickup is controlled by arm/hand."""
        hip_yaw = self._cfg("fly_to_cube_standing_leg_hip_yaw", 0.04)
        hip_pitch = self._cfg("fly_to_cube_standing_leg_hip_pitch", 0.04)
        knee = self._cfg("fly_to_cube_standing_leg_knee", -0.30)
        ankle_pitch = self._cfg("fly_to_cube_standing_leg_ankle_pitch", 0.18)
        ankle_roll = self._cfg("fly_to_cube_standing_leg_ankle_roll", 0.0)
        toe_front = self._cfg("fly_to_cube_standing_leg_toe_front", 0.04)
        toe_rear = self._cfg("fly_to_cube_standing_leg_toe_rear", -0.02)

        left = np.asarray(
            [
                -hip_yaw,
                hip_pitch,
                knee,
                ankle_pitch,
                -ankle_roll,
                toe_front,
                toe_front,
                toe_front,
                toe_rear,
            ],
            dtype=np.float32,
        )
        right = np.asarray(
            [
                hip_yaw,
                hip_pitch,
                knee,
                ankle_pitch,
                ankle_roll,
                toe_front,
                toe_front,
                toe_front,
                toe_rear,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([left, right]), -1.0, 1.0).astype(np.float32)

    def _small_cube_leg_action(
        self,
        phase: str,
        cube: np.ndarray,
        agent: np.ndarray,
        desired_z: float,
        toe_ground: Dict[str, float | str],
        hand_dist: float = 999.0,
    ) -> np.ndarray:
        """
        Leg pose for low ground pickup.

        Order per side:
            hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll,
            toe_front_inner, toe_front_mid, toe_front_outer, toe_rear
        """
        cube_on_ground = float(cube[2]) < self._cfg("fly_to_cube_ground_cube_z_threshold", 0.35)
        low_body = float(agent[2]) < desired_z + self._cfg("fly_to_cube_ground_leg_engage_z_margin", 0.20)
        active_phase = phase in ("align", "reach", "grasp", "lift")
        toe_force_sum = float(toe_ground.get("sum", 0.0))
        toe_force_max = float(toe_ground.get("max", 0.0))
        toe_force_active = (
            toe_force_sum >= self._cfg("fly_to_cube_ground_toe_contact_enter_sum", 0.12)
            or toe_force_max >= self._cfg("fly_to_cube_ground_toe_contact_enter_max", 0.055)
        )
        hard_contact = (
            toe_force_sum >= self._cfg("fly_to_cube_ground_toe_contact_hard_sum", 0.36)
            or toe_force_max >= self._cfg("fly_to_cube_ground_toe_contact_hard_max", 0.14)
        )

        if not bool(self._cfg("fly_to_cube_ground_leg_manipulation_enabled", 0.0)):
            self.ground_toe_contact_hold_steps = 0
            self._ground_leg_crouch = 0.0
            self._ground_leg_hard_contact = bool(hard_contact)
            self._ground_leg_contact_ramp = 0.0
            self._ground_toe_support_ratio = float(toe_ground.get("support_ratio", 0.0))
            self._ground_leg_balance_pitch = 0.0
            self._ground_leg_balance_roll = 0.0
            self._ground_leg_reach_lean = 0.0
            self._ground_leg_com_pitch = 0.0
            self._ground_ankle_pitch_bias = 0.0
            return self._small_cube_standing_leg_action()

        if cube_on_ground and (low_body or active_phase) and toe_force_active:
            self.ground_toe_contact_hold_steps += 1
        else:
            self.ground_toe_contact_hold_steps = max(0, int(getattr(self, "ground_toe_contact_hold_steps", 0)) - 1)

        hold_ok = self.ground_toe_contact_hold_steps >= int(self._cfg("fly_to_cube_ground_toe_contact_hold_steps", 2))
        self._ground_leg_hard_contact = bool(hard_contact)

        if not (cube_on_ground and (low_body or active_phase)):
            self._ground_leg_crouch = max(0.0, float(getattr(self, "_ground_leg_crouch", 0.0)) - self._cfg("fly_to_cube_ground_leg_release_rate", 0.035))
            self._ground_leg_contact_ramp = 0.0
            self._ground_toe_support_ratio = 0.0
            self._ground_leg_balance_pitch = 0.0
            self._ground_leg_balance_roll = 0.0
            self._ground_leg_reach_lean = 0.0
            self._ground_leg_com_pitch = 0.0
            self._ground_ankle_pitch_bias = 0.0
            return np.zeros(18, dtype=np.float32)

        preload = self._cfg("fly_to_cube_ground_leg_pre_contact_preload", 0.12)
        target_crouch = preload
        if hold_ok or hard_contact:
            target_crouch = self._cfg("fly_to_cube_ground_leg_standing_crouch", 0.24)
        elif toe_force_active:
            target_crouch = max(preload, self._cfg("fly_to_cube_ground_leg_touch_crouch", 0.12))

        current = float(getattr(self, "_ground_leg_crouch", 0.0))
        if target_crouch > current:
            rate = self._cfg("fly_to_cube_ground_leg_hard_crouch_rate", 0.12) if hard_contact else self._cfg("fly_to_cube_ground_leg_crouch_rate", 0.055)
            current += rate
        else:
            current -= self._cfg("fly_to_cube_ground_leg_release_rate", 0.020)
        max_standing_crouch = self._cfg("fly_to_cube_ground_leg_max_standing_crouch", 0.28)
        self._ground_leg_crouch = float(np.clip(current, 0.0, max_standing_crouch))

        contact_ramp_steps = max(1.0, self._cfg("fly_to_cube_ground_leg_contact_ramp_steps", 10.0))
        contact_ramp = float(np.clip(self.ground_toe_contact_hold_steps / contact_ramp_steps, 0.0, 1.0))
        if hard_contact:
            contact_ramp = 1.0
        self._ground_leg_contact_ramp = contact_ramp

        ramp = max(preload, min(float(getattr(self, "_ground_leg_crouch", 0.0)), max(contact_ramp, 0.18) if toe_force_active else preload))
        if hold_ok and phase in ("reach", "grasp", "lift"):
            ramp = max(ramp, min(max_standing_crouch, float(getattr(self, "_ground_leg_crouch", 0.0))))
        if phase == "rotate":
            ramp = min(ramp, self._cfg("fly_to_cube_ground_leg_rotate_crouch", 0.55))

        hip_yaw = self._cfg("fly_to_cube_ground_leg_hip_yaw", 0.10)
        hip_pitch = self._cfg("fly_to_cube_ground_leg_hip_pitch", 0.68)
        knee = self._cfg("fly_to_cube_ground_leg_knee", -0.95)
        ankle_pitch = self._cfg("fly_to_cube_ground_leg_ankle_pitch", 0.32)
        ankle_roll = self._cfg("fly_to_cube_ground_leg_ankle_roll", 0.04)
        toe_front = self._cfg("fly_to_cube_ground_leg_toe_front", 0.34)
        toe_rear = self._cfg("fly_to_cube_ground_leg_toe_rear", -0.18)

        support_min = max(self._cfg("fly_to_cube_ground_toe_support_min_force", 0.018), 1e-6)
        missing_gain = self._cfg("fly_to_cube_ground_toe_missing_gain", 0.28)
        rear_missing_gain = self._cfg("fly_to_cube_ground_toe_rear_missing_gain", 0.24)

        def force(key: str) -> float:
            return float(toe_ground.get(f"{key}_force", 0.0))

        def missing(key: str) -> float:
            return float(np.clip((support_min - force(key)) / support_min, 0.0, 1.0))

        left_front = force("left_front_inner") + force("left_front_mid") + force("left_front_outer")
        right_front = force("right_front_inner") + force("right_front_mid") + force("right_front_outer")
        left_rear = force("left_rear")
        right_rear = force("right_rear")
        left_total = left_front + left_rear
        right_total = right_front + right_rear
        front_total = left_front + right_front
        rear_total = left_rear + right_rear
        support_total = max(left_total + right_total, support_min)
        front_ratio = float(np.clip(front_total / support_total, 0.0, 1.0))

        front_rear_error = float(np.clip((rear_total - front_total) / support_total, -1.0, 1.0))
        side_error = float(np.clip((right_total - left_total) / support_total, -1.0, 1.0))
        roll, pitch = self._agent_roll_pitch()
        support_ratio = float(toe_ground.get("support_ratio", 0.0))
        support_target = max(self._cfg("fly_to_cube_ground_toe_support_target_ratio", 0.75), 1e-6)
        face_goal = self._cfg("fly_to_cube_fingertip_face_goal", 0.018)
        near_gap = max(self._cfg("fly_to_cube_fingertip_face_near", 0.045), 1e-6)
        hand_gap_drive = float(np.clip((float(hand_dist) - face_goal) / near_gap, 0.0, 1.0))
        support_gate = float(np.clip(support_ratio / support_target, 0.0, 1.0))
        phase_gate = 1.0 if phase in ("reach", "grasp") else 0.55 if phase == "align" else 0.35
        front_safe_max = self._cfg("fly_to_cube_ground_reach_front_force_max_ratio", 0.78)
        front_overload = float(np.clip((front_ratio - front_safe_max) / max(1.0 - front_safe_max, 1e-6), 0.0, 1.0))
        lean_drive = hand_gap_drive * support_gate * phase_gate * (1.0 - 0.75 * front_overload)
        front_target = self._cfg("fly_to_cube_ground_reach_front_force_target_ratio", 0.62)
        com_pitch_error = float(np.clip(front_target - front_ratio, -0.7, 0.7))
        com_pitch_drive = com_pitch_error * self._cfg("fly_to_cube_ground_reach_com_pitch_gain", 0.26)

        hip_pitch = float(np.clip(
            hip_pitch + lean_drive * self._cfg("fly_to_cube_ground_reach_hip_pitch_gain", 0.18),
            -1.0,
            1.0,
        ))
        knee = float(np.clip(
            knee + lean_drive * self._cfg("fly_to_cube_ground_reach_knee_gain", -0.22),
            -1.0,
            1.0,
        ))
        ankle_pitch += lean_drive * self._cfg("fly_to_cube_ground_reach_ankle_pitch_gain", 0.18)
        toe_front += lean_drive * self._cfg("fly_to_cube_ground_reach_front_toe_gain", 0.12)
        toe_rear -= front_overload * self._cfg("fly_to_cube_ground_reach_rear_toe_catch_gain", 0.18)
        self._ground_leg_reach_lean = float(lean_drive)
        self._ground_leg_com_pitch = float(com_pitch_drive)

        ankle_bias_target = (
            lean_drive * self._cfg("fly_to_cube_ground_ankle_pitch_reach_gain", 0.55)
            + com_pitch_error * self._cfg("fly_to_cube_ground_ankle_pitch_com_gain", 0.28)
            - front_overload * self._cfg("fly_to_cube_ground_ankle_pitch_overload_recover_gain", 0.22)
        )
        ankle_bias_target = float(np.clip(
            ankle_bias_target,
            self._cfg("fly_to_cube_ground_ankle_pitch_bias_min", -0.70),
            self._cfg("fly_to_cube_ground_ankle_pitch_bias_max", 0.35),
        ))
        ankle_bias = float(getattr(self, "_ground_ankle_pitch_bias", 0.0))
        ankle_rate = self._cfg("fly_to_cube_ground_ankle_pitch_bias_rate", 0.055)
        ankle_bias += float(np.clip(ankle_bias_target - ankle_bias, -ankle_rate, ankle_rate))
        self._ground_ankle_pitch_bias = ankle_bias

        balance_limit = self._cfg("fly_to_cube_ground_balance_limit", 0.28)
        ankle_pitch_corr = (
            front_rear_error * self._cfg("fly_to_cube_ground_ankle_pitch_balance_gain", 0.34)
            - pitch * self._cfg("fly_to_cube_ground_attitude_pitch_gain", 0.18)
            + com_pitch_drive
        )
        ankle_roll_corr = (
            side_error * self._cfg("fly_to_cube_ground_ankle_roll_balance_gain", 0.24)
            - roll * self._cfg("fly_to_cube_ground_attitude_roll_gain", 0.18)
        )
        ankle_pitch_corr = float(np.clip(ankle_pitch_corr, -balance_limit, balance_limit))
        ankle_roll_corr = float(np.clip(ankle_roll_corr, -balance_limit, balance_limit))
        ankle_pitch_cmd = float(np.clip(ankle_pitch + ankle_bias + ankle_pitch_corr, -1.0, 1.0))
        self._ground_toe_support_ratio = float(toe_ground.get("support_ratio", 0.0))
        self._ground_leg_balance_pitch = ankle_pitch_corr
        self._ground_leg_balance_roll = ankle_roll_corr

        left_front_inner = toe_front + missing_gain * missing("left_front_inner")
        left_front_mid = toe_front + missing_gain * missing("left_front_mid")
        left_front_outer = toe_front + missing_gain * missing("left_front_outer")
        right_front_inner = toe_front + missing_gain * missing("right_front_inner")
        right_front_mid = toe_front + missing_gain * missing("right_front_mid")
        right_front_outer = toe_front + missing_gain * missing("right_front_outer")
        left_rear_toe = toe_rear - rear_missing_gain * missing("left_rear")
        right_rear_toe = toe_rear - rear_missing_gain * missing("right_rear")

        left = np.asarray(
            [
                -hip_yaw,
                hip_pitch,
                knee,
                ankle_pitch_cmd,
                -ankle_roll + ankle_roll_corr,
                left_front_inner,
                left_front_mid,
                left_front_outer,
                left_rear_toe,
            ],
            dtype=np.float32,
        )
        right = np.asarray(
            [
                hip_yaw,
                hip_pitch,
                knee,
                ankle_pitch_cmd,
                ankle_roll + ankle_roll_corr,
                right_front_inner,
                right_front_mid,
                right_front_outer,
                right_rear_toe,
            ],
            dtype=np.float32,
        )
        crouch = np.concatenate([left, right]).astype(np.float32)
        cmd = np.clip(crouch * ramp, -1.0, 1.0).astype(np.float32)
        ankle_ramp = max(ramp, support_gate * self._cfg("fly_to_cube_ground_ankle_pitch_min_ramp", 0.85))
        cmd[3] = float(np.clip(left[3] * ankle_ramp, -1.0, 1.0))
        cmd[12] = float(np.clip(right[3] * ankle_ramp, -1.0, 1.0))
        return cmd.astype(np.float32)

    def step(self) -> Optional[ScenarioCommand]:
        owner = self.owner
        if not self.active:
            return None

        step_i = int(getattr(owner, "global_step", 0))
        if step_i - self.started_step > self.timeout_steps:
            self.stop("timeout")
            return ScenarioCommand(
                body=np.zeros(9, dtype=np.float32),
                arm=self._arm_action("align"),
                hand=self._hand_action("align"),
                leg=np.zeros(18, dtype=np.float32),
                status=dict(self.status),
            )

        agent, yaw = self._agent_xyz_yaw()
        cube = self._cube_xyz()
        delta = cube - agent
        dist_xy = float(np.linalg.norm(delta[:2]))
        if dist_xy > 1e-6:
            to_cube_xy = delta[:2] / dist_xy
        else:
            to_cube_xy = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

        desired_yaw = float(np.arctan2(to_cube_xy[1], to_cube_xy[0]))
        desired_z = self._desired_hover_z(cube)

        if self.phase in ("lift", "rotate"):
            lift = self._cfg("fly_to_cube_lift_height", 0.18)
            lift_steps = max(1.0, self._cfg("fly_to_cube_lift_steps", 70.0))
            desired_z += lift * float(np.clip(self._phase_age() / lift_steps, 0.0, 1.0))

        keepout = self._cfg("fly_to_cube_body_keepout", 0.52)
        keepout_buffer = self._cfg("fly_to_cube_body_keepout_buffer", 0.04)
        min_reach_distance = max(keepout + keepout_buffer, self._cfg("fly_to_cube_min_arm_reach_distance", 0.58))
        max_reach_distance = self._cfg("fly_to_cube_max_arm_reach_distance", 1.05)
        self.adaptive_reach_distance = float(np.clip(self.adaptive_reach_distance, min_reach_distance, max_reach_distance))

        target_xy = cube[:2] - to_cube_xy * self.adaptive_reach_distance
        target_xyz = np.asarray([target_xy[0], target_xy[1], desired_z], dtype=np.float64)
        ring_err = float(np.linalg.norm(target_xy - agent[:2]))
        z_err = float(desired_z - float(agent[2]))
        yaw_err = float((desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)

        tactile = self._tactile_features()
        tactile_sum = float(tactile["sum"])
        tactile_max = float(tactile["max"])
        contact_count = int(tactile.get("count", 0))
        left_palm_sum = float(tactile.get("left_palm_sum", 0.0))
        left_palm_max = float(tactile.get("left_palm_max", 0.0))
        left_palm_count = int(tactile.get("left_palm_count", 0))
        right_palm_sum = float(tactile.get("right_palm_sum", 0.0))
        right_palm_max = float(tactile.get("right_palm_max", 0.0))
        right_palm_count = int(tactile.get("right_palm_count", 0))
        left_finger_sum = float(tactile.get("left_finger_sum", 0.0))
        left_finger_max = float(tactile.get("left_finger_max", 0.0))
        left_finger_count = int(tactile.get("left_finger_count", 0))
        right_finger_sum = float(tactile.get("right_finger_sum", 0.0))
        right_finger_max = float(tactile.get("right_finger_max", 0.0))
        right_finger_count = int(tactile.get("right_finger_count", 0))
        palm_sum = right_palm_sum
        palm_max = right_palm_max
        palm_count = right_palm_count
        finger_sum = right_finger_sum
        finger_max = right_finger_max
        finger_count = right_finger_count
        palm_contact_active = palm_count > 0 and palm_sum >= self._cfg("fly_to_cube_palm_contact_enter_sum", 0.018)
        finger_contact_active = finger_count > 0 and finger_sum >= self._cfg("fly_to_cube_finger_contact_enter_sum", 0.012)
        contact_active = bool(palm_contact_active or finger_contact_active)
        toe_ground = self._mujoco_toe_ground_contact_features()
        toe_ground_sum = float(toe_ground.get("sum", 0.0))
        toe_ground_max = float(toe_ground.get("max", 0.0))

        fingertip_face = self._fingertip_face_distance_to_cube()
        palm_face = self._palm_face_distance_to_cube(side="right")
        try:
            fingertip_dist = float(fingertip_face.get("distance", 999.0))
            if not np.isfinite(fingertip_dist):
                fingertip_dist = self._hand_distance_to_cube(cube)
        except Exception:
            fingertip_dist = self._hand_distance_to_cube(cube)
        try:
            palm_dist = float(palm_face.get("distance", 999.0))
            if not np.isfinite(palm_dist):
                palm_dist = fingertip_dist
        except Exception:
            palm_dist = fingertip_dist

        hand_dist = palm_dist if self.phase in ("align", "reach") and not palm_contact_active else fingertip_dist
        try:
            if not np.isfinite(hand_dist):
                hand_dist = self._hand_distance_to_cube(cube)
        except Exception:
            hand_dist = self._hand_distance_to_cube(cube)
        fingertip_face_signed = float(fingertip_face.get("signed_distance", fingertip_dist))
        palm_face_signed = float(palm_face.get("signed_distance", palm_dist))

        cube_on_ground = float(cube[2]) < self._cfg("fly_to_cube_ground_cube_z_threshold", 0.35)
        support_ratio_now = float(toe_ground.get("support_ratio", 0.0))
        support_target = max(self._cfg("fly_to_cube_ground_toe_support_target_ratio", 0.75), 1e-6)
        if (
            cube_on_ground
            and self.phase in ("align", "reach", "grasp")
            and not palm_contact_active
            and bool(self._cfg("fly_to_cube_body_adapt_to_fingertip_gap", 1.0))
            and support_ratio_now >= self._cfg("fly_to_cube_ground_reach_min_support_ratio", 0.50)
        ):
            face_goal = self._cfg("fly_to_cube_fingertip_face_goal", 0.018)
            gap = max(hand_dist - face_goal, 0.0)
            near_gap = max(self._cfg("fly_to_cube_fingertip_face_near", 0.045), 1e-6)
            gap_gain = float(np.clip(gap / near_gap, 0.0, 3.0))
            adapt_rate = self._cfg("fly_to_cube_reach_distance_adapt_rate", 0.0040)
            support_gate = float(np.clip(support_ratio_now / support_target, 0.0, 1.0))
            self.adaptive_reach_distance = max(min_reach_distance, self.adaptive_reach_distance - adapt_rate * gap_gain * support_gate)
            self.adaptive_reach_distance = float(np.clip(self.adaptive_reach_distance, min_reach_distance, max_reach_distance))
            target_xy = cube[:2] - to_cube_xy * self.adaptive_reach_distance
            target_xyz = np.asarray([target_xy[0], target_xy[1], desired_z], dtype=np.float64)
            ring_err = float(np.linalg.norm(target_xy - agent[:2]))
            z_err = float(desired_z - float(agent[2]))
            yaw_err = float((desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)

        if self.phase == "approach" and (ring_err < 0.24 or self._phase_age() > int(self._cfg("fly_to_cube_approach_max_steps", 220))):
            self._set_phase("align")

        if self.phase == "align":
            align_z_tol = self._cfg("fly_to_cube_align_z_tolerance", 0.14)
            if ring_err < 0.18 and abs(z_err) < align_z_tol and abs(yaw_err) < 0.22:
                self.align_hold_steps += 1
            else:
                self.align_hold_steps = 0
            support_ready = True
            force_reach = (
                self._phase_age() > int(self._cfg("fly_to_cube_force_reach_after_steps", 55))
                and ring_err < self._cfg("fly_to_cube_force_reach_ring_err", 0.32)
                and abs(yaw_err) < self._cfg("fly_to_cube_force_reach_yaw_err", 0.42)
            )
            if support_ready and (self.align_hold_steps > int(self._cfg("fly_to_cube_align_hold_steps", 8)) or force_reach):
                self._set_phase("reach")
                self._reach_last_dist = 999.0
                self._reach_best_dist = min(float(getattr(self, "_reach_best_dist", 999.0)), hand_dist)

        if self.phase == "reach":
            if palm_contact_active:
                self.palm_contact_hold_steps += 1
            else:
                self.palm_contact_hold_steps = 0
            if self.palm_contact_hold_steps >= int(self._cfg("fly_to_cube_palm_contact_hold_steps", 2)):
                self._set_phase("grasp")
        else:
            self.palm_contact_hold_steps = int(self.palm_contact_hold_steps) if palm_contact_active else 0

        if contact_active:
            self.contact_hold_steps += 1
        else:
            self.contact_hold_steps = 0

        if self.phase == "grasp":
            if not palm_contact_active and not finger_contact_active:
                self.grasp_lost_contact_steps += 1
            else:
                self.grasp_lost_contact_steps = 0
            if self.grasp_lost_contact_steps > int(self._cfg("fly_to_cube_grasp_lost_contact_reopen_steps", 16)):
                self._set_phase("reach")
                self.grasp_hold_steps = 0
                self.finger_contact_hold_steps = 0
                self._reach_finger_base = self._cfg("fly_to_cube_reach_finger_base", 0.05)

        if self.phase == "grasp":
            if finger_contact_active:
                self.finger_contact_hold_steps += 1
            else:
                self.finger_contact_hold_steps = 0
            enough_contacts = finger_count >= int(self._cfg("fly_to_cube_grasp_contact_count", 2))
            enough_force = finger_sum >= self._cfg("fly_to_cube_grasp_touch_sum", 0.050)
            if enough_contacts or enough_force:
                self.grasp_hold_steps += 1
            else:
                self.grasp_hold_steps = 0
            if self.grasp_hold_steps > int(self._cfg("fly_to_cube_grasp_hold_steps", 18)):
                self._set_phase("lift")

        if self.phase == "lift" and self._phase_age() > int(self._cfg("fly_to_cube_lift_steps", 70)):
            self._set_phase("rotate")

        if self.phase in ("align", "reach"):
            minimizer = self._update_reach_distance_minimizer(hand_dist=hand_dist, contact_active=palm_contact_active)
        else:
            minimizer = {
                "axis": "",
                "dir": 0.0,
                "improved": False,
                "enabled": False,
            }

        self._direct_nudge_to_standoff(target_xyz, desired_yaw, dist_xy, keepout)

        agent, yaw = self._agent_xyz_yaw()
        target_delta = target_xyz - agent
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        local_target_x = c * float(target_delta[0]) + s * float(target_delta[1])
        local_target_y = -s * float(target_delta[0]) + c * float(target_delta[1])
        z_err = float(desired_z - float(agent[2]))
        yaw_err = float((desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
        pid_vz, _pid_direct_step = self._vertical_pid(desired_z, float(agent[2]))

        if self.phase == "approach":
            vx = float(np.clip(local_target_x * 1.25, -0.48, 0.48))
            vy = float(np.clip(local_target_y * 1.25, -0.36, 0.36))
            vz = float(pid_vz)
            yaw_cmd = float(np.clip(yaw_err * 0.50, -0.28, 0.28))
        elif self.phase == "align":
            vx = float(np.clip(local_target_x * 0.62, -0.16, 0.16))
            vy = float(np.clip(local_target_y * 0.62, -0.14, 0.14))
            vz = float(np.clip(pid_vz, -0.30, 0.30))
            yaw_cmd = float(np.clip(yaw_err * 0.38, -0.20, 0.20))
        else:
            vx = float(np.clip(local_target_x * 0.25, -0.055, 0.055))
            vy = float(np.clip(local_target_y * 0.25, -0.055, 0.055))
            vz = float(np.clip(pid_vz, -0.22, 0.22))
            yaw_cmd = float(np.clip(yaw_err * 0.22, -0.12, 0.12))

        if dist_xy < keepout:
            vx = -0.22
            vy = 0.0

        cube_on_ground = float(cube[2]) < self._cfg("fly_to_cube_ground_cube_z_threshold", 0.35)
        support_ratio = float(toe_ground.get("support_ratio", 0.0))
        toe_landing_active = toe_ground_sum >= self._cfg("fly_to_cube_ground_toe_contact_enter_sum", 0.12)
        if cube_on_ground and toe_landing_active and self.phase in ("align", "reach", "grasp", "lift"):
            support_target = max(self._cfg("fly_to_cube_ground_toe_support_target_ratio", 0.75), 1e-6)
            settle = float(np.clip(support_ratio / support_target, 0.0, 1.0))
            xy_scale = 0.30 + 0.35 * settle
            vx *= xy_scale
            vy *= xy_scale
            if support_ratio < support_target:
                vz = float(np.clip(vz, -0.08, 0.06))
            else:
                vz = float(np.clip(vz, -0.025, 0.08))

        if self.phase in ("align", "reach"):
            hand_pid = self._hand_contact_pid(tactile_sum=palm_sum, hand_dist=hand_dist)
            extension = float(hand_pid["extension"])
            curl = 0.0 if not palm_contact_active else float(hand_pid["curl"])
        elif self.phase in ("grasp", "lift", "rotate"):
            hand_pid = {
                "touch_norm": min(1.0, finger_sum / max(self._cfg("fly_to_cube_hand_touch_scale", 0.10), 1e-6)),
                "target_norm": 1.0,
                "err": 0.0,
                "drive": 1.0,
                "extension": 1.0,
                "curl": 1.0,
            }
            extension = 1.0
            curl = 1.0
        else:
            hand_pid = {
                "touch_norm": 0.0,
                "target_norm": 0.0,
                "err": 0.0,
                "drive": 0.0,
                "extension": 0.0,
                "curl": 0.0,
            }
            self._hand_extension = max(0.0, float(getattr(self, "_hand_extension", 0.0)) - 0.02)
            self._finger_curl_bias = max(0.0, float(getattr(self, "_finger_curl_bias", 0.0)) - 0.02)
            extension = 0.0
            curl = 0.0

        t = float(step_i - self.started_step)
        pulse = t * self._cfg("fly_to_cube_palpate_pulse_speed", 0.30)
        reach_target_world = self._cube_reach_target_world(agent) if self.phase in ("align", "reach", "grasp", "lift", "rotate") else None
        arm = self._small_cube_arm_action(self.phase, extension=extension, pulse=pulse, yaw=yaw, target_world=reach_target_world)
        hand = self._small_cube_hand_action(
            self.phase,
            extension=extension,
            curl=curl,
            pulse=pulse,
            palm_contact_active=palm_contact_active,
        )
        leg = self._small_cube_leg_action(
            self.phase,
            cube=cube,
            agent=agent,
            desired_z=desired_z,
            toe_ground=toe_ground,
            hand_dist=hand_dist,
        )

        body_pitch = self._cfg("fly_to_cube_body_pitch_cmd", -0.08)
        head_pitch = self._cfg("fly_to_cube_head_pitch_cmd", 0.22)
        head_yaw, tracked_head_pitch, head_roll = self._head_track_target_action(agent, yaw, cube)
        if cube_on_ground and self.phase in ("align", "reach", "grasp", "lift"):
            support_gate = float(np.clip(support_ratio / max(self._cfg("fly_to_cube_ground_toe_support_target_ratio", 0.75), 1e-6), 0.0, 1.0))
            face_goal = self._cfg("fly_to_cube_fingertip_face_goal", 0.018)
            near_gap = max(self._cfg("fly_to_cube_fingertip_face_near", 0.045), 1e-6)
            gap_gate = float(np.clip((hand_dist - face_goal) / near_gap, 0.0, 1.0))
            reach_gate = support_gate * (0.35 + 0.65 * gap_gate)
            target_body_pitch = self._cfg("fly_to_cube_ground_reach_body_pitch_cmd", -0.28)
            body_pitch = float((1.0 - reach_gate) * body_pitch + reach_gate * target_body_pitch)
            head_pitch = self._cfg("fly_to_cube_ground_reach_head_pitch_cmd", 0.36)
        head_pitch = float(np.clip(max(head_pitch, tracked_head_pitch), -1.0, 1.0))

        body = np.asarray(
            [vx, vy, vz, yaw_cmd, body_pitch, 0.0, head_yaw, head_pitch, head_roll],
            dtype=np.float32,
        )

        status = {
            "active": True,
            "scenario": self.name,
            "phase": self.phase,
            "phase_age": int(self._phase_age()),
            "body_dist": float(dist_xy),
            "ring_err": float(np.linalg.norm(target_xyz[:2] - agent[:2])),
            "reach_distance": float(self.adaptive_reach_distance),
            "arm_side_sign": float(self._cfg("fly_to_cube_arm_side_sign", -1.0)),
            "keepout": float(keepout),
            "hand_dist": float(hand_dist),
            "right_palm_face_dist": float(palm_dist),
            "right_palm_face_signed_dist": float(palm_face_signed),
            "right_palm_face_source": str(palm_face.get("source", "")),
            "closest_right_palm": str(palm_face.get("closest_palm", "")),
            "fingertip_face_dist": float(fingertip_dist),
            "fingertip_face_signed_dist": float(fingertip_face_signed),
            "fingertip_face_source": str(fingertip_face.get("source", "")),
            "closest_fingertip": str(fingertip_face.get("closest_tip", "")),
            "reach_target_x": float(reach_target_world[0]) if reach_target_world is not None else 0.0,
            "reach_target_y": float(reach_target_world[1]) if reach_target_world is not None else 0.0,
            "reach_target_z": float(reach_target_world[2]) if reach_target_world is not None else 0.0,
            "arm_cmd_l_yaw": float(arm[0]),
            "arm_cmd_l_pitch": float(arm[1]),
            "arm_cmd_l_elbow": float(arm[2]),
            "arm_cmd_r_yaw": float(arm[3]),
            "arm_cmd_r_pitch": float(arm[4]),
            "arm_cmd_r_elbow": float(arm[5]),
            "leg_cmd_l_hip_pitch": float(leg[1]) if leg.size > 1 else 0.0,
            "leg_cmd_l_knee": float(leg[2]) if leg.size > 2 else 0.0,
            "leg_cmd_l_ankle_pitch": float(leg[3]) if leg.size > 3 else 0.0,
            "leg_cmd_r_hip_pitch": float(leg[10]) if leg.size > 10 else 0.0,
            "leg_cmd_r_knee": float(leg[11]) if leg.size > 11 else 0.0,
            "leg_cmd_r_ankle_pitch": float(leg[12]) if leg.size > 12 else 0.0,
            "ground_pickup_leg_active": bool(
                self._cfg("fly_to_cube_ground_leg_manipulation_enabled", 0.0) > 0.5
                and np.max(np.abs(leg)) > 1e-4
            ),
            "ground_leg_mode": "manipulation" if self._cfg("fly_to_cube_ground_leg_manipulation_enabled", 0.0) > 0.5 else "standing_support",
            "toe_ground_force_sum": float(toe_ground_sum),
            "toe_ground_force_max": float(toe_ground_max),
            "toe_ground_contact_count": int(toe_ground.get("count", 0)),
            "toe_ground_contact_hold_steps": int(getattr(self, "ground_toe_contact_hold_steps", 0)),
            "ground_leg_crouch": float(getattr(self, "_ground_leg_crouch", 0.0)),
            "ground_leg_contact_ramp": float(getattr(self, "_ground_leg_contact_ramp", 0.0)),
            "ground_leg_hard_contact": bool(getattr(self, "_ground_leg_hard_contact", False)),
            "ground_toe_support_ratio": float(getattr(self, "_ground_toe_support_ratio", 0.0)),
            "ground_leg_balance_pitch": float(getattr(self, "_ground_leg_balance_pitch", 0.0)),
            "ground_leg_balance_roll": float(getattr(self, "_ground_leg_balance_roll", 0.0)),
            "ground_leg_reach_lean": float(getattr(self, "_ground_leg_reach_lean", 0.0)),
            "ground_leg_com_pitch": float(getattr(self, "_ground_leg_com_pitch", 0.0)),
            "ground_ankle_pitch_bias": float(getattr(self, "_ground_ankle_pitch_bias", 0.0)),
            "toe_ground_left_force": float(toe_ground.get("left_force", 0.0)),
            "toe_ground_right_force": float(toe_ground.get("right_force", 0.0)),
            "toe_ground_front_force": float(toe_ground.get("front_force", 0.0)),
            "toe_ground_rear_force": float(toe_ground.get("rear_force", 0.0)),
            "toe_ground_source": str(toe_ground.get("source", "")),
            "distance_minimizer_axis": str(minimizer.get("axis", "")),
            "distance_minimizer_dir": float(minimizer.get("dir", 0.0)),
            "distance_minimizer_improved": bool(minimizer.get("improved", False)),
            "distance_minimizer_enabled": bool(minimizer.get("enabled", False)),
            "distance_minimizer_best": float(getattr(self, "_reach_best_dist", 999.0)),
            "reach_palm_pitch": float(getattr(self, "_reach_palm_pitch", 0.0)),
            "reach_finger_base": float(getattr(self, "_reach_finger_base", 0.0)),
            "tactile_sum": float(tactile_sum),
            "tactile_max": float(tactile_max),
            "left_palm_touch_sum": float(left_palm_sum),
            "left_palm_touch_max": float(left_palm_max),
            "left_palm_contact_count": int(left_palm_count),
            "right_palm_touch_sum": float(right_palm_sum),
            "right_palm_touch_max": float(right_palm_max),
            "right_palm_contact_count": int(right_palm_count),
            "right_palm_contact_active": bool(palm_contact_active),
            "palm_contact_hold_steps": int(getattr(self, "palm_contact_hold_steps", 0)),
            "left_finger_touch_sum": float(left_finger_sum),
            "left_finger_touch_max": float(left_finger_max),
            "left_finger_contact_count": int(left_finger_count),
            "right_finger_touch_sum": float(right_finger_sum),
            "right_finger_touch_max": float(right_finger_max),
            "right_finger_contact_count": int(right_finger_count),
            "right_finger_contact_active": bool(finger_contact_active),
            "finger_contact_hold_steps": int(getattr(self, "finger_contact_hold_steps", 0)),
            "contact_count": int(contact_count),
            "contact_active": bool(contact_active),
            "tactile_source": str(tactile["source"]),
            "contact_hold_steps": int(self.contact_hold_steps),
            "grasp_hold_steps": int(self.grasp_hold_steps),
            "desired_z": float(desired_z),
            "agent_z": float(agent[2]),
            "z_err": float(z_err),
            "z_pid_vz": float(getattr(self, "_z_pid_vz", 0.0)),
            "hand_pid_drive": float(hand_pid["drive"]),
            "hand_extension": float(extension),
            "finger_curl": float(curl),
            "hand_touch_norm": float(hand_pid["touch_norm"]),
            "yaw_err": float(yaw_err),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "yaw_cmd": float(yaw_cmd),
            "body_pitch_cmd": float(body_pitch),
            "head_yaw_cmd": float(head_yaw),
            "head_pitch_cmd": float(head_pitch),
            "head_roll_cmd": float(head_roll),
        }
        self.status = status

        if step_i % 20 == 0:
            print(
                "[adaptive_scenario] fly_to_small_cube_grasp_rotate "
                f"phase={self.phase} age={status['phase_age']} body={dist_xy:.3f} ring={status['ring_err']:.3f} "
                f"right_palm_face={palm_dist:.3f}/{palm_face_signed:.3f} tip_face={fingertip_dist:.3f}/{fingertip_face_signed:.3f} "
                f"right_palm={palm_sum:.3f}/{palm_count} right_finger={finger_sum:.3f}/{finger_count} grasp_hold={self.grasp_hold_steps} "
                f"reach={self.adaptive_reach_distance:.3f} r_arm=({arm[3]:.2f},{arm[4]:.2f},{arm[5]:.2f}) "
                f"toe_ground={toe_ground_sum:.3f}/{toe_ground_max:.3f} hold={self.ground_toe_contact_hold_steps} "
                f"leg=({status['leg_cmd_l_hip_pitch']:.2f},{status['leg_cmd_l_knee']:.2f},{status['leg_cmd_l_ankle_pitch']:.2f}) "
                f"crouch={status['ground_leg_crouch']:.2f}/{status['ground_leg_contact_ramp']:.2f} hard={int(status['ground_leg_hard_contact'])} "
                f"support={status['ground_toe_support_ratio']:.2f} lean={status['ground_leg_reach_lean']:.2f} "
                f"ankle_bias={status['ground_ankle_pitch_bias']:.2f} bal=({status['ground_leg_balance_pitch']:.2f},{status['ground_leg_balance_roll']:.2f}) "
                f"body_pitch={status['body_pitch_cmd']:.2f} head=({status['head_yaw_cmd']:.2f},{status['head_pitch_cmd']:.2f}) "
                f"finger_base={float(getattr(self, '_reach_finger_base', 0.0)):.2f} "
                f"min={minimizer.get('axis','')}:{float(minimizer.get('dir',0.0)):.0f} "
                f"z={agent[2]:.3f}->{desired_z:.3f} vz={vz:.3f} src={status['tactile_source']}"
            )

        return ScenarioCommand(body=body, arm=arm, hand=hand, leg=leg, status=status)


class FlyToTetrahedronInspectScenario(BaseScenario):
    """Inspect the suspended tetrahedron/cube pair from a stable 2 m standoff."""

    name = "fly_to_tetrahedron_inspect"

    _TETRA_VERTICES = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, owner):
        super().__init__(owner)
        self.phase = "approach"
        self.phase_started_step = int(getattr(owner, "global_step", 0))
        self.timeout_steps = int(self._cfg("fly_to_tetrahedron_timeout_steps", 0))
        self.standoff_distance = self._cfg("fly_to_tetrahedron_standoff_distance", 2.0)
        self.face_steps = max(8, int(self._cfg("fly_to_tetrahedron_face_steps", 90)))
        self.gaze_switch_steps = max(1, int(self._cfg("fly_to_tetrahedron_gaze_switch_steps", 500)))
        self.rotate_tetrahedron = False
        self.rotate_cube = False
        self.fly_cube = False
        self._cube_flight_phase = 0.0
        self._approach_dir_xy = np.asarray([0.0, 1.0], dtype=np.float64)
        self._tetra_center = np.asarray([0.0, -2.4, 2.6], dtype=np.float64)
        self._cube_center = np.asarray([1.3, -2.8, 2.6], dtype=np.float64)
        self._cube_home_center = self._cube_center.copy()
        self._tetra_static_quat = np.asarray([0.92388, 0.33141, 0.19134, 0.03813], dtype=np.float64)
        self._cube_static_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _cfg(self, key: str, default: float) -> float:
        cfg = getattr(self.owner, "cfg", None)
        try:
            section = getattr(cfg, "adaptive_scenario_controller", None)
            section_val = _adaptive_section_value(section, key)
            if section_val is not _MISSING:
                val = _cfg_float_value(section_val, key, "section")
                if val is _MISSING:
                    raise ValueError(f"invalid adaptive_scenario_controller.{key}")
                #print("\033[32m section - adaptive_scenario_controller \033[0m", key , val) 
                return val
        except Exception:
            print("\033[31m ######## error section - adaptive_scenario_controller Exception \033[0m")
            pass
        try:
            root_val = _config_value(cfg, key, default)
            val = _cfg_float_value(root_val, key, "root/default")
            if val is not _MISSING:
                return val
        except Exception:
            pass
        return float(default)

    @staticmethod
    def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(arr))
        if n < 1e-9 or not np.isfinite(n):
            return np.asarray(fallback, dtype=np.float64).reshape(arr.shape)
        return arr / n

    @staticmethod
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        return q / max(float(np.linalg.norm(q)), 1e-9)

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
        w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
        return np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = FlyToTetrahedronInspectScenario._normalize(axis, np.asarray([0.0, 0.0, 1.0]))
        s = float(np.sin(0.5 * angle_rad))
        return FlyToTetrahedronInspectScenario._quat_normalize(
            np.asarray([np.cos(0.5 * angle_rad), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)
        )

    @staticmethod
    def _quat_from_to(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        a = FlyToTetrahedronInspectScenario._normalize(v_from, np.asarray([1.0, 0.0, 0.0]))
        b = FlyToTetrahedronInspectScenario._normalize(v_to, np.asarray([1.0, 0.0, 0.0]))
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot > 1.0 - 1e-7:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if dot < -1.0 + 1e-7:
            axis = np.cross(a, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(a, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
            return FlyToTetrahedronInspectScenario._axis_angle_quat(axis, np.pi)
        axis = np.cross(a, b)
        return FlyToTetrahedronInspectScenario._quat_normalize(
            np.asarray([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
        )

    @staticmethod
    def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        q0 = FlyToTetrahedronInspectScenario._quat_normalize(q0)
        q1 = FlyToTetrahedronInspectScenario._quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        t = float(np.clip(t, 0.0, 1.0))
        if dot > 0.9995:
            return FlyToTetrahedronInspectScenario._quat_normalize(q0 + t * (q1 - q0))
        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta_0 = max(float(np.sin(theta_0)), 1e-9)
        theta = theta_0 * t
        return FlyToTetrahedronInspectScenario._quat_normalize(
            (np.sin(theta_0 - theta) / sin_theta_0) * q0 + (np.sin(theta) / sin_theta_0) * q1
        )

    def _mocap_id_for_body(self, body_name: str) -> int:
        world = getattr(self.owner, "world", None)
        if world is None:
            return -1
        if hasattr(world, "_mocap_id_for_body"):
            try:
                return int(world._mocap_id_for_body(body_name, fallback=-1))
            except Exception:
                pass
        try:
            import mujoco
            bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                return int(world.model.body_mocapid[bid])
        except Exception:
            pass
        return -1

    def _body_id(self, body_name: str) -> int:
        try:
            import mujoco
            return int(mujoco.mj_name2id(self.owner.world.model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        except Exception:
            return -1

    def _agent_xyz_yaw(self) -> Tuple[np.ndarray, float]:
        owner = self.owner
        ctrl = getattr(owner, "dynamic_agent_rig_controller", None)
        if ctrl is not None and hasattr(ctrl, "qpos_adr"):
            try:
                qpos = owner.world.data.qpos[ctrl.qpos_adr:ctrl.qpos_adr + 7]
                xyz = np.asarray(qpos[:3], dtype=np.float64).copy()
                qw, qx, qy, qz = np.asarray(qpos[3:7], dtype=np.float64)
                return xyz, float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            except Exception:
                pass
        xyz = np.asarray(getattr(owner.world, "cam_pos", np.zeros(3)), dtype=np.float64).copy()
        return xyz, float(np.deg2rad(float(getattr(owner.world, "yaw_deg", 0.0))))

    def _eye_xyz(self, fallback: np.ndarray) -> np.ndarray:
        world = getattr(self.owner, "world", None)
        if world is None:
            return np.asarray(fallback, dtype=np.float64).reshape(3).copy()
        try:
            import mujoco
            mujoco.mj_forward(world.model, world.data)
            points = []
            for name in ("cam_left", "cam_right"):
                cid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cid >= 0:
                    points.append(np.asarray(world.data.cam_xpos[cid], dtype=np.float64).reshape(3))
            if points:
                eye = np.mean(np.stack(points, axis=0), axis=0)
                if np.isfinite(eye).all():
                    return eye.copy()
        except Exception:
            pass
        return np.asarray(fallback, dtype=np.float64).reshape(3).copy()

    def _object_xyz(self, body_name: str, fallback: np.ndarray) -> np.ndarray:
        world = getattr(self.owner, "world", None)
        if world is None:
            return np.asarray(fallback, dtype=np.float64).reshape(3).copy()
        mid = self._mocap_id_for_body(body_name)
        try:
            if mid >= 0:
                arr = np.asarray(world.data.mocap_pos[mid], dtype=np.float64).reshape(3).copy()
                if np.isfinite(arr).all():
                    return arr
        except Exception:
            pass
        bid = self._body_id(body_name)
        try:
            if bid >= 0:
                arr = np.asarray(world.data.xpos[bid], dtype=np.float64).reshape(3).copy()
                if np.isfinite(arr).all():
                    return arr
        except Exception:
            pass
        return np.asarray(fallback, dtype=np.float64).reshape(3).copy()

    def _mocap_quat_for_body(self, body_name: str, fallback: np.ndarray) -> np.ndarray:
        world = getattr(self.owner, "world", None)
        if world is None:
            return self._quat_normalize(fallback)
        mid = self._mocap_id_for_body(body_name)
        try:
            if mid >= 0:
                arr = np.asarray(world.data.mocap_quat[mid], dtype=np.float64).reshape(4).copy()
                if np.isfinite(arr).all():
                    return self._quat_normalize(arr)
        except Exception:
            pass
        return self._quat_normalize(fallback)

    def _set_mocap_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray) -> bool:
        world = getattr(self.owner, "world", None)
        if world is None:
            return False
        mid = self._mocap_id_for_body(body_name)
        if mid < 0:
            return False
        try:
            world.data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64).reshape(3)
            world.data.mocap_quat[mid] = self._quat_normalize(quat)
            return True
        except Exception as e:
            if not hasattr(self.owner, "_floating_object_mocap_warned"):
                print(f"[adaptive_scenario] floating object mocap pose failed for {body_name}: {e}")
                self.owner._floating_object_mocap_warned = True
            return False

    def _scene_center(self) -> np.ndarray:
        return 0.5 * (self._tetra_center + self._cube_home_center)

    def _perpendicular_approach_dir(self, agent: np.ndarray) -> np.ndarray:
        line = np.asarray(self._cube_home_center[:2] - self._tetra_center[:2], dtype=np.float64)
        line_norm = float(np.linalg.norm(line))
        if line_norm < 1e-9:
            return self._normalize(agent[:2] - self._scene_center()[:2], np.asarray([0.0, 1.0]))

        perp = np.asarray([-line[1], line[0]], dtype=np.float64) / line_norm
        from_center = np.asarray(agent[:2] - self._scene_center()[:2], dtype=np.float64)
        if float(np.dot(from_center, perp)) < 0.0:
            perp = -perp
        return self._normalize(perp, np.asarray([0.0, 1.0]))

    def _desired_agent_pose(self, center: np.ndarray) -> np.ndarray:
        target = np.asarray(center, dtype=np.float64).reshape(3).copy()
        target[:2] = target[:2] + self._approach_dir_xy * self.standoff_distance
        target[2] = float(center[2] + self._cfg("fly_to_tetrahedron_agent_z_offset", 0.0))
        return target

    def _gaze_target(self, step_i: int) -> tuple[np.ndarray, str]:
        forced = str(getattr(self, "forced_gaze_target", "") or "").strip().lower()
        if forced == "cube":
            return self._cube_center.copy(), "cube"
        if forced in ("tetra", "tetrahedron"):
            return self._tetra_center.copy(), "tetrahedron"
        if self.fly_cube:
            return self._cube_center.copy(), "cube"
        if self.rotate_tetrahedron and not self.rotate_cube:
            return self._tetra_center.copy(), "tetrahedron"
        if self.rotate_cube and not self.rotate_tetrahedron:
            return self._cube_center.copy(), "cube"
        if self.rotate_tetrahedron and self.rotate_cube:
            elapsed = max(0, int(step_i) - int(self.started_step))
            if (elapsed // self.gaze_switch_steps) % 2 == 0:
                return self._cube_center.copy(), "cube"
            return self._tetra_center.copy(), "tetrahedron"
        return self._scene_center(), "middle"

    def _gaze_yaw_pitch(self, agent: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        delta = np.asarray(target, dtype=np.float64).reshape(3) - np.asarray(agent, dtype=np.float64).reshape(3)
        yaw = float(np.arctan2(delta[1], delta[0]))
        xy = max(float(np.linalg.norm(delta[:2])), 1e-9)
        pitch = float(np.arctan2(delta[2], xy))
        max_pitch = float(np.deg2rad(self._cfg("fly_to_tetrahedron_max_body_pitch_deg", 40.0)))
        return yaw, float(np.clip(pitch, -max_pitch, max_pitch))

    def _head_track_action(self, body_yaw: float, body_pitch: float, target_yaw: float, target_pitch: float) -> tuple[float, float, float]:
        yaw_err = float((target_yaw - body_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        pitch_err = float(target_pitch - body_pitch)

        yaw_gain = self._cfg("fly_to_tetrahedron_head_yaw_gain", 1.35)
        pitch_gain = self._cfg("fly_to_tetrahedron_head_pitch_gain", 1.25)

        # Manual head channels are normalized into the joint ranges.
        # The head pitch joint uses MuJoCo's +Y hinge convention: positive
        # pitch lowers the camera ray, so gaze elevation needs the opposite sign.
        head_yaw = float(np.clip(np.rad2deg(yaw_err) / 45.0 * yaw_gain, -1.0, 1.0))
        pitch_target_deg = float(np.clip(-np.rad2deg(pitch_err) * pitch_gain, -70.0, 50.0))
        head_pitch = float(np.clip(2.0 * (pitch_target_deg + 70.0) / 120.0 - 1.0, -1.0, 1.0))
        return head_yaw, head_pitch, 0.0

    def _face_quat(self, face_index: int, to_agent: np.ndarray) -> np.ndarray:
        normals = -self._TETRA_VERTICES
        local_normal = self._normalize(normals[int(face_index) % 4], np.asarray([1.0, 0.0, 0.0]))
        return self._quat_from_to(local_normal, to_agent)

    def _tetra_quat_for_step(self, tetra: np.ndarray, agent: np.ndarray) -> tuple[np.ndarray, int, float]:
        elapsed = max(0, int(getattr(self.owner, "global_step", 0)) - int(self.started_step))
        face_i = int(elapsed // self.face_steps) % 4
        face_next = (face_i + 1) % 4
        u = (elapsed % self.face_steps) / float(self.face_steps)
        smooth = float(u * u * (3.0 - 2.0 * u))
        to_agent = self._normalize(agent - tetra, np.asarray([0.0, 1.0, 0.0]))
        q = self._quat_slerp(self._face_quat(face_i, to_agent), self._face_quat(face_next, to_agent), smooth)
        spin = self._axis_angle_quat(to_agent, float(elapsed) * self._cfg("fly_to_tetrahedron_spin_rad_per_step", 0.035))
        return self._quat_normalize(self._quat_mul(spin, q)), face_i, float(u)

    def _cube_quat_for_step(self, cube: np.ndarray, agent: np.ndarray) -> np.ndarray:
        elapsed = max(0, int(getattr(self.owner, "global_step", 0)) - int(self.started_step))
        to_agent = self._normalize(agent - cube, np.asarray([0.0, 1.0, 0.0]))
        diagonal = self._normalize(np.asarray([0.7, 1.0, 0.35], dtype=np.float64), np.asarray([1.0, 0.0, 0.0]))
        q_view_spin = self._axis_angle_quat(to_agent, elapsed * self._cfg("fly_to_tetrahedron_cube_spin_rad_per_step", 0.030))
        q_tumble = self._axis_angle_quat(diagonal, elapsed * self._cfg("fly_to_tetrahedron_cube_tumble_rad_per_step", 0.020))
        return self._quat_normalize(self._quat_mul(q_view_spin, q_tumble))

    def _advance_cube_flight(self) -> np.ndarray:
        if not self.fly_cube:
            return self._cube_center.copy()
        self._cube_flight_phase += self._cfg("fly_to_tetrahedron_cube_flight_rad_per_step", 0.018)
        t = float(self._cube_flight_phase)
        offset = np.asarray(
            [
                self._cfg("fly_to_tetrahedron_cube_flight_radius_x", 0.75) * np.sin(t),
                self._cfg("fly_to_tetrahedron_cube_flight_radius_y", 0.55) * np.sin(1.35 * t),
                self._cfg("fly_to_tetrahedron_cube_flight_radius_z", 0.28) * np.sin(0.85 * t),
            ],
            dtype=np.float64,
        )
        self._cube_center = self._cube_home_center + offset
        return self._cube_center.copy()

    def apply_options(self, options: dict | None = None) -> None:
        options = dict(options or {})
        self.rotate_tetrahedron = bool(options.get("rotate_tetrahedron", options.get("rotate_tetra", self.rotate_tetrahedron)))
        self.rotate_cube = bool(options.get("rotate_cube", self.rotate_cube))
        self.fly_cube = bool(options.get("fly_cube", self.fly_cube))
        target_name = options.get("target_name", options.get("target", None))
        if target_name is not None:
            self.forced_gaze_target = str(target_name)
        self.status.update(
            rotate_tetrahedron=bool(self.rotate_tetrahedron),
            rotate_cube=bool(self.rotate_cube),
            fly_cube=bool(self.fly_cube),
            forced_gaze_target=str(getattr(self, "forced_gaze_target", "")),
        )

    def _set_phase(self, phase: str) -> None:
        if self.phase != phase:
            self.phase = phase
            self.phase_started_step = int(getattr(self.owner, "global_step", 0))

    def start(self) -> None:
        super().start()
        options = getattr(self, "options", {}) if isinstance(getattr(self, "options", {}), dict) else {}
        self.apply_options(options)
        self.phase = "approach"
        self.phase_started_step = int(getattr(self.owner, "global_step", 0))
        self.standoff_distance = self._cfg("fly_to_tetrahedron_standoff_distance", 2.0)
        self.face_steps = max(8, int(self._cfg("fly_to_tetrahedron_face_steps", 90)))
        self.gaze_switch_steps = max(1, int(self._cfg("fly_to_tetrahedron_gaze_switch_steps", 500)))
        self._tetra_center = self._object_xyz("obj_tetrahedron_mocap", self._tetra_center)
        self._cube_center = self._object_xyz("obj_cube_mocap", self._cube_center)
        self._cube_home_center = self._cube_center.copy()
        self._tetra_static_quat = self._mocap_quat_for_body("obj_tetrahedron_mocap", self._tetra_static_quat)
        self._cube_static_quat = self._mocap_quat_for_body("obj_cube_mocap", self._cube_static_quat)
        self.forced_gaze_target = str(options.get("target_name", options.get("target", "")) or "")

        agent, _yaw = self._agent_xyz_yaw()
        self._approach_dir_xy = self._perpendicular_approach_dir(agent)

        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_arm_action = np.zeros(6, dtype=np.float32)
        self.owner._ipc_manual_hand_action = np.full(44, 0.5, dtype=np.float32)
        self.owner._ipc_manual_leg_action = np.zeros(18, dtype=np.float32)
        print(
            "[adaptive_scenario] fly_to_tetrahedron_inspect started "
            f"rotate_tetra={int(self.rotate_tetrahedron)} rotate_cube={int(self.rotate_cube)} fly_cube={int(self.fly_cube)}"
        )

    def step(self) -> Optional[ScenarioCommand]:
        if not self.active:
            return None

        step_i = int(getattr(self.owner, "global_step", 0))
        if self.timeout_steps > 0 and step_i - self.started_step > self.timeout_steps:
            self.stop("timeout")
            return ScenarioCommand(
                body=np.zeros(9, dtype=np.float32),
                arm=np.zeros(6, dtype=np.float32),
                hand=np.full(44, 0.5, dtype=np.float32),
                leg=np.zeros(18, dtype=np.float32),
                status=dict(self.status),
            )

        owner = self.owner
        world = getattr(owner, "world", None)
        tetra = self._tetra_center.copy()
        cube = self._advance_cube_flight()
        scene_center = self._scene_center()
        gaze_target, gaze_target_name = self._gaze_target(step_i)
        agent, yaw = self._agent_xyz_yaw()
        desired_agent = self._desired_agent_pose(scene_center)
        desired_body_yaw, _desired_body_pitch = self._gaze_yaw_pitch(agent, scene_center)
        desired_body_pitch = 0.0
        eye = self._eye_xyz(agent)
        desired_gaze_yaw, desired_gaze_pitch = self._gaze_yaw_pitch(eye, gaze_target)

        if world is not None and hasattr(world, "cam_pos"):
            cur = np.asarray(world.cam_pos, dtype=np.float64).reshape(3)
            err = desired_agent - cur
            step = np.zeros(3, dtype=np.float64)
            step[:2] = np.clip(err[:2] * 0.18, -self._cfg("fly_to_tetrahedron_max_direct_xy_step", 0.050), self._cfg("fly_to_tetrahedron_max_direct_xy_step", 0.050))
            step[2] = float(np.clip(err[2] * 0.18, -self._cfg("fly_to_tetrahedron_max_direct_z_step", 0.050), self._cfg("fly_to_tetrahedron_max_direct_z_step", 0.050)))
            world.cam_pos[:] = cur + step

            yaw_now = float(np.deg2rad(float(getattr(world, "yaw_deg", 0.0))))
            yaw_err_now = float((desired_body_yaw - yaw_now + np.pi) % (2.0 * np.pi) - np.pi)
            world.yaw_deg = float(getattr(world, "yaw_deg", 0.0)) + float(np.rad2deg(np.clip(yaw_err_now * 0.22, -0.075, 0.075)))
            if hasattr(world, "pitch_deg"):
                pitch_now = float(np.deg2rad(float(getattr(world, "pitch_deg", 0.0))))
                world.pitch_deg = float(getattr(world, "pitch_deg", 0.0)) + float(np.rad2deg(np.clip((0.0 - pitch_now) * 0.24, -0.060, 0.060)))
            if hasattr(world, "roll_deg"):
                world.roll_deg = float(0.90 * float(getattr(world, "roll_deg", 0.0)))
            if hasattr(world, "_clamp_flight_zone"):
                world._clamp_flight_zone()
            if hasattr(world, "_update_rig_pose"):
                world._update_rig_pose()

        agent, yaw = self._agent_xyz_yaw()
        eye = self._eye_xyz(agent)
        gaze_target, gaze_target_name = self._gaze_target(step_i)
        desired_body_yaw, _desired_body_pitch = self._gaze_yaw_pitch(agent, scene_center)
        desired_body_pitch = 0.0
        desired_gaze_yaw, desired_gaze_pitch = self._gaze_yaw_pitch(eye, gaze_target)
        elapsed = max(0, int(getattr(owner, "global_step", 0)) - int(self.started_step))
        if self.rotate_tetrahedron:
            tetra_quat, face_i, face_u = self._tetra_quat_for_step(tetra, agent)
        else:
            tetra_quat, face_i, face_u = self._tetra_static_quat, -1, 0.0
        tetra_angle = float(elapsed) * float(self._cfg("fly_to_tetrahedron_spin_rad_per_step", 0.035))
        cube_angle = float(elapsed) * float(self._cfg("fly_to_tetrahedron_cube_spin_rad_per_step", 0.030))
        cube_tumble_angle = float(elapsed) * float(self._cfg("fly_to_tetrahedron_cube_tumble_rad_per_step", 0.020))
        cube_quat = self._cube_quat_for_step(cube, agent) if self.rotate_cube else self._cube_static_quat
        tetra_pose_ok = self._set_mocap_pose("obj_tetrahedron_mocap", tetra, tetra_quat)
        cube_pose_ok = self._set_mocap_pose("obj_cube_mocap", cube, cube_quat)

        if world is not None:
            try:
                import mujoco
                mujoco.mj_forward(world.model, world.data)
            except Exception:
                pass

        target_delta = desired_agent - agent
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        local_x = c * float(target_delta[0]) + s * float(target_delta[1])
        local_y = -s * float(target_delta[0]) + c * float(target_delta[1])
        local_z = float(target_delta[2])
        body_yaw_err = float((desired_body_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
        current_pitch = float(np.deg2rad(float(getattr(world, "pitch_deg", 0.0)))) if world is not None else 0.0
        body_pitch_err = float(0.0 - current_pitch)

        arrive_dist = float(np.linalg.norm(target_delta))
        if self.phase == "approach" and arrive_dist < self._cfg("fly_to_tetrahedron_arrive_radius", 0.10) and abs(body_yaw_err) < 0.18:
            self._set_phase("inspect")

        if self.phase == "approach":
            vx = float(np.clip(local_x * 1.10, -0.38, 0.38))
            vy = float(np.clip(local_y * 1.10, -0.30, 0.30))
            vz = float(np.clip(local_z * 1.10, -0.28, 0.28))
            yaw_cmd = float(np.clip(body_yaw_err * 0.45, -0.24, 0.24))
        else:
            vx = float(np.clip(local_x * 0.35, -0.08, 0.08))
            vy = float(np.clip(local_y * 0.35, -0.08, 0.08))
            vz = float(np.clip(local_z * 0.35, -0.08, 0.08))
            yaw_cmd = float(np.clip(body_yaw_err * 0.30, -0.14, 0.14))

        head_yaw, head_pitch, head_roll = self._head_track_action(yaw, 0.0, desired_gaze_yaw, desired_gaze_pitch)
        body_pitch = float(np.clip(body_pitch_err * 0.20, -0.08, 0.08))

        body = np.asarray([vx, vy, vz, yaw_cmd, body_pitch, 0.0, head_yaw, head_pitch, head_roll], dtype=np.float32)
        arm = np.zeros(6, dtype=np.float32)
        hand = np.full(44, 0.5, dtype=np.float32)
        leg = np.zeros(18, dtype=np.float32)

        status = {
            "active": True,
            "scenario": self.name,
            "phase": self.phase,
            "body_dist": float(np.linalg.norm(scene_center - agent)),
            "gaze_distance": float(np.linalg.norm(gaze_target - eye)),
            "depth_focus_half_range": float(self._cfg("fly_to_tetrahedron_depth_focus_half_range", 0.85)),
            "tetra_spin_rad_per_step": float(self._cfg("fly_to_tetrahedron_spin_rad_per_step", 0.035)),
            "tetra_face_steps": int(self.face_steps),
            "tetra_angle": float(tetra_angle),
            "tetra_qw": float(tetra_quat[0]),
            "tetra_qx": float(tetra_quat[1]),
            "tetra_qy": float(tetra_quat[2]),
            "tetra_qz": float(tetra_quat[3]),
            "cube_qw": float(cube_quat[0]),
            "cube_qx": float(cube_quat[1]),
            "cube_qy": float(cube_quat[2]),
            "cube_qz": float(cube_quat[3]),
            "cube_spin_rad_per_step": float(self._cfg("fly_to_tetrahedron_cube_spin_rad_per_step", 0.030)),
            "cube_tumble_rad_per_step": float(self._cfg("fly_to_tetrahedron_cube_tumble_rad_per_step", 0.020)),
            "cube_flight_rad_per_step": float(self._cfg("fly_to_tetrahedron_cube_flight_rad_per_step", 0.018)),
            "cube_angle": float(cube_angle),
            "cube_tumble_angle": float(cube_tumble_angle),
            "standoff_distance": float(self.standoff_distance),
            "target_dist": float(arrive_dist),
            "eye_x": float(eye[0]),
            "eye_y": float(eye[1]),
            "eye_z": float(eye[2]),
            "tetra_x": float(tetra[0]),
            "tetra_y": float(tetra[1]),
            "tetra_z": float(tetra[2]),
            "cube_x": float(cube[0]),
            "cube_y": float(cube[1]),
            "cube_z": float(cube[2]),
            "tetra_face_index": int(face_i),
            "tetra_face_progress": float(face_u),
            "tetra_pose_ok": bool(tetra_pose_ok),
            "cube_pose_ok": bool(cube_pose_ok),
            "rotate_tetrahedron": bool(self.rotate_tetrahedron),
            "rotate_cube": bool(self.rotate_cube),
            "fly_cube": bool(self.fly_cube),
            "cube_flight_phase": float(self._cube_flight_phase),
            "gaze_target": str(gaze_target_name),
            "desired_body_yaw_deg": float(np.rad2deg(desired_body_yaw)),
            "desired_body_pitch_deg": float(np.rad2deg(desired_body_pitch)),
            "desired_gaze_yaw_deg": float(np.rad2deg(desired_gaze_yaw)),
            "desired_gaze_pitch_deg": float(np.rad2deg(desired_gaze_pitch)),
            "yaw_err": float(body_yaw_err),
            "pitch_err": float(body_pitch_err),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "head_yaw_cmd": float(head_yaw),
            "head_pitch_cmd": float(head_pitch),
        }
        self.status = status

        if step_i % 20 == 0:
            print(
                "\033[32m [adaptive_scenario] fly_to_tetrahedron_inspect \033[0m"
                f"phase={self.phase} target={arrive_dist:.3f} body={status['body_dist']:.3f} "
                f"face={face_i} u={face_u:.2f} gaze_target={gaze_target_name} "
                f"spin={status['tetra_spin_rad_per_step']:.3f} face_steps={status['tetra_face_steps']} "
                f"cube_spin={status['cube_spin_rad_per_step']:.3f} cube_tumble={status['cube_tumble_rad_per_step']:.3f} "
                f"cube_flight={status['cube_flight_rad_per_step']:.3f} "
                f"cmd=({vx:.2f},{vy:.2f},{vz:.2f}) mocap=({int(tetra_pose_ok)},{int(cube_pose_ok)})"
            )

        return ScenarioCommand(body=body, arm=arm, hand=hand, leg=leg, status=status)


class AdaptiveScenarioController:
    """
    Scenario manager.

    Add future scenarios by registering another BaseScenario subclass in
    self.registry. Runtime only calls start(name), stop(reason), update().
    """

    def __init__(self, owner):
        self.owner = owner
        self.registry = {
            FlyToCubePalpateScenario.name: FlyToCubePalpateScenario,
            FlyToSmallCubeGraspRotateScenario.name: FlyToSmallCubeGraspRotateScenario,
            FlyToTetrahedronInspectScenario.name: FlyToTetrahedronInspectScenario,
        }
        self.active: Optional[BaseScenario] = None

    def start(self, name: str, options: dict | None = None) -> None:
        if name not in self.registry:
            raise KeyError(f"unknown adaptive scenario: {name}")
        if self.active is not None and self.active.active and getattr(self.active, "name", "") == name:
            if hasattr(self.active, "apply_options"):
                self.active.apply_options(options)
                self.owner._fly_to_cube_palpate_active = True
                self.owner._fly_to_cube_palpate_status = dict(self.active.status)
                print(f"[adaptive_scenario] updated {name} options={dict(options or {})}")
                return
        self.active = self.registry[name](self.owner)
        self.active.options = dict(options or {})
        self.active.start()
        self.owner._fly_to_cube_palpate_active = True
        print(f"[adaptive_scenario] started {name}")

    def stop(self, reason: str = "stopped") -> None:
        if self.active is not None:
            self.active.stop(reason)
            self.owner._fly_to_cube_palpate_status = dict(self.active.status)
        self.owner._fly_to_cube_palpate_active = False
        self.active = None
        self.owner._ipc_manual_body_action = np.zeros(9, dtype=np.float32)
        self.owner._ipc_manual_arm_action = np.zeros(6, dtype=np.float32)
        self.owner._ipc_manual_hand_action = np.full(44, 0.5, dtype=np.float32)
        self.owner._ipc_manual_leg_action = np.zeros(18, dtype=np.float32)
        print(f"[adaptive_scenario] stopped: {reason}")

    def update(self) -> None:
        if self.active is None or not self.active.active:
            self.owner._fly_to_cube_palpate_active = False
            return

        cmd = self.active.step()
        if cmd is None:
            return

        self.owner._ipc_manual_actions_enabled = True
        self.owner._ipc_manual_body_action = cmd.body.astype(np.float32)
        self.owner._ipc_manual_arm_action = cmd.arm.astype(np.float32)
        self.owner._ipc_manual_hand_action = cmd.hand.astype(np.float32)
        self.owner._ipc_manual_leg_action = cmd.leg.astype(np.float32)

        self.owner._fly_to_cube_palpate_active = bool(cmd.status.get("active", True))
        self.owner._fly_to_cube_palpate_status = dict(cmd.status)

# ---------------------------------------------------------------------------
# Small cube adaptive palm/cube servo V5.
# ---------------------------------------------------------------------------
try:
    from src.modules.m15_counterfactual_imagination_planning.small_cube_grasp_rotate_finger_gait import install_small_cube_grasp_rotate_gait
    install_small_cube_grasp_rotate_gait(FlyToSmallCubeGraspRotateScenario)
except NameError:
    pass
except Exception as e:
    print(f"[small_cube_adaptive_servo_v5] install skipped: {e}")

# ---------------------------------------------------------------------------
# Small cube chase-until-grasp V6.
# Keeps chasing after palm touch until real stable grasp.
# ---------------------------------------------------------------------------
try:
    from src.modules.m15_counterfactual_imagination_planning.small_cube_adaptive_chase import install_small_cube_chase_until_grasp

    install_small_cube_chase_until_grasp(FlyToSmallCubeGraspRotateScenario)
except NameError:
    pass
except Exception as e:
    print(f"[small_cube_chase_v6] install skipped: {e}")

# ---------------------------------------------------------------------------
# Small cube arm reach servo V7.
# Explicit adaptive arm extension: hand reaches, not only body circles.
# ---------------------------------------------------------------------------
try:
    from src.modules.m15_counterfactual_imagination_planning.small_cube_arm_reach_servo import install_small_cube_arm_reach_servo

    install_small_cube_arm_reach_servo(FlyToSmallCubeGraspRotateScenario)
except NameError:
    pass
except Exception as e:
    print(f"[small_cube_arm_reach_v7] install skipped: {e}")
