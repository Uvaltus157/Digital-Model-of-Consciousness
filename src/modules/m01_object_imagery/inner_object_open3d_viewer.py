
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch


@dataclass
class InnerObjectOpen3DViewerV2Config:
    enabled: bool = True
    window_name: str = "inner object Open3D"
    width: int = 960
    height: int = 760
    update_every_steps: int = 2
    point_size: float = 5.0
    voxel_threshold: float = 0.60
    max_voxel_points: int = 1200
    show_voxels: bool = True
    use_internal_color: bool = True
    max_slots: int = 4
    slot_spacing: float = 2.6
    show_slot_snapshots: bool = False
    view_zoom: float = 0.42
    view_z_near: float = 0.01
    view_z_far: float = 100.0
    export_dir: str = "exports/inner_object_3d"
    show_long_dynamic_debug: bool = False
    debug_panel_x: float = -1.80
    debug_panel_y: float = 1.40
    debug_panel_z: float = 0.0
    debug_text_scale: float = 0.030


class InnerObjectOpen3DViewerV2:
    """
    Open3D window for the internal 3D object hypothesis.

    New in V2:
      - point cloud is colored by internal color_rgb
      - multiple slot snapshots can be displayed at once
      - current model can be exported to PLY / PCD
    """
    def __init__(self, cfg: Optional[InnerObjectOpen3DViewerV2Config] = None):
        self.cfg = cfg or InnerObjectOpen3DViewerV2Config()
        self.available = False
        self._warned_unavailable = False
        self.vis = None
        self.created = False
        self.o3d = None
        self.axis = None
        self.current_pc = None
        self.current_vox = None
        self.slot_pcs = []
        self.slot_voxs = []
        self._last_current_export = None
        self.debug_text_pc = None

        try:
            import open3d as o3d
            self.o3d = o3d
            self.available = True
        except Exception as e:
            self.available = False
            self._import_error = e

    def _ensure(self):
        if not self.available:
            if not self._warned_unavailable:
                err = getattr(self, "_import_error", "unknown error")
                print(f"[inner_object_open3d] Open3D unavailable: {err}")
                self._warned_unavailable = True
            return False

        if self.created:
            return True

        self.vis = self.o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=self.cfg.window_name,
            width=int(self.cfg.width),
            height=int(self.cfg.height),
            visible=True,
        )

        self.current_pc = self.o3d.geometry.PointCloud()
        self.current_vox = self.o3d.geometry.PointCloud()
        self.axis = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        if bool(getattr(self.cfg, "show_long_dynamic_debug", False)):
            self.debug_text_pc = self.o3d.geometry.PointCloud()
        else:
            self.debug_text_pc = None

        self.vis.add_geometry(self.current_pc)
        self.vis.add_geometry(self.current_vox)
        self.vis.add_geometry(self.axis)
        if self.debug_text_pc is not None:
            self.vis.add_geometry(self.debug_text_pc)

        self.slot_pcs = []
        self.slot_voxs = []
        if bool(getattr(self.cfg, "show_slot_snapshots", False)):
            for _ in range(int(self.cfg.max_slots)):
                pc = self.o3d.geometry.PointCloud()
                vox = self.o3d.geometry.PointCloud()
                self.slot_pcs.append(pc)
                self.slot_voxs.append(vox)
                self.vis.add_geometry(pc)
                self.vis.add_geometry(vox)

        try:
            opt = self.vis.get_render_option()
            opt.point_size = float(self.cfg.point_size)
            opt.background_color = np.array([0.04, 0.05, 0.07], dtype=np.float64)
        except Exception:
            pass
        self._apply_view()

        self.created = True
        return True

    def close(self):
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass
        self.vis = None
        self.current_pc = None
        self.current_vox = None
        self.axis = None
        self.debug_text_pc = None
        self.slot_pcs = []
        self.slot_voxs = []
        self.created = False

    def _tensor_to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _voxel_centers(self, occ: np.ndarray):
        idx = np.argwhere(occ >= float(self.cfg.voxel_threshold))
        if idx.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        R = occ.shape[0]
        pts = (idx.astype(np.float32) / max(1, R - 1)) * 2.0 - 1.0
        vals = occ[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
        if pts.shape[0] > int(self.cfg.max_voxel_points):
            order = np.argsort(vals)[::-1][: int(self.cfg.max_voxel_points)]
            pts = pts[order]
            vals = vals[order]
        return pts, vals

    def _base_color(self, obj: Dict) -> np.ndarray:
        if self.cfg.use_internal_color and "color_rgb" in obj:
            col = self._tensor_to_numpy(obj["color_rgb"])
            if col.ndim == 2:
                col = col[0]
            col = np.clip(col.reshape(3), 0.0, 1.0)
            return col.astype(np.float32)
        return np.array([0.35, 0.65, 1.0], dtype=np.float32)

    def _debug_imit_kind(self, obj: Dict) -> str:
        kind = str(obj.get("debug_imit_shape_kind", "") or "").lower().strip()
        if kind in ("tetra", "tetrahedron", "pyramid"):
            return "tetrahedron"
        if kind in ("morph", "cube_tetra"):
            return "morph"
        return "cube" if kind in ("cube", "box", "cuboid", "hexahedron") else ""

    def _line_points(self, a: np.ndarray, b: np.ndarray, n: int = 18) -> np.ndarray:
        t = np.linspace(0.0, 1.0, max(2, int(n)), dtype=np.float32)[:, None]
        return (1.0 - t) * a[None, :] + t * b[None, :]

    def _debug_imit_primitive_points(self, kind: str) -> np.ndarray:
        kind = self._debug_imit_kind({"debug_imit_shape_kind": kind})
        if kind == "tetrahedron":
            verts = np.asarray(
                [
                    [0.0, 0.0, 0.85],
                    [0.80, 0.0, -0.45],
                    [-0.40, 0.70, -0.45],
                    [-0.40, -0.70, -0.45],
                ],
                dtype=np.float32,
            )
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]
            return np.concatenate([self._line_points(verts[i], verts[j], 24) for i, j in edges], axis=0)

        cube_vals = [-0.65, 0.65]
        cube_verts = np.asarray([[x, y, z] for x in cube_vals for y in cube_vals for z in cube_vals], dtype=np.float32)
        cube_edges = []
        for i, a in enumerate(cube_verts):
            for j, b in enumerate(cube_verts):
                if j <= i:
                    continue
                if int(np.sum(np.abs(a - b) > 1e-6)) == 1:
                    cube_edges.append((i, j))
        cube = np.concatenate([self._line_points(cube_verts[i], cube_verts[j], 18) for i, j in cube_edges], axis=0)

        if kind == "morph":
            tetra = self._debug_imit_primitive_points("tetrahedron")
            n = min(cube.shape[0], tetra.shape[0])
            return np.concatenate([0.55 * cube[:n] + 0.45 * tetra[:n], cube[::3], tetra[::3]], axis=0)
        return cube

    def _debug_imit_geometry(self, obj: Dict, offset: np.ndarray, fade: float):
        kind = self._debug_imit_kind(obj)
        if not kind:
            return None
        pts = self._debug_imit_primitive_points(kind).astype(np.float32) + offset[None, :]
        if kind == "tetrahedron":
            base = np.array([1.0, 0.58, 0.20], dtype=np.float32)
        elif kind == "morph":
            base = np.array([0.85, 0.45, 1.0], dtype=np.float32)
        else:
            base = np.array([0.20, 0.85, 1.0], dtype=np.float32)
        colors = np.repeat((base * float(fade))[None, :], pts.shape[0], axis=0)
        return pts, np.clip(colors, 0.0, 1.0)

    def _geometry_from_obj(self, obj: Dict, offset=None, fade: float = 1.0):
        if offset is None:
            offset = np.zeros(3, dtype=np.float32)
        else:
            offset = np.asarray(offset, dtype=np.float32)

        if bool(obj.get("debug_imit_fallback_shape", False)):
            fallback = self._debug_imit_geometry(obj, offset, fade)
            if fallback is not None:
                pts, pt_colors = fallback
                return pts, pt_colors, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        pts = self._tensor_to_numpy(obj["point_cloud"])
        if pts.ndim == 3:
            pts = pts[0]
        conf = self._tensor_to_numpy(obj.get("point_conf", None))
        if conf is None:
            conf = np.ones((pts.shape[0], 1), dtype=np.float32)
        elif conf.ndim == 3:
            conf = conf[0]
        conf = np.clip(conf.reshape(-1, 1), 0.0, 1.0)

        pts = pts.astype(np.float32) + offset[None, :]

        base_color = self._base_color(obj)
        pt_colors = np.clip(base_color[None, :] * (0.35 + 0.65 * conf) * float(fade), 0.0, 1.0)

        voxel_pts = np.zeros((0, 3), dtype=np.float32)
        voxel_colors = np.zeros((0, 3), dtype=np.float32)
        if self.cfg.show_voxels and "voxel_occ" in obj:
            occ = self._tensor_to_numpy(obj["voxel_occ"])
            if occ.ndim == 5:
                occ = occ[0, 0]
            elif occ.ndim == 4:
                occ = occ[0]
            voxel_pts, vvals = self._voxel_centers(occ)
            voxel_pts = voxel_pts + offset[None, :]
            if voxel_pts.shape[0] > 0:
                voxel_colors = np.clip(base_color[None, :] * (0.25 + 0.75 * vvals[:, None]) * float(fade), 0.0, 1.0)

        return pts, pt_colors, voxel_pts, voxel_colors


    def _scalar_debug_value(self, obj: Dict, key: str, default=None):
        try:
            if key not in obj:
                return default
            x = obj.get(key)
            if isinstance(x, torch.Tensor):
                if x.numel() == 0:
                    return default
                return float(x.detach().cpu().reshape(-1)[0].item())
            arr = np.asarray(x)
            if arr.size == 0:
                return default
            return float(arr.reshape(-1)[0])
        except Exception:
            return default

    def _fmt_debug_float(self, value, ndigits: int = 3) -> str:
        try:
            if value is None:
                return "-"
            value = float(value)
            if not np.isfinite(value):
                return "-"
            if abs(value) >= 100:
                return f"{value:.0f}"
            if abs(value) >= 10:
                return f"{value:.1f}"
            return f"{value:.{ndigits}f}"
        except Exception:
            return "-"

    def _long_dynamic_debug_lines(self, obj: Dict) -> List[str]:
        ready = self._scalar_debug_value(obj, "long_dynamic_ready", None)
        write = self._scalar_debug_value(obj, "long_dynamic_slot_update_allowed", None)
        props = self._scalar_debug_value(obj, "semantic_proposal_count", None)
        conf = self._scalar_debug_value(obj, "long_dynamic_confidence", None)
        streak = self._scalar_debug_value(obj, "long_dynamic_ready_streak", None)
        steps = self._scalar_debug_value(obj, "long_dynamic_steps", None)
        active_steps = self._scalar_debug_value(obj, "long_dynamic_active_steps", None)
        dz = self._scalar_debug_value(obj, "long_dynamic_dz", None)
        dyn = self._scalar_debug_value(obj, "dynamic_score", None)
        nov = self._scalar_debug_value(obj, "scene_novelty", None)
        inter = self._scalar_debug_value(obj, "interaction", None)
        slot = self._scalar_debug_value(obj, "semantic_updated_slot", None)

        def flag(v):
            if v is None:
                return "-"
            return "1" if float(v) >= 0.5 else "0"

        return [
            "LONG DYNAMIC",
            f"READY:{flag(ready)} WRITE:{flag(write)}",
            f"PROPS:{self._fmt_debug_float(props, 0)} SLOT:{self._fmt_debug_float(slot, 0)}",
            f"CONF:{self._fmt_debug_float(conf)} STREAK:{self._fmt_debug_float(streak, 0)}",
            f"STEPS:{self._fmt_debug_float(steps, 0)} ACT:{self._fmt_debug_float(active_steps, 0)}",
            f"DYN:{self._fmt_debug_float(dyn)} NOV:{self._fmt_debug_float(nov)}",
            f"INT:{self._fmt_debug_float(inter)} DZ:{self._fmt_debug_float(dz)}",
        ]

    def _glyph_pattern(self, ch: str):
        font = {
            "A":["01110","10001","10001","11111","10001","10001","10001"],
            "B":["11110","10001","10001","11110","10001","10001","11110"],
            "C":["01111","10000","10000","10000","10000","10000","01111"],
            "D":["11110","10001","10001","10001","10001","10001","11110"],
            "E":["11111","10000","10000","11110","10000","10000","11111"],
            "F":["11111","10000","10000","11110","10000","10000","10000"],
            "G":["01111","10000","10000","10011","10001","10001","01111"],
            "H":["10001","10001","10001","11111","10001","10001","10001"],
            "I":["11111","00100","00100","00100","00100","00100","11111"],
            "J":["00111","00010","00010","00010","00010","10010","01100"],
            "K":["10001","10010","10100","11000","10100","10010","10001"],
            "L":["10000","10000","10000","10000","10000","10000","11111"],
            "M":["10001","11011","10101","10101","10001","10001","10001"],
            "N":["10001","11001","10101","10011","10001","10001","10001"],
            "O":["01110","10001","10001","10001","10001","10001","01110"],
            "P":["11110","10001","10001","11110","10000","10000","10000"],
            "Q":["01110","10001","10001","10001","10101","10010","01101"],
            "R":["11110","10001","10001","11110","10100","10010","10001"],
            "S":["01111","10000","10000","01110","00001","00001","11110"],
            "T":["11111","00100","00100","00100","00100","00100","00100"],
            "U":["10001","10001","10001","10001","10001","10001","01110"],
            "V":["10001","10001","10001","10001","10001","01010","00100"],
            "W":["10001","10001","10001","10101","10101","10101","01010"],
            "X":["10001","10001","01010","00100","01010","10001","10001"],
            "Y":["10001","10001","01010","00100","00100","00100","00100"],
            "Z":["11111","00001","00010","00100","01000","10000","11111"],
            "0":["01110","10001","10011","10101","11001","10001","01110"],
            "1":["00100","01100","00100","00100","00100","00100","01110"],
            "2":["01110","10001","00001","00010","00100","01000","11111"],
            "3":["11110","00001","00001","01110","00001","00001","11110"],
            "4":["00010","00110","01010","10010","11111","00010","00010"],
            "5":["11111","10000","10000","11110","00001","00001","11110"],
            "6":["01110","10000","10000","11110","10001","10001","01110"],
            "7":["11111","00001","00010","00100","01000","01000","01000"],
            "8":["01110","10001","10001","01110","10001","10001","01110"],
            "9":["01110","10001","10001","01111","00001","00001","01110"],
            ":":["00000","00100","00100","00000","00100","00100","00000"],
            ".":["00000","00000","00000","00000","00000","00100","00100"],
            "-":["00000","00000","00000","11111","00000","00000","00000"],
            "/":["00001","00010","00010","00100","01000","01000","10000"],
            " ":["00000","00000","00000","00000","00000","00000","00000"],
        }
        return font.get(str(ch).upper(), font[" "])

    def _debug_text_points(self, obj: Dict, lines: List[str]):
        scale = float(getattr(self.cfg, "debug_text_scale", 0.030))
        x0 = float(getattr(self.cfg, "debug_panel_x", -1.80))
        y0 = float(getattr(self.cfg, "debug_panel_y", 1.40))
        z0 = float(getattr(self.cfg, "debug_panel_z", 0.0))
        pts, colors = [], []
        for li, line in enumerate(lines):
            y_line = y0 - li * scale * 9.0
            for ci, ch in enumerate(str(line).upper()[:32]):
                pattern = self._glyph_pattern(ch)
                x_char = x0 + ci * scale * 6.0
                for gy, row in enumerate(pattern):
                    for gx, bit in enumerate(row):
                        if bit != "1":
                            continue
                        pts.append([x_char + gx * scale, y_line - gy * scale, z0])
                        if "READY:1" in line or "WRITE:1" in line:
                            col = [0.35, 1.0, 0.45]
                        elif "READY:0" in line or "WRITE:0" in line:
                            col = [1.0, 0.35, 0.25]
                        else:
                            col = [0.85, 0.92, 1.0]
                        colors.append(col)
        ready = self._scalar_debug_value(obj, "long_dynamic_ready", 0.0)
        write = self._scalar_debug_value(obj, "long_dynamic_slot_update_allowed", 0.0)
        lamp_y = y0 + scale * 2.0
        for i, flag_value in enumerate([ready, write]):
            cx = x0 + i * scale * 8.0
            for a in range(6):
                for b in range(6):
                    pts.append([cx + a * scale * 0.8, lamp_y - b * scale * 0.8, z0])
                    colors.append([0.1, 1.0, 0.2] if float(flag_value or 0.0) >= 0.5 else [1.0, 0.15, 0.1])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32), np.asarray(colors, dtype=np.float32)

    def _update_long_dynamic_debug_panel(self, obj: Dict):
        if getattr(self, "debug_text_pc", None) is None:
            return
        if not bool(getattr(self.cfg, "show_long_dynamic_debug", False)):
            if getattr(self, "debug_text_pc", None) is not None:
                self._set_pc(self.debug_text_pc, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return
        lines = self._long_dynamic_debug_lines(obj)
        pts, cols = self._debug_text_points(obj, lines)
        self._set_pc(self.debug_text_pc, pts, cols)

    def _set_pc(self, geom, points: np.ndarray, colors: np.ndarray):
        geom.points = self.o3d.utility.Vector3dVector(points.astype(np.float64))
        geom.colors = self.o3d.utility.Vector3dVector(colors.astype(np.float64))

    def _apply_view(self):
        if self.vis is None:
            return
        try:
            ctr = self.vis.get_view_control()
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_front([0.55, -0.75, 0.35])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(float(getattr(self.cfg, "view_zoom", 0.42)))
            if hasattr(ctr, "set_constant_z_near"):
                ctr.set_constant_z_near(float(getattr(self.cfg, "view_z_near", 0.01)))
            if hasattr(ctr, "set_constant_z_far"):
                ctr.set_constant_z_far(float(getattr(self.cfg, "view_z_far", 100.0)))
        except Exception:
            pass

    def update(self, current_obj: Dict, slot_snapshots: Optional[List[Dict]] = None):
        if not self._ensure():
            return

        pts, cols, vpts, vcols = self._geometry_from_obj(current_obj, fade=1.0)
        self._update_long_dynamic_debug_panel(current_obj)
        self._set_pc(self.current_pc, pts, cols)
        self._set_pc(self.current_vox, vpts, vcols)

        if bool(getattr(self.cfg, "show_slot_snapshots", False)):
            snaps = slot_snapshots or []
            snaps = snaps[: len(self.slot_pcs)]
            for i, (pc, vox) in enumerate(zip(self.slot_pcs, self.slot_voxs)):
                if i < len(snaps):
                    snap = snaps[i]
                    offset = np.array([(i + 1) * float(self.cfg.slot_spacing), 0.0, 0.0], dtype=np.float32)
                    fade = max(0.35, 0.85 - 0.12 * i)
                    spts, scols, svpts, svcols = self._geometry_from_obj(snap, offset=offset, fade=fade)
                else:
                    spts = np.zeros((0, 3), dtype=np.float32)
                    scols = np.zeros((0, 3), dtype=np.float32)
                    svpts = np.zeros((0, 3), dtype=np.float32)
                    svcols = np.zeros((0, 3), dtype=np.float32)
                self._set_pc(pc, spts, scols)
                self._set_pc(vox, svpts, svcols)

        self.vis.update_geometry(self.current_pc)
        self.vis.update_geometry(self.current_vox)
        for pc, vox in zip(self.slot_pcs, self.slot_voxs):
            self.vis.update_geometry(pc)
            self.vis.update_geometry(vox)
        if self.axis is not None:
            self.vis.update_geometry(self.axis)
        if self.debug_text_pc is not None:
            self.vis.update_geometry(self.debug_text_pc)

        self.vis.poll_events()
        self.vis.update_renderer()
        self._last_current_export = current_obj

    def export_current(self, fmt: str = "ply") -> Optional[str]:
        if not self.available or self._last_current_export is None:
            return None
        fmt = fmt.lower().strip(".")
        if fmt not in {"ply", "pcd"}:
            raise ValueError(f"Unsupported format: {fmt}")

        export_dir = Path(self.cfg.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        obj = self._last_current_export
        pts, cols, vpts, vcols = self._geometry_from_obj(obj, fade=1.0)
        all_pts = pts
        all_cols = cols
        if self.cfg.show_voxels and vpts.shape[0] > 0:
            all_pts = np.concatenate([all_pts, vpts], axis=0)
            all_cols = np.concatenate([all_cols, vcols], axis=0)

        pc = self.o3d.geometry.PointCloud()
        pc.points = self.o3d.utility.Vector3dVector(all_pts.astype(np.float64))
        pc.colors = self.o3d.utility.Vector3dVector(np.clip(all_cols, 0.0, 1.0).astype(np.float64))

        path = export_dir / f"inner_object_current.{fmt}"
        self.o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
        return str(path)
