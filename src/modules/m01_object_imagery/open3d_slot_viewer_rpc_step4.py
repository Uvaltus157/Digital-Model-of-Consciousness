from __future__ import annotations

import argparse, base64, json, time, urllib.request, zlib
import numpy as np

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _load_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception as e:
        raise SystemExit(f"Open3D import failed: {e}")


def _rpc(url: str, method: str, params: dict | None = None) -> dict | None:
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with _NO_PROXY_OPENER.open(req, timeout=1.0) as r:
            out = json.loads(r.read().decode("utf-8"))
        return out.get("result")
    except Exception as e:
        print("RPC unavailable:", e)
        return None


def _dec(spec: dict | None) -> np.ndarray:
    if not isinstance(spec, dict):
        return np.zeros((0, 3), dtype=np.float32)
    try:
        raw = zlib.decompress(base64.b64decode(spec["data"].encode("ascii")))
        return np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(tuple(spec["shape"])).astype(np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def _frame(res: dict | None, sid: int) -> dict:
    for f in (res or {}).get("slots", []):
        if int(f.get("slot_id", -1)) == int(sid):
            return f
    return {}


def _pts(f: dict) -> np.ndarray:
    x = _dec(f.get("xyz"))
    if x.ndim != 2 or x.shape[-1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    x = x[:, :3]
    return x[np.isfinite(x).all(axis=1)].astype(np.float32)


def _cols(f: dict, n: int, tint: tuple[float, float, float]) -> np.ndarray:
    c = _dec(f.get("color"))
    if c.ndim != 2 or c.shape[-1] < 3 or c.shape[0] != n:
        c = np.ones((n, 3), dtype=np.float32) * 0.7
    c = np.nan_to_num(c[:, :3], nan=0.7, posinf=1.0, neginf=0.0)
    return np.clip(0.65 * np.clip(c, 0, 1) + 0.35 * np.asarray(tint, dtype=np.float32).reshape(1, 3), 0, 1)


def _bbox(o3d, pts: np.ndarray, tint: tuple[float, float, float]):
    if pts.shape[0] <= 1:
        return None
    b = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    b.color = tint
    return b


def _mesh(o3d, pts: np.ndarray, tint: tuple[float, float, float]):
    if pts.shape[0] < 4:
        return None
    try:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        mesh, _ = pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(tint)
        return mesh
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Inner Object Open3D RPC Step4 Viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--slot", default="both", choices=["0", "1", "both"])
    p.add_argument("--mode", default="deformed", choices=["raw", "deformed"])
    p.add_argument("--refresh", type=float, default=0.08)
    p.add_argument("--point-size", type=float, default=7.0)
    p.add_argument("--offset", type=float, default=1.25)
    args = p.parse_args()

    o3d = _load_open3d()
    url = f"http://{args.host}:{args.port}/"
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Inner Object Open3D RPC Step4 {args.host}:{args.port}", width=1280, height=760)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.asarray([0.02, 0.025, 0.035], dtype=np.float32)

    pcd0, pcd1 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    vis.add_geometry(pcd0)
    vis.add_geometry(pcd1)
    state = {"slot": args.slot, "mode": args.mode, "points": True, "mesh": False, "bbox": True, "last": "", "first": True, "dyn": []}

    vis.register_key_callback(ord("R"), lambda _v: state.update(mode="raw") or False)
    vis.register_key_callback(ord("D"), lambda _v: state.update(mode="deformed") or False)
    vis.register_key_callback(ord("0"), lambda _v: state.update(slot="0") or False)
    vis.register_key_callback(ord("1"), lambda _v: state.update(slot="1") or False)
    vis.register_key_callback(ord("B"), lambda _v: state.update(slot="both") or False)
    vis.register_key_callback(ord("P"), lambda _v: state.update(points=not state["points"]) or False)
    vis.register_key_callback(ord("M"), lambda _v: state.update(mesh=not state["mesh"]) or False)
    vis.register_key_callback(ord("X"), lambda _v: state.update(bbox=not state["bbox"]) or False)

    print("[Inner Object Open3D RPC Step4]", url)
    print("keys: 0 slot0 | 1 slot1 | B both | R raw | D deformed | P points | M mesh | X bbox")

    while True:
        res = _rpc(url, "slot_viewer.get_both_slots", {"mode": state["mode"]})
        if res:
            for g in state["dyn"]:
                try:
                    vis.remove_geometry(g, reset_bounding_box=False)
                except Exception:
                    pass
            state["dyn"] = []
            f0, f1 = _frame(res, 0), _frame(res, 1)
            nonempty = False
            for sid, f, pcd, visible, xoff, tint in [
                (0, f0, pcd0, state["slot"] in ("0", "both"), -args.offset / 2 if state["slot"] == "both" else 0.0, (0.2, 0.9, 1.0)),
                (1, f1, pcd1, state["slot"] in ("1", "both"), args.offset / 2 if state["slot"] == "both" else 0.0, (1.0, 0.8, 0.2)),
            ]:
                pts = _pts(f)
                if visible and pts.shape[0]:
                    pts = pts.copy()
                    pts[:, 0] += float(xoff)
                draw_pts = pts if (visible and state["points"]) else np.zeros((0, 3), dtype=np.float32)
                cols = _cols(f, draw_pts.shape[0], tint)
                pcd.points = o3d.utility.Vector3dVector(draw_pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                vis.update_geometry(pcd)
                nonempty = nonempty or draw_pts.shape[0] > 0
                if visible and state["bbox"]:
                    b = _bbox(o3d, pts, tint)
                    if b is not None:
                        vis.add_geometry(b, reset_bounding_box=False)
                        state["dyn"].append(b)
                if visible and state["mesh"]:
                    m = _mesh(o3d, pts, tint)
                    if m is not None:
                        vis.add_geometry(m, reset_bounding_box=False)
                        state["dyn"].append(m)

            title = (
                f"mode={state['mode']} slot={state['slot']} p={int(state['points'])} m={int(state['mesh'])} x={int(state['bbox'])} | "
                f"s0={f0.get('target_name','?')} n={f0.get('point_count',0)} phase={float(f0.get('playback_phase',0.0)):.3f} live={int(bool(f0.get('live_xyz_changed', False)))} | "
                f"s1={f1.get('target_name','?')} n={f1.get('point_count',0)} phase={float(f1.get('playback_phase',0.0)):.3f} live={int(bool(f1.get('live_xyz_changed', False)))}"
            )
            if title != state["last"]:
                print(title)
                state["last"] = title
            if nonempty and state["first"]:
                vis.reset_view_point(True)
                state["first"] = False

        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(max(0.01, float(args.refresh)))

    vis.destroy_window()


if __name__ == "__main__":
    main()
