from __future__ import annotations

"""M2 scenario imitator for the Open3D JSON-RPC slot viewer.

This is a deterministic debug feeder for the M2 -> slot Gaussian -> Open3D RPC
path. It does not train model weights. It creates synthetic scenario objects as
Gaussian states for slot 0/1 and publishes them through Slot4DJsonRpcStreamer so
the separate Open3D RPC viewer can display concrete objects immediately.
"""

from typing import Any, Dict, Iterable
import math
import numpy as np
import torch


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _normalize_kind(kind: str) -> str:
    k = str(kind or "cube_tetra").lower().strip()
    if k in ("clear", "none", "off", "stop"):
        return "clear"
    if k in ("tetra", "tetrahedron", "pyramid"):
        return "tetrahedron"
    if k in ("cube", "box", "cuboid", "hexahedron"):
        return "cube"
    if k in ("morph", "cube_tetra", "cube+tetra", "cube_and_tetra"):
        return k
    return "cube_tetra"


def _line_points(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(2, int(n)), dtype=np.float32)[:, None]
    return (1.0 - t) * a[None, :] + t * b[None, :]


def _cube_points(samples_per_edge: int = 32) -> np.ndarray:
    vals = [-0.55, 0.55]
    verts = np.asarray([[x, y, z] for x in vals for y in vals for z in vals], dtype=np.float32)
    edges = []
    for i, a in enumerate(verts):
        for j, b in enumerate(verts):
            if j <= i:
                continue
            if int(np.sum(np.abs(a - b) > 1e-6)) == 1:
                edges.append((i, j))
    return np.concatenate([_line_points(verts[i], verts[j], samples_per_edge) for i, j in edges], axis=0)


def _tetra_points(samples_per_edge: int = 40) -> np.ndarray:
    verts = np.asarray(
        [
            [0.0, 0.0, 0.75],
            [0.75, 0.0, -0.45],
            [-0.38, 0.66, -0.45],
            [-0.38, -0.66, -0.45],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]
    return np.concatenate([_line_points(verts[i], verts[j], samples_per_edge) for i, j in edges], axis=0)


def _shape_points(kind: str, *, alpha: float = 0.5, density: int = 1) -> np.ndarray:
    density = max(1, min(8, int(density)))
    kind = _normalize_kind(kind)
    cube = _cube_points(28 * density)
    tetra = _tetra_points(34 * density)
    if kind == "tetrahedron":
        pts = tetra
    elif kind == "morph":
        n = min(len(cube), len(tetra))
        a = max(0.0, min(1.0, float(alpha)))
        pts = np.concatenate([(1.0 - a) * cube[:n] + a * tetra[:n], cube[::4], tetra[::4]], axis=0)
    else:
        pts = cube
    pts = pts.astype(np.float32)
    pts[:, 2] += 2.0
    return pts


def _shape_color(kind: str, n: int) -> np.ndarray:
    kind = _normalize_kind(kind)
    if kind == "tetrahedron":
        base = np.asarray([1.0, 0.65, 0.18], dtype=np.float32)
    elif kind == "morph":
        base = np.asarray([0.85, 0.45, 1.0], dtype=np.float32)
    else:
        base = np.asarray([0.18, 0.85, 1.0], dtype=np.float32)
    return np.repeat(base[None, :], int(n), axis=0).astype(np.float32)


class M2ScenarioImitRuntimeMixin:
    def _m2_scenario_device(self) -> torch.device:
        return torch.device(getattr(self, "device", torch.device("cpu")))

    def _ensure_m2_scenario_gaussian_reconstructor(self) -> None:
        if hasattr(self, "slot_gaussian_reconstructor"):
            return
        if hasattr(self, "_ensure_slot_gaussian_reconstruction"):
            self._ensure_slot_gaussian_reconstruction()
            return
        from src.modules.m01_object_imagery.slot_gaussian_reconstruction import SlotGaussianReconstructor

        self.slot_gaussian_reconstructor = SlotGaussianReconstructor(device=self._m2_scenario_device())

    def _m2_slot_items(self, payload: Dict[str, Any]) -> list[dict[str, Any]]:
        kind = _normalize_kind(str(payload.get("kind", "cube_tetra")))
        alpha = max(0.0, min(1.0, _safe_float(payload.get("alpha", 0.5), 0.5)))
        if kind in ("cube", "tetrahedron", "morph"):
            slot = int(_safe_float(payload.get("slot", 0 if kind == "cube" else 1), 0))
            return [{"slot": max(0, min(1, slot)), "kind": kind, "alpha": alpha}]
        return [
            {"slot": 0, "kind": "cube", "alpha": 0.0},
            {"slot": 1, "kind": "tetrahedron", "alpha": 1.0},
        ]

    def _install_m2_scenario_slot_states(self, items: Iterable[dict[str, Any]], *, density: int) -> list[dict[str, Any]]:
        from src.modules.m01_object_imagery.slot_gaussian_reconstruction import SlotGaussianMetrics, SlotGaussianState

        self._ensure_m2_scenario_gaussian_reconstructor()
        recon = self.slot_gaussian_reconstructor
        device = self._m2_scenario_device()
        installed = []
        for item in items:
            slot = int(item.get("slot", 0))
            kind = _normalize_kind(str(item.get("kind", "cube")))
            alpha = _safe_float(item.get("alpha", 0.5), 0.5)
            pts_np = _shape_points(kind, alpha=alpha, density=density)
            col_np = _shape_color(kind, len(pts_np))
            pts = torch.as_tensor(pts_np, dtype=torch.float32, device=device)
            col = torch.as_tensor(col_np, dtype=torch.float32, device=device)
            state = SlotGaussianState(slot_id=slot, target_name=kind, xyz=pts, color=col, lr=1.0e-3)
            with torch.no_grad():
                state.opacity_logit.fill_(2.2)
                state.log_scale.fill_(-3.2)
            state.updates = 1
            recon.states[slot] = state
            metrics = SlotGaussianMetrics(
                slot_id=slot,
                target_name=kind,
                initialized=True,
                gaussian_count=int(state.gaussian_count),
                updates=int(state.updates),
                rgb_loss=0.0,
                depth_loss=0.0,
                total_loss=0.0,
                render_valid=True,
                backend="m2_scenario_imit",
                requested_backend="m2_scenario_imit",
                fallback_used=False,
            )
            recon.last_metrics[slot] = metrics
            installed.append({"slot": slot, "kind": kind, "points": int(state.gaussian_count), "alpha": float(alpha)})
        return installed

    def _publish_m2_scenario_rpc(self) -> Dict[str, Any]:
        if hasattr(self, "_ensure_slot_4d_jsonrpc_streamer"):
            self._ensure_slot_4d_jsonrpc_streamer()
        elif not hasattr(self, "slot_4d_jsonrpc_streamer"):
            from src.modules.m02_event_dream_replay.slot_4d_jsonrpc_stream import Slot4DJsonRpcStreamer

            self.slot_4d_jsonrpc_streamer = Slot4DJsonRpcStreamer()
            self.slot_4d_jsonrpc_streamer.start()
        streamer = getattr(self, "slot_4d_jsonrpc_streamer", None)
        if streamer is None:
            return {"started": False, "updated": False, "reason": "streamer_missing"}
        return dict(streamer.publish(
            self.slot_gaussian_reconstructor,
            deformation_trainer=getattr(self, "slot_4d_deformation_trainer", None),
            playback_renderer=getattr(self, "slot_4d_playback_renderer", None),
            live_step=int(getattr(self, "live_step", getattr(self, "global_step", 0))),
        ))

    def _clear_m2_scenario_rpc_slots(self) -> Dict[str, Any]:
        if hasattr(self, "slot_gaussian_reconstructor"):
            try:
                self.slot_gaussian_reconstructor.states.pop(0, None)
                self.slot_gaussian_reconstructor.states.pop(1, None)
                self.slot_gaussian_reconstructor.last_metrics.pop(0, None)
                self.slot_gaussian_reconstructor.last_metrics.pop(1, None)
            except Exception:
                pass
        streamer = getattr(self, "slot_4d_jsonrpc_streamer", None)
        if streamer is not None:
            try:
                with streamer.lock:
                    streamer.frames = {}
                    streamer.last_update_unix = 0.0
            except Exception:
                pass
        return {"updated": True, "reason": "cleared", "slot_0_points": 0, "slot_1_points": 0}

    def request_m2_scenario_imit(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = dict(payload or {})
        kind = _normalize_kind(str(payload.get("kind", "cube_tetra")))
        if kind == "clear":
            rpc = self._clear_m2_scenario_rpc_slots()
            state = {
                "active": False,
                "kind": "clear",
                "items": [],
                "rpc": rpc,
                "layout": "imit",
            }
            self._m2_scenario_imit_state = state
            if hasattr(self, "write_module_debug_status"):
                self.write_module_debug_status()
            print("[m2_scenario_imit] cleared")
            return dict(state)

        density = int(_safe_float(payload.get("density", 1), 1))
        items = self._m2_slot_items(payload)
        installed = self._install_m2_scenario_slot_states(items, density=density)
        rpc = self._publish_m2_scenario_rpc()
        state = {
            "active": True,
            "kind": kind,
            "items": installed,
            "rpc": rpc,
            "layout": "imit",
            "target": "M2 scenario imit -> SlotGaussianState -> Slot4DJsonRpcStreamer -> Open3D RPC",
            "source": str(payload.get("source", "ipc")),
            "global_step": int(getattr(self, "global_step", 0)),
            "host": str(rpc.get("host", "127.0.0.1")),
            "port": int(rpc.get("port", 8771) or 8771),
        }
        self._m2_scenario_imit_state = state
        if hasattr(self, "write_module_debug_status"):
            self.write_module_debug_status()
        print(
            "[m2_scenario_imit][request] "
            f"kind={kind} items={[(i['kind'], i['slot'], i['points']) for i in installed]} "
            f"rpc_updated={bool(rpc.get('updated', False))}"
        )
        return dict(state)

    def m2_scenario_imit_status(self) -> Dict[str, Any]:
        state = getattr(self, "_m2_scenario_imit_state", None)
        if not isinstance(state, dict):
            return {"active": False, "kind": "", "items": [], "layout": "imit"}
        return dict(state)


__all__ = ["M2ScenarioImitRuntimeMixin"]
