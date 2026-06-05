from __future__ import annotations

from typing import Any, Dict
import time
import traceback


def _compact(value: Any, *, max_items: int = 6) -> Any:
    try:
        import torch
        if torch.is_tensor(value):
            flat = value.detach().float().reshape(-1)
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "sample": [float(x) for x in flat[:max_items].cpu().tolist()],
            }
    except Exception:
        pass
    if isinstance(value, dict):
        out = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = f"{len(value) - max_items} more"
                break
            out[str(k)] = _compact(v, max_items=max_items)
        return out
    if isinstance(value, (list, tuple)):
        return [_compact(v, max_items=max_items) for v in list(value)[:max_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_module_lab_from_payload(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    module = str(payload.get("module", "all")).lower().strip() or "all"
    started = time.time()
    try:
        if module in ("behavioral", "scenario", "scenarios", "unconscious_scenarios"):
            from scripts.module_lab.scenario_unconscious_replay import run_all
            result = run_all()
            return {
                "ok": bool(result.get("status") == "ok"),
                "module": module,
                "kind": "behavioral_scenarios",
                "duration_sec": round(time.time() - started, 4),
                "result": _compact(result, max_items=12),
            }

        from scripts.module_lab.run_module_lab import LABS
        if module == "all":
            selected = ("m11", "m13", "m4", "m02", "m05", "loop")
            result = {name: LABS[name]() for name in selected if name in LABS}
        else:
            if module not in LABS:
                raise KeyError(f"unknown module lab {module!r}; available={sorted(LABS)}")
            result = LABS[module]()

        return {
            "ok": True,
            "module": module,
            "kind": "module_lab",
            "duration_sec": round(time.time() - started, 4),
            "result": _compact(result, max_items=12),
        }
    except Exception as e:
        return {
            "ok": False,
            "module": module,
            "kind": "module_lab",
            "duration_sec": round(time.time() - started, 4),
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
        }


__all__ = ["run_module_lab_from_payload"]
