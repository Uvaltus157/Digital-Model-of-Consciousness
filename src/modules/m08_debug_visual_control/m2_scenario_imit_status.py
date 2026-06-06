from __future__ import annotations

from typing import Any, Dict


def build_m2_scenario_imit_status(runtime: Any) -> Dict[str, Any]:
    if hasattr(runtime, "m2_scenario_imit_status"):
        try:
            out = runtime.m2_scenario_imit_status()
            if isinstance(out, dict):
                out = dict(out)
            else:
                out = {}
        except Exception as e:
            out = {"active": False, "error": str(e)}
    else:
        out = {"active": False, "installed": False}

    out.setdefault("layout", "imit")
    out["installed"] = bool(hasattr(runtime, "request_m2_scenario_imit"))
    out["global_step"] = int(getattr(runtime, "global_step", 0))

    streamer = getattr(runtime, "slot_4d_jsonrpc_streamer", None)
    if streamer is not None and hasattr(streamer, "status"):
        try:
            out["streamer"] = streamer.status()
        except Exception:
            out["streamer"] = {}
    return out


__all__ = ["build_m2_scenario_imit_status"]
