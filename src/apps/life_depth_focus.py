from __future__ import annotations

import numpy as np
import torch


class LifeDepthFocusMixin:
    def _floating_depth_focus_params(self) -> tuple[float | None, float | None, str]:
        scenario = getattr(self, "_fly_to_cube_palpate_status", None)
        if not isinstance(scenario, dict):
            return None, None, ""
        is_active = bool(getattr(self, "_fly_to_cube_palpate_active", False) or scenario.get("active", False))
        if not is_active:
            return None, None, ""
        if str(scenario.get("scenario", "")) != "fly_to_tetrahedron_inspect":
            return None, None, ""
        try:
            focus_depth = float(scenario.get("gaze_distance", 0.0))
            if not np.isfinite(focus_depth) or focus_depth <= 0.0:
                return None, None, ""
            half_range = max(0.10, float(scenario.get("depth_focus_half_range", 0.85)))
            return focus_depth, half_range, str(scenario.get("gaze_target", ""))
        except Exception:
            return None, None, ""

    def apply_focused_depth_observation(self, obs: dict) -> dict:
        """
        During floating-object inspection the learning input should be a focused
        depth map, not the raw metric renderer depth. Keep depth_raw for debug
        views that still want meters.
        """
        if not isinstance(obs, dict) or "depth" not in obs:
            return obs

        focus_depth, half_range, focus_label = self._floating_depth_focus_params()
        if focus_depth is None:
            return obs

        depth = obs.get("depth")
        if not torch.is_tensor(depth):
            return obs

        try:
            raw = depth.detach()
            valid = torch.isfinite(raw) & (raw > 1e-6)
            if not bool(valid.any().detach().cpu().item()):
                return obs

            lo = max(0.0, float(focus_depth) - float(half_range))
            hi = float(focus_depth) + float(half_range)
            focused = (torch.clamp(raw.float(), min=lo, max=hi) - lo) / max(hi - lo, 1e-6)
            focused = torch.where(valid, focused, torch.ones_like(focused))

            out = dict(obs)
            out["depth_raw"] = raw
            out["depth"] = focused.clamp(0.0, 1.0)
            out["depth_focus_applied"] = True
            out["depth_focus_depth"] = float(focus_depth)
            out["depth_focus_half_range"] = float(half_range)
            out["depth_focus_label"] = str(focus_label)
            return out
        except Exception as e:
            if not hasattr(self, "_focused_depth_warned"):
                print(f"[depth_focus] failed to focus observation depth: {e}")
                self._focused_depth_warned = True
            return obs


__all__ = ["LifeDepthFocusMixin"]
