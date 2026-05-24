from __future__ import annotations

import torch

from src.modules.m04_long_dynamic_memory.passport_debug_visualizer import PassportDebugVisualizer, PassportDebugVisualizerConfig


class PassportDebugRuntimeMixin:
    """
    Runtime glue for live-vs-passport replay debugging.

    It compares:
        live z_obj decode
    with:
        DynamicObjectPassport replay_z decode

    This answers:
        does the passport really reproduce the same internal object?
    """

    def _passport_debug_enabled(self) -> bool:
        cfg = getattr(self.cfg, "passport_debug_visualizer", None)
        return bool(getattr(cfg, "enabled", False))

    def _ensure_passport_debug_visualizer(self) -> None:
        if hasattr(self, "passport_debug_viz") and self.passport_debug_viz is not None:
            return

        cfg = getattr(self.cfg, "passport_debug_visualizer", None)
        self.passport_debug_viz = PassportDebugVisualizer(PassportDebugVisualizerConfig(
            enabled=bool(getattr(cfg, "enabled", True)),
            window_name=str(getattr(cfg, "window_name", "passport debug: live vs replay")),
            width=int(getattr(cfg, "width", 1560)),
            height=int(getattr(cfg, "height", 980)),
            delay_ms=int(getattr(cfg, "delay_ms", 1)),
            show_every_steps=int(getattr(cfg, "show_every_steps", 1)),
        ))

    def update_passport_debug_visualizer(self, obj: dict) -> dict:
        try:
            cfg = getattr(self.cfg, "passport_debug_visualizer", None)
            if not self._passport_debug_enabled():
                try:
                    if hasattr(self, "passport_debug_viz") and self.passport_debug_viz is not None:
                        self.passport_debug_viz.close()
                except Exception:
                    pass
                return obj

            if not isinstance(obj, dict):
                return obj

            every = max(1, int(getattr(cfg, "show_every_steps", 1)))
            if int(getattr(self, "global_step", 0)) % every != 0:
                return obj

            z_live = obj.get("z_obj")
            z_replay = obj.get("passport_inner_world_z")
            if not torch.is_tensor(z_live) or not torch.is_tensor(z_replay):
                return obj

            self._ensure_passport_debug_visualizer()

            # Live decoded object is already obj. For consistency, decode live z again
            # through the same head so both sides use the same path.
            live_extra = {k: v for k, v in obj.items() if k != "z_obj"}
            live_decoded = self.inner_object_system.decode_z(z_live, live_extra)

            replay_extra = dict(live_extra)
            replay_extra["z_obj"] = z_replay
            replay_decoded = self.inner_object_system.decode_z(z_replay, replay_extra)

            self.passport_debug_viz.draw(
                obj=obj,
                live_decoded=live_decoded,
                replay_decoded=replay_decoded,
                global_step=int(getattr(self, "global_step", 0)),
            )

            # Numeric metrics for status/logs.
            try:
                a = z_live.detach().float().reshape(1, -1)
                b = z_replay.detach().float().to(a.device).reshape(1, -1)
                d = min(a.shape[-1], b.shape[-1])
                a = a[:, :d]
                b = b[:, :d]
                dist = float((a - b).norm(dim=-1).mean().cpu().item())
                cos = float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().cpu().item())
                obj["passport_debug_z_distance"] = torch.tensor([[dist]], device=z_live.device, dtype=z_live.dtype)
                obj["passport_debug_z_cosine"] = torch.tensor([[cos]], device=z_live.device, dtype=z_live.dtype)
                obj["passport_debug_active"] = torch.tensor([[1.0]], device=z_live.device, dtype=z_live.dtype)
            except Exception:
                pass

            return obj

        except Exception as e:
            if not hasattr(self, "_passport_debug_warned"):
                print(f"[passport_debug] update failed: {e}")
                self._passport_debug_warned = True
            return obj
