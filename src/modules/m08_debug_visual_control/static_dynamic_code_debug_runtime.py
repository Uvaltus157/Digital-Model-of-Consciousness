from __future__ import annotations

import torch

from src.modules.m08_debug_visual_control.static_dynamic_code_visualizer import StaticDynamicCodeVisualizer, StaticDynamicCodeVisualizerConfig


class StaticDynamicCodeDebugRuntimeMixin:
    """
    Adds explicit names to the code ladder:

        z_static
            primary static sensory/proposal code

        z_dynamic
            dynamic object code from ObjectSlotMemory / DynamicObjectPassport

        scenario_z
            imagined temporal code

    It also opens a heatmap viewer for z_static and the following layers.
    """

    def _static_dynamic_code_viz_enabled(self) -> bool:
        cfg = getattr(self.cfg, "static_dynamic_code_visualizer", None)
        # Config means "feature exists"; runtime flag means "window is requested".
        # The PyQt pult toggles self.show_static_dynamic_code_window through IPC.
        if not bool(getattr(cfg, "enabled", True)):
            return False
        return bool(getattr(self, "show_static_dynamic_code_window", False))

    def _ensure_static_dynamic_code_visualizer(self) -> None:
        if hasattr(self, "static_dynamic_code_viz") and self.static_dynamic_code_viz is not None:
            return

        cfg = getattr(self.cfg, "static_dynamic_code_visualizer", None)
        self.static_dynamic_code_viz = StaticDynamicCodeVisualizer(StaticDynamicCodeVisualizerConfig(
            enabled=bool(getattr(cfg, "enabled", True)),
            window_name=str(getattr(cfg, "window_name", "static/dynamic code heatmaps")),
            width=int(getattr(cfg, "width", 1500)),
            height=int(getattr(cfg, "height", 900)),
            delay_ms=int(getattr(cfg, "delay_ms", 1)),
            show_every_steps=int(getattr(cfg, "show_every_steps", 1)),
            heatmap_height=int(getattr(cfg, "heatmap_height", 120)),
        ))

    def annotate_static_dynamic_codes(self, obj: dict, vision=None) -> dict:
        """
        Attach explicit debug fields to inner_object.

        `vision` is the proposal tensor from build_inner_object_vision_proposals():
            [B, P, D] or [B, D]

        We define:
            z_static = first proposal / scene summary
            z_dynamic = passport_inner_world_z if available, else z_obj
            scenario_z = scenario_z if available, else inner_mind_z
        """
        try:
            if not isinstance(obj, dict):
                return obj

            if torch.is_tensor(vision):
                v = vision.detach()
                if v.ndim == 3:
                    z_static = v[:, 0, :]
                elif v.ndim == 2:
                    z_static = v
                elif v.ndim == 1:
                    z_static = v.unsqueeze(0)
                else:
                    z_static = None

                if z_static is not None:
                    obj["z_static"] = z_static
                    obj["static_sensor_code"] = z_static
                    obj["z_static_dim"] = torch.tensor([[float(z_static.shape[-1])]], device=z_static.device, dtype=z_static.dtype)

            # Dynamic object code is first-order object identity code.
            if torch.is_tensor(obj.get("passport_inner_world_z")):
                obj["z_dynamic"] = obj["passport_inner_world_z"]
                obj["dynamic_object_code"] = obj["passport_inner_world_z"]
            elif torch.is_tensor(obj.get("z_obj")):
                obj["z_dynamic"] = obj["z_obj"]
                obj["dynamic_object_code"] = obj["z_obj"]

            # Scenario code is temporal/imagined code.
            if torch.is_tensor(obj.get("scenario_z")):
                obj["scenario_code"] = obj["scenario_z"]
            elif torch.is_tensor(obj.get("inner_mind_z")):
                obj["scenario_code"] = obj["inner_mind_z"]

            return obj
        except Exception as e:
            if not hasattr(self, "_static_dynamic_annotate_warned"):
                print(f"[static_dynamic_code] annotate failed: {e}")
                self._static_dynamic_annotate_warned = True
            return obj

    def update_static_dynamic_code_visualizer(self, obj: dict) -> dict:
        try:
            cfg = getattr(self.cfg, "static_dynamic_code_visualizer", None)
            if not self._static_dynamic_code_viz_enabled():
                try:
                    if hasattr(self, "static_dynamic_code_viz") and self.static_dynamic_code_viz is not None:
                        self.static_dynamic_code_viz.close()
                except Exception:
                    pass
                return obj

            if not isinstance(obj, dict):
                return obj

            every = max(1, int(getattr(cfg, "show_every_steps", 1)))
            if int(getattr(self, "global_step", 0)) % every != 0:
                return obj

            self._ensure_static_dynamic_code_visualizer()

            # Static/Dynamic only calls imshow().
            # HighGUI events are pumped once globally after all visualizers.
            self.static_dynamic_code_viz.draw(
                obj,
                global_step=int(getattr(self, "global_step", 0)),
                pump_events=False,
            )

            ref = obj.get("z_obj")
            device = ref.device if torch.is_tensor(ref) else getattr(self, "device", "cpu")
            dtype = ref.dtype if torch.is_tensor(ref) else torch.float32
            obj["static_dynamic_code_viz_active"] = torch.tensor([[1.0]], device=device, dtype=dtype)
            return obj

        except Exception as e:
            if not hasattr(self, "_static_dynamic_code_viz_warned"):
                print(f"[static_dynamic_code] visualizer failed: {e}")
                self._static_dynamic_code_viz_warned = True
            return obj
