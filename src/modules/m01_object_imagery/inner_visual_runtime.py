from __future__ import annotations

import cv2
from src.platform.gui.opencv_gui_thread import close_cv2_window
import numpy as np
import torch


class InnerVisualRuntimeMixin:

    def _close_inner_world_gui_windows(self) -> None:
        try:
            close_cv2_window("dreamer inner world v3")
            close_cv2_window("dreamer inner world")
        except Exception:
            pass

    def _close_latent_semantic_gui_window(self) -> None:
        try:
            if hasattr(self, "latent_semantic_viz") and self.latent_semantic_viz is not None:
                self.latent_semantic_viz.close()
        except Exception:
            pass
        try:
            close_cv2_window(self.cfg.latent_semantic_map.window_name)
        except Exception:
            pass

    def update_inner_world_window(self, out):
        # exactly the same flow as V5.7, but visualizer now renders object_imagery too
        if self.inner_viz is None or not self.show_inner_world_window:
            if not bool(getattr(self, "show_inner_world_window", False)):
                self._close_inner_world_gui_windows()
            return
        if self.global_step % max(1, self.cfg.inner_world.show_every_steps) != 0:
            return

        symbolic = out.get("symbolic_report")
        key = self.inner_viz.show(out, symbolic, predicted_text=out.get('decoded_report'), target_text=out.get('target_report'), delay_ms=1)
        if key in (27, ord("q")):
            self.shutdown = True

        if self.cfg.inner_world.save_frames and self.global_step % max(1, self.cfg.inner_world.save_every_steps) == 0:
            from pathlib import Path
            path = Path(self.cfg.inner_world.out_dir) / f"inner_world_{self.global_step:07d}.png"
            self.inner_viz.save(str(path), out, symbolic, predicted_text=out.get('decoded_report'), target_text=out.get('target_report'))


    def update_latent_semantic_window(self, out):
        cfg = self.cfg.latent_semantic_map
        if not cfg.enabled:
            return
        if not self.show_latent_semantic_window:
            self._close_latent_semantic_gui_window()
            return
        if self.global_step % max(1, cfg.show_every_steps) != 0:
            return
        key = self.latent_semantic_viz.show(out, delay_ms=cfg.delay_ms)
        if key in (27, ord("q")):
            self.shutdown = True
        if self.cfg.inner_world.save_frames and self.global_step % max(1, self.cfg.inner_world.save_every_steps) == 0:
            from pathlib import Path
            path = Path(self.cfg.inner_world.out_dir) / f"latent_semantic_{self.global_step:07d}.png"
            self.latent_semantic_viz.save(str(path))

