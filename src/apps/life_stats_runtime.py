from __future__ import annotations

import numpy as np
import torch

from src.apps.life_stats_builders import (
    build_base_life_stats,
    build_conscious_module_life_stats,
    build_object_memory_life_stats,
    build_runtime_control_life_stats,
)


class LifeStatsRuntimeMixin:
    def _life_tensor_float(self, value, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().float().reshape(-1)[0].cpu().item())
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _life_tensor_bool(self, value, default: bool = False) -> bool:
        return bool(self._life_tensor_float(value, 1.0 if default else 0.0) > 0.5)

    def maybe_update_inner_world(
        self,
        obs: dict,
        out: dict,
        decoded_report: str,
        target_report: str,
        inner_report_confidence: float,
    ) -> None:
        out["decoded_report"] = decoded_report
        out["target_report"] = target_report
        self.update_inner_world_window(out)

    def maybe_update_camera_preview(self, obs: dict, out: dict, *, show_window: bool = True) -> None:
        if not show_window:
            return
        self.update_camera_preview_window(obs)

    def maybe_update_action_outputs(self, obs: dict, out: dict) -> None:
        self.update_action_output_window()

    def maybe_update_inner_object_visualizer(self, obs: dict, out: dict) -> None:
        # This method also owns the event-code visualizer, because both need the
        # current inner-object packet and share the same lazy compute path.
        self.update_inner_object_window(obs, out)

    def maybe_update_inner_object_open3d(self, obs: dict, out: dict) -> None:
        self.update_inner_object_open3d_window(out)

    def maybe_update_latent_semantic_map(self, obs: dict, out: dict) -> None:
        self.update_latent_semantic_window(out)

    def maybe_update_static_dynamic_code_visualizer(self, obs: dict, out: dict) -> None:
        obj = out.get("inner_object")
        if isinstance(obj, dict):
            out["inner_object"] = self.update_static_dynamic_code_visualizer(obj)

    def maybe_update_event_code_visualizer(self, obs: dict, out: dict) -> None:
        # Event-code drawing is handled in maybe_update_inner_object_visualizer()
        # to avoid rendering the same frame twice in one life step.
        return

    def build_latest_life_stats(
        self,
        *,
        obs: dict,
        out: dict,
        emotion: dict,
        novelty_score: float,
        chosen_action: int,
        decoded_report: str,
        target_report: str,
        inner_report_confidence: float,
        self_confidence: float,
    ) -> dict:
        stats: dict = {}
        stats.update(build_base_life_stats(
            self,
            obs=obs,
            out=out,
            emotion=emotion,
            novelty_score=novelty_score,
            chosen_action=chosen_action,
            decoded_report=decoded_report,
            target_report=target_report,
            inner_report_confidence=inner_report_confidence,
            self_confidence=self_confidence,
        ))
        stats.update(build_object_memory_life_stats(self, obs=obs, out=out))
        stats.update(build_runtime_control_life_stats(self, obs=obs, out=out))
        stats.update(build_conscious_module_life_stats(self, out=out))

        inner_diag = getattr(self, "_latest_inner_speech_diagnostics", {})
        if isinstance(inner_diag, dict):
            stats.update(inner_diag)
        return stats

    def finalize_life_step_side_effects(self, *, obs: dict, out: dict, emotion: dict, decoded_report: str, target_report: str, inner_report_confidence: float) -> None:
        self.write_module_debug_status()
        self.poll_ipc_control_messages()

        calls = [
            ("maybe_print_action_signal_trace", (out,), {}),
            ("maybe_print_emotional_drive_trace", (emotion,), {}),
            ("maybe_print_long_dynamic_memory_trace", (out,), {}),
            ("maybe_print_event_dream_replay_trace", (out,), {}),
            ("maybe_print_thought_chain_trace", (out,), {}),
            ("maybe_print_inner_speech_trace", (out,), {}),
            ("maybe_print_global_broadcast_trace", (out,), {}),
            ("maybe_print_metacognition_trace", (out,), {}),
            ("maybe_print_autobiographical_memory_trace", (out,), {}),
            ("maybe_print_semantic_action_trace", (out,), {}),
            ("maybe_print_inner_object_trace", (out,), {}),
            ("maybe_update_inner_world", (obs, out, decoded_report, target_report, inner_report_confidence), {}),
            ("maybe_update_camera_preview", (obs, out), {"show_window": self.show_camera_preview_window}),
            ("maybe_update_action_outputs", (obs, out), {}),
            ("maybe_update_inner_object_visualizer", (obs, out), {}),
            ("maybe_update_inner_object_open3d", (obs, out), {}),
            ("maybe_update_latent_semantic_map", (obs, out), {}),
            ("maybe_update_static_dynamic_code_visualizer", (obs, out), {}),
            ("maybe_update_event_code_visualizer", (obs, out), {}),
        ]
        for name, args, kwargs in calls:
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    warn = f"_{name}_warned"
                    if not hasattr(self, warn):
                        print(f"[{name}] skipped: {e}")
                        setattr(self, warn, True)

        if self.global_step % self.cfg.life.report_every_steps == 0:
            serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.latest_stats.items()}
            self.log_event({"step": self.global_step, "event": "life_tick", **serial})

        if hasattr(self, "log_tetra_life_tick_diagnostics"):
            self.log_tetra_life_tick_diagnostics()

        try:
            self.maybe_save_checkpoint(force=False, owner="life")
        except AttributeError:
            pass
        except Exception as e:
            if self.global_step % max(1, self.cfg.life.report_every_steps) == 0:
                print(f"[checkpoint] periodic save failed: {e}")

        train_once_if_ready = getattr(self, "train_once_if_ready", None)
        if callable(train_once_if_ready):
            train_once_if_ready()


__all__ = ["LifeStatsRuntimeMixin"]
