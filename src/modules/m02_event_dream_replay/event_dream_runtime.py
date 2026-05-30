from __future__ import annotations

import torch

from src.modules.m02_event_dream_replay.event_dream_replay import (
    EventDreamReplay,
    EventDreamReplayConfig,
)


class EventDreamReplayRuntimeMixin:
    def ensure_event_dream_replay_ready(self) -> None:
        if hasattr(self, "event_dream_replay") and self.event_dream_replay is not None:
            return
        cfg_obj = getattr(self.cfg, "event_dream_replay", None)
        self.event_dream_replay = EventDreamReplay(EventDreamReplayConfig(
            enabled=bool(getattr(cfg_obj, "enabled", True)),
            replay_context_dim=int(getattr(cfg_obj, "replay_context_dim", getattr(self.cfg.self_core, "focus_context_dim", 256))),
            event_code_dim=int(getattr(cfg_obj, "event_code_dim", 8)),
            replay_threshold=float(getattr(cfg_obj, "replay_threshold", 0.35)),
            focus_blend=float(getattr(cfg_obj, "focus_blend", 0.15)),
            blend_replay_into_focus=bool(getattr(cfg_obj, "blend_replay_into_focus", True)),
            use_m13_context=bool(getattr(cfg_obj, "use_m13_context", True)),
            use_event_memory=bool(getattr(cfg_obj, "use_event_memory", True)),
            max_recent_events_scan=int(getattr(cfg_obj, "max_recent_events_scan", 16)),
        ))
        print("[event_dream_replay] initialized")

    def compute_event_dream_replay(self, obs: dict, out: dict):
        del obs
        cfg_obj = getattr(self.cfg, "event_dream_replay", None)
        if not bool(getattr(cfg_obj, "enabled", True)):
            return None
        self.ensure_event_dream_replay_ready()
        dream_mode = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
        packet = self.event_dream_replay.compute(
            out=out,
            event_memory=getattr(self, "event_latent_memory", None),
            dream_mode=dream_mode,
        )
        out["event_dream_replay"] = packet

        focus = out.get("focus_context")
        replay_context = packet.get("replay_context")
        should = packet.get("should_replay")
        cfg = self.event_dream_replay.cfg
        if (
            bool(cfg.blend_replay_into_focus)
            and torch.is_tensor(focus)
            and torch.is_tensor(replay_context)
            and torch.is_tensor(should)
        ):
            try:
                gate = float(should.detach().reshape(-1)[0].cpu().item())
                if gate > 0.5:
                    if replay_context.shape[-1] != focus.shape[-1]:
                        if replay_context.shape[-1] > focus.shape[-1]:
                            replay_context = replay_context[..., : focus.shape[-1]]
                        else:
                            pad = torch.zeros(*replay_context.shape[:-1], focus.shape[-1] - replay_context.shape[-1], device=focus.device, dtype=focus.dtype)
                            replay_context = torch.cat([replay_context.to(focus.device, focus.dtype), pad], dim=-1)
                    out["pre_event_dream_focus_context"] = focus
                    out["focus_context"] = focus + float(cfg.focus_blend) * replay_context.to(focus.device, focus.dtype)
                    out["focus_context_source"] = "m02_event_dream_replay"
                    packet["blended_into_focus"] = torch.tensor([1.0], device=focus.device)
                else:
                    packet["blended_into_focus"] = torch.tensor([0.0], device=focus.device)
            except Exception as e:
                if not hasattr(self, "_event_dream_blend_warned"):
                    print(f"[event_dream_replay] focus blend skipped: {e}")
                    self._event_dream_blend_warned = True
        return packet

    def maybe_print_event_dream_replay_trace(self, out: dict) -> None:
        cfg_obj = getattr(self.cfg, "event_dream_replay", None)
        every = int(getattr(cfg_obj, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        packet = out.get("event_dream_replay")
        if not isinstance(packet, dict):
            print(f"[event_dream_replay step={self.global_step}] no replay output")
            return

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[event_dream_replay step={self.global_step}] "
            f"source={packet.get('replay_source', '')} "
            f"kind={packet.get('selected_event_kind', '')} "
            f"gate={f(packet.get('replay_gate')):.0f} "
            f"salience={f(packet.get('event_salience')):.3f} "
            f"pressure={f(packet.get('dream_pressure')):.3f} "
            f"blend={f(packet.get('blended_into_focus')):.0f} | "
            f"{packet.get('selected_event_sentence', '')}"
        )


__all__ = ["EventDreamReplayRuntimeMixin"]
