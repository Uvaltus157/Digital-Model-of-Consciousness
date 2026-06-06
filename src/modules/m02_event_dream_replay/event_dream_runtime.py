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
        self_core_cfg = getattr(self.cfg, "self_core", None)
        default_context_dim = getattr(self_core_cfg, "focus_context_dim", 256)
        self.event_dream_replay = EventDreamReplay(EventDreamReplayConfig(
            enabled=bool(getattr(cfg_obj, "enabled", True)),
            replay_context_dim=int(getattr(cfg_obj, "replay_context_dim", default_context_dim)),
            event_code_dim=int(getattr(cfg_obj, "event_code_dim", 8)),
            replay_threshold=float(getattr(cfg_obj, "replay_threshold", 0.35)),
            focus_blend=float(getattr(cfg_obj, "focus_blend", 0.15)),
            blend_replay_into_focus=bool(getattr(cfg_obj, "blend_replay_into_focus", False)),
            use_m13_context=bool(getattr(cfg_obj, "use_m13_context", True)),
            use_m4_context=bool(getattr(cfg_obj, "use_m4_context", True)),
            m4_context_weight=float(getattr(cfg_obj, "m4_context_weight", 0.20)),
            use_event_memory=bool(getattr(cfg_obj, "use_event_memory", True)),
            max_recent_events_scan=int(getattr(cfg_obj, "max_recent_events_scan", 16)),
            seed_to_m5_boundary=bool(getattr(cfg_obj, "seed_to_m5_boundary", True)),
            seed_gate_gain=float(getattr(cfg_obj, "seed_gate_gain", 1.0)),
            apply_stage=str(getattr(cfg_obj, "apply_stage", "pre_observe")),
            seed_only_in_sleep=bool(getattr(cfg_obj, "seed_only_in_sleep", True)),
        ))
        if not hasattr(self, "_event_dream_next_focus_seed"):
            self._event_dream_next_focus_seed = None
            self._event_dream_next_focus_gate = None
            self._event_dream_seed_step = -1
        print("[event_dream_replay] initialized")

    def _event_dream_stage_allowed(self, stage: str) -> bool:
        cfg = getattr(getattr(self, "cfg", None), "event_dream_replay", None)
        apply_stage = str(getattr(cfg, "apply_stage", "pre_observe"))
        if apply_stage not in ("both", "pre_observe", "main"):
            apply_stage = "pre_observe"
        return apply_stage == "both" or str(stage) == apply_stage

    def get_event_dream_focus_seed(self, stage: str = "model_step"):
        cfg = getattr(getattr(self, "cfg", None), "event_dream_replay", None)
        if not bool(getattr(cfg, "enabled", True)):
            return None, None
        if bool(getattr(cfg, "seed_only_in_sleep", True)):
            is_sleep = bool(self.is_full_sleep_mode()) if hasattr(self, "is_full_sleep_mode") else False
            if not is_sleep:
                return None, None
        if not self._event_dream_stage_allowed(stage):
            return None, None
        seed = getattr(self, "_event_dream_next_focus_seed", None)
        gate = getattr(self, "_event_dream_next_focus_gate", None)
        if not torch.is_tensor(seed):
            return None, None
        return seed.detach(), gate.detach() if torch.is_tensor(gate) else gate

    def get_m5_focus_seed(self, stage: str = "model_step"):
        """Common M5 FocusFeedbackBoundary seed bus.

        Priority:
          1. M5 latent prototype simulator seed when active.
          2. M2 dream/replay seed in sleep mode.
          3. M15 conscious loop seed when available.
        Both enter M5 through the same focus_context_seed/gate inputs.
        """
        if hasattr(self, "get_m5_latent_prototype_focus_seed"):
            seed, gate = self.get_m5_latent_prototype_focus_seed(stage=stage)
            if torch.is_tensor(seed):
                return seed, gate
        seed, gate = self.get_event_dream_focus_seed(stage=stage)
        if torch.is_tensor(seed):
            return seed, gate
        if hasattr(self, "get_conscious_loop_focus_seed"):
            return self.get_conscious_loop_focus_seed(stage=stage)
        return None, None

    def _store_event_dream_m5_seed(self, packet: dict) -> None:
        self._event_dream_next_focus_seed = None
        self._event_dream_next_focus_gate = None
        cfg = self.event_dream_replay.cfg
        if not bool(getattr(cfg, "seed_to_m5_boundary", True)):
            return
        replay_context = packet.get("replay_context")
        replay_gate = packet.get("replay_gate", packet.get("should_replay"))
        dream_pressure = packet.get("dream_pressure")
        if not torch.is_tensor(replay_context):
            return
        if torch.is_tensor(replay_gate):
            gate = replay_gate.detach().float()
        else:
            gate = torch.tensor([[float(replay_gate or 0.0)]], dtype=torch.float32, device=replay_context.device)
        if torch.is_tensor(dream_pressure):
            gate = gate * dream_pressure.detach().float().to(gate.device)
        gate = gate * float(getattr(cfg, "seed_gate_gain", 1.0))
        if gate.ndim == 0:
            gate = gate.reshape(1, 1)
        elif gate.ndim == 1:
            gate = gate.reshape(-1, 1)
        if float(gate.detach().reshape(-1)[0].cpu().item()) <= 0.0:
            return
        self._event_dream_next_focus_seed = replay_context.detach()
        self._event_dream_next_focus_gate = gate.detach()
        self._event_dream_seed_step = int(getattr(self, "global_step", -1))
        packet["next_focus_context_seed"] = self._event_dream_next_focus_seed
        packet["next_focus_context_seed_gate"] = self._event_dream_next_focus_gate
        packet["target_m5_boundary"] = "FocusFeedbackBoundary(workspace_seed + preconscious_seed)"
        packet["seed_source"] = "m02_event_dream_replay"

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
        self._store_event_dream_m5_seed(packet)

        focus = out.get("focus_context")
        replay_context = packet.get("replay_context")
        should = packet.get("should_replay")
        cfg = self.event_dream_replay.cfg
        if (
            bool(cfg.blend_replay_into_focus)
            and not bool(getattr(cfg, "seed_to_m5_boundary", True))
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
