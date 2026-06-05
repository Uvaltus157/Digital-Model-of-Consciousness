from __future__ import annotations

import torch

from src.modules.m13_autobiographical_memory.autobiographical_memory import (
    AutobiographicalMemory,
    AutobiographicalMemoryConfig,
)


class AutobiographicalMemoryRuntimeMixin:
    def ensure_autobiographical_memory_ready(self) -> None:
        if hasattr(self, "autobiographical_memory") and self.autobiographical_memory is not None:
            return
        cfg_obj = getattr(self.cfg, "autobiographical_memory", None)
        cfg = AutobiographicalMemoryConfig(
            enabled=bool(getattr(cfg_obj, "enabled", True)),
            memory_dim=int(getattr(cfg_obj, "memory_dim", getattr(self.cfg.self_core, "focus_context_dim", 256))),
            max_episodes=int(getattr(cfg_obj, "max_episodes", 512)),
            retrieval_topk=int(getattr(cfg_obj, "retrieval_topk", 3)),
            write_every_steps=int(getattr(cfg_obj, "write_every_steps", 1)),
            blend_retrieved_into_focus=bool(getattr(cfg_obj, "blend_retrieved_into_focus", False)),
            focus_blend=float(getattr(cfg_obj, "focus_blend", 0.20)),
            min_relevance_for_blend=float(getattr(cfg_obj, "min_relevance_for_blend", 0.05)),
        )
        self.autobiographical_memory = AutobiographicalMemory(cfg)
        print("[autobiographical_memory] initialized")

    def compute_autobiographical_retrieval(self, obs: dict, out: dict):
        del obs
        cfg_obj = getattr(self.cfg, "autobiographical_memory", None)
        if not bool(getattr(cfg_obj, "enabled", True)):
            return None
        self.ensure_autobiographical_memory_ready()
        memory = self.autobiographical_memory.retrieve(out)
        memory["stage"] = "pre_self_retrieval"
        out["autobiographical_memory"] = memory

        focus = out.get("focus_context")
        retrieved = memory.get("retrieved_context")
        relevance = memory.get("retrieval_relevance")
        cfg = self.autobiographical_memory.cfg
        if (
            bool(cfg.blend_retrieved_into_focus)
            and torch.is_tensor(focus)
            and torch.is_tensor(retrieved)
            and torch.is_tensor(relevance)
        ):
            try:
                rel = float(relevance.detach().reshape(-1)[0].cpu().item())
                if rel >= float(cfg.min_relevance_for_blend):
                    if retrieved.shape[-1] != focus.shape[-1]:
                        if retrieved.shape[-1] > focus.shape[-1]:
                            retrieved = retrieved[..., : focus.shape[-1]]
                        else:
                            pad = torch.zeros(*retrieved.shape[:-1], focus.shape[-1] - retrieved.shape[-1], device=focus.device, dtype=focus.dtype)
                            retrieved = torch.cat([retrieved.to(focus.device, focus.dtype), pad], dim=-1)
                    out["pre_autobiographical_focus_context"] = focus
                    out["focus_context"] = focus + float(cfg.focus_blend) * retrieved.to(focus.device, focus.dtype)
                    out["focus_context_source"] = "m13_autobiographical_retrieval"
                    memory["blended_into_focus"] = torch.tensor([1.0], device=focus.device)
                else:
                    memory["blended_into_focus"] = torch.tensor([0.0], device=focus.device)
            except Exception as e:
                if not hasattr(self, "_autobiographical_blend_warned"):
                    print(f"[autobiographical_memory] focus blend skipped: {e}")
                    self._autobiographical_blend_warned = True
        return memory

    def write_autobiographical_episode(self, obs: dict, out: dict):
        cfg_obj = getattr(self.cfg, "autobiographical_memory", None)
        if not bool(getattr(cfg_obj, "enabled", True)):
            return None
        self.ensure_autobiographical_memory_ready()
        every = max(1, int(getattr(cfg_obj, "write_every_steps", getattr(self.autobiographical_memory.cfg, "write_every_steps", 1))))
        if int(getattr(self, "global_step", 0)) % every != 0:
            return None
        write = self.autobiographical_memory.write_episode(obs=obs, out=out, global_step=int(getattr(self, "global_step", 0)))
        memory = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}
        memory.update(write)
        memory["stage"] = "post_step_written"
        out["autobiographical_memory"] = memory
        return write

    def maybe_print_autobiographical_memory_trace(self, out: dict) -> None:
        cfg_obj = getattr(self.cfg, "autobiographical_memory", None)
        every = int(getattr(cfg_obj, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        memory = out.get("autobiographical_memory")
        if not isinstance(memory, dict):
            print(f"[autobiographical_memory step={self.global_step}] no memory output")
            return

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[autobiographical_memory step={self.global_step}] "
            f"episodes={f(memory.get('episode_count', memory.get('retrieved_episode_count'))):.0f} "
            f"relevance={f(memory.get('retrieval_relevance')):.3f} "
            f"blend={f(memory.get('blended_into_focus')):.0f} "
            f"summary={memory.get('summary', memory.get('last_summary', ''))}"
        )


__all__ = ["AutobiographicalMemoryRuntimeMixin"]
