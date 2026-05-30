from __future__ import annotations

"""
M13 Autobiographical Memory.

Architecture role:
    M13 stores and retrieves self-relevant episodes: what was in focus, what the
    agent felt, what chain/action was chosen, whether it doubted or held action,
    and whether the outcome looked good/bad. It provides a retrieved context back
    to the pre-self branch before M15/M10/M9.

This first implementation is deliberately lightweight and runtime-local. It uses
cosine retrieval over episode vectors and can later be replaced by a trainable
long-term memory backend without changing the out["autobiographical_memory"]
contract.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def pad_or_trim_episode(x: Optional[torch.Tensor], dim: int, *, device=None, dtype=None, batch_size: int = 1) -> torch.Tensor:
    if x is None:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(batch_size, dim, device=device, dtype=dtype)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device, dtype=dtype or torch.float32)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    x = x.float()
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _scalar(value, default: float = 0.0) -> float:
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


def _first_tensor(*values) -> Optional[torch.Tensor]:
    for value in values:
        if torch.is_tensor(value):
            return value
    return None


@dataclass
class AutobiographicalMemoryConfig:
    enabled: bool = True
    memory_dim: int = 256
    max_episodes: int = 512
    retrieval_topk: int = 3
    write_every_steps: int = 1
    blend_retrieved_into_focus: bool = True
    focus_blend: float = 0.20
    min_relevance_for_blend: float = 0.05


class AutobiographicalMemory:
    def __init__(self, cfg: Optional[AutobiographicalMemoryConfig] = None) -> None:
        self.cfg = cfg or AutobiographicalMemoryConfig()
        self.episodes: List[Dict] = []
        self.write_count: int = 0

    def _device_from(self, out: Dict) -> torch.device:
        tensor = _first_tensor(
            out.get("focus_context"),
            out.get("workspace_out"),
            out.get("object_repr"),
        )
        if tensor is not None:
            return tensor.device
        for value in out.values():
            if torch.is_tensor(value):
                return value.device
            if isinstance(value, dict):
                for nested in value.values():
                    if torch.is_tensor(nested):
                        return nested.device
        return torch.device("cpu")

    def encode_query(self, out: Dict) -> torch.Tensor:
        device = self._device_from(out)
        c = self.cfg
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        thought_chain = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        broadcast = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}

        parts = [
            pad_or_trim_episode(out.get("focus_context"), c.memory_dim, device=device),
            pad_or_trim_episode(out.get("workspace_out"), c.memory_dim, device=device),
            pad_or_trim_episode(out.get("object_repr"), c.memory_dim, device=device),
            pad_or_trim_episode(affect.get("affect_latents"), c.memory_dim, device=device),
            pad_or_trim_episode(self_core.get("self_state"), c.memory_dim, device=device),
            pad_or_trim_episode(thought_chain.get("plan_context"), c.memory_dim, device=device),
            pad_or_trim_episode(broadcast.get("broadcast_latent"), c.memory_dim, device=device),
        ]
        # Stable non-trainable episode sketch: average normalized available views.
        stacked = torch.stack(parts, dim=0)
        query = stacked.mean(dim=0)
        return pad_or_trim_episode(query, c.memory_dim, device=device)

    def retrieve(self, out: Dict) -> Dict[str, torch.Tensor | int | str]:
        device = self._device_from(out)
        c = self.cfg
        query = self.encode_query(out)
        if not self.episodes:
            return {
                "retrieved_context": torch.zeros_like(query),
                "retrieval_relevance": torch.zeros(query.shape[0], 1, device=device),
                "retrieved_episode_count": torch.tensor([0.0], device=device),
                "retrieved_indices": torch.zeros(1, 0, dtype=torch.long, device=device),
                "summary": "no autobiographical episodes yet",
            }

        bank = torch.stack([ep["vector"].to(device) for ep in self.episodes], dim=0).reshape(len(self.episodes), -1)
        q = query.reshape(query.shape[0], -1)
        sims = F.cosine_similarity(q[:, None, :], bank[None, :, :], dim=-1)
        k = max(1, min(int(c.retrieval_topk), bank.shape[0]))
        vals, idx = torch.topk(sims, k=k, dim=-1)
        weights = torch.softmax(vals, dim=-1)
        selected = bank[idx.reshape(-1)].reshape(q.shape[0], k, -1)
        retrieved = torch.sum(weights.unsqueeze(-1) * selected, dim=1)
        relevance = vals.mean(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        best = self.episodes[int(idx.reshape(-1)[0].detach().cpu().item())]
        return {
            "retrieved_context": retrieved,
            "retrieval_relevance": relevance,
            "retrieved_episode_count": torch.tensor([float(len(self.episodes))], device=device),
            "retrieved_indices": idx.long(),
            "summary": str(best.get("summary", "")),
        }

    def write_episode(self, *, obs: Dict, out: Dict, global_step: int = 0) -> Dict[str, torch.Tensor | str | int]:
        del obs
        device = self._device_from(out)
        vector = self.encode_query(out).detach().reshape(-1).to(device)
        if vector.norm().detach().cpu().item() <= 1e-8:
            vector = torch.zeros(int(self.cfg.memory_dim), dtype=torch.float32, device=device)

        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        emotion = out.get("emotion", {}) if isinstance(out.get("emotion"), dict) else {}
        metacog = out.get("metacognition", {}) if isinstance(out.get("metacognition"), dict) else {}
        action = out.get("conscious_action", {}) if isinstance(out.get("conscious_action"), dict) else {}
        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        bc = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}

        summary = (
            f"step={int(global_step)} "
            f"source={bc.get('selected_source', '')} "
            f"valence={_scalar(emotion.get('emotional_valence'), 0.0):.2f} "
            f"panic={_scalar(affect.get('panic_latent'), 0.0):.2f} "
            f"doubt={_scalar(metacog.get('doubt'), 0.0):.2f} "
            f"action={action.get('reason', '')} "
            f"chain={_scalar(tc.get('best_chain_score'), 0.0):.2f}"
        )
        episode = {
            "vector": vector.detach().cpu(),
            "step": int(global_step),
            "summary": summary,
            "valence": _scalar(emotion.get("emotional_valence"), 0.0),
            "arousal": _scalar(emotion.get("emotional_arousal"), 0.0),
            "panic": _scalar(affect.get("panic_latent"), 0.0),
            "doubt": _scalar(metacog.get("doubt"), 0.0),
            "action_reason": str(action.get("reason", "")),
            "selected_source": str(bc.get("selected_source", "")),
        }
        self.episodes.append(episode)
        if len(self.episodes) > int(self.cfg.max_episodes):
            self.episodes = self.episodes[-int(self.cfg.max_episodes):]
        self.write_count += 1
        return {
            "episode_written": torch.tensor([1.0], device=device),
            "episode_count": torch.tensor([float(len(self.episodes))], device=device),
            "write_count": torch.tensor([float(self.write_count)], device=device),
            "last_summary": summary,
        }


__all__ = [
    "AutobiographicalMemory",
    "AutobiographicalMemoryConfig",
    "pad_or_trim_episode",
]
