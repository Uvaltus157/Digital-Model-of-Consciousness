from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class EventScenarioDecoderConfig:
    """
    Level 4 decoder: episode/sentence memory -> replayable latent scenario.

    This is the first decoder, but still deterministic:
        - it does not yet learn a neural EventDecoder;
        - it reads saved episode sentences and raw event z_before/z_after;
        - it produces a sequence of latent frames that can be sent to
          ObjectImaginationHead2D/Object3DHead later.

    Next step after this:
        trainable EventDecoder(sentence_code -> z trajectory).
    """
    enabled: bool = True
    max_replay_steps: int = 32
    loop: bool = True
    interpolate_steps: int = 3
    prefer_current_episode: bool = True
    min_events_to_decode: int = 1


def _clone_cpu_to_device(x: Any, device: torch.device | str, dtype=torch.float32):
    try:
        if torch.is_tensor(x):
            return x.detach().to(device=device, dtype=dtype)
    except Exception:
        pass
    return None


class EventScenarioDecoder:
    """
    Deterministic scenario decoder.

    Input:
        EventSentenceMemory episode with event entries.

    Output:
        scenario_state:
            sequence of z latent vectors,
            sentences,
            roles,
            cursor,
            current z.

    This is like reading a DNA-like event script and reconstructing the
    latent motion through memory.
    """

    def __init__(self, cfg: Optional[EventScenarioDecoderConfig] = None):
        self.cfg = cfg or EventScenarioDecoderConfig()
        self.cursor: int = 0
        self.last_episode_id: Optional[int] = None
        self.last_sequence_len: int = 0

    def _event_z_pair(self, ev: Dict[str, Any], device, dtype) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        z0 = _clone_cpu_to_device(ev.get("z_before"), device, dtype)
        z1 = _clone_cpu_to_device(ev.get("z_after"), device, dtype)
        if z0 is None:
            z0 = _clone_cpu_to_device(ev.get("z_obj"), device, dtype)
        if z1 is None:
            dz = _clone_cpu_to_device(ev.get("delta_z"), device, dtype)
            if z0 is not None and dz is not None:
                z1 = z0 + dz
        return z0, z1

    def _interpolate(self, z0: torch.Tensor, z1: torch.Tensor, steps: int) -> List[torch.Tensor]:
        if steps <= 1:
            return [z1]
        out: List[torch.Tensor] = []
        for i in range(steps):
            a = float(i + 1) / float(steps)
            out.append(((1.0 - a) * z0 + a * z1).detach())
        return out

    def build_sequence_from_episode(
        self,
        episode: Dict[str, Any],
        *,
        device,
        dtype=torch.float32,
    ) -> Dict[str, Any]:
        if not isinstance(episode, dict):
            return {}

        events = list(episode.get("sentences", []) or [])
        if len(events) < int(self.cfg.min_events_to_decode):
            return {}

        max_steps = max(1, int(self.cfg.max_replay_steps))
        interp_steps = max(1, int(self.cfg.interpolate_steps))

        z_seq: List[torch.Tensor] = []
        sentences: List[str] = []
        roles: List[Dict[str, Any]] = []
        event_refs: List[Dict[str, Any]] = []

        for ev in events:
            if len(z_seq) >= max_steps:
                break

            z0, z1 = self._event_z_pair(ev.get("raw_event", ev), device, dtype)
            # The EventSentenceMemory record may not keep raw_event. In that case
            # try the record itself.
            if z0 is None or z1 is None:
                z0b, z1b = self._event_z_pair(ev, device, dtype)
                z0 = z0 if z0 is not None else z0b
                z1 = z1 if z1 is not None else z1b

            if z0 is None and z1 is None:
                continue
            if z0 is None:
                z0 = z1
            if z1 is None:
                z1 = z0

            frames = self._interpolate(z0, z1, interp_steps)
            for z in frames:
                if len(z_seq) >= max_steps:
                    break
                if z.ndim == 1:
                    z = z.unsqueeze(0)
                z_seq.append(z.detach())
                sentences.append(str(ev.get("sentence", "") or ev.get("semantic_code_sentence", "")))
                r = ev.get("roles", {})
                roles.append(r if isinstance(r, dict) else {})
                event_refs.append(ev)

        if not z_seq:
            return {}

        z_stack = torch.cat([z if z.ndim == 2 else z.view(1, -1) for z in z_seq], dim=0)
        eid = int(episode.get("episode_id", 0) or 0)

        if self.last_episode_id != eid:
            self.cursor = 0
            self.last_episode_id = eid

        self.last_sequence_len = int(z_stack.shape[0])

        return {
            "episode_id": eid,
            "episode_summary": str(episode.get("summary", "")),
            "z_sequence": z_stack,
            "sentences": sentences,
            "roles": roles,
            "event_refs": event_refs,
            "sequence_len": int(z_stack.shape[0]),
        }

    def select_episode(self, sentence_memory: Any) -> Optional[Dict[str, Any]]:
        if sentence_memory is None:
            return None
        try:
            if bool(self.cfg.prefer_current_episode):
                ep = getattr(sentence_memory, "current_episode", None)
                if isinstance(ep, dict) and ep.get("sentences"):
                    return ep
            episodes = list(getattr(sentence_memory, "episodes", []) or [])
            if episodes:
                return episodes[-1]
        except Exception:
            pass
        return None

    def decode_next(
        self,
        sentence_memory: Any,
        *,
        device,
        dtype=torch.float32,
    ) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        ep = self.select_episode(sentence_memory)
        if ep is None:
            return {}

        seq = self.build_sequence_from_episode(ep, device=device, dtype=dtype)
        if not seq:
            return {}

        z_sequence = seq["z_sequence"]
        n = int(z_sequence.shape[0])
        if n <= 0:
            return {}

        idx = int(self.cursor)
        idx = max(0, min(idx, n - 1))
        z = z_sequence[idx:idx + 1].detach()

        if bool(self.cfg.loop):
            self.cursor = (idx + 1) % n
        else:
            self.cursor = min(idx + 1, n - 1)

        sentence = ""
        try:
            sentence = str(seq.get("sentences", [""])[idx])
        except Exception:
            sentence = ""

        roles = {}
        try:
            roles = seq.get("roles", [{}])[idx]
            if not isinstance(roles, dict):
                roles = {}
        except Exception:
            roles = {}

        return {
            "scenario_active": True,
            "scenario_episode_id": int(seq.get("episode_id", 0)),
            "scenario_cursor": int(idx),
            "scenario_sequence_len": int(n),
            "scenario_z": z,
            "scenario_sentence": sentence,
            "scenario_roles": roles,
            "scenario_summary": str(seq.get("episode_summary", "")),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "cursor": int(self.cursor),
            "last_episode_id": self.last_episode_id,
            "last_sequence_len": int(self.last_sequence_len),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        cfg = state.get("cfg", {})
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if hasattr(self.cfg, k):
                    try:
                        setattr(self.cfg, k, v)
                    except Exception:
                        pass
        self.cursor = int(state.get("cursor", 0) or 0)
        self.last_episode_id = state.get("last_episode_id", None)
        self.last_sequence_len = int(state.get("last_sequence_len", 0) or 0)
