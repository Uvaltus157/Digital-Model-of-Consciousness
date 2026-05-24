from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EventSentenceMemoryConfig:
    """
    Level 3 memory: store chains of composed event sentences.

    Level 1:
        latent transition -> tokens

    Level 2:
        tokens -> roles -> sentence

    Level 3:
        sentence stream -> episodes / scenarios / narrative memory
    """
    enabled: bool = True
    max_sentences: int = 512
    max_episodes: int = 64
    episode_gap_steps: int = 25
    new_episode_on_slot_change: bool = False
    keep_raw_event: bool = False


class EventSentenceMemory:
    """
    Stores semantic/code sentences as an ordered memory.

    It is deterministic runtime memory for now:
        - no optimizer,
        - no loss,
        - saved inside event_latent_memory checkpoint state.

    The main product is:
        sentence_memory.latest_episode_summary()
    """

    def __init__(self, cfg: Optional[EventSentenceMemoryConfig] = None):
        self.cfg = cfg or EventSentenceMemoryConfig()
        self.sentences = deque(maxlen=max(1, int(self.cfg.max_sentences)))
        self.episodes = deque(maxlen=max(1, int(self.cfg.max_episodes)))
        self.current_episode: Optional[Dict[str, Any]] = None
        self.last_step: Optional[int] = None
        self.last_slot: Optional[int] = None
        self.episode_counter: int = 0

    def __len__(self) -> int:
        return len(self.sentences)

    def _start_episode(self, step: int, slot: int, reason: str) -> Dict[str, Any]:
        if self.current_episode is not None:
            self._finish_current_episode()

        self.episode_counter += 1
        self.current_episode = {
            "episode_id": int(self.episode_counter),
            "start_step": int(step),
            "end_step": int(step),
            "reason": str(reason),
            "slots": [],
            "tokens": [],
            "verbs": {},
            "contexts": {},
            "sentences": [],
            "summary": "",
        }
        return self.current_episode

    def _finish_current_episode(self) -> None:
        if self.current_episode is None:
            return
        ep = dict(self.current_episode)
        ep["summary"] = self._summarize_episode(ep)
        self.episodes.append(ep)

    def _summarize_episode(self, ep: Dict[str, Any]) -> str:
        eid = int(ep.get("episode_id", 0))
        start = int(ep.get("start_step", 0))
        end = int(ep.get("end_step", start))
        slots = ep.get("slots", [])
        tokens = ep.get("tokens", [])
        verbs = ep.get("verbs", {})
        contexts = ep.get("contexts", {})
        n = len(ep.get("sentences", []))

        slot_txt = ",".join([f"SLOT_{s}" for s in slots[:6]]) if slots else "none"
        token_txt = ",".join(tokens[:6]) if tokens else "none"
        verb_txt = ",".join([f"{k}:{v}" for k, v in list(verbs.items())[:5]]) if verbs else "none"
        ctx_txt = ",".join([f"{k}:{v}" for k, v in list(contexts.items())[:5]]) if contexts else "none"

        return (
            f"EP_{eid:04d} steps={start}-{end} n={n} "
            f"slots=[{slot_txt}] tokens=[{token_txt}] verbs=[{verb_txt}] contexts=[{ctx_txt}]"
        )

    def _should_start_new_episode(self, step: int, slot: int) -> tuple[bool, str]:
        if self.current_episode is None:
            return True, "first_sentence"
        if self.last_step is None:
            return False, ""
        gap = int(step) - int(self.last_step)
        if gap > int(self.cfg.episode_gap_steps):
            return True, f"gap>{int(self.cfg.episode_gap_steps)}"
        if bool(self.cfg.new_episode_on_slot_change) and self.last_slot is not None and int(slot) != int(self.last_slot):
            return True, "slot_change"
        return False, ""

    def add(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {}

        step = int(event.get("step", 0) or 0)
        slot = int(event.get("slot", 0) or 0)
        token = str(event.get("slot_token", "") or f"SLOT_{slot}")
        sentence = str(
            event.get("semantic_sentence", "")
            or event.get("semantic_code_sentence", "")
            or event.get("sentence", "")
            or ""
        )
        roles = event.get("sentence_roles", {})
        if not isinstance(roles, dict):
            roles = {}

        new_ep, reason = self._should_start_new_episode(step, slot)
        if new_ep:
            self._start_episode(step, slot, reason)

        assert self.current_episode is not None

        verb = str(roles.get("verb", event.get("kind", "unknown")) or "unknown")
        context = str(roles.get("context", "unknown") or "unknown")

        record = {
            "step": step,
            "slot": slot,
            "slot_token": token,
            "sentence": sentence,
            "semantic_code_sentence": str(event.get("semantic_code_sentence", "")),
            "roles": roles,
            "kind": str(event.get("kind", "")),
            "episode_id": int(self.current_episode.get("episode_id", 0)),
        }
        if bool(getattr(self.cfg, "keep_raw_event", False)):
            record["raw_event"] = event

        self.sentences.append(record)

        ep = self.current_episode
        ep["end_step"] = step
        if slot not in ep["slots"]:
            ep["slots"].append(slot)
        if token and token not in ep["tokens"]:
            ep["tokens"].append(token)
        verbs = dict(ep.get("verbs", {}))
        verbs[verb] = int(verbs.get(verb, 0)) + 1
        ep["verbs"] = verbs
        contexts = dict(ep.get("contexts", {}))
        contexts[context] = int(contexts.get(context, 0)) + 1
        ep["contexts"] = contexts
        ep["sentences"].append(record)
        ep["summary"] = self._summarize_episode(ep)

        self.last_step = step
        self.last_slot = slot

        return {
            "sentence_memory_size": len(self.sentences),
            "episode_id": int(ep.get("episode_id", 0)),
            "episode_size": len(ep.get("sentences", [])),
            "episode_summary": ep.get("summary", ""),
            "latest_sentence": sentence,
        }

    def latest_sentences(self, n: int = 8) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return list(self.sentences)[-int(n):]

    def latest_episode_summary(self) -> str:
        if self.current_episode is not None:
            return str(self.current_episode.get("summary", ""))
        if self.episodes:
            return str(self.episodes[-1].get("summary", ""))
        return ""

    def get_state_view(self, max_sentences: int = 12, max_episodes: int = 8) -> Dict[str, Any]:
        episodes = list(self.episodes)
        if self.current_episode is not None:
            episodes = episodes + [self.current_episode]
        return {
            "sentence_count": len(self.sentences),
            "episode_count": len(episodes),
            "latest_sentences": self.latest_sentences(max_sentences),
            "episodes": episodes[-int(max_episodes):],
            "current_episode": self.current_episode,
            "latest_episode_summary": self.latest_episode_summary(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cfg": dict(self.cfg.__dict__),
            "sentences": list(self.sentences),
            "episodes": list(self.episodes),
            "current_episode": self.current_episode,
            "last_step": self.last_step,
            "last_slot": self.last_slot,
            "episode_counter": self.episode_counter,
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

        self.sentences = deque(state.get("sentences", []), maxlen=max(1, int(self.cfg.max_sentences)))
        self.episodes = deque(state.get("episodes", []), maxlen=max(1, int(self.cfg.max_episodes)))
        self.current_episode = state.get("current_episode", None)
        self.last_step = state.get("last_step", None)
        self.last_slot = state.get("last_slot", None)
        self.episode_counter = int(state.get("episode_counter", len(self.episodes)))
