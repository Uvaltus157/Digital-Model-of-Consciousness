from __future__ import annotations

"""Lightweight memory container for M13 Autobiographical Memory."""

from dataclasses import dataclass, field
from typing import List, Optional

from .state import SelfEpisode


@dataclass
class PersonalTimeline:
    """Append-only timeline of self episodes.

    This is a small in-memory scaffold. Persistent storage can be added later
    without changing the public episode state types.
    """

    episodes: List[SelfEpisode] = field(default_factory=list)

    def append(self, episode: SelfEpisode) -> None:
        self.episodes.append(episode)

    def latest(self) -> Optional[SelfEpisode]:
        return self.episodes[-1] if self.episodes else None

    def __len__(self) -> int:
        return len(self.episodes)
