from __future__ import annotations

"""Runtime scaffold for M13 Autobiographical Memory."""

import time
from typing import Any, Optional

from .memory import PersonalTimeline
from .state import AutobiographicalEventTag, SelfEpisode


class AutobiographicalMemoryRuntime:
    """Small dependency-light runtime for recording self episodes."""

    def __init__(self, timeline: Optional[PersonalTimeline] = None):
        self.timeline = timeline or PersonalTimeline()
        self._counter = 0

    def record_episode(
        self,
        self_state: Any = None,
        action: Any = None,
        event: Any = None,
        outcome: Any = None,
        tags: Optional[list[AutobiographicalEventTag]] = None,
        **metadata: Any,
    ) -> SelfEpisode:
        self._counter += 1
        episode = SelfEpisode(
            episode_id=f"EP_{self._counter:06d}",
            timestamp=time.time(),
            self_state=self_state,
            action=action,
            event=event,
            outcome=outcome,
            tags=list(tags or []),
            metadata=dict(metadata),
        )
        self.timeline.append(episode)
        return episode
