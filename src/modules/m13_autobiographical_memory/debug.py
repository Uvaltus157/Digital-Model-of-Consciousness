from __future__ import annotations

"""Debug helpers for M13 Autobiographical Memory."""

from typing import Any, Dict

from .memory import PersonalTimeline
from .state import SelfEpisode


def summarize_episode(episode: SelfEpisode) -> Dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "timestamp": float(episode.timestamp),
        "num_tags": len(episode.tags),
        "tag_types": [tag.tag_type for tag in episode.tags],
        "has_self_state": episode.self_state is not None,
        "has_action": episode.action is not None,
        "has_event": episode.event is not None,
        "has_outcome": episode.outcome is not None,
    }


def summarize_timeline(timeline: PersonalTimeline) -> Dict[str, Any]:
    latest = timeline.latest()
    return {
        "num_episodes": len(timeline),
        "latest": summarize_episode(latest) if latest else None,
    }
