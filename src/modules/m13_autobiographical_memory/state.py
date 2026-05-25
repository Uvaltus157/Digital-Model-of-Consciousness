from __future__ import annotations

"""State types for M13 Autobiographical Memory."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AutobiographicalEventTag:
    """Meaning tag attached to a self episode."""

    tag_type: str
    value: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfEpisode:
    """A remembered episode involving self, action, event and outcome."""

    episode_id: str
    timestamp: float
    self_state: Optional[Any] = None
    action: Optional[Any] = None
    event: Optional[Any] = None
    outcome: Optional[Any] = None
    tags: List[AutobiographicalEventTag] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
