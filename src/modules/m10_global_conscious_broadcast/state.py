from __future__ import annotations

"""State types for M10 Global Conscious Broadcast.

M10 is the future boundary where candidate contents from object imagery,
events, self-state, memory, motivation, and metacognition compete for global
availability.

This file is intentionally dependency-light and does not alter current runtime
behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConsciousCandidate:
    """A candidate content item that may enter global broadcast."""

    source_module: str
    content_type: str
    content: Any
    salience: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BroadcastPacket:
    """Selected globally available content for one broadcast step."""

    active_object: Optional[Any] = None
    active_event: Optional[Any] = None
    active_self_state: Optional[Any] = None
    active_goal: Optional[Any] = None
    candidates: List[ConsciousCandidate] = field(default_factory=list)
    selected_candidate: Optional[ConsciousCandidate] = None
    reason_selected: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
