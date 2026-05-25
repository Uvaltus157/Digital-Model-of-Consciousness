from __future__ import annotations

"""Runtime scaffold for M10 Global Conscious Broadcast.

The current project does not yet route runtime control through this module.
This scaffold is dependency-light and safe to import. It provides a small
selection function that can later be replaced by a neural/attention-based
competition layer.
"""

from typing import Iterable

from .state import BroadcastPacket, ConsciousCandidate


class GlobalConsciousBroadcastRuntime:
    """Minimal broadcast selector for future integration."""

    def select(self, candidates: Iterable[ConsciousCandidate]) -> BroadcastPacket:
        items = list(candidates)
        if not items:
            return BroadcastPacket(candidates=[])
        selected = max(items, key=lambda c: (float(c.salience), float(c.confidence)))
        return BroadcastPacket(
            candidates=items,
            selected_candidate=selected,
            reason_selected=selected.reason or "highest salience/confidence candidate",
        )
