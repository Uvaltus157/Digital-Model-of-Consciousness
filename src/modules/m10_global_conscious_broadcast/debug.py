from __future__ import annotations

"""Debug helpers for M10 Global Conscious Broadcast."""

from typing import Any, Dict

from .state import BroadcastPacket


def summarize_broadcast_packet(packet: BroadcastPacket) -> Dict[str, Any]:
    """Return a JSON-friendly summary of a broadcast packet."""
    selected = packet.selected_candidate
    return {
        "num_candidates": len(packet.candidates),
        "selected_source": selected.source_module if selected else None,
        "selected_type": selected.content_type if selected else None,
        "selected_salience": float(selected.salience) if selected else 0.0,
        "selected_confidence": float(selected.confidence) if selected else 0.0,
        "reason_selected": packet.reason_selected,
    }
