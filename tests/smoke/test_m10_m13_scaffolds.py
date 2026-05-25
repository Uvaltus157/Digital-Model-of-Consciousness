from __future__ import annotations

from src.modules.m10_global_conscious_broadcast.debug import summarize_broadcast_packet
from src.modules.m10_global_conscious_broadcast.runtime import GlobalConsciousBroadcastRuntime
from src.modules.m10_global_conscious_broadcast.state import ConsciousCandidate
from src.modules.m13_autobiographical_memory.debug import summarize_timeline
from src.modules.m13_autobiographical_memory.runtime import AutobiographicalMemoryRuntime
from src.modules.m13_autobiographical_memory.state import AutobiographicalEventTag


def test_m10_selects_highest_salience_candidate() -> None:
    runtime = GlobalConsciousBroadcastRuntime()
    candidates = [
        ConsciousCandidate("M1", "object", {"slot": 1}, salience=0.3, confidence=0.9),
        ConsciousCandidate("M11", "drive", {"urge": "explore"}, salience=0.8, confidence=0.5, reason="novelty"),
    ]
    packet = runtime.select(candidates)
    assert packet.selected_candidate is not None
    assert packet.selected_candidate.source_module == "M11"
    assert packet.reason_selected == "novelty"

    summary = summarize_broadcast_packet(packet)
    assert summary["num_candidates"] == 2
    assert summary["selected_source"] == "M11"


def test_m10_empty_selection_is_safe() -> None:
    packet = GlobalConsciousBroadcastRuntime().select([])
    assert packet.selected_candidate is None
    assert packet.candidates == []


def test_m13_records_episode_and_debug_summary() -> None:
    runtime = AutobiographicalMemoryRuntime()
    episode = runtime.record_episode(
        self_state={"agency": 0.9},
        action={"kind": "look"},
        event={"object": "OBJ_001"},
        outcome={"stabilized": True},
        tags=[AutobiographicalEventTag("discovery", "object_stabilized")],
    )
    assert episode.episode_id == "EP_000001"
    assert len(runtime.timeline) == 1

    summary = summarize_timeline(runtime.timeline)
    assert summary["num_episodes"] == 1
    assert summary["latest"]["episode_id"] == "EP_000001"
    assert summary["latest"]["tag_types"] == ["discovery"]
