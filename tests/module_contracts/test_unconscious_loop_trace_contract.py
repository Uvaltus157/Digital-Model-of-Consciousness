from __future__ import annotations

from scripts.module_lab.module_fixture_factory import make_unconscious_loop_out


def test_unconscious_loop_trace_packet_and_format():
    from src.modules.m02_event_dream_replay.unconscious_loop_trace import (
        build_unconscious_loop_trace_packet,
        format_unconscious_loop_trace,
    )

    out = make_unconscious_loop_out()
    out["event_dream_replay"] = {
        "replay_gate": out["long_dynamic_memory"]["dynamic_memory_gate"],
        "should_replay": out["long_dynamic_memory"]["dynamic_memory_gate"],
        "dream_pressure": out["long_dynamic_memory"]["dynamic_memory_gate"],
        "event_salience": out["long_dynamic_memory"]["dynamic_memory_gate"],
        "replay_context": out["focus_context"],
        "replay_source": "test",
        "selected_identity_token": "obj:test",
    }

    packet = build_unconscious_loop_trace_packet(out, sleep_mode=True, sensor_state="sleep")
    assert packet["sleep"] is True
    assert "m11" in packet
    assert "m13" in packet
    assert "m4" in packet
    assert "m2" in packet
    assert "m5_seed" in packet

    line = format_unconscious_loop_trace(packet, step=1)
    assert "[unconscious_loop step=1]" in line
    assert "m11:" in line
    assert "m2:" in line
    assert "m5_seed:" in line
