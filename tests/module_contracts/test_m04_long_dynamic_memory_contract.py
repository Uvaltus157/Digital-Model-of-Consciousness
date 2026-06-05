from __future__ import annotations

from scripts.module_lab.module_fixture_factory import (
    FakePassportManager,
    assert_gate,
    assert_tensor,
    make_fake_inner_object,
    make_fake_m5_out,
)


def test_m04_long_dynamic_memory_contract():
    from src.modules.m04_long_dynamic_memory.long_dynamic_memory import LongDynamicMemory, LongDynamicMemoryConfig

    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=256))
    out = make_fake_m5_out()
    obj = make_fake_inner_object()
    passport_manager = FakePassportManager(context_dim=256)

    packet = m4.compute(
        out=out,
        obj=obj,
        passport_manager=passport_manager,
        event_memory=None,
        dream_mode=True,
        global_step=1,
    )

    for key in ("dynamic_identity_context", "dynamic_memory_gate", "identity_token", "identity_stability", "identity_novelty", "passport_slot", "selected_sentence"):
        assert key in packet, f"missing {key}"

    assert_tensor("dynamic_identity_context", packet["dynamic_identity_context"], (1, 256))
    assert_gate("dynamic_memory_gate", packet["dynamic_memory_gate"], 0.0, 1.0)
    assert packet["identity_token"] == "obj:test"
