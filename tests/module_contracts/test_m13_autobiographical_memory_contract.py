from __future__ import annotations

from scripts.module_lab.module_fixture_factory import assert_gate, assert_tensor, make_fake_affect, make_fake_m5_out


def test_m13_autobiographical_memory_empty_and_retrieval_contract():
    from src.modules.m13_autobiographical_memory.autobiographical_memory import AutobiographicalMemory, AutobiographicalMemoryConfig

    m13 = AutobiographicalMemory(AutobiographicalMemoryConfig(memory_dim=256, max_episodes=16, retrieval_topk=1))
    out = make_fake_m5_out()
    out["affect"] = make_fake_affect()

    empty = m13.retrieve(out)
    assert_tensor("empty retrieved_context", empty["retrieved_context"], (1, 256))
    assert_gate("empty retrieval_relevance", empty["retrieval_relevance"], 0.0, 0.0)

    m13.write_episode(obs={}, out=out, global_step=1)
    got = m13.retrieve(out)

    assert_tensor("retrieved_context", got["retrieved_context"], (1, 256))
    assert_gate("retrieval_relevance", got["retrieval_relevance"], -1.0, 1.0)
    assert "summary" in got
    assert_tensor("retrieved_episode_count", got["retrieved_episode_count"])
