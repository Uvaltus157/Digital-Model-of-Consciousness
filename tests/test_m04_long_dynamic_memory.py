from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from src.modules.m04_long_dynamic_memory.dynamic_object_passport import (
    DynamicObjectPassportConfig,
    DynamicObjectPassportManager,
)
from src.modules.m04_long_dynamic_memory.long_dynamic_memory import (
    LongDynamicMemory,
    LongDynamicMemoryConfig,
)


def test_m4_long_dynamic_memory_exports_identity_context_and_passport() -> None:
    torch.manual_seed(4)
    manager = DynamicObjectPassportManager(DynamicObjectPassportConfig(latent_dim=16, min_dynamic_score=0.0, min_confidence_to_create=0.0))
    event_memory = SimpleNamespace(
        events=deque([], maxlen=8),
        last_event={
            "semantic_sentence": "SENT SUBJ=OBJ_000 VERB=move_changes",
            "sentence": "EVT t=1 SLOT_0 OBJ_000 KIND=self_motion_transition",
        },
        sentence_memory=SimpleNamespace(latest_episode_summary=lambda: "episode with moving object"),
    )
    obj = {
        "z_obj": torch.randn(1, 16),
        "active_slot_index": torch.tensor([[0]]),
        "slot_token": "OBJ_000",
        "confidence": torch.tensor([[0.8]]),
        "event_delta_norm": torch.tensor([[0.5]]),
        "touch_strength": torch.tensor([[0.2]]),
        "vision_strength": torch.tensor([[0.7]]),
    }
    out = {
        "inner_object": obj,
        "focus_context": torch.randn(1, 16),
        "event_dream_replay": {"replay_context": torch.randn(1, 16), "event_salience": torch.tensor([[0.4]])},
    }

    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=16))
    packet = m4.compute(
        out=out,
        obj=obj,
        passport_manager=manager,
        event_memory=event_memory,
        dream_mode=False,
        global_step=10,
    )

    assert packet["dynamic_identity_context"].shape == (1, 16)
    assert packet["dynamic_memory_gate"].shape == (1, 1)
    assert packet["should_bind_identity"].shape == (1, 1)
    assert packet["identity_stability"].shape == (1, 1)
    assert packet["identity_similarity"].shape == (1, 1)
    assert packet["identity_novelty"].shape == (1, 1)
    assert packet["passport_count"].shape == (1, 1)
    assert packet["passport_slot"].shape == (1, 1)
    assert packet["identity_token"] == "OBJ_000"
    assert packet["selected_sentence"] == "SENT SUBJ=OBJ_000 VERB=move_changes"
    assert packet["episode_summary"] == "episode with moving object"
    assert packet["replay_z"].shape == (1, 16)
    assert len(manager.passports) == 1
