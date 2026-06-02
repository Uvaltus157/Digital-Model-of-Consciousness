from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch

from src.modules.m02_event_dream_replay.event_dream_replay import (
    EventDreamReplay,
    EventDreamReplayConfig,
)
from src.modules.m04_long_dynamic_memory.dynamic_object_passport import (
    DynamicObjectPassportConfig,
    DynamicObjectPassportManager,
)
from src.modules.m04_long_dynamic_memory.long_dynamic_memory import (
    LongDynamicMemory,
    LongDynamicMemoryConfig,
)
from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_memory_thought import (
    ConsciousDreamerMemoryThought,
    ConsciousDreamerMemoryThoughtConfig,
)
from src.modules.m07_inner_speech_thoughts.inner_speech_decoder import (
    InnerSpeechDecoder,
    InnerSpeechDecoderConfig,
    render_inner_speech_text,
)
from src.modules.m09_self_core.models.self_core import SelfCore, SelfCoreConfig
from src.modules.m10_global_conscious_broadcast.broadcast_gate import (
    GlobalBroadcastConfig,
    GlobalConsciousBroadcastGate,
)
from src.modules.m11_motivational_homeostasis.emotional_drive_bivalent import EmotionalDrive
from src.modules.m12_metacognition_monitor.metacognition_monitor import (
    MetacognitionMonitor,
    MetacognitionMonitorConfig,
)
from src.modules.m13_autobiographical_memory.autobiographical_memory import (
    AutobiographicalMemory,
    AutobiographicalMemoryConfig,
)
from src.modules.m14_semantic_grounding.semantic_action_grounding import (
    SemanticActionGrounding,
)
from src.modules.m15_counterfactual_imagination_planning.thought_chain_controller import (
    ThoughtChainController,
    ThoughtChainControllerConfig,
)


def test_unified_system_is_primary_runtime_class_source_contract() -> None:
    source = Path("src/apps/runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "UnifiedSystem" in class_names
    assert "UnifiedSystem" + "V510" not in source

    unified = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnifiedSystem")
    base_names = [base.id for base in unified.bases if isinstance(base, ast.Name)]
    assert "UnifiedRuntimeBase" in base_names
    assert "UnifiedSystem" + "V57" not in source


def test_unified_system_has_current_runtime_methods_source_contract() -> None:
    source = Path("src/apps/runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    unified = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnifiedSystem")
    base_names = [base.id for base in unified.bases if isinstance(base, ast.Name)]

    required_mixins = {
        "LongDynamicMemoryRuntimeMixin",
        "EventDreamReplayRuntimeMixin",
        "AutobiographicalMemoryRuntimeMixin",
        "ThoughtChainRuntimeMixin",
        "GlobalBroadcastRuntimeMixin",
        "InnerSpeechRuntimeMixin",
        "MetacognitionRuntimeMixin",
        "SemanticActionRuntimeMixin",
    }
    assert required_mixins.issubset(set(base_names))


def test_conscious_system_uses_versionless_primary_class_source_contract() -> None:
    source = Path("src/modules/m05_world_model_attention_workspace/legacy/conscious_system.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ConsciousSystem" in class_names
    assert "ConsciousSystemV5" not in class_names

    alias_found = False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "ConsciousSystemV5"
                and isinstance(node.value, ast.Name)
                and node.value.id == "ConsciousSystem"
            ):
                alias_found = True
    assert alias_found

    wrapper_source = Path("src/modules/m05_world_model_attention_workspace/models/conscious_system.py").read_text(
        encoding="utf-8"
    )
    wrapper_tree = ast.parse(wrapper_source)
    assert not any(isinstance(node, ast.ClassDef) for node in wrapper_tree.body)
    assert "legacy.conscious_system" in wrapper_source


def _small_m5_config() -> ConsciousDreamerMemoryThoughtConfig:
    cfg = ConsciousDreamerMemoryThoughtConfig()
    cfg.data.image_height = 16
    cfg.data.image_width = 16
    cfg.data.pose_dim = 7
    cfg.data.body_state_dim = 8
    cfg.data.tactile_dim = 5
    cfg.data.hand_motor_dim = 4
    cfg.data.embodied_dim = 3
    cfg.data.action_dim = 6

    cfg.latent.vision_dim = 8
    cfg.latent.pose_dim = 4
    cfg.latent.body_dim = 4
    cfg.latent.tactile_dim = 4
    cfg.latent.hand_motor_dim = 4
    cfg.latent.object_state_dim = 4
    cfg.latent.action_embed_dim = 4
    cfg.latent.modality_dim = 8
    cfg.latent.fused_dim = 16
    cfg.latent.rssm_dim = 16

    cfg.conscious.workspace_dim = 12
    cfg.conscious.body_context_dim = 10
    cfg.conscious.model_reflection_dim = 10
    cfg.conscious.thought_dim = 8
    cfg.conscious.value_dim = 6
    cfg.conscious.report_dim = 6
    cfg.conscious.object_repr_dim = 7
    cfg.conscious.imagination_horizon = 2
    cfg.conscious.attention_heads = 2

    cfg.thought_memory.thought_steps = 2
    cfg.thought_memory.memory_slots = 8
    cfg.thought_memory.memory_dim = 9
    return cfg


def test_m5_outputs_owned_focus_context() -> None:
    torch.manual_seed(1)
    cfg = _small_m5_config()
    model = ConsciousDreamerMemoryThought(cfg)
    state = model.initial_state(batch_size=1, device="cpu")

    out = model.step(
        left=torch.randn(1, 3, 16, 16),
        right=torch.randn(1, 3, 16, 16),
        depth=torch.randn(1, 1, 16, 16),
        pose=torch.randn(1, 7),
        body_state=torch.randn(1, cfg.data.body_state_dim),
        tactile=torch.randn(1, cfg.data.tactile_dim),
        hand_motor=torch.randn(1, cfg.data.hand_motor_dim),
        embodied_action=torch.randn(1, cfg.data.embodied_dim),
        object_state=torch.randn(1, 9),
        state=state,
        write_memory=False,
    )

    assert "focus_context" in out
    assert out["focus_context"].ndim == 2
    assert out["focus_context"].shape[0] == 1
    assert out["focus_context"].shape[-1] > cfg.conscious.workspace_dim
    assert "symbolic_report" not in out


def test_m9_binds_focus_context_and_affect_latents() -> None:
    torch.manual_seed(2)
    cfg = SelfCoreConfig(
        body_state_dim=8,
        action_dim=3,
        tactile_dim=5,
        vestibular_dim=4,
        object_latent_dim=7,
        workspace_dim=12,
        focus_context_dim=16,
        affect_latent_dim=6,
        hidden_dim=32,
        self_dim=10,
        subjective_dim=5,
    )
    model = SelfCore(cfg)
    prev = model.initial_state(batch_size=1, device="cpu")

    out = model(
        prev,
        body_state=torch.randn(1, 8),
        action=torch.randn(1, 3),
        tactile=torch.randn(1, 5),
        vestibular=torch.randn(1, 4),
        object_latent=torch.randn(1, 7),
        workspace=torch.randn(1, 12),
        focus_context=torch.randn(1, 16),
        affect_latents=torch.randn(1, 6),
    )

    assert out["self_bound_context"].shape == (1, cfg.self_dim + cfg.focus_context_dim + cfg.affect_latent_dim)
    assert out["affect_binding_score"].shape == (1, 1)
    assert out["subjective_affect_state"].shape == (1, cfg.subjective_dim)
    assert out["affect_latents"].shape == (1, cfg.affect_latent_dim)


def test_m10_selects_global_broadcast_for_self_binding() -> None:
    torch.manual_seed(3)
    cfg = GlobalBroadcastConfig(
        focus_context_dim=16,
        affect_latent_dim=6,
        thought_dim=10,
        plan_context_dim=12,
        hidden_dim=32,
    )
    gate = GlobalConsciousBroadcastGate(cfg)

    out = gate(
        focus_context=torch.randn(1, 16),
        raw_focus_context=torch.randn(1, 16),
        active_thought=torch.randn(1, 10),
        plan_context=torch.randn(1, 12),
        affect_latents=torch.randn(1, 6),
        best_chain_score=torch.tensor([[0.7]]),
        predicted_affect_delta=torch.tensor([[0.4]]),
        no_viable_chain=torch.tensor([[0.0]]),
        panic_trigger=torch.tensor([[0.0]]),
    )

    assert out["broadcast_latent"].shape == (1, cfg.focus_context_dim)
    assert out["source_logits"].shape == (1, 4)
    assert out["source_weights"].shape == (1, 4)
    assert out["priority"].shape == (1, 1)
    assert out["broadcast_gate"].shape == (1, 1)
    assert out["selected_source"] in gate.SOURCE_NAMES


def test_m12_monitors_confidence_doubt_and_verification_need() -> None:
    torch.manual_seed(4)
    cfg = MetacognitionMonitorConfig(
        self_dim=10,
        focus_context_dim=16,
        affect_latent_dim=6,
        report_latent_dim=8,
        plan_context_dim=12,
        hidden_dim=32,
    )
    monitor = MetacognitionMonitor(cfg)
    out = monitor(
        self_state=torch.randn(1, 10),
        focus_context=torch.randn(1, 16),
        affect_latents=torch.randn(1, 6),
        report_latent=torch.randn(1, 8),
        plan_context=torch.randn(1, 12),
        scalar_features=torch.tensor([[0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.7, 0.2, 1.0, 0.0, 0.8, 0.8]]),
    )

    assert out["metacognitive_confidence"].shape == (1, 1)
    assert out["doubt"].shape == (1, 1)
    assert out["verification_need"].shape == (1, 1)
    assert out["action_hold"].shape == (1, 1)
    assert out["should_verify"].shape == (1, 1)
    assert out["should_hold_action"].shape == (1, 1)
    assert out["high_doubt"].shape == (1, 1)


def test_m13_writes_and_retrieves_autobiographical_episode() -> None:
    torch.manual_seed(5)
    memory = AutobiographicalMemory(AutobiographicalMemoryConfig(memory_dim=16, max_episodes=8, retrieval_topk=1))
    out = {
        "focus_context": torch.randn(1, 16),
        "workspace_out": torch.randn(1, 16),
        "object_repr": torch.randn(1, 16),
        "affect": {"affect_latents": torch.randn(1, 6), "panic_latent": torch.tensor([[0.2]])},
        "emotion": {"emotional_valence": 0.4, "emotional_arousal": 0.5},
        "metacognition": {"doubt": torch.tensor([[0.1]])},
        "conscious_action": {"reason": "action_allowed"},
        "thought_chain": {"plan_context": torch.randn(1, 12), "best_chain_score": torch.tensor([[0.8]])},
        "broadcast": {"broadcast_latent": torch.randn(1, 16), "selected_source": "m15_focus"},
    }

    write = memory.write_episode(obs={}, out=out, global_step=7)
    retrieved = memory.retrieve(out)

    assert write["episode_written"].shape == (1,)
    assert float(write["episode_count"].item()) == 1.0
    assert retrieved["retrieved_context"].shape == (1, 16)
    assert retrieved["retrieval_relevance"].shape == (1, 1)
    assert float(retrieved["retrieved_episode_count"].item()) == 1.0
    assert "step=7" in write["last_summary"]


def test_m14_softens_action_and_outputs_semantic_intent() -> None:
    guard = SemanticActionGrounding()
    out = {
        "decoded_report": "I should check before acting",
        "metacognition": {
            "action_hold": torch.tensor([[0.9]]),
            "verification_need": torch.tensor([[0.8]]),
            "doubt": torch.tensor([[0.9]]),
            "should_hold_action": torch.tensor([[1.0]]),
            "should_verify": torch.tensor([[1.0]]),
            "high_doubt": torch.tensor([[1.0]]),
        },
        "thought_chain": {
            "no_viable_chain": torch.tensor([[0.0]]),
            "predicted_affect_delta": torch.tensor([[0.2]]),
            "best_chain_score": torch.tensor([[0.6]]),
        },
        "affect": {
            "panic_latent": torch.tensor([[0.0]]),
            "fear_latent": torch.tensor([[0.0]]),
            "stress_latent": torch.tensor([[0.0]]),
            "curiosity_latent": torch.tensor([[0.3]]),
            "comfort_latent": torch.tensor([[0.1]]),
        },
        "broadcast": {"urgency": torch.tensor([[0.4]]), "priority": torch.tensor([[0.7]]), "selected_source": "m15_focus"},
        "autobiographical_memory": {"retrieval_relevance": torch.tensor([[0.4]]), "summary": "step=1 source=m15_focus"},
        "self_core": {"agency_score": torch.tensor([[0.7]])},
        "inner_speech": {"confidence": torch.tensor([[0.6]])},
    }

    action = guard.compute(out)

    assert action["applied_action_scale"].shape == (1, 1)
    assert float(action["applied_action_scale"].item()) <= 0.25
    assert float(action["verify_before_action"].item()) == 1.0
    assert action["reason"] in {"verify_before_action", "metacognitive_hold"}
    assert action["semantic_intent"] in guard.INTENT_ORDER
    assert action["semantic_intent_scores"].shape == (1, len(guard.INTENT_ORDER))
    assert action["semantic_confidence"].shape == (1, 1)
    assert action["grounding_confidence"].shape == (1, 1)
    assert action["expected_outcome"].shape == (1, 1)
    assert isinstance(action["goal_text"], str) and action["goal_text"]
    assert action["target_source"] == "m15_focus"


def test_m15_searches_chain_before_m9_and_enhances_focus_context() -> None:
    torch.manual_seed(6)
    cfg = ThoughtChainControllerConfig(
        self_bound_context_dim=20,
        subjective_affect_dim=5,
        focus_context_dim=16,
        affect_latent_dim=6,
        hidden_dim=32,
        thought_dim=10,
        plan_context_dim=12,
        chain_len=3,
    )
    model = ThoughtChainController(cfg)

    out = model(
        focus_context=torch.randn(1, 16),
        affect_latents=torch.randn(1, 6),
    )

    assert out["active_thought_chain"].shape == (1, cfg.chain_len, cfg.thought_dim)
    assert out["candidate_thought_chain"].shape == (1, cfg.chain_len, cfg.thought_dim)
    assert out["active_thought"].shape == (1, cfg.thought_dim)
    assert out["plan_context"].shape == (1, cfg.plan_context_dim)
    assert out["enhanced_focus_context"].shape == (1, cfg.focus_context_dim)
    assert out["best_chain"].shape == (1, cfg.chain_len, cfg.thought_dim)
    assert out["best_chain_score"].shape == (1, 1)
    assert out["predicted_affect_delta"].shape == (1, 1)
    assert out["no_viable_chain"].shape == (1, 1)
    assert out["panic_trigger"].shape == (1, 1)
    assert set(out["thought_chain_metrics"].keys()) == {
        "stability",
        "urgency",
        "self_relevance",
        "planning_readiness",
    }


def test_m4_m13_m2_m15_share_focus_context_dimension() -> None:
    torch.manual_seed(8)
    focus_dim = 16
    focus_context = torch.randn(1, focus_dim)
    affect = {
        "affect_latents": torch.randn(1, 6),
        "panic_latent": torch.tensor([[0.1]]),
        "stress_latent": torch.tensor([[0.2]]),
        "curiosity_latent": torch.tensor([[0.8]]),
    }
    out = {
        "focus_context": focus_context,
        "workspace_out": torch.randn(1, focus_dim),
        "object_repr": torch.randn(1, focus_dim),
        "values": {"coherence": torch.tensor([[0.5]]), "curiosity": torch.tensor([[0.4]])},
        "affect": affect,
        "emotion": {"emotional_valence": 0.3, "emotional_arousal": 0.4},
        "metacognition": {"doubt": torch.tensor([[0.2]])},
    }
    obj = {
        "z_obj": torch.randn(1, focus_dim),
        "active_slot_index": torch.tensor([[0]]),
        "slot_token": "OBJ_000",
        "confidence": torch.tensor([[0.9]]),
        "event_delta_norm": torch.tensor([[0.6]]),
        "touch_strength": torch.tensor([[0.4]]),
        "vision_strength": torch.tensor([[0.7]]),
    }
    out["inner_object"] = obj

    event_memory = SimpleNamespace(
        events=deque([{
            "slot": 0,
            "confidence": 0.9,
            "delta_norm": 0.6,
            "action_norm": 0.3,
            "contact_norm": 0.4,
            "vision_strength": 0.7,
            "touch_strength": 0.4,
            "dream_mode": False,
            "event_code": torch.tensor([[1.0, 0.9, 0.6, 0.3, 0.4, 0.7, 0.4, 0.0]]),
            "semantic_sentence": "SENT SUBJ=OBJ_000 VERB=move_changes",
            "kind": "self_motion_transition",
            "slot_token": "OBJ_000",
        }], maxlen=8),
        last_event=None,
        sentence_memory=SimpleNamespace(latest_episode_summary=lambda: "episode with dynamic object"),
    )
    passport_manager = DynamicObjectPassportManager(
        DynamicObjectPassportConfig(latent_dim=focus_dim, min_dynamic_score=0.0, min_confidence_to_create=0.0)
    )

    m4 = LongDynamicMemory(LongDynamicMemoryConfig(context_dim=focus_dim))
    out["long_dynamic_memory"] = m4.compute(
        out=out,
        obj=obj,
        passport_manager=passport_manager,
        event_memory=event_memory,
        dream_mode=False,
        global_step=1,
    )
    out["focus_context"] = out["focus_context"] + 0.1 * out["long_dynamic_memory"]["dynamic_identity_context"]

    memory = AutobiographicalMemory(AutobiographicalMemoryConfig(memory_dim=focus_dim, max_episodes=8, retrieval_topk=1))
    memory.write_episode(obs={}, out=out, global_step=2)
    out["autobiographical_memory"] = memory.retrieve(out)
    out["focus_context"] = out["focus_context"] + 0.1 * out["autobiographical_memory"]["retrieved_context"]

    replay = EventDreamReplay(EventDreamReplayConfig(replay_context_dim=focus_dim, replay_threshold=0.0))
    out["event_dream_replay"] = replay.compute(out=out, event_memory=event_memory, dream_mode=False)
    out["focus_context"] = out["focus_context"] + 0.1 * out["event_dream_replay"]["replay_context"]

    thought = ThoughtChainController(ThoughtChainControllerConfig(
        self_bound_context_dim=20,
        subjective_affect_dim=5,
        focus_context_dim=focus_dim,
        affect_latent_dim=6,
        hidden_dim=32,
        thought_dim=10,
        plan_context_dim=12,
        chain_len=3,
    ))
    out["thought_chain"] = thought(
        focus_context=out["focus_context"],
        affect_latents=out["affect"]["affect_latents"],
    )

    assert out["long_dynamic_memory"]["dynamic_identity_context"].shape == (1, focus_dim)
    assert out["autobiographical_memory"]["retrieved_context"].shape == (1, focus_dim)
    assert out["event_dream_replay"]["replay_context"].shape == (1, focus_dim)
    assert out["thought_chain"]["enhanced_focus_context"].shape == (1, focus_dim)


def test_m7_verbalizes_self_bound_thought_inputs() -> None:
    torch.manual_seed(7)
    cfg = InnerSpeechDecoderConfig(
        active_thought_dim=10,
        plan_context_dim=12,
        self_bound_context_dim=20,
        subjective_affect_dim=5,
        affect_latent_dim=6,
        hidden_dim=32,
        report_latent_dim=8,
        vocab_size=32,
        max_tokens=6,
    )
    model = InnerSpeechDecoder(cfg)

    out = model(
        active_thought=torch.randn(1, 10),
        plan_context=torch.randn(1, 12),
        self_bound_context=torch.randn(1, 20),
        subjective_affect_state=torch.randn(1, 5),
        affect_latents=torch.randn(1, 6),
    )

    assert out["report_latent"].shape == (1, cfg.report_latent_dim)
    assert out["confidence"].shape == (1, 1)
    assert out["text_token_ids"].shape == (1, cfg.max_tokens)

    text = render_inner_speech_text(
        self_core={
            "agency_score": torch.tensor([[0.8]]),
            "body_ownership_score": torch.tensor([[0.7]]),
            "self_continuity_score": torch.tensor([[0.7]]),
            "focus_binding_score": torch.tensor([[0.8]]),
            "affect_binding_score": torch.tensor([[0.7]]),
        },
        thought_chain={"thought_chain_metrics": {"planning_readiness": torch.tensor([[0.8]])}},
        affect={"comfort_latent": torch.tensor([[0.6]]), "panic_latent": torch.tensor([[0.1]]), "valence": torch.tensor([[0.3]])},
        confidence=0.5,
    )
    assert "I am causing this action" in text
    assert "I can form a plan" in text


def test_emotional_drive_reuses_cached_packet_without_second_ema_update() -> None:
    drive = EmotionalDrive()
    out = {
        "values": {"coherence": torch.tensor([[0.5]]), "curiosity": torch.tensor([[0.2]])},
        "workspace_out": torch.randn(1, 12),
        "object_repr": torch.randn(1, 7),
        "preconscious_thoughts": {"thought_candidate": torch.randn(1, 8)},
        "memory": {"memory_context": torch.randn(1, 9)},
        "preconscious_reflection_out": {"model_confidence": torch.tensor([[0.4]])},
    }
    obs = {"tactile": torch.zeros(1, 5)}

    first = drive.compute(out, obs)
    out["emotion"] = first
    ema_after_first = drive.ema_uncertainty

    second = drive.compute(out, obs)

    assert second is first
    assert drive.ema_uncertainty == ema_after_first
    assert second["_emotion_cache_reusable"] is True
    assert "affect_latents" in second["affect"]
