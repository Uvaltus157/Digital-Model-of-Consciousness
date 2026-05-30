from __future__ import annotations

import numpy as np
import torch


def _d(value) -> dict:
    return value if isinstance(value, dict) else {}


def build_base_life_stats(
    rt,
    *,
    obs: dict,
    out: dict,
    emotion: dict,
    novelty_score: float,
    chosen_action: int,
    decoded_report: str,
    target_report: str,
    inner_report_confidence: float,
    self_confidence: float,
) -> dict:
    affect = _d(out.get("affect"))
    return {
        "step": rt.global_step,
        "training": bool(rt.training_enabled) and not (rt.is_full_sleep_mode() if hasattr(rt, "is_full_sleep_mode") else False),
        "sensor_state": rt.sensor_state_label() if hasattr(rt, "sensor_state_label") else "unknown",
        "quality": rt.quality.get(),
        "train_steps": rt.train_steps,
        "focus_idx": int(out["focus"]["focus_idx"].item()),
        "action": int(chosen_action),
        "curiosity": float(out["values"]["curiosity"].item()),
        "coherence": float(out["values"]["coherence"].item()),
        "self_confidence": float(self_confidence),
        "inner_report_confidence": float(inner_report_confidence),
        "decoded_report": decoded_report,
        "target_report": target_report,
        "novelty_score": float(novelty_score),
        "emotion_valence": float(emotion["emotional_valence"]),
        "emotion_arousal": float(emotion["emotional_arousal"]),
        "intrinsic_reward": float(emotion["intrinsic_reward"].detach().cpu().item()),
        "meaning_progress": float(emotion["meaning_progress"]),
        "gap_fill_reward": float(emotion["gap_fill_reward"]),
        "misunderstanding": float(emotion["misunderstanding"]),
        "confusion_increase": float(emotion["confusion_increase"]),
        "coherence_loss": float(emotion["coherence_loss"]),
        "alignment_loss": float(emotion["alignment_loss"]),
        "chaotic_touch": float(emotion["chaotic_touch"]),
        "multimodal_alignment": float(emotion["multimodal_alignment"]),
        "contact_pleasure": float(emotion["contact_pleasure"]),
        "uncertainty": float(emotion["uncertainty"]),
        "affect_latent_norm": float(affect.get("affect_latents", torch.zeros(1, device=rt.device)).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(affect.get("affect_latents")) else 0.0,
        "affect_panic": rt._life_tensor_float(affect.get("panic_latent"), 0.0),
        "affect_comfort": rt._life_tensor_float(affect.get("comfort_latent"), 0.0),
    }


def build_object_memory_life_stats(rt, *, obs: dict, out: dict) -> dict:
    inner_object = _d(out.get("inner_object"))
    return {
        "object_decoder_loss": float(getattr(rt, "latest_object_decoder_stats", {}).get("object_decoder_loss", 0.0) or 0.0),
        "object_rgb_loss": float(getattr(rt, "latest_object_decoder_stats", {}).get("object_rgb_loss", 0.0) or 0.0),
        "object_depth_loss": float(getattr(rt, "latest_object_decoder_stats", {}).get("object_depth_loss", 0.0) or 0.0),
        "object_mask_loss": float(getattr(rt, "latest_object_decoder_stats", {}).get("object_mask_loss", 0.0) or 0.0),
        "inner_action_active": bool(inner_object.get("inner_action_active", False)),
        "inner_action_confidence": rt._life_tensor_float(inner_object.get("inner_action_confidence"), 0.0),
        "inner_trust_value": rt._life_tensor_float(inner_object.get("inner_trust_value"), 0.0),
        "inner_trust_alpha": rt._life_tensor_float(inner_object.get("inner_trust_alpha"), 0.0),
        "inner_trust_allowed": rt._life_tensor_bool(inner_object.get("inner_trust_allowed"), False),
        "inner_trust_applied_to_policy": rt._life_tensor_bool(inner_object.get("inner_trust_applied_to_policy"), False),
        "inner_trust_reason": str(inner_object.get("inner_trust_reason", "")),
        "passport_active": rt._life_tensor_bool(inner_object.get("passport_active"), False),
        "passport_token": str(inner_object.get("passport_token", inner_object.get("m4_identity_token", ""))),
        "passport_count": rt._life_tensor_float(inner_object.get("passport_count", inner_object.get("m4_passport_count")), 0.0),
        "passport_replay_active": rt._life_tensor_bool(inner_object.get("passport_replay_active"), False),
        "passport_second_order_decoded": rt._life_tensor_bool(inner_object.get("passport_second_order_decoded"), False),
        "passport_debug_active": rt._life_tensor_bool(inner_object.get("passport_debug_active"), False),
        "passport_debug_z_distance": rt._life_tensor_float(inner_object.get("passport_debug_z_distance"), 0.0),
        "passport_debug_z_cosine": rt._life_tensor_float(inner_object.get("passport_debug_z_cosine"), 0.0),
        "inner_real_action_trace_path": str(inner_object.get("inner_real_action_trace_path", "")),
        "inner_real_action_body_norm": float(inner_object.get("inner_action_body", torch.zeros(1, device=rt.device)).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(inner_object.get("inner_action_body")) else 0.0,
        "inner_real_action_hand_norm": float(inner_object.get("inner_action_hand", torch.zeros(1, device=rt.device)).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(inner_object.get("inner_action_hand")) else 0.0,
        "touch_sum": float(obs.get("tactile", torch.zeros(1, device=rt.device)).sum().item()),
        "object_repr_norm": float(out.get("object_repr", torch.zeros(1, device=rt.device)).norm(dim=-1).mean().item()) if torch.is_tensor(out.get("object_repr")) else 0.0,
        "memory_used": float(out.get("memory", {}).get("memory_usage", torch.zeros(1, device=rt.device)).sum().item()) if isinstance(out.get("memory"), dict) and torch.is_tensor(out.get("memory", {}).get("memory_usage")) else 0.0,
    }


def build_runtime_control_life_stats(rt, *, obs: dict, out: dict) -> dict:
    return {
        "exploration_active": bool(out.get("exploration", {}).get("active", False)),
        "exploration_boost": float(out.get("exploration", {}).get("boost", 0.0)),
        "dynamic_rig_enabled": bool(rt.dynamic_agent_rig_controller is not None),
        "rig_z": float(rt.world.data.qpos[rt.dynamic_agent_rig_controller.qpos_adr + 2]) if rt.dynamic_agent_rig_controller is not None else 0.0,
        "hover_height": float(rt.cfg.dynamic_agent_rig.hover_height),
        "hover_target_z": float(getattr(rt.dynamic_agent_rig_controller, "hover_target_z", 0.0)) if rt.dynamic_agent_rig_controller is not None else 0.0,
        "manual_override": bool(getattr(rt, "_ipc_manual_actions_enabled", False)),
        "pyqt_manual_actions_enabled": bool(getattr(rt, "_ipc_manual_actions_enabled", False)),
        "manual_body_len": int(len(rt._ipc_manual_body_action)) if getattr(rt, "_ipc_manual_body_action", None) is not None else 0,
        "manual_arm_len": int(len(rt._ipc_manual_arm_action)) if getattr(rt, "_ipc_manual_arm_action", None) is not None else 0,
        "manual_hand_len": int(len(rt._ipc_manual_hand_action)) if getattr(rt, "_ipc_manual_hand_action", None) is not None else 0,
        "manual_leg_len": int(len(rt._ipc_manual_leg_action)) if getattr(rt, "_ipc_manual_leg_action", None) is not None else 0,
        "fly_to_cube_palpate_active": bool(getattr(rt, "_fly_to_cube_palpate_active", False)),
        "fly_to_cube_palpate_phase": str(getattr(rt, "_fly_to_cube_palpate_status", {}).get("phase", "")) if isinstance(getattr(rt, "_fly_to_cube_palpate_status", {}), dict) else "",
        "fly_to_cube_palpate_dist": float(getattr(rt, "_fly_to_cube_palpate_status", {}).get("body_dist", 0.0)) if isinstance(getattr(rt, "_fly_to_cube_palpate_status", {}), dict) else 0.0,
        "fly_to_cube_palpate_touch": float(getattr(rt, "_fly_to_cube_palpate_status", {}).get("tactile_sum", 0.0)) if isinstance(getattr(rt, "_fly_to_cube_palpate_status", {}), dict) else 0.0,
        "module_debug": bool(getattr(rt, "show_module_debug_window", False)),
        "module_training_flags": dict(getattr(rt, "module_training_gate", None).flags) if hasattr(rt, "module_training_gate") else {},
        "module_training_seq": int(getattr(rt, "last_module_training_seq", 0)),
        "module_trainable_total": int(getattr(rt, "module_training_gate", None).count_trainable().get("total", 0)) if hasattr(rt, "module_training_gate") else 0,
        "contact_damping": bool(rt.cfg.dynamic_agent_rig.contact_angular_damping_enabled),
        "vestibular_norm": float(obs.get("vestibular", torch.zeros(1, 24, device=rt.device)).norm(dim=-1).mean().detach().cpu().item()),
        "balance_reward": float(obs.get("balance_reward", torch.zeros(1, device=rt.device)).mean().detach().cpu().item()),
        "flight_z": float(getattr(rt.world, "cam_pos", np.zeros(3))[2]),
        "roll_deg": float(getattr(rt.world, "roll_deg", 0.0)),
        "head_yaw": float(getattr(rt.world, "head_ctrl", np.zeros(3))[0]),
        "head_pitch": float(getattr(rt.world, "head_ctrl", np.zeros(3))[1]),
        "head_roll": float(getattr(rt.world, "head_ctrl", np.zeros(3))[2]),
        "flight_z_min": float(rt.cfg.mocap_flight_bounds.min_z),
        "flight_z_max": float(rt.cfg.mocap_flight_bounds.max_z),
        "body_state_dim": int(obs.get("body_state", torch.zeros(1, 0, device=rt.device)).shape[-1]),
        "leg_ctrl_norm": float(out.get("leg_ctrl", rt.prev_leg_motor).norm(dim=-1).mean().detach().cpu().item()) if torch.is_tensor(out.get("leg_ctrl", rt.prev_leg_motor)) else 0.0,
        "embodied_dim": int(rt.cfg.embodied_dim),
        "modality_weights": out.get("attention", {}).get("modality_weights", torch.zeros(1, 1, device=rt.device))[0].detach().cpu().numpy() if isinstance(out.get("attention"), dict) and torch.is_tensor(out.get("attention", {}).get("modality_weights")) else np.zeros(1, dtype=np.float32),
        "sphere": rt.world.get_object_pos("sphere") if hasattr(rt.world, "get_object_pos") else np.zeros(3, dtype=np.float32),
    }


def build_conscious_module_life_stats(rt, *, out: dict) -> dict:
    self_core = _d(out.get("self_core"))
    replay2 = _d(out.get("event_dream_replay"))
    memory4 = _d(out.get("long_dynamic_memory"))
    broadcast = _d(out.get("broadcast"))
    metacog = _d(out.get("metacognition"))
    memory13 = _d(out.get("autobiographical_memory"))
    conscious_action = _d(out.get("conscious_action"))
    thought_chain = _d(out.get("thought_chain"))

    return {
        "self_agency": rt._life_tensor_float(self_core.get("agency_score"), 0.0),
        "self_ownership": rt._life_tensor_float(self_core.get("body_ownership_score"), 0.0),
        "self_continuity": rt._life_tensor_float(self_core.get("self_continuity_score"), 0.0),
        "self_core_present": bool(hasattr(rt, "self_core") and rt.self_core is not None),
        "self_core_trainable": int(sum(p.numel() for p in rt.self_core.parameters() if p.requires_grad)) if hasattr(rt, "self_core") and rt.self_core is not None else 0,
        "m2_replay_gate": rt._life_tensor_float(replay2.get("replay_gate"), 0.0),
        "m2_should_replay": rt._life_tensor_bool(replay2.get("should_replay"), False),
        "m2_event_salience": rt._life_tensor_float(replay2.get("event_salience"), 0.0),
        "m2_dream_pressure": rt._life_tensor_float(replay2.get("dream_pressure"), 0.0),
        "m2_blended_into_focus": rt._life_tensor_bool(replay2.get("blended_into_focus"), False),
        "m2_replay_source": str(replay2.get("replay_source", "")),
        "m2_selected_event_kind": str(replay2.get("selected_event_kind", "")),
        "m2_selected_event_sentence": str(replay2.get("selected_event_sentence", "")),
        "m4_identity_token": str(memory4.get("identity_token", "")),
        "m4_dynamic_memory_gate": rt._life_tensor_float(memory4.get("dynamic_memory_gate"), 0.0),
        "m4_identity_stability": rt._life_tensor_float(memory4.get("identity_stability"), 0.0),
        "m4_identity_similarity": rt._life_tensor_float(memory4.get("identity_similarity"), 0.0),
        "m4_identity_novelty": rt._life_tensor_float(memory4.get("identity_novelty"), 0.0),
        "m4_identity_dynamic_score": rt._life_tensor_float(memory4.get("identity_dynamic_score"), 0.0),
        "m4_should_bind_identity": rt._life_tensor_bool(memory4.get("should_bind_identity"), False),
        "m4_passport_count": rt._life_tensor_float(memory4.get("passport_count"), 0.0),
        "m4_passport_slot": rt._life_tensor_float(memory4.get("passport_slot"), 0.0),
        "m4_blended_into_focus": rt._life_tensor_bool(memory4.get("blended_into_focus"), False),
        "m4_selected_sentence": str(memory4.get("selected_sentence", "")),
        "m4_episode_summary": str(memory4.get("episode_summary", "")),
        "m10_selected_source": str(broadcast.get("selected_source", "")),
        "m10_priority": rt._life_tensor_float(broadcast.get("priority"), 0.0),
        "m10_gate": rt._life_tensor_float(broadcast.get("broadcast_gate"), 0.0),
        "m12_confidence": rt._life_tensor_float(metacog.get("metacognitive_confidence"), 0.0),
        "m12_doubt": rt._life_tensor_float(metacog.get("doubt"), 0.0),
        "m12_verify": rt._life_tensor_float(metacog.get("verification_need"), 0.0),
        "m12_hold": rt._life_tensor_float(metacog.get("action_hold"), 0.0),
        "m13_episode_count": rt._life_tensor_float(memory13.get("episode_count", memory13.get("retrieved_episode_count")), 0.0),
        "m13_retrieval_relevance": rt._life_tensor_float(memory13.get("retrieval_relevance"), 0.0),
        "m13_blended_into_focus": rt._life_tensor_bool(memory13.get("blended_into_focus"), False),
        "m13_last_summary": str(memory13.get("last_summary", memory13.get("summary", ""))),
        "m14_semantic_intent": str(conscious_action.get("semantic_intent", "")),
        "m14_target_source": str(conscious_action.get("target_source", "")),
        "m14_goal_text": str(conscious_action.get("goal_text", "")),
        "m14_grounding_confidence": rt._life_tensor_float(conscious_action.get("grounding_confidence"), 0.0),
        "m14_expected_outcome": rt._life_tensor_float(conscious_action.get("expected_outcome"), 0.0),
        "m14_action_scale": rt._life_tensor_float(conscious_action.get("applied_action_scale"), 1.0),
        "m14_reason": str(conscious_action.get("reason", "")),
        "m15_best_chain_score": rt._life_tensor_float(thought_chain.get("best_chain_score"), 0.0),
        "m15_predicted_affect_delta": rt._life_tensor_float(thought_chain.get("predicted_affect_delta"), 0.0),
    }


__all__ = [
    "build_base_life_stats",
    "build_object_memory_life_stats",
    "build_runtime_control_life_stats",
    "build_conscious_module_life_stats",
]
