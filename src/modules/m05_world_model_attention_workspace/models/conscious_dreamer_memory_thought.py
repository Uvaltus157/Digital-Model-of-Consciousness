from __future__ import annotations

"""
conscious_dreamer_memory_thought.py

M5 memory / preconscious-candidate layer.

Architecture rule:
    M5 is a preconscious world-model and action-preparation module.
    It does not create autobiographical memory, self-aware thought, or inner
    speech. Those appear only after M9 self-binding and M7/M11 access.

This layer adds:
- PreconsciousThoughtLoop
- PreconsciousEpisodeMemory
- body_context / model_reflection_context naming
- preconscious_reflection_out naming
- focus_context owned by M5
"""

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer_core import (
    AttentionController,
    ConsciousDreamerCore,
    ConsciousDreamerCoreConfig,
    RSSMCore,
    Workspace,
    BodyContextModel,
    ObjectRepresentation,
    PreconsciousReflectionLoop,
    ImaginationCore,
    DecoderHeads,
    VisionEncoder,
    MLPEncoder,
)


@dataclass
class ThoughtMemoryConfig:
    thought_steps: int = 4
    memory_slots: int = 256
    memory_dim: int = 256
    write_decay: float = 0.985
    use_memory_write: bool = True


@dataclass
class ConsciousDreamerMemoryThoughtConfig(ConsciousDreamerCoreConfig):
    thought_memory: ThoughtMemoryConfig = field(default_factory=ThoughtMemoryConfig)


class PreconsciousThoughtLoop(nn.Module):
    def __init__(self, workspace_dim: int, thought_dim: int, reflection_dim: int, object_dim: int, memory_dim: int, steps: int = 4) -> None:
        super().__init__()
        self.steps = int(steps)
        self.drive = nn.Sequential(
            nn.Linear(workspace_dim + thought_dim + reflection_dim + object_dim + memory_dim, thought_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(thought_dim),
        )
        self.core = nn.GRUCell(thought_dim, thought_dim)
        self.gate = nn.Sequential(nn.Linear(thought_dim * 2, thought_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(thought_dim)

    def forward(self, workspace, seed, reflection, object_repr, memory_context):
        drive = self.drive(torch.cat([workspace, seed, reflection, object_repr, memory_context], dim=-1))
        h = seed
        seq = []
        for _ in range(self.steps):
            cand = self.core(drive, h)
            g = self.gate(torch.cat([h, cand], dim=-1))
            h = self.norm(g * cand + (1.0 - g) * h)
            seq.append(h)
        return {
            "candidate": h,
            "candidate_sequence": torch.stack(seq, dim=1),
            "candidate_delta": h - seed,
        }


# Backward-compatible Python class alias inside implementation only.
ThoughtLoop = PreconsciousThoughtLoop


class PreconsciousEpisodeMemory(nn.Module):
    """
    Online preconscious episode cache.

    This is not autobiographical memory. It stores sensorimotor/planning traces
    before M9 self-binding. True autobiographical memory should be written after
    self_core.self_bound_context exists.
    """
    def __init__(self, query_dim: int, episode_dim: int, memory_dim: int = 256, slots: int = 256, write_decay: float = 0.985) -> None:
        super().__init__()
        self.slots = int(slots)
        self.memory_dim = int(memory_dim)
        self.write_decay = float(write_decay)
        self.write_index = 0

        self.query_proj = nn.Sequential(nn.Linear(query_dim, memory_dim), nn.LayerNorm(memory_dim))
        self.episode_proj = nn.Sequential(nn.Linear(episode_dim, memory_dim), nn.LayerNorm(memory_dim), nn.Tanh())
        self.out_proj = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.ReLU(inplace=True), nn.LayerNorm(memory_dim))
        self.importance = nn.Sequential(nn.Linear(episode_dim, 1), nn.Sigmoid())

        self.register_buffer("memory", torch.zeros(slots, memory_dim))
        self.register_buffer("usage", torch.zeros(slots))
        self.register_buffer("age", torch.zeros(slots))

    def read(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query_proj(query)
        mem = self.memory.detach().clone().to(q.device, q.dtype)
        usage = self.usage.detach().clone().to(q.device, q.dtype)

        if float(usage.sum().detach().cpu().item()) <= 1e-6:
            return {
                "memory_context": torch.zeros_like(q),
                "memory_weights": torch.zeros(q.shape[0], self.slots, device=q.device, dtype=q.dtype),
            }

        scores = torch.matmul(q, mem.t()) / (self.memory_dim ** 0.5)
        scores = scores + torch.log(usage.clamp_min(1e-6)).unsqueeze(0)
        weights = torch.softmax(scores, dim=-1)
        context = self.out_proj(torch.matmul(weights, mem))
        return {"memory_context": context, "memory_weights": weights}

    @torch.no_grad()
    def write(self, episode: torch.Tensor) -> None:
        ep = self.episode_proj(episode).detach()
        imp = self.importance(episode).detach().reshape(-1)

        for b in range(ep.shape[0]):
            idx = self.write_index % self.slots
            alpha = float((1.0 - self.write_decay) * (0.25 + 0.75 * imp[b].clamp(0, 1)).cpu().item())
            if self.usage[idx] <= 1e-6:
                self.memory[idx].copy_(ep[b].to(self.memory.device, self.memory.dtype))
            else:
                self.memory[idx].mul_(1.0 - alpha).add_(ep[b].to(self.memory.device, self.memory.dtype), alpha=alpha)
            self.usage[idx] = torch.clamp(self.usage[idx] + 0.05, 0.0, 1.0)
            self.age[idx] = 0.0
            self.write_index += 1

        self.age.add_(1.0)
        self.usage.mul_(0.9995)


# Deprecated alias for old imports/checkpoints. New code uses PreconsciousEpisodeMemory.
AutobiographicalMemory = PreconsciousEpisodeMemory


class ConsciousPlanner(nn.Module):
    def __init__(self, workspace_dim: int, thought_dim: int, reflection_dim: int, object_dim: int, memory_dim: int, value_dim: int, action_dim: int, embodied_dim: int, hand_motor_dim: int) -> None:
        super().__init__()
        self.value_latent = nn.Sequential(
            nn.Linear(workspace_dim + thought_dim + object_dim + memory_dim, value_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(value_dim),
        )
        planner_in = workspace_dim + thought_dim + reflection_dim + object_dim + memory_dim + value_dim + 2
        self.action_logits = nn.Linear(planner_in, action_dim)
        self.focus_logits = nn.Linear(planner_in, 8)
        self.embodied = nn.Sequential(nn.Linear(planner_in, 256), nn.ReLU(inplace=True), nn.Linear(256, embodied_dim), nn.Tanh())
        self.hand = nn.Sequential(nn.Linear(planner_in, 256), nn.ReLU(inplace=True), nn.Linear(256, hand_motor_dim), nn.Sigmoid())
        self.curiosity = nn.Sequential(nn.Linear(planner_in, 1), nn.Sigmoid())
        self.coherence = nn.Sequential(nn.Linear(planner_in, 1), nn.Sigmoid())

    def forward(self, workspace, preconscious_candidate, reflection, object_repr, memory_context, imagined_value, imagined_touch):
        value_latent = self.value_latent(torch.cat([workspace, preconscious_candidate, object_repr, memory_context], dim=-1))
        iv = imagined_value.mean(dim=-1, keepdim=True)
        it = imagined_touch.mean(dim=-1, keepdim=True)
        x = torch.cat([workspace, preconscious_candidate, reflection, object_repr, memory_context, value_latent, iv, it], dim=-1)
        action_logits = self.action_logits(x)
        focus_logits = self.focus_logits(x)
        return {
            "value_latent": value_latent,
            "action_logits": action_logits,
            "action_ids": torch.argmax(action_logits, dim=-1),
            "focus_logits": focus_logits,
            "focus_idx": torch.argmax(focus_logits, dim=-1),
            "embodied_targets": self.embodied(x),
            "hand_ctrl": self.hand(x),
            "curiosity": self.curiosity(x),
            "coherence": self.coherence(x),
        }


class ConsciousDreamerMemoryThought(ConsciousDreamerCore):
    def __init__(self, cfg: ConsciousDreamerMemoryThoughtConfig) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        d = cfg.data
        l = cfg.latent
        c = cfg.conscious
        tm = cfg.thought_memory

        self.vision_encoder = VisionEncoder(d.image_channels * 2 + 1, l.vision_dim)
        self.pose_encoder = MLPEncoder(d.pose_dim, l.pose_dim)
        self.body_encoder = MLPEncoder(d.body_state_dim, l.body_dim)
        self.tactile_encoder = MLPEncoder(d.tactile_dim, l.tactile_dim)
        self.hand_motor_encoder = MLPEncoder(d.hand_motor_dim + d.embodied_dim, l.hand_motor_dim)
        self.object_state_encoder = MLPEncoder(9, l.object_state_dim)
        self.action_embed = nn.Embedding(d.action_dim, l.action_embed_dim)

        self.attention = AttentionController(
            dims=[l.vision_dim, l.pose_dim, l.body_dim, l.tactile_dim, l.hand_motor_dim, l.object_state_dim, l.action_embed_dim],
            modality_dim=l.modality_dim,
            heads=c.attention_heads,
            workspace_dim=c.workspace_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(c.workspace_dim + l.modality_dim, l.fused_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(l.fused_dim),
        )
        self.rssm = RSSMCore(l.fused_dim, l.rssm_dim)
        self.workspace = Workspace(l.rssm_dim, c.workspace_dim, c.workspace_dim, c.thought_dim, c.report_dim)
        self.body_context_model = BodyContextModel(l.rssm_dim, l.body_dim, l.tactile_dim, c.body_context_dim, c.model_reflection_dim)
        self.object_repr = ObjectRepresentation(c.workspace_dim, c.body_context_dim, l.tactile_dim, l.vision_dim, l.object_state_dim, c.object_repr_dim)
        self.preconscious_reflection_loop = PreconsciousReflectionLoop(c.workspace_dim, c.thought_dim, c.body_context_dim, c.object_repr_dim, c.model_reflection_dim)

        memory_query_dim = c.workspace_dim + c.thought_dim + c.body_context_dim + c.object_repr_dim
        episode_dim = c.workspace_dim + c.thought_dim + c.body_context_dim + c.object_repr_dim + l.tactile_dim + d.embodied_dim + d.hand_motor_dim + c.value_dim
        self.preconscious_memory = PreconsciousEpisodeMemory(memory_query_dim, episode_dim, tm.memory_dim, tm.memory_slots, tm.write_decay)
        self.memory = self.preconscious_memory

        self.thought_loop = PreconsciousThoughtLoop(c.workspace_dim, c.thought_dim, c.model_reflection_dim, c.object_repr_dim, tm.memory_dim, tm.thought_steps)
        self.imagination = ImaginationCore(l.rssm_dim, c.workspace_dim, c.thought_dim, c.object_repr_dim, l.tactile_dim, d.action_dim, c.imagination_horizon)
        self.planner = ConsciousPlanner(c.workspace_dim, c.thought_dim, c.model_reflection_dim, c.object_repr_dim, tm.memory_dim, c.value_dim, d.action_dim, d.embodied_dim, d.hand_motor_dim)
        self.decoder = DecoderHeads(l.rssm_dim, d.image_channels, d.image_height, d.image_width)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Dict[str, torch.Tensor]:
        device = torch.device(device)
        return {
            "rssm": torch.zeros(batch_size, self.cfg.latent.rssm_dim, device=device),
            "prev_action_ids": torch.zeros(batch_size, dtype=torch.long, device=device),
            "prev_embodied_action": torch.zeros(batch_size, self.cfg.data.embodied_dim, device=device),
            "prev_hand_motor": torch.zeros(batch_size, self.cfg.data.hand_motor_dim, device=device),
        }

    def _zeros(self, ref: torch.Tensor, *shape: int) -> torch.Tensor:
        return torch.zeros(*shape, device=ref.device, dtype=ref.dtype)

    def build_focus_context(
        self,
        *,
        workspace: torch.Tensor,
        preconscious_seed: torch.Tensor,
        preconscious_candidate: torch.Tensor,
        candidate_delta: torch.Tensor,
        body_context: torch.Tensor,
        model_reflection_context: torch.Tensor,
        reflection: torch.Tensor,
        object_repr: torch.Tensor,
        tactile_latent: torch.Tensor,
        memory_context: torch.Tensor,
        value_latent: torch.Tensor,
        curiosity: torch.Tensor,
        coherence: torch.Tensor,
        focus_logits: torch.Tensor,
        action_logits: torch.Tensor,
        modality_weights: torch.Tensor,
        imagined_value: torch.Tensor,
        imagined_touch: torch.Tensor,
        embodied_targets: torch.Tensor,
        hand_ctrl: torch.Tensor,
    ) -> torch.Tensor:
        """
        M5-owned focus field.

        M9 must not manually reconstruct focus from M5 internals. Downstream
        self-binding receives this single tensor as out["focus_context"]. M15 can
        later add/replace active thought-chain content by modifying this field at
        the M5 focus boundary rather than sending thought latents directly to M9.
        """
        parts = [
            workspace,
            preconscious_seed,
            preconscious_candidate,
            candidate_delta,
            body_context,
            model_reflection_context,
            reflection,
            object_repr,
            tactile_latent,
            memory_context,
            value_latent,
            curiosity,
            coherence,
            focus_logits,
            action_logits,
            modality_weights,
            imagined_value,
            imagined_touch,
            embodied_targets,
            hand_ctrl,
        ]
        flat_parts = []
        for part in parts:
            if torch.is_tensor(part):
                if part.ndim > 2:
                    part = part.reshape(part.shape[0], -1)
                flat_parts.append(part.float())
        if not flat_parts:
            return torch.zeros(workspace.shape[0], self.cfg.conscious.workspace_dim, device=workspace.device, dtype=workspace.dtype)
        return torch.cat(flat_parts, dim=-1)

    def step(self, left, right, pose, body_state, state, tactile=None, hand_motor=None, embodied_action=None, depth=None, object_state=None, action_override=None, write_memory: bool = True) -> Dict:
        b = left.shape[0]
        d = self.cfg.data

        tactile = tactile if tactile is not None else self._zeros(left, b, d.tactile_dim)
        hand_motor = hand_motor if hand_motor is not None else state.get("prev_hand_motor", self._zeros(left, b, d.hand_motor_dim))
        embodied_action = embodied_action if embodied_action is not None else state.get("prev_embodied_action", self._zeros(left, b, d.embodied_dim))

        if object_state is None:
            object_state = body_state[:, 6:15] if body_state.shape[-1] >= 15 else self._zeros(left, b, 9)

        action_ids_in = action_override.reshape(-1).long() if action_override is not None else state.get("prev_action_ids", torch.zeros(b, dtype=torch.long, device=left.device))

        vision = self.vision_encoder(left, right, depth)
        pose_latent = self.pose_encoder(pose)
        body_latent = self.body_encoder(body_state)
        tactile_latent = self.tactile_encoder(tactile)
        motor_latent = self.hand_motor_encoder(torch.cat([hand_motor, embodied_action], dim=-1))
        object_state_latent = self.object_state_encoder(object_state)
        action_latent = self.action_embed(action_ids_in)

        attn = self.attention([vision, pose_latent, body_latent, tactile_latent, motor_latent, object_state_latent, action_latent])
        fused = self.fusion(torch.cat([attn["workspace_seed"], attn["context"]], dim=-1))

        prev_rssm = state.get("rssm", self._zeros(left, b, self.cfg.latent.rssm_dim))
        rssm_out = self.rssm(fused, prev_rssm)
        rssm = rssm_out["state"]

        ws = self.workspace(rssm, attn["workspace_seed"])
        body_ctx = self.body_context_model(rssm, body_latent, tactile_latent)
        body_context = body_ctx["body_context"]
        model_reflection_context = body_ctx["model_reflection_context"]
        preconscious_seed = ws["preconscious_seed"]
        obj = self.object_repr(ws["workspace"], body_context, tactile_latent, vision, object_state_latent)

        refl0 = self.preconscious_reflection_loop(ws["workspace"], preconscious_seed, body_context, obj, model_reflection_context)

        mem_query = torch.cat([ws["workspace"], preconscious_seed, body_context, obj], dim=-1)
        mem_read = self.preconscious_memory.read(mem_query)
        memory_context = mem_read["memory_context"]

        preconscious_loop = self.thought_loop(ws["workspace"], preconscious_seed, refl0["reflection"], obj, memory_context)
        preconscious_candidate = preconscious_loop["candidate"]

        refl = self.preconscious_reflection_loop(ws["workspace"], preconscious_candidate, body_context, obj, refl0["reflection"])
        imagined = self.imagination(rssm, ws["workspace"], preconscious_candidate, obj, tactile_latent, action_ids_in)
        plan = self.planner(ws["workspace"], preconscious_candidate, refl["reflection"], obj, memory_context, imagined["imagined_value"], imagined["imagined_touch"])
        decoder = self.decoder(rssm, left.shape[-2:])

        focus_context = self.build_focus_context(
            workspace=ws["workspace"],
            preconscious_seed=preconscious_seed,
            preconscious_candidate=preconscious_candidate,
            candidate_delta=preconscious_loop["candidate_delta"],
            body_context=body_context,
            model_reflection_context=model_reflection_context,
            reflection=refl["reflection"],
            object_repr=obj,
            tactile_latent=tactile_latent,
            memory_context=memory_context,
            value_latent=plan["value_latent"],
            curiosity=plan["curiosity"],
            coherence=plan["coherence"],
            focus_logits=plan["focus_logits"],
            action_logits=plan["action_logits"],
            modality_weights=attn["modality_weights"],
            imagined_value=imagined["imagined_value"],
            imagined_touch=imagined["imagined_touch"],
            embodied_targets=plan["embodied_targets"],
            hand_ctrl=plan["hand_ctrl"],
        )

        episode = torch.cat([
            ws["workspace"].detach(),
            preconscious_candidate.detach(),
            body_context.detach(),
            obj.detach(),
            tactile_latent.detach(),
            embodied_action.detach(),
            hand_motor.detach(),
            plan["value_latent"].detach(),
        ], dim=-1)
        if self.cfg.thought_memory.use_memory_write and write_memory:
            self.preconscious_memory.write(episode)

        next_state = {
            "rssm": rssm.detach(),
            "prev_action_ids": plan["action_ids"].detach(),
            "prev_embodied_action": plan["embodied_targets"].detach(),
            "prev_hand_motor": plan["hand_ctrl"].detach(),
        }

        return {
            "state": next_state,
            "obs_embed": fused,
            "rssm": rssm,
            "prior": rssm_out["prior"],
            "posterior": rssm_out["posterior"],
            "attention": {
                "tokens": attn["tokens"],
                "modality_weights": attn["modality_weights"],
                "attn_matrix": attn["attn_matrix"],
            },
            "preconscious_memory": {
                "memory_context": memory_context,
                "memory_weights": mem_read["memory_weights"],
                "memory_usage": self.preconscious_memory.usage.detach().clone(),
            },
            "memory": {
                "memory_context": memory_context,
                "memory_weights": mem_read["memory_weights"],
                "memory_usage": self.preconscious_memory.usage.detach().clone(),
            },
            "workspace_out": ws["workspace"],
            "focus_context": focus_context,
            "preconscious_thoughts": {
                "thought_candidate": preconscious_candidate,
                "workspace_seed": preconscious_seed,
                "candidate_sequence": preconscious_loop["candidate_sequence"],
                "candidate_delta": preconscious_loop["candidate_delta"],
            },
            "preconscious_report": ws["report"],
            "body_context": {
                "body_context": body_context,
                "model_reflection_context": model_reflection_context,
            },
            "preconscious_reflection_out": {
                "reflection": refl["reflection"],
                "initial_reflection": refl0["reflection"],
                "model_confidence": refl["model_confidence"],
            },
            "object_repr": obj,
            "tactile_latent": tactile_latent,
            "values": {
                "value_latent": plan["value_latent"],
                "curiosity": plan["curiosity"],
                "coherence": plan["coherence"],
            },
            "focus": {
                "focus_logits": plan["focus_logits"],
                "focus_idx": plan["focus_idx"],
                "attention_focus_logits": attn["focus_logits"],
                "attention_focus_idx": attn["focus_idx"],
            },
            "action_logits": plan["action_logits"],
            "action_ids": plan["action_ids"],
            "embodied_targets": plan["embodied_targets"],
            "hand_ctrl": plan["hand_ctrl"],
            "imagined": imagined,
            "decoder": decoder,
        }

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)


def make_memory_thought_config_from_world(image_height=128, image_width=192, body_state_dim=None, tactile_dim=None, hand_motor_dim=None, embodied_dim=None, action_dim=None):
    cfg = ConsciousDreamerMemoryThoughtConfig()
    required_dims = {
        "body_state_dim": body_state_dim,
        "tactile_dim": tactile_dim,
        "hand_motor_dim": hand_motor_dim,
        "embodied_dim": embodied_dim,
        "action_dim": action_dim,
    }
    missing = [k for k, v in required_dims.items() if v is None]
    if missing:
        raise ValueError(
            "make_memory_thought_config_from_world() does not own model dimensions. "
            "Read them from runner.yaml / UnifiedConfig and pass them explicitly. "
            f"Missing: {missing}"
        )
    cfg.data.image_height = image_height
    cfg.data.image_width = image_width
    cfg.data.body_state_dim = body_state_dim
    cfg.data.tactile_dim = tactile_dim
    cfg.data.hand_motor_dim = hand_motor_dim
    cfg.data.embodied_dim = embodied_dim
    cfg.data.action_dim = action_dim
    return cfg

__all__ = [
    "PreconsciousThoughtLoop",
    "ThoughtLoop",
    "PreconsciousEpisodeMemory",
    "AutobiographicalMemory",
    "ConsciousDreamerMemoryThought",
    "ConsciousDreamerMemoryThoughtConfig",
    "make_memory_thought_config_from_world",
]
