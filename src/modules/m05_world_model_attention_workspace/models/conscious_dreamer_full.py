from __future__ import annotations

"""
conscious_dreamer_full.py

Base M5 world-model stack for the embodied MuJoCo project.

M5 is intentionally preconscious:
- separate multimodal encoders
- AttentionController
- RSSM world state
- workspace and preconscious thought seed
- body context, not SelfCore
- model reflection context, not self-awareness
- object representation
- embodied/base/arm action prediction
- realistic hand motor prediction
- stable output dictionary contract

True self-binding is owned by M9 SelfCore. True inner speech is owned by M7
and should be created only after M9 binds focused content to the body/self model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DreamerDataConfig:
    image_channels: int = 3
    image_height: int = 128
    image_width: int = 192
    pose_dim: int = 7
    body_state_dim: int = 49
    tactile_dim: int = 42
    hand_motor_dim: int = 44
    embodied_dim: int = 11
    action_dim: int = 24


@dataclass
class DreamerLatentConfig:
    vision_dim: int = 256
    pose_dim: int = 64
    body_dim: int = 128
    tactile_dim: int = 128
    hand_motor_dim: int = 128
    object_state_dim: int = 128
    action_embed_dim: int = 64
    modality_dim: int = 192
    fused_dim: int = 512
    rssm_dim: int = 512


@dataclass
class ConsciousConfig:
    workspace_dim: int = 256
    body_context_dim: int = 192
    model_reflection_dim: int = 192
    thought_dim: int = 192
    value_dim: int = 128
    report_dim: int = 128
    object_repr_dim: int = 128
    plan_dim: int = 192
    imagination_horizon: int = 5
    attention_heads: int = 4

    @property
    def body_self_dim(self) -> int:
        return self.body_context_dim

    @body_self_dim.setter
    def body_self_dim(self, value: int) -> None:
        self.body_context_dim = int(value)

    @property
    def reflective_self_dim(self) -> int:
        return self.model_reflection_dim

    @reflective_self_dim.setter
    def reflective_self_dim(self, value: int) -> None:
        self.model_reflection_dim = int(value)


@dataclass
class ConsciousDreamerCoreConfig:
    data: DreamerDataConfig = field(default_factory=DreamerDataConfig)
    latent: DreamerLatentConfig = field(default_factory=DreamerLatentConfig)
    conscious: ConsciousConfig = field(default_factory=ConsciousConfig)


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int = 7, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
            nn.Linear(128 * 4 * 6, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if depth is None:
            depth = torch.zeros(left.shape[0], 1, left.shape[-2], left.shape[-1], device=left.device, dtype=left.dtype)
        if depth.shape[-2:] != left.shape[-2:]:
            depth = F.interpolate(depth, size=left.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([left, right, depth], dim=1))


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionController(nn.Module):
    modality_names = ["vision", "pose", "body", "tactile", "hand_motor", "object", "action"]

    def __init__(self, dims: list[int], modality_dim: int = 192, heads: int = 4, workspace_dim: int = 256) -> None:
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, modality_dim), nn.LayerNorm(modality_dim), nn.ReLU(inplace=True))
            for d in dims
        ])
        self.modality_attn = nn.MultiheadAttention(modality_dim, heads, batch_first=True)
        self.score = nn.Linear(modality_dim, 1)
        self.to_workspace_seed = nn.Sequential(nn.Linear(modality_dim, workspace_dim), nn.ReLU(inplace=True), nn.LayerNorm(workspace_dim))
        self.focus_head = nn.Linear(modality_dim, 8)

    def forward(self, modalities: list[torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = torch.stack([proj(x) for proj, x in zip(self.projectors, modalities)], dim=1)
        attended, attn_matrix = self.modality_attn(tokens, tokens, tokens, need_weights=True)
        weights = torch.softmax(self.score(attended).squeeze(-1), dim=-1)
        context = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        focus_logits = self.focus_head(context)
        return {
            "tokens": attended,
            "context": context,
            "workspace_seed": self.to_workspace_seed(context),
            "modality_weights": weights,
            "attn_matrix": attn_matrix,
            "focus_logits": focus_logits,
            "focus_idx": torch.argmax(focus_logits, dim=-1),
        }


class RSSMCore(nn.Module):
    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_dim, state_dim)
        self.post = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.ReLU(inplace=True), nn.Linear(state_dim, state_dim))
        self.prior = nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(inplace=True), nn.Linear(state_dim, state_dim))

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.gru(x, prev_state)
        post = torch.tanh(self.post(torch.cat([x, h], dim=-1)))
        prior = torch.tanh(self.prior(prev_state))
        return {"state": torch.tanh(h + 0.25 * post), "prior": prior, "posterior": post}


class Workspace(nn.Module):
    def __init__(self, rssm_dim: int, workspace_seed_dim: int, workspace_dim: int, thought_dim: int, report_dim: int) -> None:
        super().__init__()
        self.workspace = nn.Sequential(nn.Linear(rssm_dim + workspace_seed_dim, workspace_dim), nn.ReLU(inplace=True), nn.LayerNorm(workspace_dim))
        self.preconscious_seed = nn.Sequential(nn.Linear(workspace_dim, thought_dim), nn.ReLU(inplace=True), nn.LayerNorm(thought_dim))
        self.report = nn.Sequential(nn.Linear(workspace_dim + thought_dim, report_dim), nn.Tanh())

    def forward(self, rssm: torch.Tensor, workspace_seed: torch.Tensor) -> Dict[str, torch.Tensor]:
        w = self.workspace(torch.cat([rssm, workspace_seed], dim=-1))
        seed = self.preconscious_seed(w)
        return {"workspace": w, "preconscious_seed": seed, "report": self.report(torch.cat([w, seed], dim=-1))}


class BodyContextModel(nn.Module):
    def __init__(self, rssm_dim: int, body_latent_dim: int, tactile_dim: int, body_context_dim: int, model_reflection_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(rssm_dim + body_latent_dim + tactile_dim, body_context_dim), nn.ReLU(inplace=True), nn.LayerNorm(body_context_dim))
        self.model_reflection = nn.Sequential(nn.Linear(rssm_dim + body_context_dim, model_reflection_dim), nn.ReLU(inplace=True), nn.LayerNorm(model_reflection_dim))

    def forward(self, rssm: torch.Tensor, body_latent: torch.Tensor, tactile_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        body_context = self.body(torch.cat([rssm, body_latent, tactile_latent], dim=-1))
        model_reflection_context = self.model_reflection(torch.cat([rssm, body_context], dim=-1))
        return {"body_context": body_context, "model_reflection_context": model_reflection_context}


class ObjectRepresentation(nn.Module):
    def __init__(self, workspace_dim: int, body_context_dim: int, tactile_dim: int, vision_dim: int, object_state_dim: int, object_repr_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + body_context_dim + tactile_dim + vision_dim + object_state_dim, 320),
            nn.ReLU(inplace=True), nn.Linear(320, object_repr_dim), nn.LayerNorm(object_repr_dim), nn.ReLU(inplace=True),
        )

    def forward(self, workspace, body_context, tactile_latent, vision_latent, object_state_latent):
        return self.net(torch.cat([workspace, body_context, tactile_latent, vision_latent, object_state_latent], dim=-1))


class PreconsciousReflectionLoop(nn.Module):
    def __init__(self, workspace_dim: int, thought_dim: int, body_dim: int, object_dim: int, model_reflection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + thought_dim + body_dim + object_dim + model_reflection_dim, 320),
            nn.ReLU(inplace=True), nn.Linear(320, model_reflection_dim), nn.LayerNorm(model_reflection_dim), nn.ReLU(inplace=True),
        )
        self.confidence = nn.Sequential(nn.Linear(model_reflection_dim, 1), nn.Sigmoid())

    def forward(self, workspace, preconscious_seed, body_context, object_repr, model_reflection_context):
        reflection = self.net(torch.cat([workspace, preconscious_seed, body_context, object_repr, model_reflection_context], dim=-1))
        return {"reflection": reflection, "model_confidence": self.confidence(reflection)}


ReflectiveLoop = PreconsciousReflectionLoop


class ImaginationCore(nn.Module):
    def __init__(self, rssm_dim: int, workspace_dim: int, thought_dim: int, object_dim: int, tactile_dim: int, action_dim: int, horizon: int = 5) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_embed = nn.Embedding(action_dim, 64)
        self.cell = nn.GRUCell(workspace_dim + thought_dim + object_dim + tactile_dim + 64, rssm_dim)
        self.value = nn.Sequential(nn.Linear(rssm_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.touch_pred = nn.Sequential(nn.Linear(rssm_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, rssm, workspace, preconscious_candidate, object_repr, tactile_latent, action_ids):
        h = rssm
        action_emb = self.action_embed(action_ids.long().reshape(-1))
        x = torch.cat([workspace, preconscious_candidate, object_repr, tactile_latent, action_emb], dim=-1)
        states, values, touches = [], [], []
        for _ in range(self.horizon):
            h = self.cell(x, h)
            states.append(h)
            values.append(self.value(h))
            touches.append(self.touch_pred(h))
        return {"imagined_states": torch.stack(states, dim=1), "imagined_value": torch.cat(values, dim=-1), "imagined_touch": torch.cat(touches, dim=-1)}


class ConsciousPlanner(nn.Module):
    def __init__(self, workspace_dim: int, thought_dim: int, model_reflection_dim: int, object_dim: int, value_dim: int, action_dim: int, embodied_dim: int, hand_motor_dim: int) -> None:
        super().__init__()
        planner_in = workspace_dim + thought_dim + model_reflection_dim + object_dim + value_dim + 2
        self.value_latent = nn.Sequential(nn.Linear(workspace_dim + thought_dim + object_dim, value_dim), nn.ReLU(inplace=True), nn.LayerNorm(value_dim))
        self.action_logits = nn.Linear(planner_in, action_dim)
        self.focus_logits = nn.Linear(planner_in, 8)
        self.embodied = nn.Sequential(nn.Linear(planner_in, 256), nn.ReLU(inplace=True), nn.Linear(256, embodied_dim), nn.Tanh())
        self.hand = nn.Sequential(nn.Linear(planner_in, 256), nn.ReLU(inplace=True), nn.Linear(256, hand_motor_dim), nn.Sigmoid())
        self.curiosity = nn.Sequential(nn.Linear(planner_in, 1), nn.Sigmoid())
        self.coherence = nn.Sequential(nn.Linear(planner_in, 1), nn.Sigmoid())

    def forward(self, workspace, preconscious_candidate, model_reflection, object_repr, imagined_value, imagined_touch):
        value_latent = self.value_latent(torch.cat([workspace, preconscious_candidate, object_repr], dim=-1))
        x = torch.cat([workspace, preconscious_candidate, model_reflection, object_repr, value_latent, imagined_value.mean(dim=-1, keepdim=True), imagined_touch.mean(dim=-1, keepdim=True)], dim=-1)
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


class DecoderHeads(nn.Module):
    def __init__(self, rssm_dim: int, image_channels: int, h: int, w: int) -> None:
        super().__init__()
        bh, bw = max(4, h // 16), max(4, w // 16)
        self.bh, self.bw = bh, bw
        self.fc = nn.Sequential(nn.Linear(rssm_dim, 128 * bh * bw), nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.rgb = nn.Conv2d(32, image_channels, 3, padding=1)
        self.depth = nn.Conv2d(32, 1, 3, padding=1)
        self.reward = nn.Linear(rssm_dim, 1)
        self.cont = nn.Sequential(nn.Linear(rssm_dim, 1), nn.Sigmoid())

    def forward(self, rssm: torch.Tensor, target_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        b = rssm.shape[0]
        x = self.deconv(self.fc(rssm).view(b, 128, self.bh, self.bw))
        rgb = torch.sigmoid(self.rgb(x))
        depth = F.softplus(self.depth(x))
        if rgb.shape[-2:] != target_hw:
            rgb = F.interpolate(rgb, size=target_hw, mode="bilinear", align_corners=False)
            depth = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=False)
        return {"rgb": rgb, "depth": depth, "reward": self.reward(rssm), "continue": self.cont(rssm)}


class ConsciousDreamerCore(nn.Module):
    def __init__(self, cfg: ConsciousDreamerCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, l, c = cfg.data, cfg.latent, cfg.conscious
        self.vision_encoder = VisionEncoder(d.image_channels * 2 + 1, l.vision_dim)
        self.pose_encoder = MLPEncoder(d.pose_dim, l.pose_dim)
        self.body_encoder = MLPEncoder(d.body_state_dim, l.body_dim)
        self.tactile_encoder = MLPEncoder(d.tactile_dim, l.tactile_dim)
        self.hand_motor_encoder = MLPEncoder(d.hand_motor_dim + d.embodied_dim, l.hand_motor_dim)
        self.object_state_encoder = MLPEncoder(9, l.object_state_dim)
        self.action_embed = nn.Embedding(d.action_dim, l.action_embed_dim)
        self.attention = AttentionController([l.vision_dim, l.pose_dim, l.body_dim, l.tactile_dim, l.hand_motor_dim, l.object_state_dim, l.action_embed_dim], l.modality_dim, c.attention_heads, c.workspace_dim)
        self.fusion = nn.Sequential(nn.Linear(c.workspace_dim + l.modality_dim, l.fused_dim), nn.ReLU(inplace=True), nn.LayerNorm(l.fused_dim))
        self.rssm = RSSMCore(l.fused_dim, l.rssm_dim)
        self.workspace = Workspace(l.rssm_dim, c.workspace_dim, c.workspace_dim, c.thought_dim, c.report_dim)
        self.body_context_model = BodyContextModel(l.rssm_dim, l.body_dim, l.tactile_dim, c.body_context_dim, c.model_reflection_dim)
        self.object_repr = ObjectRepresentation(c.workspace_dim, c.body_context_dim, l.tactile_dim, l.vision_dim, l.object_state_dim, c.object_repr_dim)
        self.preconscious_reflection_loop = PreconsciousReflectionLoop(c.workspace_dim, c.thought_dim, c.body_context_dim, c.object_repr_dim, c.model_reflection_dim)
        self.imagination = ImaginationCore(l.rssm_dim, c.workspace_dim, c.thought_dim, c.object_repr_dim, l.tactile_dim, d.action_dim, c.imagination_horizon)
        self.planner = ConsciousPlanner(c.workspace_dim, c.thought_dim, c.model_reflection_dim, c.object_repr_dim, c.value_dim, d.action_dim, d.embodied_dim, d.hand_motor_dim)
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

    def step(self, left: torch.Tensor, right: torch.Tensor, pose: torch.Tensor, body_state: torch.Tensor, state: Dict[str, torch.Tensor], tactile: Optional[torch.Tensor] = None, hand_motor: Optional[torch.Tensor] = None, embodied_action: Optional[torch.Tensor] = None, depth: Optional[torch.Tensor] = None, object_state: Optional[torch.Tensor] = None, action_override: Optional[torch.Tensor] = None) -> Dict:
        b, d = left.shape[0], self.cfg.data
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
        rssm_out = self.rssm(fused, state.get("rssm", self._zeros(left, b, self.cfg.latent.rssm_dim)))
        rssm = rssm_out["state"]
        ws = self.workspace(rssm, attn["workspace_seed"])
        body_ctx = self.body_context_model(rssm, body_latent, tactile_latent)
        preconscious_seed = ws["preconscious_seed"]
        obj = self.object_repr(ws["workspace"], body_ctx["body_context"], tactile_latent, vision, object_state_latent)
        refl = self.preconscious_reflection_loop(ws["workspace"], preconscious_seed, body_ctx["body_context"], obj, body_ctx["model_reflection_context"])
        imagined = self.imagination(rssm, ws["workspace"], preconscious_seed, obj, tactile_latent, action_ids_in)
        plan = self.planner(ws["workspace"], preconscious_seed, refl["reflection"], obj, imagined["imagined_value"], imagined["imagined_touch"])
        decoder = self.decoder(rssm, left.shape[-2:])
        next_state = {"rssm": rssm.detach(), "prev_action_ids": plan["action_ids"].detach(), "prev_embodied_action": plan["embodied_targets"].detach(), "prev_hand_motor": plan["hand_ctrl"].detach()}
        return {
            "state": next_state,
            "obs_embed": fused,
            "rssm": rssm,
            "prior": rssm_out["prior"],
            "posterior": rssm_out["posterior"],
            "attention": {"tokens": attn["tokens"], "modality_weights": attn["modality_weights"], "attn_matrix": attn["attn_matrix"]},
            "workspace_out": ws["workspace"],
            "preconscious_thoughts": {"thought_candidate": preconscious_seed, "workspace_seed": preconscious_seed},
            "preconscious_report": ws["report"],
            "body_context": body_ctx,
            "preconscious_reflection_out": {"reflection": refl["reflection"], "model_confidence": refl["model_confidence"]},
            "object_repr": obj,
            "tactile_latent": tactile_latent,
            "values": {"value_latent": plan["value_latent"], "curiosity": plan["curiosity"], "coherence": plan["coherence"]},
            "focus": {"focus_logits": plan["focus_logits"], "focus_idx": plan["focus_idx"], "attention_focus_logits": attn["focus_logits"], "attention_focus_idx": attn["focus_idx"]},
            "action_logits": plan["action_logits"], "action_ids": plan["action_ids"], "embodied_targets": plan["embodied_targets"], "hand_ctrl": plan["hand_ctrl"],
            "imagined": imagined, "decoder": decoder,
        }

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)


def make_core_config_from_world(image_height: int = 128, image_width: int = 192, body_state_dim: int = 49, tactile_dim: int = 42, hand_motor_dim: int = 44, embodied_dim: int = 11, action_dim: int = 24) -> ConsciousDreamerCoreConfig:
    cfg = ConsciousDreamerCoreConfig()
    cfg.data.image_height = image_height
    cfg.data.image_width = image_width
    cfg.data.body_state_dim = body_state_dim
    cfg.data.tactile_dim = tactile_dim
    cfg.data.hand_motor_dim = hand_motor_dim
    cfg.data.embodied_dim = embodied_dim
    cfg.data.action_dim = action_dim
    return cfg

__all__ = [
    "DreamerDataConfig",
    "DreamerLatentConfig",
    "ConsciousConfig",
    "ConsciousDreamerCore",
    "ConsciousDreamerCoreConfig",
    "VisionEncoder",
    "MLPEncoder",
    "AttentionController",
    "RSSMCore",
    "Workspace",
    "BodyContextModel",
    "ObjectRepresentation",
    "PreconsciousReflectionLoop",
    "ReflectiveLoop",
    "ImaginationCore",
    "ConsciousPlanner",
    "DecoderHeads",
    "make_core_config_from_world",
]


if __name__ == "__main__":
    cfg = make_core_config_from_world()
    model = ConsciousDreamerCore(cfg)
    state = model.initial_state(1, "cpu")
    out = model.step(
        torch.zeros(1, 3, 128, 192), torch.zeros(1, 3, 128, 192), torch.zeros(1, 7), torch.zeros(1, cfg.data.body_state_dim), state,
        tactile=torch.zeros(1, cfg.data.tactile_dim), hand_motor=torch.zeros(1, cfg.data.hand_motor_dim), embodied_action=torch.zeros(1, cfg.data.embodied_dim),
    )
    print("workspace", out["workspace_out"].shape)
    print("attention weights", out["attention"]["modality_weights"].shape)
    print("object_repr", out["object_repr"].shape)
    print("imagined_value", out["imagined"]["imagined_value"].shape)
    print("embodied", out["embodied_targets"].shape)
    print("hand_ctrl", out["hand_ctrl"].shape)
