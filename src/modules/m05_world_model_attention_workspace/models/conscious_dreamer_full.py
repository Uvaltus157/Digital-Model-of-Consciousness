from __future__ import annotations

"""
conscious_dreamer_full.py

Full ConsciousDreamerV2 stack for the embodied MuJoCo project.

Includes:
- separate multimodal encoders
- AttentionControllerV2
- ConsciousPlannerV2
- ImaginationCoreV2
- ReflectiveLoopV2
- object representation
- embodied/base/arm action prediction
- realistic hand motor prediction
- V1-like output dictionary compatibility
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DreamerV2DataConfig:
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
class DreamerV2LatentConfig:
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
class ConsciousV2Config:
    workspace_dim: int = 256
    body_self_dim: int = 192
    reflective_self_dim: int = 192
    thought_dim: int = 192
    value_dim: int = 128
    report_dim: int = 128
    object_repr_dim: int = 128
    plan_dim: int = 192
    imagination_horizon: int = 5
    attention_heads: int = 4


@dataclass
class ConsciousDreamerV2Config:
    data: DreamerV2DataConfig = field(default_factory=DreamerV2DataConfig)
    latent: DreamerV2LatentConfig = field(default_factory=DreamerV2LatentConfig)
    conscious: ConsciousV2Config = field(default_factory=ConsciousV2Config)


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


class AttentionControllerV2(nn.Module):
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


class RSSMCoreV2(nn.Module):
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


class WorkspaceV2(nn.Module):
    def __init__(self, rssm_dim: int, workspace_seed_dim: int, workspace_dim: int, thought_dim: int, report_dim: int) -> None:
        super().__init__()
        self.workspace = nn.Sequential(nn.Linear(rssm_dim + workspace_seed_dim, workspace_dim), nn.ReLU(inplace=True), nn.LayerNorm(workspace_dim))
        self.thought = nn.Sequential(nn.Linear(workspace_dim, thought_dim), nn.ReLU(inplace=True), nn.LayerNorm(thought_dim))
        self.report = nn.Sequential(nn.Linear(workspace_dim + thought_dim, report_dim), nn.Tanh())

    def forward(self, rssm: torch.Tensor, workspace_seed: torch.Tensor) -> Dict[str, torch.Tensor]:
        w = self.workspace(torch.cat([rssm, workspace_seed], dim=-1))
        thought = self.thought(w)
        return {"workspace": w, "thought": thought, "report": self.report(torch.cat([w, thought], dim=-1))}


class SelfModelV2(nn.Module):
    def __init__(self, rssm_dim: int, body_latent_dim: int, tactile_dim: int, body_self_dim: int, reflective_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(rssm_dim + body_latent_dim + tactile_dim, body_self_dim), nn.ReLU(inplace=True), nn.LayerNorm(body_self_dim))
        self.reflective = nn.Sequential(nn.Linear(rssm_dim + body_self_dim, reflective_dim), nn.ReLU(inplace=True), nn.LayerNorm(reflective_dim))

    def forward(self, rssm: torch.Tensor, body_latent: torch.Tensor, tactile_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        body_self = self.body(torch.cat([rssm, body_latent, tactile_latent], dim=-1))
        reflective = self.reflective(torch.cat([rssm, body_self], dim=-1))
        return {"body_self": body_self, "reflective_self": reflective}


class ObjectRepresentationV2(nn.Module):
    def __init__(self, workspace_dim: int, body_self_dim: int, tactile_dim: int, vision_dim: int, object_state_dim: int, object_repr_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + body_self_dim + tactile_dim + vision_dim + object_state_dim, 320),
            nn.ReLU(inplace=True), nn.Linear(320, object_repr_dim), nn.LayerNorm(object_repr_dim), nn.ReLU(inplace=True),
        )

    def forward(self, workspace, body_self, tactile_latent, vision_latent, object_state_latent):
        return self.net(torch.cat([workspace, body_self, tactile_latent, vision_latent, object_state_latent], dim=-1))


class ReflectiveLoopV2(nn.Module):
    def __init__(self, workspace_dim: int, thought_dim: int, body_dim: int, object_dim: int, reflective_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + thought_dim + body_dim + object_dim + reflective_dim, 320),
            nn.ReLU(inplace=True), nn.Linear(320, reflective_dim), nn.LayerNorm(reflective_dim), nn.ReLU(inplace=True),
        )
        self.confidence = nn.Sequential(nn.Linear(reflective_dim, 1), nn.Sigmoid())

    def forward(self, workspace, thought, body_self, object_repr, reflective_self):
        reflection = self.net(torch.cat([workspace, thought, body_self, object_repr, reflective_self], dim=-1))
        return {"reflection": reflection, "self_confidence": self.confidence(reflection)}


class ImaginationCoreV2(nn.Module):
    def __init__(self, rssm_dim: int, workspace_dim: int, thought_dim: int, object_dim: int, tactile_dim: int, action_dim: int, horizon: int = 5) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_embed = nn.Embedding(action_dim, 64)
        self.cell = nn.GRUCell(workspace_dim + thought_dim + object_dim + tactile_dim + 64, rssm_dim)
        self.value = nn.Sequential(nn.Linear(rssm_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.touch_pred = nn.Sequential(nn.Linear(rssm_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, rssm, workspace, thought, object_repr, tactile_latent, action_ids):
        h = rssm
        action_emb = self.action_embed(action_ids.long().reshape(-1))
        x = torch.cat([workspace, thought, object_repr, tactile_latent, action_emb], dim=-1)
        states, values, touches = [], [], []
        for _ in range(self.horizon):
            h = self.cell(x, h)
            states.append(h)
            values.append(self.value(h))
            touches.append(self.touch_pred(h))
        return {"imagined_states": torch.stack(states, dim=1), "imagined_value": torch.cat(values, dim=-1), "imagined_touch": torch.cat(touches, dim=-1)}


class ConsciousPlannerV2(nn.Module):
    def __init__(self, workspace_dim: int, thought_dim: int, reflective_dim: int, object_dim: int, value_dim: int, action_dim: int, embodied_dim: int, hand_motor_dim: int) -> None:
        super().__init__()
        planner_in = workspace_dim + thought_dim + reflective_dim + object_dim + value_dim + 2
        self.value_latent = nn.Sequential(nn.Linear(workspace_dim + thought_dim + object_dim, value_dim), nn.ReLU(inplace=True), nn.LayerNorm(value_dim))
        self.action_logits = nn.Linear(planner_in, action_dim)
        self.focus_logits = nn.Linear(planner_in, 8)
        self.embodied = nn.Sequential(nn.Linear(planner_in, 256), nn.ReLU(inplace=True), nn.Linear(256, embodied_dim), nn.Tanh())
        self.hand = nn.Sequential(nn.Linear(planner_in, 256), nn.ReLU(inplace=True), nn.Linear(256, hand_motor_dim), nn.Sigmoid())
        self.curiosity = nn.Sequential(nn.Linear(planner_in, 1), nn.Sigmoid())
        self.coherence = nn.Sequential(nn.Linear(planner_in, 1), nn.Sigmoid())

    def forward(self, workspace, thought, reflection, object_repr, imagined_value, imagined_touch):
        value_latent = self.value_latent(torch.cat([workspace, thought, object_repr], dim=-1))
        x = torch.cat([workspace, thought, reflection, object_repr, value_latent, imagined_value.mean(dim=-1, keepdim=True), imagined_touch.mean(dim=-1, keepdim=True)], dim=-1)
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


class DecoderHeadsV2(nn.Module):
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


class ConsciousDreamerV2Full(nn.Module):
    def __init__(self, cfg: ConsciousDreamerV2Config) -> None:
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
        self.attention = AttentionControllerV2([l.vision_dim, l.pose_dim, l.body_dim, l.tactile_dim, l.hand_motor_dim, l.object_state_dim, l.action_embed_dim], l.modality_dim, c.attention_heads, c.workspace_dim)
        self.fusion = nn.Sequential(nn.Linear(c.workspace_dim + l.modality_dim, l.fused_dim), nn.ReLU(inplace=True), nn.LayerNorm(l.fused_dim))
        self.rssm = RSSMCoreV2(l.fused_dim, l.rssm_dim)
        self.workspace = WorkspaceV2(l.rssm_dim, c.workspace_dim, c.workspace_dim, c.thought_dim, c.report_dim)
        self.self_model = SelfModelV2(l.rssm_dim, l.body_dim, l.tactile_dim, c.body_self_dim, c.reflective_self_dim)
        self.object_repr = ObjectRepresentationV2(c.workspace_dim, c.body_self_dim, l.tactile_dim, l.vision_dim, l.object_state_dim, c.object_repr_dim)
        self.reflective_loop = ReflectiveLoopV2(c.workspace_dim, c.thought_dim, c.body_self_dim, c.object_repr_dim, c.reflective_self_dim)
        self.imagination = ImaginationCoreV2(l.rssm_dim, c.workspace_dim, c.thought_dim, c.object_repr_dim, l.tactile_dim, d.action_dim, c.imagination_horizon)
        self.planner = ConsciousPlannerV2(c.workspace_dim, c.thought_dim, c.reflective_self_dim, c.object_repr_dim, c.value_dim, d.action_dim, d.embodied_dim, d.hand_motor_dim)
        self.decoder = DecoderHeadsV2(l.rssm_dim, d.image_channels, d.image_height, d.image_width)

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
        selves = self.self_model(rssm, body_latent, tactile_latent)
        obj = self.object_repr(ws["workspace"], selves["body_self"], tactile_latent, vision, object_state_latent)
        refl = self.reflective_loop(ws["workspace"], ws["thought"], selves["body_self"], obj, selves["reflective_self"])
        imagined = self.imagination(rssm, ws["workspace"], ws["thought"], obj, tactile_latent, action_ids_in)
        plan = self.planner(ws["workspace"], ws["thought"], refl["reflection"], obj, imagined["imagined_value"], imagined["imagined_touch"])
        decoder = self.decoder(rssm, left.shape[-2:])
        next_state = {"rssm": rssm.detach(), "prev_action_ids": plan["action_ids"].detach(), "prev_embodied_action": plan["embodied_targets"].detach(), "prev_hand_motor": plan["hand_ctrl"].detach()}
        return {
            "state": next_state, "obs_embed": fused, "rssm": rssm, "prior": rssm_out["prior"], "posterior": rssm_out["posterior"],
            "attention": {"tokens": attn["tokens"], "modality_weights": attn["modality_weights"], "attn_matrix": attn["attn_matrix"]},
            "workspace_out": ws["workspace"], "thoughts": {"thought": ws["thought"]}, "report": ws["report"],
            "selves": {"body_self": selves["body_self"], "reflective_self": selves["reflective_self"]},
            "reflection_out": {"reflection": refl["reflection"], "self_confidence": refl["self_confidence"]},
            "object_repr": obj, "tactile_latent": tactile_latent,
            "values": {"value_latent": plan["value_latent"], "curiosity": plan["curiosity"], "coherence": plan["coherence"]},
            "focus": {"focus_logits": plan["focus_logits"], "focus_idx": plan["focus_idx"], "attention_focus_logits": attn["focus_logits"], "attention_focus_idx": attn["focus_idx"]},
            "action_logits": plan["action_logits"], "action_ids": plan["action_ids"], "embodied_targets": plan["embodied_targets"], "hand_ctrl": plan["hand_ctrl"],
            "imagined": imagined, "decoder": decoder,
        }

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)


def make_v2_full_config_from_world(image_height: int = 128, image_width: int = 192, body_state_dim: int = 49, tactile_dim: int = 42, hand_motor_dim: int = 44, embodied_dim: int = 11, action_dim: int = 24) -> ConsciousDreamerV2Config:
    cfg = ConsciousDreamerV2Config()
    cfg.data.image_height = image_height
    cfg.data.image_width = image_width
    cfg.data.body_state_dim = body_state_dim
    cfg.data.tactile_dim = tactile_dim
    cfg.data.hand_motor_dim = hand_motor_dim
    cfg.data.embodied_dim = embodied_dim
    cfg.data.action_dim = action_dim
    return cfg


ConsciousDreamerV2 = ConsciousDreamerV2Full


if __name__ == "__main__":
    cfg = make_v2_full_config_from_world()
    model = ConsciousDreamerV2Full(cfg)
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
