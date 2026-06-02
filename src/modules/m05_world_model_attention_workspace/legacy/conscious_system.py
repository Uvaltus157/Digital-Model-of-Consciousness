from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


# ============================================================
# Config
# ============================================================
@dataclass
class DataConfig:
    train_dir: str = "dataset_dynamic_self/train"
    val_dir: str = "dataset_dynamic_self/val"
    seq_len: int = 16
    image_size: Tuple[int, int] = (128, 192)
    pose_dim: int = 7
    hand_state_dim: int = 8
    action_dim: int = 24
    num_objects: int = 8


@dataclass
class ModelConfig:
    obs_dim: int = 256
    world_dim: int = 192
    body_self_dim: int = 96
    agent_self_dim: int = 128
    reflective_self_dim: int = 96
    narrative_self_dim: int = 96
    workspace_dim: int = 192
    thought_dim: int = 160
    goal_dim: int = 64
    value_dim: int = 48
    memory_dim: int = 160
    report_dim: int = 64
    num_world_slots: int = 8
    num_action_bins: int = 24
    num_counterfactuals: int = 6
    num_thought_steps: int = 4
    render_size: Tuple[int, int] = (64, 96)
    num_types: int = 4


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class ConsciousSystemConfig:
    mode: str = "print"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ============================================================
# Helpers
# ============================================================
def mean_pool_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1) if x.ndim == 3 else x


# ============================================================
# Perception and world model
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class PerceptionSystem(nn.Module):
    def __init__(self, obs_dim: int, pose_dim: int) -> None:
        super().__init__()
        img_dim = obs_dim // 2
        self.cnn = SmallCNN(img_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(img_dim * 2 + 64, obs_dim), nn.ReLU(inplace=True),
            nn.Linear(obs_dim, obs_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        lf = self.cnn(left)
        rf = self.cnn(right)
        pf = self.pose_mlp(pose)
        return self.fuse(torch.cat([lf, rf, pf], dim=-1))


class WorldModel(nn.Module):
    def __init__(self, world_dim: int, obs_dim: int, action_dim: int, num_world_slots: int) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim + action_dim, world_dim)
        self.cross_attn = nn.MultiheadAttention(world_dim, num_heads=4, batch_first=True)
        self.gru = nn.GRUCell(world_dim, world_dim)
        self.ff = nn.Sequential(
            nn.Linear(world_dim, world_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(world_dim * 2, world_dim),
        )
        self.norm = nn.LayerNorm(world_dim)
        self.init_world = nn.Parameter(torch.randn(1, num_world_slots, world_dim) * 0.02)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.init_world.expand(batch_size, -1, -1).to(device)

    def forward(self, world_slots: torch.Tensor, percept: torch.Tensor, prev_action_embed: torch.Tensor) -> torch.Tensor:
        cond = self.obs_proj(torch.cat([percept, prev_action_embed], dim=-1)).unsqueeze(1)
        attended, _ = self.cross_attn(query=world_slots, key=cond, value=cond)
        b, k, d = attended.shape
        updated = self.gru(attended.reshape(b * k, d), world_slots.reshape(b * k, d)).view(b, k, d)
        return self.norm(updated + self.ff(updated))


# ============================================================
# Hierarchical self model
# ============================================================
class BodySelfModel(nn.Module):
    def __init__(self, body_dim: int, pose_dim: int, hand_state_dim: int, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim + hand_state_dim + obs_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, body_dim),
        )

    def forward(self, pose: torch.Tensor, hand_state: torch.Tensor, percept: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pose, hand_state, percept], dim=-1))


class AgentSelfModel(nn.Module):
    def __init__(self, agent_dim: int, body_dim: int, world_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(body_dim + world_dim, 160), nn.ReLU(inplace=True),
            nn.Linear(160, agent_dim),
        )

    def forward(self, body_self: torch.Tensor, world_slots: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([body_self, mean_pool_tokens(world_slots)], dim=-1))


class ReflectiveSelfModel(nn.Module):
    def __init__(self, reflective_dim: int, agent_dim: int, workspace_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(agent_dim + workspace_dim + memory_dim, 160), nn.ReLU(inplace=True),
            nn.Linear(160, reflective_dim),
        )

    def forward(self, agent_self: torch.Tensor, workspace: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([agent_self, workspace, memory], dim=-1))


class NarrativeSelfModel(nn.Module):
    def __init__(self, narrative_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, narrative_dim),
        )

    def forward(self, autobiographical_memory: torch.Tensor) -> torch.Tensor:
        return self.net(autobiographical_memory)


# ============================================================
# Attention, thought, workspace
# ============================================================
class AttentionController(nn.Module):
    def __init__(self, world_dim: int, agent_dim: int, reflective_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(world_dim + agent_dim + reflective_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, world_slots: torch.Tensor, agent_self: torch.Tensor, reflective_self: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, k, d = world_slots.shape
        a = agent_self.unsqueeze(1).expand(-1, k, -1)
        r = reflective_self.unsqueeze(1).expand(-1, k, -1)
        logits = self.score(torch.cat([world_slots, a, r], dim=-1)).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        focus_slot = torch.gather(world_slots, 1, idx[:, None, None].expand(-1, 1, d))[:, 0]
        return {"focus_logits": logits, "focus_probs": probs, "focus_idx": idx, "focus_slot": focus_slot}


class GlobalWorkspace(nn.Module):
    def __init__(self, workspace_dim: int, world_dim: int, body_dim: int, agent_dim: int, reflective_dim: int, narrative_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(world_dim + body_dim + agent_dim + reflective_dim + narrative_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, workspace_dim),
        )
        self.gru = nn.GRUCell(workspace_dim, workspace_dim)

    def forward(self, focus_slot: torch.Tensor, body_self: torch.Tensor, agent_self: torch.Tensor, reflective_self: torch.Tensor, narrative_self: torch.Tensor, prev_workspace: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([focus_slot, body_self, agent_self, reflective_self, narrative_self], dim=-1))
        return self.gru(x, prev_workspace)


class ThoughtLoop(nn.Module):
    def __init__(self, thought_dim: int, goal_dim: int, workspace_dim: int, memory_dim: int, value_dim: int) -> None:
        super().__init__()
        self.goal_proj = nn.Sequential(
            nn.Linear(workspace_dim + memory_dim + value_dim, 160), nn.ReLU(inplace=True),
            nn.Linear(160, goal_dim),
        )
        self.gru = nn.GRUCell(goal_dim + workspace_dim + memory_dim + value_dim, thought_dim)
        self.post = nn.Sequential(
            nn.Linear(thought_dim, thought_dim), nn.ReLU(inplace=True),
            nn.Linear(thought_dim, thought_dim),
        )

    def forward(self, workspace: torch.Tensor, memory: torch.Tensor, value_latent: torch.Tensor, num_steps: int) -> Dict[str, torch.Tensor]:
        goal = self.goal_proj(torch.cat([workspace, memory, value_latent], dim=-1))
        thought = torch.zeros(workspace.shape[0], self.gru.hidden_size, device=workspace.device)
        trace = []
        for _ in range(num_steps):
            x = torch.cat([goal, workspace, memory, value_latent], dim=-1)
            thought = self.gru(x, thought)
            thought = thought + self.post(thought)
            trace.append(thought)
        return {"goal": goal, "thought": thought, "thought_trace": torch.stack(trace, dim=1)}


# ============================================================
# Values, imagination, action
# ============================================================
class ValueSystem(nn.Module):
    def __init__(self, value_dim: int, workspace_dim: int, reflective_dim: int, narrative_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + reflective_dim + narrative_dim, 160), nn.ReLU(inplace=True),
            nn.Linear(160, value_dim),
        )
        self.curiosity = nn.Linear(value_dim, 1)
        self.control = nn.Linear(value_dim, 1)
        self.coherence = nn.Linear(value_dim, 1)
        self.salience = nn.Linear(value_dim, 1)

    def forward(self, workspace: torch.Tensor, reflective_self: torch.Tensor, narrative_self: torch.Tensor) -> Dict[str, torch.Tensor]:
        v = self.net(torch.cat([workspace, reflective_self, narrative_self], dim=-1))
        return {
            "value_latent": v,
            "curiosity": torch.sigmoid(self.curiosity(v)),
            "control_value": torch.sigmoid(self.control(v)),
            "coherence": torch.sigmoid(self.coherence(v)),
            "salience": torch.sigmoid(self.salience(v)),
        }


class CounterfactualSimulator(nn.Module):
    def __init__(self, world_dim: int, agent_dim: int, workspace_dim: int, action_dim: int, num_action_bins: int, num_counterfactuals: int) -> None:
        super().__init__()
        self.num_counterfactuals = num_counterfactuals
        self.action_embed = nn.Embedding(num_action_bins, action_dim)
        self.pred = nn.Sequential(
            nn.Linear(world_dim + agent_dim + workspace_dim + action_dim, 192), nn.ReLU(inplace=True),
            nn.Linear(192, world_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(world_dim + workspace_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, world_slots: torch.Tensor, agent_self: torch.Tensor, workspace: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = world_slots.shape[0]
        world_summary = mean_pool_tokens(world_slots)
        action_ids = torch.arange(self.num_counterfactuals, device=world_slots.device).unsqueeze(0).expand(b, -1)
        action_ids = action_ids % self.action_embed.num_embeddings
        action_embed = self.action_embed(action_ids)
        ws = world_summary.unsqueeze(1).expand(-1, self.num_counterfactuals, -1)
        ag = agent_self.unsqueeze(1).expand(-1, self.num_counterfactuals, -1)
        wk = workspace.unsqueeze(1).expand(-1, self.num_counterfactuals, -1)
        pred_world = self.pred(torch.cat([ws, ag, wk, action_embed], dim=-1))
        scores = self.score(torch.cat([pred_world, wk], dim=-1)).squeeze(-1)
        best_idx = scores.argmax(dim=-1)
        best_action = torch.gather(action_ids, 1, best_idx[:, None])[:, 0]
        return {
            "cf_action_ids": action_ids,
            "cf_world": pred_world,
            "cf_scores": scores,
            "best_cf_action": best_action,
        }


class ActionPolicy(nn.Module):
    def __init__(self, num_action_bins: int, thought_dim: int, workspace_dim: int, value_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(thought_dim + workspace_dim + value_dim, 192), nn.ReLU(inplace=True),
            nn.Linear(192, num_action_bins),
        )

    def forward(self, thought: torch.Tensor, workspace: torch.Tensor, value_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(torch.cat([thought, workspace, value_latent], dim=-1))
        return {"action_logits": logits, "action_probs": F.softmax(logits, dim=-1)}


# ============================================================
# Memory and reflection
# ============================================================
class ReflectionSystem(nn.Module):
    def __init__(self, reflective_dim: int, workspace_dim: int, action_dim: int, world_dim: int, value_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + action_dim + world_dim + value_dim, 192), nn.ReLU(inplace=True),
            nn.Linear(192, reflective_dim),
        )
        self.agency = nn.Linear(reflective_dim, 1)
        self.pred_error = nn.Linear(reflective_dim, 1)

    def forward(self, workspace: torch.Tensor, action_embed: torch.Tensor, world_delta: torch.Tensor, value_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        r = self.net(torch.cat([workspace, action_embed, mean_pool_tokens(world_delta), value_latent], dim=-1))
        return {
            "reflection": r,
            "agency_score": torch.sigmoid(self.agency(r)),
            "prediction_error": self.pred_error(r),
        }


class AutobiographicalMemory(nn.Module):
    def __init__(self, memory_dim: int, workspace_dim: int, reflective_dim: int, value_dim: int, report_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(workspace_dim + reflective_dim + value_dim + report_dim, memory_dim)
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.summary = nn.Sequential(
            nn.Linear(memory_dim, memory_dim), nn.ReLU(inplace=True),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, workspace: torch.Tensor, reflection: torch.Tensor, value_latent: torch.Tensor, report: torch.Tensor, prev_memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.in_proj(torch.cat([workspace, reflection, value_latent, report], dim=-1))
        mem = self.gru(x, prev_memory)
        return {"memory": mem, "summary": self.summary(mem)}


class SelfRevisionSystem(nn.Module):
    def __init__(self, agent_dim: int, reflective_dim: int, value_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(reflective_dim + value_dim + memory_dim, 160), nn.ReLU(inplace=True),
            nn.Linear(160, agent_dim),
        )

    def forward(self, reflection: torch.Tensor, value_latent: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.delta(torch.cat([reflection, value_latent, memory], dim=-1))


class LanguageReportHead(nn.Module):
    def __init__(self, report_dim: int, workspace_dim: int, reflective_dim: int, narrative_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim + reflective_dim + narrative_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, report_dim),
        )

    def forward(self, workspace: torch.Tensor, reflection: torch.Tensor, narrative_self: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([workspace, reflection, narrative_self], dim=-1))


# ============================================================
# Optional decoders
# ============================================================
class RenderHead(nn.Module):
    def __init__(self, world_dim: int, workspace_dim: int, render_size: Tuple[int, int]) -> None:
        super().__init__()
        h, w = render_size
        self.h, self.w = h, w
        self.query = nn.Sequential(
            nn.Linear(workspace_dim, world_dim), nn.ReLU(inplace=True),
            nn.Linear(world_dim, world_dim),
        )
        self.cross_attn = nn.MultiheadAttention(world_dim, 4, batch_first=True)
        self.rgb_head = nn.Sequential(nn.Linear(world_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 3 * h * w))
        self.depth_head = nn.Sequential(nn.Linear(world_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, h * w))

    def forward(self, world_slots: torch.Tensor, workspace: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.query(workspace).unsqueeze(1)
        fused, _ = self.cross_attn(query=q, key=world_slots, value=world_slots)
        fused = fused[:, 0]
        return {
            "rgb": torch.sigmoid(self.rgb_head(fused).view(-1, 3, self.h, self.w)),
            "depth": F.softplus(self.depth_head(fused).view(-1, 1, self.h, self.w)),
        }


# ============================================================
# Full conscious system
# ============================================================
class ConsciousSystem(nn.Module):
    def __init__(self, cfg: ConsciousSystemConfig) -> None:
        super().__init__()
        c = cfg.model
        d = cfg.data

        self.action_embed = nn.Embedding(c.num_action_bins, d.action_dim)

        self.perception = PerceptionSystem(c.obs_dim, d.pose_dim)
        self.world_model = WorldModel(c.world_dim, c.obs_dim, d.action_dim, c.num_world_slots)

        self.body_self = BodySelfModel(c.body_self_dim, d.pose_dim, d.hand_state_dim, c.obs_dim)
        self.agent_self = AgentSelfModel(c.agent_self_dim, c.body_self_dim, c.world_dim)
        self.reflective_self = ReflectiveSelfModel(c.reflective_self_dim, c.agent_self_dim, c.workspace_dim, c.memory_dim)
        self.narrative_self = NarrativeSelfModel(c.narrative_self_dim, c.memory_dim)

        self.attention = AttentionController(c.world_dim, c.agent_self_dim, c.reflective_self_dim)
        self.workspace = GlobalWorkspace(c.workspace_dim, c.world_dim, c.body_self_dim, c.agent_self_dim, c.reflective_self_dim, c.narrative_self_dim)
        self.value_system = ValueSystem(c.value_dim, c.workspace_dim, c.reflective_self_dim, c.narrative_self_dim)
        self.thought_loop = ThoughtLoop(c.thought_dim, c.goal_dim, c.workspace_dim, c.memory_dim, c.value_dim)
        self.counterfactual = CounterfactualSimulator(c.world_dim, c.agent_self_dim, c.workspace_dim, d.action_dim, c.num_action_bins, c.num_counterfactuals)
        self.policy = ActionPolicy(c.num_action_bins, c.thought_dim, c.workspace_dim, c.value_dim)
        self.reflection = ReflectionSystem(c.reflective_self_dim, c.workspace_dim, d.action_dim, c.world_dim, c.value_dim)
        self.autobiography = AutobiographicalMemory(c.memory_dim, c.workspace_dim, c.reflective_self_dim, c.value_dim, c.report_dim)
        self.self_revision = SelfRevisionSystem(c.agent_self_dim, c.reflective_self_dim, c.value_dim, c.memory_dim)
        self.language_report = LanguageReportHead(c.report_dim, c.workspace_dim, c.reflective_self_dim, c.narrative_self_dim)
        self.render = RenderHead(c.world_dim, c.workspace_dim, c.render_size)

        self.init_workspace = nn.Parameter(torch.zeros(1, c.workspace_dim))
        self.init_memory = nn.Parameter(torch.zeros(1, c.memory_dim))
        self.init_prev_action = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "world_slots": self.world_model.initial_state(batch_size, device),
            "workspace": self.init_workspace.expand(batch_size, -1).to(device),
            "autobiography": self.init_memory.expand(batch_size, -1).to(device),
            "prev_action_ids": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def step(
        self,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
        pose_t: torch.Tensor,
        hand_state_t: torch.Tensor,
        state: Dict[str, torch.Tensor],
        action_override: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        prev_action_embed = self.action_embed(state["prev_action_ids"])

        percept = self.perception(left_t, right_t, pose_t)
        world_slots = self.world_model(state["world_slots"], percept, prev_action_embed)

        body_self = self.body_self(pose_t, hand_state_t, percept)
        agent_self = self.agent_self(body_self, world_slots)
        narrative_self = self.narrative_self(state["autobiography"])
        reflective_self = self.reflective_self(agent_self, state["workspace"], state["autobiography"])

        focus = self.attention(world_slots, agent_self, reflective_self)
        workspace = self.workspace(focus["focus_slot"], body_self, agent_self, reflective_self, narrative_self, state["workspace"])
        values = self.value_system(workspace, reflective_self, narrative_self)
        thoughts = self.thought_loop(workspace, state["autobiography"], values["value_latent"], num_steps=cfg_obj.model.num_thought_steps if False else 4)
        imagined = self.counterfactual(world_slots, agent_self, workspace)
        policy = self.policy(thoughts["thought"], workspace, values["value_latent"])

        # Bias action logits using best counterfactual and salience
        action_logits = policy["action_logits"].clone()
        action_logits = action_logits.scatter_add(1, imagined["best_cf_action"][:, None], values["salience"] * 0.5)
        action_probs = F.softmax(action_logits, dim=-1)
        action_ids = action_override if action_override is not None else action_probs.argmax(dim=-1)
        action_embed = self.action_embed(action_ids)

        world_delta = world_slots - state["world_slots"]
        reflection = self.reflection(workspace, action_embed, world_delta, values["value_latent"])

        provisional_report = self.language_report(workspace, reflection["reflection"], narrative_self)
        memory_out = self.autobiography(workspace, reflection["reflection"], values["value_latent"], provisional_report, state["autobiography"])
        narrative_self = self.narrative_self(memory_out["memory"])
        report = self.language_report(workspace, reflection["reflection"], narrative_self)

        agent_self = agent_self + self.self_revision(reflection["reflection"], values["value_latent"], memory_out["memory"])
        render_out = self.render(world_slots, workspace)

        next_state = {
            "world_slots": world_slots,
            "workspace": workspace,
            "autobiography": memory_out["memory"],
            "prev_action_ids": action_ids,
        }

        return {
            "percept": percept,
            "world_slots": world_slots,
            "body_self": body_self,
            "agent_self": agent_self,
            "reflective_self": reflective_self,
            "narrative_self": narrative_self,
            "focus": focus,
            "workspace_out": workspace,
            "values": values,
            "thoughts": thoughts,
            "imagined": imagined,
            "policy": {"action_logits": action_logits, "action_probs": action_probs},
            "action_ids": action_ids,
            "reflection_out": reflection,
            "memory_out": memory_out,
            "report": report,
            "render": render_out,
            "state": next_state,
        }

    def forward_sequence(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        pose: torch.Tensor,
        hand_state: torch.Tensor,
        action_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        b, t = left.shape[:2]
        state = self.initial_state(b, left.device)

        rgbs = []
        depths = []
        acts = []
        focus_idx = []
        curiosity = []
        coherence = []
        cf_best = []
        reports = []
        workspaces = []
        memories = []
        thought_traces = []

        for i in range(t):
            out = self.step(
                left[:, i],
                right[:, i],
                pose[:, i],
                hand_state[:, i],
                state,
                None if action_ids is None else action_ids[:, i],
            )
            state = out["state"]
            rgbs.append(out["render"]["rgb"])
            depths.append(out["render"]["depth"])
            acts.append(out["action_ids"])
            focus_idx.append(out["focus"]["focus_idx"])
            curiosity.append(out["values"]["curiosity"])
            coherence.append(out["values"]["coherence"])
            cf_best.append(out["imagined"]["best_cf_action"])
            reports.append(out["report"])
            workspaces.append(out["workspace_out"])
            memories.append(out["memory_out"]["summary"])
            thought_traces.append(out["thoughts"]["thought_trace"])

        return {
            "rgb": torch.stack(rgbs, dim=1),
            "depth": torch.stack(depths, dim=1),
            "action_ids": torch.stack(acts, dim=1),
            "focus_idx": torch.stack(focus_idx, dim=1),
            "curiosity": torch.stack(curiosity, dim=1),
            "coherence": torch.stack(coherence, dim=1),
            "best_cf_action": torch.stack(cf_best, dim=1),
            "report": torch.stack(reports, dim=1),
            "workspace": torch.stack(workspaces, dim=1),
            "memory_summary": torch.stack(memories, dim=1),
            "thought_trace": torch.stack(thought_traces, dim=1),
        }


ConsciousSystemV5 = ConsciousSystem


# ============================================================
# Demo
# ============================================================
@hydra.main(version_base=None, config_path="../../config", config_name="conscious_system")
def main(cfg_raw) -> None:
    base = OmegaConf.structured(ConsciousSystemConfig())
    cfg = OmegaConf.merge(base, cfg_raw)
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    cfg_obj: ConsciousSystemConfig = OmegaConf.to_object(cfg)

    model = ConsciousSystem(cfg_obj).to(cfg_obj.train.device)

    if cfg_obj.mode == "print":
        b, t = 2, cfg_obj.data.seq_len
        h, w = cfg_obj.data.image_size
        device = torch.device(cfg_obj.train.device)
        left = torch.randn(b, t, 3, h, w, device=device)
        right = torch.randn(b, t, 3, h, w, device=device)
        pose = torch.randn(b, t, cfg_obj.data.pose_dim, device=device)
        hand_state = torch.randn(b, t, cfg_obj.data.hand_state_dim, device=device)
        out = model.forward_sequence(left, right, pose, hand_state)
        for k in [
            "rgb", "depth", "action_ids", "focus_idx", "curiosity",
            "coherence", "best_cf_action", "report", "workspace",
            "memory_summary", "thought_trace",
        ]:
            print(k, tuple(out[k].shape))
    else:
        print(model)


if __name__ == "__main__":
    main()
