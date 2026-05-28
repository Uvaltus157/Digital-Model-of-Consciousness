from __future__ import annotations

"""
unified_conscious_viewer.py

Legacy V5.7 compatibility module.

This file still provides the V5.7 base classes used by the newer V5.10 runner
compatibility layer, but model creation now goes through the canonical M5
ConsciousDreamer API instead of importing versioned V21/V22/V23 names directly.

Current app-level rule:
    import ConsciousDreamer / ConsciousDreamerConfig from
    src.modules.m05_world_model_attention_workspace.models.conscious_dreamer

Historical implementation files remain inside M5 as internal layers only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import json
import os
import random
import time

import hydra
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from src.shared.console_colors import install_colored_errors

install_colored_errors()

from src.modules.m05_world_model_attention_workspace.models.conscious_dreamer import (
    ConsciousDreamer,
    make_conscious_dreamer_config_from_world,
)
from src.modules.m07_inner_speech_thoughts.inner_world_visualizer import (
    DreamerInnerWorldVisualizer,
    InnerWorldVizConfig,
)
from src.platform.mujoco_world.realistic_hand_control import RealisticHandBridge
from src.shared.unified_conscious_utils import (
    LifeConfig,
    MujocoWorldConfig,
    NoveltyConfig,
    NoveltyDetector,
    QualityMeter,
    ReplayBuffer,
    ReplayConfig,
    RuntimeConfig,
    TrainLoopConfig,
    ViewerConfig,
    action_names,
    quat_from_forward_up,
    yaw_pitch_to_forward,
)


@dataclass
class InnerWorldWindowConfig:
    enabled: bool = True
    width: int = 1800
    height: int = 1100
    show_every_steps: int = 1
    save_frames: bool = False
    save_every_steps: int = 100
    out_dir: str = "inner_world_frames"


@dataclass
class UnifiedV57Config:
    mode: str = "run"
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    mujoco_world: MujocoWorldConfig = field(default_factory=MujocoWorldConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(out_dir="runs/unified_conscious_viewer"))
    inner_world: InnerWorldWindowConfig = field(default_factory=InnerWorldWindowConfig)

    action_dim: int = 24
    embodied_dim: int = 11
    hand_motor_dim: int = 34
    tactile_dim: int = 42
    body_state_dim: int = 49


class MujocoLiveWorldV57:
    def __init__(self, device: str, cfg: MujocoWorldConfig, embodied_dim: int = 11, hand_motor_dim: int = 34) -> None:
        self.device = device
        self.cfg = cfg
        self.embodied_dim = embodied_dim
        self.hand_motor_dim = hand_motor_dim

        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        xml_path = project_root / "src" / "platform" / "mujoco_world" / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=cfg.height, width=cfg.width)
        self.hand_bridge = RealisticHandBridge(self.model)

        self.cam_pos = np.array(cfg.start_pos, dtype=np.float64)
        self.yaw_deg = float(cfg.start_yaw_deg)
        self.pitch_deg = float(cfg.start_pitch_deg)

        self.focus_idx = 1
        self.curiosity_drive = 0.5
        self.planned_action_id = 0
        self.focus_world_hint = np.array([0.0, 1.3, 0.55], dtype=np.float64)

        self.latest_tactile = np.zeros(42, dtype=np.float32)
        self.latest_hand_ctrl = np.zeros(hand_motor_dim, dtype=np.float32)
        self.latest_embodied = np.zeros(embodied_dim, dtype=np.float32)

        self.body_ids = {
            "box": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_box_body"),
            "sphere": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_sphere_body"),
            "cylinder": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_cylinder_body"),
        }
        self.site_ids = {
            "left_hand": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site"),
            "right_hand": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site"),
        }
        self.act_ids = {
            "left_shoulder_yaw": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_left_shoulder_yaw"),
            "left_shoulder_pitch": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_left_shoulder_pitch"),
            "left_elbow": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_left_elbow"),
            "right_shoulder_yaw": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_right_shoulder_yaw"),
            "right_shoulder_pitch": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_right_shoulder_pitch"),
            "right_elbow": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_right_elbow"),
        }
        self.prev_arm_ctrl = np.zeros(6, dtype=np.float64)

    def close(self) -> None:
        self.renderer.close()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.cam_pos = np.array(self.cfg.start_pos, dtype=np.float64)
        self.yaw_deg = float(self.cfg.start_yaw_deg)
        self.pitch_deg = float(self.cfg.start_pitch_deg)
        self.focus_world_hint = self.get_object_pos("sphere")
        self._update_rig_pose()
        mujoco.mj_forward(self.model, self.data)
        return self.observe(
            action_id=0,
            embodied_targets=np.zeros(self.embodied_dim, dtype=np.float32),
            hand_controls=np.zeros(self.hand_motor_dim, dtype=np.float32),
        )

    def get_object_pos(self, key: str = "sphere") -> np.ndarray:
        bid = self.body_ids.get(key, -1)
        if bid is not None and bid >= 0:
            return self.data.xpos[bid].copy()
        return np.zeros(3, dtype=np.float64)

    def _scene_focus_point(self, focus_idx: int) -> np.ndarray:
        keys = ["box", "sphere", "cylinder"]
        return self.get_object_pos(keys[int(focus_idx) % len(keys)])

    def _mocap_id_for_body(self, body_name: str, fallback: int = 0) -> int:
        try:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
            if bid >= 0:
                mid = int(self.model.body_mocapid[bid])
                if mid >= 0:
                    return mid
        except Exception:
            pass
        return int(fallback)

    def set_attention_drive(self, focus_idx: int, curiosity_drive: float, planned_action_id: int) -> None:
        self.focus_idx = int(focus_idx)
        self.curiosity_drive = float(np.clip(curiosity_drive, 0.0, 1.0))
        self.planned_action_id = int(planned_action_id)
        self.focus_world_hint = self._scene_focus_point(self.focus_idx)

    def _update_rig_pose(self) -> None:
        forward = yaw_pitch_to_forward(self.yaw_deg, self.pitch_deg)
        quat = quat_from_forward_up(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        mid = self._mocap_id_for_body("agent_rig", fallback=0)
        self.data.mocap_pos[mid] = self.cam_pos
        self.data.mocap_quat[mid] = quat

    def _apply_embodied_base_and_arm(self, embodied: np.ndarray) -> None:
        embodied = np.asarray(embodied, dtype=np.float64).reshape(self.embodied_dim)
        self.latest_embodied = embodied.astype(np.float32)

        base = embodied[:5]
        arm_raw = embodied[5:11]

        forward = yaw_pitch_to_forward(self.yaw_deg, self.pitch_deg)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-9)

        self.cam_pos = (
            self.cam_pos
            + forward * (base[0] * self.cfg.base.move_gain)
            + right * (base[1] * self.cfg.base.move_gain)
            + up * (base[2] * self.cfg.base.lift_gain)
        )
        self.yaw_deg += float(base[3]) * self.cfg.base.yaw_gain_deg
        self.pitch_deg = float(np.clip(
            self.pitch_deg + float(base[4]) * self.cfg.base.pitch_gain_deg,
            -self.cfg.base.pitch_limit_deg,
            self.cfg.base.pitch_limit_deg,
        ))

        ranges = np.array([
            [-1.48353, 1.48353],
            [-1.22173, 1.22173],
            [0.0872665, 2.53073],
            [-1.48353, 1.48353],
            [-1.22173, 1.22173],
            [0.0872665, 2.53073],
        ], dtype=np.float64)
        arm01 = np.clip((arm_raw + 1.0) * 0.5, 0.0, 1.0)
        arm_ctrl = ranges[:, 0] + arm01 * (ranges[:, 1] - ranges[:, 0])
        self.prev_arm_ctrl = 0.75 * self.prev_arm_ctrl + 0.25 * arm_ctrl

        keys = [
            "left_shoulder_yaw",
            "left_shoulder_pitch",
            "left_elbow",
            "right_shoulder_yaw",
            "right_shoulder_pitch",
            "right_elbow",
        ]
        for i, k in enumerate(keys):
            aid = self.act_ids.get(k, -1)
            if aid >= 0:
                self.data.ctrl[aid] = float(self.prev_arm_ctrl[i])

    def _render_named_camera(self, cam_name: str):
        self.renderer.update_scene(self.data, camera=cam_name)
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=cam_name)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        rgb = np.ascontiguousarray(np.flipud(rgb).copy())
        depth = np.ascontiguousarray(np.flipud(depth).copy())
        return rgb, depth

    def observe(self, action_id: int, embodied_targets: np.ndarray, hand_controls: np.ndarray):
        self._apply_embodied_base_and_arm(embodied_targets)
        self.hand_bridge.apply(self.data, hand_controls, smoothing=0.25)
        self.latest_hand_ctrl = np.asarray(hand_controls, dtype=np.float32).reshape(self.hand_motor_dim)

        self._update_rig_pose()

        substeps = int(getattr(self.cfg, "mj_substeps", 4))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        tactile = self.hand_bridge.read_tactile(self.data)
        self.latest_tactile = tactile

        left_rgb, left_depth = self._render_named_camera("cam_left")
        right_rgb, _ = self._render_named_camera("cam_right")

        forward = yaw_pitch_to_forward(self.yaw_deg, self.pitch_deg)
        quat = quat_from_forward_up(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))

        left_hand = self.data.site_xpos[self.site_ids["left_hand"]].copy()
        right_hand = self.data.site_xpos[self.site_ids["right_hand"]].copy()
        sphere = self.get_object_pos("sphere")
        box = self.get_object_pos("box")
        cylinder = self.get_object_pos("cylinder")

        body_state = np.concatenate([
            left_hand.astype(np.float32),
            right_hand.astype(np.float32),
            sphere.astype(np.float32),
            box.astype(np.float32),
            cylinder.astype(np.float32),
            self.latest_hand_ctrl.astype(np.float32),
        ])

        pose = np.concatenate([self.cam_pos.astype(np.float32), quat.astype(np.float32)], axis=0)
        reward = 0.05 * (1.0 / (1.0 + float(np.linalg.norm(sphere - self.cam_pos))))
        reward += 0.01 * float(tactile.sum())

        left_rgb_chw = np.ascontiguousarray(left_rgb.transpose(2, 0, 1))
        right_rgb_chw = np.ascontiguousarray(right_rgb.transpose(2, 0, 1))
        left_depth_chw = np.ascontiguousarray(left_depth[None, ...])

        return {
            "left": torch.from_numpy(left_rgb_chw).float().unsqueeze(0).to(self.device) / 255.0,
            "right": torch.from_numpy(right_rgb_chw).float().unsqueeze(0).to(self.device) / 255.0,
            "pose": torch.from_numpy(np.ascontiguousarray(pose)).float().unsqueeze(0).to(self.device),
            "body_state": torch.from_numpy(np.ascontiguousarray(body_state)).float().unsqueeze(0).to(self.device),
            "tactile": torch.from_numpy(np.ascontiguousarray(tactile)).float().unsqueeze(0).to(self.device),
            "hand_motor": torch.from_numpy(np.ascontiguousarray(self.latest_hand_ctrl)).float().unsqueeze(0).to(self.device),
            "embodied_action": torch.from_numpy(np.ascontiguousarray(self.latest_embodied)).float().unsqueeze(0).to(self.device),
            "object_state": torch.from_numpy(np.ascontiguousarray(np.concatenate([sphere, box, cylinder]).astype(np.float32))).float().unsqueeze(0).to(self.device),
            "reward": torch.tensor([[reward]], device=self.device, dtype=torch.float32),
            "done": torch.tensor([[0.0]], device=self.device, dtype=torch.float32),
            "action_id": torch.tensor([action_id], device=self.device, dtype=torch.long),
            "depth": torch.from_numpy(left_depth_chw).float().unsqueeze(0).to(self.device),
        }


class UnifiedSystemV57:
    def __init__(self, cfg: UnifiedV57Config) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.runtime.device)
        random.seed(cfg.runtime.seed)
        torch.manual_seed(cfg.runtime.seed)

        self.model_cfg = make_conscious_dreamer_config_from_world(
            image_height=cfg.mujoco_world.height,
            image_width=cfg.mujoco_world.width,
            body_state_dim=cfg.body_state_dim,
            tactile_dim=cfg.tactile_dim,
            hand_motor_dim=cfg.hand_motor_dim,
            embodied_dim=cfg.embodied_dim,
            action_dim=cfg.action_dim,
        )
        self.model = ConsciousDreamer(self.model_cfg).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.replay = ReplayBuffer(cfg.replay.capacity)
        self.quality = QualityMeter(ema_decay=0.98)
        self.novelty = NoveltyDetector(cfg.novelty)

        self.world = MujocoLiveWorldV57(
            self.device,
            cfg.mujoco_world,
            embodied_dim=cfg.embodied_dim,
            hand_motor_dim=cfg.hand_motor_dim,
        )

        viz_cfg = InnerWorldVizConfig(width=cfg.inner_world.width, height=cfg.inner_world.height)
        self.inner_viz = DreamerInnerWorldVisualizer(viz_cfg) if cfg.inner_world.enabled else None
        if cfg.inner_world.save_frames:
            Path(cfg.inner_world.out_dir).mkdir(parents=True, exist_ok=True)

        self.out_dir = Path(cfg.runtime.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "live_log.jsonl"

        self.shutdown = False
        self.training_enabled = True
        self.global_step = 0
        self.train_steps = 0

        self.state = self.model.initial_state(batch_size=1, device=self.device)
        self.prev_embodied_action = torch.zeros(1, cfg.embodied_dim, device=self.device)
        self.prev_hand_motor = torch.zeros(1, cfg.hand_motor_dim, device=self.device)

        self.latest_stats = None
        self.latest_out = None
        self.last_print_time = 0.0

    def log_event(self, payload: Dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_checkpoint(self, name: str = "last.pt") -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "train_steps": self.train_steps,
                "quality": self.quality.get(),
                "config": OmegaConf.to_container(OmegaConf.structured(self.cfg), resolve=True),
            },
            self.out_dir / name,
        )

    def compute_step_loss(self, out: Dict, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_rgb = out["decoder"]["rgb"]
        target_rgb = obs["left"]
        if target_rgb.shape[-2:] != pred_rgb.shape[-2:]:
            target_rgb = nn.functional.interpolate(target_rgb, size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False)
        rgb_loss = nn.functional.l1_loss(pred_rgb, target_rgb)

        pred_depth = out["decoder"]["depth"]
        target_depth = obs["depth"]
        if target_depth.shape[-2:] != pred_depth.shape[-2:]:
            target_depth = nn.functional.interpolate(target_depth, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
        depth_loss = nn.functional.l1_loss(pred_depth, target_depth)

        reward_loss = nn.functional.mse_loss(out["decoder"]["reward"], obs["reward"])
        continue_loss = nn.functional.mse_loss(out["decoder"]["continue"], 1.0 - obs["done"])

        hand_reg = out["hand_ctrl"].mean()
        embodied_reg = out["embodied_targets"].abs().mean()
        object_reg = out["object_repr"].abs().mean()
        report_reg = out["symbolic_report"]["report_latent"].abs().mean()

        return (
            rgb_loss
            + depth_loss
            + 0.1 * reward_loss
            + 0.1 * continue_loss
            + 0.003 * hand_reg
            + 0.002 * embodied_reg
            + 0.002 * object_reg
            + 0.001 * report_reg
        )

    def model_step(self, obs: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor], action_override=None, write_memory: bool = True) -> Dict:
        return self.model.step(
            left=obs["left"],
            right=obs["right"],
            pose=obs["pose"],
            body_state=obs["body_state"],
            state=state,
            tactile=obs["tactile"],
            hand_motor=self.prev_hand_motor,
            embodied_action=self.prev_embodied_action,
            depth=obs["depth"],
            object_state=obs["object_state"],
            action_override=action_override,
            write_memory=write_memory,
        )

    def update_inner_world_window(self, out: Dict) -> None:
        if self.inner_viz is None:
            return
        if self.global_step % max(1, self.cfg.inner_world.show_every_steps) != 0:
            return

        symbolic = out.get("symbolic_report")
        key = self.inner_viz.show(out, symbolic, delay_ms=1)
        if key in (27, ord("q")):
            self.shutdown = True

        if self.cfg.inner_world.save_frames and self.global_step % max(1, self.cfg.inner_world.save_every_steps) == 0:
            path = Path(self.cfg.inner_world.out_dir) / f"inner_world_{self.global_step:07d}.png"
            self.inner_viz.save(str(path), out, symbolic)

    def maybe_print_status(self):
        now = time.time()
        if self.latest_stats is not None and (now - self.last_print_time) >= self.cfg.viewer.print_status_every_sec:
            s = self.latest_stats
            sp = s["sphere"]
            mw = s["modality_weights"]
            print(
                f"step={s['step']} training={s['training']} quality={s['quality']:.4f} "
                f"focus={s['focus_idx']} action={action_names.get(s['action'], s['action'])} "
                f"curiosity={s['curiosity']:.3f} coherence={s['coherence']:.3f} self={s['self_confidence']:.3f} "
                f"speech={s['inner_report_confidence']:.3f} touch={s['touch_sum']:.2f} obj={s['object_repr_norm']:.3f} "
                f"mem={s['memory_used']:.2f} sphere=({sp[0]:.2f},{sp[1]:.2f},{sp[2]:.2f}) "
                f"attn[v,t,m,obj]=({mw[0]:.2f},{mw[3]:.2f},{mw[4]:.2f},{mw[5]:.2f})"
            )
            self.last_print_time = now


@hydra.main(version_base=None, config_path="../../config", config_name="runner")
def main(cfg_raw) -> None:
    raw = OmegaConf.create(OmegaConf.to_container(cfg_raw, resolve=False))
    base = OmegaConf.structured(UnifiedV57Config())

    allowed_top_keys = {
        "mode",
        "novelty",
        "replay",
        "life",
        "train",
        "mujoco_world",
        "viewer",
        "runtime",
        "inner_world",
        "action_dim",
        "embodied_dim",
        "hand_motor_dim",
        "tactile_dim",
        "body_state_dim",
    }

    allowed_nested_keys = {
        "novelty": set(OmegaConf.structured(NoveltyConfig()).keys()),
        "replay": set(OmegaConf.structured(ReplayConfig()).keys()),
        "life": set(OmegaConf.structured(LifeConfig()).keys()),
        "train": set(OmegaConf.structured(TrainLoopConfig()).keys()),
        "mujoco_world": set(OmegaConf.structured(MujocoWorldConfig()).keys()),
        "viewer": set(OmegaConf.structured(ViewerConfig()).keys()),
        "runtime": set(OmegaConf.structured(RuntimeConfig()).keys()),
        "inner_world": set(OmegaConf.structured(InnerWorldWindowConfig()).keys()),
    }

    clean_dict = {}
    for key in raw.keys():
        if key not in allowed_top_keys:
            continue
        if key in allowed_nested_keys:
            if OmegaConf.is_config(raw[key]) or isinstance(raw[key], dict):
                clean_dict[key] = {
                    sub_key: raw[key][sub_key]
                    for sub_key in raw[key].keys()
                    if sub_key in allowed_nested_keys[key]
                }
            continue
        clean_dict[key] = raw[key]

    clean = OmegaConf.create(clean_dict)
    cfg = OmegaConf.merge(base, clean)

    print("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    cfg_obj: UnifiedV57Config = OmegaConf.to_object(cfg)

    system = UnifiedSystemV57(cfg_obj)
    if hasattr(system, "run"):
        system.run()
    else:
        raise RuntimeError("UnifiedSystemV57 has no run() method. Use src.apps.runner for the V5.10 runtime.")


if __name__ == "__main__":
    main()
