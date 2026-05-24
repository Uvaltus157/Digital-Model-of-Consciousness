from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, Tuple
import json
import math
import random
import time

import hydra

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
#import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf


action_names = {i: f"action_{i}" for i in range(24)}


# ============================================================
# Config
# ============================================================
@dataclass
class NoveltyConfig:
    enabled: bool = True
    threshold: float = 0.18
    ema_decay: float = 0.98
    min_gap_steps: int = 20


@dataclass
class ReplayConfig:
    capacity: int = 30000
    batch_size: int = 8
    min_ready: int = 128
    recent_bias: float = 0.35


@dataclass
class LifeConfig:
    fps: float = 20.0
    max_steps: int = 1000000
    checkpoint_every_steps: int = 250
    report_every_steps: int = 25
    mj_substeps: int = 4


@dataclass
class TrainLoopConfig:
    enabled: bool = True
    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_sleep_sec: float = 0.01
    gradient_clip: float = 1.0
    stop_train_quality: float = 0.08
    restart_train_quality: float = 0.12
    stabilization_patience: int = 40
    #device: str="cpu"
    #batch_size: int = 128 
    #epochs: int = 1000
        
@dataclass
class ArmActuatorConfig:
    shoulder_yaw_range_deg: Tuple[float, float] = (-85.0, 85.0)
    shoulder_pitch_range_deg: Tuple[float, float] = (-70.0, 70.0)
    elbow_range_deg: Tuple[float, float] = (5.0, 145.0)
    upper_len: float = 0.42
    fore_len: float = 0.40
    reach_margin: float = 0.03
    curiosity_freq_hz: float = 0.45
    action_smoothing: float = 0.25


@dataclass
class BaseActuatorConfig:
    move_gain: float = 0.10
    lift_gain: float = 0.06
    yaw_gain_deg: float = 3.5
    pitch_gain_deg: float = 2.5
    pitch_limit_deg: float = 70.0
    smoothing: float = 0.25


@dataclass
class MujocoWorldConfig:
    width: int = 192
    height: int = 128
    hfov_deg: float = 78.0
    baseline: float = 0.30
    human_mesh_path: str = "assets/human.obj"
    start_pos: Tuple[float, float, float] = (-6.0, 0.0, 2.6)
    start_yaw_deg: float = 0.0
    start_pitch_deg: float = -8.0
    push_gain: float = 0.75
    friction: float = 0.82
    novel_impulse_gain: float = 1.6
    grasp_dist: float = 0.14
    base: BaseActuatorConfig = field(default_factory=BaseActuatorConfig)
    arm: ArmActuatorConfig = field(default_factory=ArmActuatorConfig)


@dataclass
class ViewerConfig:
    auto_inject_novelty: bool = True
    inject_novelty_every_sec: float = 18.0
    print_status_every_sec: float = 2.0
    allow_mujoco_window: bool = True
    # MuJoCo's passive viewer already owns an internal UI thread; wrapping it
    # in another thread is experimental and can freeze mouse interaction on
    # some GLFW/OpenGL backends.
    mujoco_threaded: bool = False
    mujoco_sync_fps: float = 8.0
    mujoco_sync_every_steps: int = 1
    no_display_fallback: bool = True
    #show_overlay_window: bool = True
    #overlay_delay_ms: float = 3.0
        
@dataclass
class RuntimeConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    out_dir: str = "runs/unified_conscious_viewer_v51"



# ============================================================
# Math helpers
# ============================================================
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def camera_fovy_deg(hfov_deg: float, width: int, height: int) -> float:
    aspect = width / height
    vfov = 2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) / aspect)
    return math.degrees(vfov)


def yaw_pitch_to_forward(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return normalize(np.array([cy * cp, sy * cp, sp], dtype=np.float64))


def rig_axes(yaw_deg: float, pitch_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = yaw_pitch_to_forward(yaw_deg, pitch_deg)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    return forward, right, up


def local_to_world(rig_pos: np.ndarray, yaw_deg: float, pitch_deg: float, local: Tuple[float, float, float]) -> np.ndarray:
    forward, right, up = rig_axes(yaw_deg, pitch_deg)
    lx, ly, lz = local
    return rig_pos + forward * lx + right * ly + up * lz


def quat_from_rot(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_from_forward_up(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    x_axis = normalize(forward)
    z_axis = up_hint - np.dot(up_hint, x_axis) * x_axis
    if np.linalg.norm(z_axis) < 1e-8:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        z_axis = alt_up - np.dot(alt_up, x_axis) * x_axis
    z_axis = normalize(z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return quat_from_rot(np.column_stack([x_axis, y_axis, z_axis]))


def simple_arm_ik_local(target_local: np.ndarray, cfg: ArmActuatorConfig) -> np.ndarray:
    x, y, z = map(float, target_local)
    yaw = math.atan2(y, x + 1e-9)
    planar = math.hypot(x, y)
    dz = -z
    dist = math.hypot(planar, dz)
    max_reach = cfg.upper_len + cfg.fore_len - cfg.reach_margin
    min_reach = abs(cfg.upper_len - cfg.fore_len) + 1e-4
    dist = clamp(dist, min_reach, max_reach)
    pitch_dir = math.atan2(dz, planar + 1e-9)
    cos_elbow = clamp((cfg.upper_len**2 + cfg.fore_len**2 - dist**2) / (2 * cfg.upper_len * cfg.fore_len), -1.0, 1.0)
    elbow = math.pi - math.acos(cos_elbow)
    cos_sh = clamp((cfg.upper_len**2 + dist**2 - cfg.fore_len**2) / (2 * cfg.upper_len * dist), -1.0, 1.0)
    shoulder_offset = math.acos(cos_sh)
    shoulder_pitch = pitch_dir - shoulder_offset
    yaw = clamp(yaw, math.radians(cfg.shoulder_yaw_range_deg[0]), math.radians(cfg.shoulder_yaw_range_deg[1]))
    shoulder_pitch = clamp(shoulder_pitch, math.radians(cfg.shoulder_pitch_range_deg[0]), math.radians(cfg.shoulder_pitch_range_deg[1]))
    elbow = clamp(elbow, math.radians(cfg.elbow_range_deg[0]), math.radians(cfg.elbow_range_deg[1]))
    return np.array([yaw, shoulder_pitch, elbow], dtype=np.float64)

# ============================================================
# Utility classes
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        from collections import deque
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()

    def add(self, item: Dict[str, torch.Tensor]) -> None:
        with self.lock:
            self.buffer.append(item)

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)

    def sample(self, batch_size: int, recent_bias: float = 0.0):
        with self.lock:
            n = len(self.buffer)
            if n == 0:
                return []
            if recent_bias > 0 and random.random() < recent_bias:
                k = min(n, max(batch_size * 4, batch_size))
                pool = list(self.buffer)[-k:]
                return random.sample(pool, min(batch_size, len(pool)))
            return random.sample(list(self.buffer), min(batch_size, n))


class QualityMeter:
    def __init__(self, ema_decay: float = 0.98) -> None:
        self.ema_decay = ema_decay
        self.value = 1.0
        self.initialized = False
        self.lock = Lock()

    def update(self, x: float) -> float:
        with self.lock:
            if not self.initialized:
                self.value = x
                self.initialized = True
            else:
                self.value = self.ema_decay * self.value + (1.0 - self.ema_decay) * x
            return self.value

    def get(self) -> float:
        with self.lock:
            return float(self.value)


class NoveltyDetector:
    def __init__(self, cfg: NoveltyConfig) -> None:
        self.cfg = cfg
        self.baseline = 0.0
        self.initialized = False
        self.last_novel_step = -10**9
        self.lock = Lock()

    def score(self, obs_embed: torch.Tensor, workspace: torch.Tensor, imagined_value: torch.Tensor) -> float:
        s1 = float(obs_embed.abs().mean().item())
        s2 = float(workspace.abs().mean().item())
        s3 = float(imagined_value.abs().mean().item())
        return 0.35 * s1 + 0.35 * s2 + 0.30 * s3

    def is_novel(self, step: int, score: float) -> bool:
        if not self.cfg.enabled:
            return False
        with self.lock:
            if not self.initialized:
                self.baseline = score
                self.initialized = True
                return False
            self.baseline = self.cfg.ema_decay * self.baseline + (1.0 - self.cfg.ema_decay) * score
            surprise = abs(score - self.baseline)
            if surprise > self.cfg.threshold and (step - self.last_novel_step) >= self.cfg.min_gap_steps:
                self.last_novel_step = step
                return True
            return False

