from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dynamic_self_world_dataset_v2 import DynamicSelfWorldDataset, DynamicSelfWorldDatasetConfig
from src.modules.m05_world_model_attention_workspace.legacy.conscious_system import (
    ConsciousSystem,
    ConsciousSystemConfig,
)


action_names = {i: f"action_{i}" for i in range(24)}


@dataclass
class ViewerConfig:
    ckpt: str = "runs/conscious_system/best.pt"
    dataset_root: str = "dataset_dynamic_self/val"
    index: int = 0
    device: str = "cpu"
    delay_ms: int = 180
    loop: bool = True
    canvas_width: int = 1400
    canvas_height: int = 900
    trail_len: int = 24
    panel_bg: Tuple[int, int, int] = (8, 10, 16)


@dataclass
class AppConfig:
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    system: ConsciousSystemConfig = field(default_factory=ConsciousSystemConfig)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _norm01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    mn, mx = float(v.min()), float(v.max())
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def project_vec2(v: np.ndarray, cx: int, cy: int, scale: float) -> tuple[int, int]:
    x = int(cx + float(v[0]) * scale)
    y = int(cy - float(v[1]) * scale)
    return x, y


def draw_text_block(img: np.ndarray, lines, x=12, y0=24, fs=0.55, th=1, color=(230, 230, 230)):
    y = y0
    for line in lines:
        cv2.putText(img, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
        y += 22


def draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, label: str, color=(80, 220, 255)):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    cv2.rectangle(img, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.putText(img, f"{label}: {value:.3f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def draw_subjective_space(
    img: np.ndarray,
    workspace_t: np.ndarray,
    memory_t: np.ndarray,
    report_t: np.ndarray,
    thought_trace_t: np.ndarray,
    curiosity_t: float,
    coherence_t: float,
    focus_idx_t: int,
    best_cf_action_t: int,
    action_t: int,
    trails: Dict[str, list[tuple[int, int]]],
):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # workspace core
    ws2 = workspace_t[:2] if workspace_t.shape[0] >= 2 else np.array([0.0, 0.0], dtype=np.float32)
    mem2 = memory_t[:2] if memory_t.shape[0] >= 2 else np.array([0.0, 0.0], dtype=np.float32)
    rep2 = report_t[:2] if report_t.shape[0] >= 2 else np.array([0.0, 0.0], dtype=np.float32)

    ws_p = project_vec2(ws2, cx, cy, 140)
    mem_p = project_vec2(mem2, cx + 250, cy + 140, 90)
    rep_p = project_vec2(rep2, cx - 250, cy + 140, 90)

    trails["workspace"].append(ws_p)
    trails["memory"].append(mem_p)
    trails["report"].append(rep_p)
    for k in trails:
        if len(trails[k]) > 24:
            trails[k] = trails[k][-24:]

    # ambient reflection waves
    wave_r = int(70 + 40 * float(curiosity_t))
    cv2.circle(img, (cx, cy), wave_r, (70, 110, 220), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), wave_r + 26, (50, 70, 140), 1, cv2.LINE_AA)

    # workspace nucleus
    core_r = int(26 + 16 * float(coherence_t))
    cv2.circle(img, (cx, cy), core_r + 8, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.circle(img, ws_p, core_r, (255, 240, 120), -1, cv2.LINE_AA)
    cv2.putText(img, "GLOBAL WORKSPACE", (cx - 84, cy - core_r - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 240, 120), 1, cv2.LINE_AA)

    # memory and report nodes
    cv2.circle(img, mem_p, 16, (120, 255, 160), -1, cv2.LINE_AA)
    cv2.circle(img, rep_p, 16, (255, 140, 120), -1, cv2.LINE_AA)
    cv2.putText(img, "MEMORY", (mem_p[0] - 28, mem_p[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 160), 1, cv2.LINE_AA)
    cv2.putText(img, "REPORT", (rep_p[0] - 26, rep_p[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 140, 120), 1, cv2.LINE_AA)
    cv2.line(img, ws_p, mem_p, (120, 255, 160), 1, cv2.LINE_AA)
    cv2.line(img, ws_p, rep_p, (255, 140, 120), 1, cv2.LINE_AA)

    # trails
    for name, color in [("workspace", (255, 240, 120)), ("memory", (120, 255, 160)), ("report", (255, 140, 120))]:
        pts = trails[name]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    # thought particles orbiting the workspace
    if thought_trace_t.ndim == 2:
        for i in range(thought_trace_t.shape[0]):
            th = thought_trace_t[i]
            if th.shape[0] < 2:
                continue
            px, py = project_vec2(th[:2], cx, cy, 80 + 15 * i)
            cv2.circle(img, (px, py), 4, (80, 255, 255), -1, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (px, py), (40, 110, 110), 1, cv2.LINE_AA)

    # focus / action / counterfactual badges
    badges_y = 60
    badges = [
        (f"focus={int(focus_idx_t)}", (90, 200, 255)),
        (f"act={action_names.get(int(action_t), str(int(action_t)))}", (120, 180, 255)),
        (f"best_cf={action_names.get(int(best_cf_action_t), str(int(best_cf_action_t)))}", (120, 255, 180)),
    ]
    x = 28
    for text, color in badges:
        tw = max(130, 10 * len(text))
        cv2.rectangle(img, (x, badges_y), (x + tw, badges_y + 28), color, -1)
        cv2.putText(img, text, (x + 8, badges_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)
        x += tw + 10


def render_frame(
    cfg: ViewerConfig,
    left_t: torch.Tensor,
    pred_rgb_t: torch.Tensor,
    target_depth_t: torch.Tensor,
    pred_depth_t: torch.Tensor,
    workspace_t: torch.Tensor,
    memory_t: torch.Tensor,
    report_t: torch.Tensor,
    thought_trace_t: torch.Tensor,
    curiosity_t: float,
    coherence_t: float,
    focus_idx_t: int,
    best_cf_action_t: int,
    action_t: int,
    t_idx: int,
    trails: Dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    canvas = np.full((cfg.canvas_height, cfg.canvas_width, 3), cfg.panel_bg, dtype=np.uint8)

    gt = cv2.cvtColor(to_uint8_rgb(left_t), cv2.COLOR_RGB2BGR)
    pr = cv2.cvtColor(to_uint8_rgb(pred_rgb_t), cv2.COLOR_RGB2BGR)
    gd = depth_to_bgr(target_depth_t)
    pd = depth_to_bgr(pred_depth_t)

    # top-left panels
    canvas[20:20 + gt.shape[0], 20:20 + gt.shape[1]] = gt
    canvas[20:20 + pr.shape[0], 40 + gt.shape[1]:40 + gt.shape[1] + pr.shape[1]] = pr
    canvas[40 + gt.shape[0]:40 + gt.shape[0] + gd.shape[0], 20:20 + gd.shape[1]] = gd
    canvas[40 + pr.shape[0]:40 + pr.shape[0] + pd.shape[0], 40 + gd.shape[1]:40 + gd.shape[1] + pd.shape[1]] = pd

    cv2.putText(canvas, "GT RGB", (20, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PRED RGB", (40 + gt.shape[1], 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT DEPTH", (20, 36 + gt.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PRED DEPTH", (40 + gd.shape[1], 36 + pr.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # right: subjective space
    subj = canvas[:, 520:]
    draw_subjective_space(
        subj,
        _to_np(workspace_t),
        _to_np(memory_t),
        _to_np(report_t),
        _to_np(thought_trace_t),
        curiosity_t,
        coherence_t,
        focus_idx_t,
        best_cf_action_t,
        action_t,
        trails,
    )

    # bottom bars
    draw_bar(canvas, 20, 760, 240, 16, curiosity_t, "curiosity", (80, 220, 255))
    draw_bar(canvas, 20, 800, 240, 16, coherence_t, "coherence", (120, 255, 180))

    lines = [
        f"t={t_idx}",
        f"workspace_norm={float(workspace_t.norm().item()):.3f}",
        f"memory_norm={float(memory_t.norm().item()):.3f}",
        f"report_norm={float(report_t.norm().item()):.3f}",
        "space: pause | n: next | q/esc: quit",
    ]
    draw_text_block(canvas, lines, x=280, y0=770, fs=0.52, th=1)

    return canvas


@hydra.main(version_base=None, config_path="../../config", config_name="conscious_system_tools")
def main(cfg_raw) -> None:
    base = OmegaConf.structured(ToolConfig())
    cfg = OmegaConf.merge(base, cfg_raw)
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    cfg_obj: ToolConfig = OmegaConf.to_object(cfg)

    ds = DynamicSelfWorldDataset(
        DynamicSelfWorldDatasetConfig(
            root=cfg_obj.viewer.dataset_root,
            seq_len=cfg_obj.system.data.seq_len,
            pose_dim=cfg_obj.system.data.pose_dim,
            use_depth=True,
            max_objects=cfg_obj.system.data.num_objects,
            include_hand_state=True,
        )
    )
    sample = ds[cfg_obj.viewer.index]
    batch = {k: (v.unsqueeze(0).to(cfg_obj.viewer.device) if isinstance(v, torch.Tensor) and v.ndim > 0 else v) for k, v in sample.items()}

    model = ConsciousSystem(cfg_obj.system).to(cfg_obj.viewer.device)
    ckpt = torch.load(Path(cfg_obj.viewer.ckpt), map_location=cfg_obj.viewer.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        outputs = model.forward_sequence(
            batch["left"],
            batch["right"],
            batch["pose"],
            batch["hand_state"],
            batch.get("target_action_ids"),
        )

    left = batch["left"][0]
    target_depth = batch["target_depth"][0]
    pred_rgb = outputs["rgb"][0]
    pred_depth = outputs["depth"][0]
    pred_action = outputs["action_ids"][0]
    focus_idx = outputs["focus_idx"][0]
    curiosity = outputs["curiosity"][0]
    coherence = outputs["coherence"][0]
    best_cf_action = outputs["best_cf_action"][0]
    report = outputs["report"][0]
    workspace = outputs["workspace"][0]
    memory = outputs["memory_summary"][0]
    thought_trace = outputs["thought_trace"][0]
    seq_len = left.shape[0]

    trails = {"workspace": [], "memory": [], "report": []}
    paused = False
    t = 0
    cv2.namedWindow("subjective stream viewer", cv2.WINDOW_NORMAL)

    while True:
        panel = render_frame(
            cfg_obj.viewer,
            left[t],
            pred_rgb[t],
            target_depth[t],
            pred_depth[t],
            workspace[t],
            memory[t],
            report[t],
            thought_trace[t],
            float(curiosity[t].item()),
            float(coherence[t].item()),
            int(focus_idx[t].item()),
            int(best_cf_action[t].item()),
            int(pred_action[t].item()),
            t,
            trails,
        )
        cv2.imshow("subjective stream viewer", panel)
        key = cv2.waitKey(0 if paused else cfg_obj.viewer.delay_ms) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
            continue
        if key == ord('n'):
            t = (t + 1) % seq_len
            continue
        if not paused:
            t += 1
            if t >= seq_len:
                if cfg_obj.viewer.loop:
                    t = 0
                    trails = {"workspace": [], "memory": [], "report": []}
                else:
                    break

    try:
        cv2.destroyWindow("subjective stream viewer")
    except Exception:
        pass


if __name__ == "__main__":
    main()
