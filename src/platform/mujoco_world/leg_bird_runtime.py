from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class LegBirdRuntimeMixin:
    def init_leg_control_head(self):
        """
        Separate neural head for bird legs.

        It reads the current shared latent / workspace state and produces:
            out["leg_ctrl"] with shape [B, leg_motor_dim]

        This separates:
            hand_ctrl -> hands/fingers
            leg_ctrl  -> bird legs/toes
        """
        if not self.cfg.leg_control.enabled:
            self.leg_control_head = None
            return

        # Workspace dimension is inferred lazily on first forward if needed.
        # Prefer cfg.hidden_dim if workspace shape is unknown here.
        in_dim = int(getattr(self.cfg.model, "hidden_dim", self.cfg.leg_control.hidden_dim)) if hasattr(self.cfg, "model") else int(self.cfg.leg_control.hidden_dim)

        self.leg_control_head = nn.Sequential(
            nn.Linear(in_dim, self.cfg.leg_control.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.leg_control.hidden_dim, self.cfg.leg_control.leg_motor_dim),
            nn.Tanh(),
        ).to(self.device)

        # Include it in the same optimizer, so it learns with the rest of the model.
        try:
            self.optimizer.add_param_group({"params": self.leg_control_head.parameters()})
            print("[leg_control] separate leg_control_head added to optimizer")
        except Exception as e:
            print(f"[leg_control] optimizer add_param_group skipped: {e}")

        self.prev_leg_motor = torch.zeros(1, self.cfg.leg_control.leg_motor_dim, device=self.device)


    def compute_leg_control(self, out: dict) -> torch.Tensor:
        if not getattr(self.cfg.leg_control, "enabled", True) or self.leg_control_head is None:
            return torch.zeros(1, self.cfg.leg_control.leg_motor_dim, device=self.device)

        latent = out.get("workspace_out", None)
        if latent is None:
            latent = out.get("object_repr", None)

        if latent is None:
            return self.prev_leg_motor

        if latent.ndim > 2:
            latent = latent.reshape(latent.shape[0], -1)

        # If actual latent dim differs from lazy guessed input dim, rebuild once.
        first_linear = self.leg_control_head[0]
        if int(latent.shape[-1]) != int(first_linear.in_features):
            self.leg_control_head = nn.Sequential(
                nn.Linear(int(latent.shape[-1]), self.cfg.leg_control.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.cfg.leg_control.hidden_dim, self.cfg.leg_control.leg_motor_dim),
                nn.Tanh(),
            ).to(self.device)
            try:
                self.optimizer.add_param_group({"params": self.leg_control_head.parameters()})
                print(f"[leg_control] rebuilt head for in_dim={int(latent.shape[-1])}")
            except Exception as e:
                print(f"[leg_control] rebuilt but optimizer add skipped: {e}")

        leg = 0.35 * self.leg_control_head(latent.detach())
        alpha = float(self.cfg.leg_control.smoothing)
        leg = (1.0 - alpha) * self.prev_leg_motor + alpha * leg
        self.prev_leg_motor = leg.detach()
        return leg


    def init_bird_leg_actuators(self):
        import mujoco
        self.bird_leg_act_ids = []
        names = []
        for side in ("left", "right"):
            names.extend([
                f"act_{side}_hip_yaw",
                f"act_{side}_hip_pitch",
                f"act_{side}_knee",
                f"act_{side}_ankle_pitch",
                f"act_{side}_ankle_roll",
                f"act_{side}_toe_front_inner_joint",
                f"act_{side}_toe_front_mid_joint",
                f"act_{side}_toe_front_outer_joint",
                f"act_{side}_toe_rear_joint",
            ])
        for name in names:
            aid = mujoco.mj_name2id(self.world.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.bird_leg_act_ids.append(aid)
        self.prev_bird_leg_ctrl = np.zeros(len(self.bird_leg_act_ids), dtype=np.float64)
        found = sum(1 for a in self.bird_leg_act_ids if a >= 0)
        print(f"[bird_body] leg actuators found: {found}/{len(self.bird_leg_act_ids)}")


    def apply_bird_leg_controls(self, leg_ctrl_tensor):
        if not getattr(self.cfg.bird_body, "enabled", True):
            return
        if not hasattr(self, "bird_leg_act_ids"):
            self.init_bird_leg_actuators()

        ctrl = leg_ctrl_tensor.detach().cpu().numpy().reshape(-1)
        n = len(self.bird_leg_act_ids)
        if n <= 0:
            return

        if ctrl.size >= n:
            leg_raw = ctrl[:n]
        else:
            leg_raw = np.zeros(n, dtype=np.float64)
            leg_raw[:ctrl.size] = ctrl

        alpha = float(self.cfg.bird_body.leg_smoothing)
        leg_raw = np.clip(leg_raw, -1.0, 1.0)
        self.prev_bird_leg_ctrl = (1.0 - alpha) * self.prev_bird_leg_ctrl + alpha * leg_raw

        for aid, value in zip(self.bird_leg_act_ids, self.prev_bird_leg_ctrl):
            if aid >= 0:
                self.world.data.ctrl[aid] = float(value)

