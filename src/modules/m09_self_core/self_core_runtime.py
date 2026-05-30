from __future__ import annotations

import torch

from src.modules.m09_self_core.models.self_core import (
    SelfCore,
    SelfCoreConfig,
    build_self_experience_text,
    pad_or_trim_selfcore,
)


class SelfCoreRuntimeMixin:
    def ensure_self_core_ready(self):
        if hasattr(self, "self_core") and self.self_core is not None:
            return
        self.self_core = SelfCore(SelfCoreConfig(
            enabled=self.cfg.self_core.enabled,
            body_state_dim=self.cfg.body_state_dim,
            action_dim=self.cfg.embodied_dim,
            tactile_dim=self.cfg.tactile_dim,
            vestibular_dim=24,
            object_latent_dim=self.cfg.self_core.object_latent_dim,
            workspace_dim=self.cfg.self_core.workspace_dim,
            focus_context_dim=getattr(self.cfg.self_core, "focus_context_dim", self.cfg.self_core.workspace_dim),
            affect_latent_dim=getattr(self.cfg.self_core, "affect_latent_dim", 12),
            hidden_dim=self.cfg.self_core.hidden_dim,
            self_dim=self.cfg.self_core.self_dim,
        )).to(self.device)
        self.self_core_state = self.self_core.initial_state(batch_size=1, device=self.device)
        try:
            if hasattr(self, "module_training_gate"):
                self.module_training_gate.apply()
                self.rebuild_optimizer_from_trainable_modules()
            else:
                self.optimizer.add_param_group({"params": self.self_core.parameters()})
        except Exception as e:
            print(f"[self_core] lazy optimizer attach skipped: {e}")
        print("[self_core] lazy initialized")

    def _build_self_core_focus_context(self, out: dict, workspace: torch.Tensor) -> torch.Tensor:
        """
        Read the M5-owned focus_context.

        Stage-2 architecture rule:
            M5 owns out["focus_context"].
            M9 must not manually reconstruct focus from M5 internals.

        If focus_context is absent, return a zero tensor instead of silently
        rebuilding the old implicit focus packet. That keeps older partial runs
        from crashing while making missing M5 focus easy to detect in stats.
        """
        target_dim = int(getattr(self.cfg.self_core, "focus_context_dim", self.cfg.self_core.workspace_dim))
        focus_context = out.get("focus_context")
        if torch.is_tensor(focus_context):
            if focus_context.ndim > 2:
                focus_context = focus_context.reshape(focus_context.shape[0], -1)
            return pad_or_trim_selfcore(focus_context.float(), target_dim, device=self.device)

        batch = int(workspace.shape[0]) if torch.is_tensor(workspace) and workspace.ndim >= 1 else 1
        return torch.zeros(batch, target_dim, device=self.device)

    def _build_self_core_affect_latents(self, out: dict, workspace: torch.Tensor) -> torch.Tensor:
        """
        Read the M10/M11 affect packet.

        Stage-4 architecture rule:
            M9 binds M5 focus_context with M10/M11 affect_latents.
            M9 should not know the internal emotion scalar formulas.
        """
        target_dim = int(getattr(self.cfg.self_core, "affect_latent_dim", 12))
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        affect_latents = affect.get("affect_latents")
        if torch.is_tensor(affect_latents):
            if affect_latents.ndim > 2:
                affect_latents = affect_latents.reshape(affect_latents.shape[0], -1)
            return pad_or_trim_selfcore(affect_latents.float(), target_dim, device=self.device)

        batch = int(workspace.shape[0]) if torch.is_tensor(workspace) and workspace.ndim >= 1 else 1
        return torch.zeros(batch, target_dim, device=self.device)

    def compute_self_core(self, obs: dict, out: dict):
        if not self.cfg.self_core.enabled:
            return None
        self.ensure_self_core_ready()

        body_state = obs.get("body_state", torch.zeros(1, self.cfg.body_state_dim, device=self.device))
        tactile = obs.get("tactile", torch.zeros(1, self.cfg.tactile_dim, device=self.device))
        vestibular = obs.get("vestibular", torch.zeros(1, 24, device=self.device))
        action = out.get("embodied_targets", torch.zeros(1, self.cfg.embodied_dim, device=self.device))

        # object latent: prefer inner object slot if present
        object_latent = None
        inner_obj = out.get("inner_object")
        if isinstance(inner_obj, dict) and "z_obj" in inner_obj:
            object_latent = inner_obj["z_obj"]
        if object_latent is None:
            object_latent = torch.zeros(1, self.cfg.self_core.object_latent_dim, device=self.device)

        # workspace latent: try common keys; otherwise zeros
        workspace = None
        for key in ("workspace_out", "workspace", "global_workspace", "belief_state"):
            if key in out and torch.is_tensor(out[key]):
                workspace = out[key]
                break
        if workspace is None:
            workspace = torch.zeros(1, self.cfg.self_core.workspace_dim, device=self.device)
        if workspace.ndim > 2:
            workspace = workspace.reshape(workspace.shape[0], -1)

        focus_context = self._build_self_core_focus_context(out, workspace)
        affect_latents = self._build_self_core_affect_latents(out, workspace)

        sc = self.self_core(
            self.self_core_state,
            body_state=body_state,
            action=action,
            tactile=tactile,
            vestibular=vestibular,
            object_latent=object_latent,
            workspace=workspace,
            focus_context=focus_context,
            affect_latents=affect_latents,
        )
        self.self_core_state = {
            "self_state": sc["self_state"].detach(),
            "predicted_self_state": sc["predicted_self_state"].detach(),
            "agency_score": sc["agency_score"].detach(),
            "body_ownership_score": sc["body_ownership_score"].detach(),
            "self_continuity_score": sc["self_continuity_score"].detach(),
        }
        sc["self_experience_text"] = build_self_experience_text(sc)
        sc["focus_context_present"] = torch.tensor(
            [1.0 if torch.is_tensor(out.get("focus_context")) else 0.0],
            device=self.device,
            dtype=sc["self_state"].dtype,
        )
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        sc["affect_latents_present"] = torch.tensor(
            [1.0 if torch.is_tensor(affect.get("affect_latents")) else 0.0],
            device=self.device,
            dtype=sc["self_state"].dtype,
        )
        out["self_core"] = sc
        out["self_experience_text"] = sc["self_experience_text"]
        return sc

    def maybe_print_self_core_trace(self, out: dict):
        if not self.cfg.self_core.enabled:
            return
        if self.global_step % max(1, self.cfg.self_core.print_every_steps) != 0:
            return
        sc = out.get("self_core")
        if not isinstance(sc, dict):
            print(f"[self_core step={self.global_step}] no self_core output")
            return
        def f(name):
            try:
                return float(sc[name].detach().cpu().reshape(-1)[0].item())
            except Exception:
                return 0.0
        print(
            f"[self_core step={self.global_step}] "
            f"agency={f('agency_score'):.3f} "
            f"ownership={f('body_ownership_score'):.3f} "
            f"continuity={f('self_continuity_score'):.3f} "
            f"focus_binding={f('focus_binding_score'):.3f} "
            f"affect_binding={f('affect_binding_score'):.3f} "
            f"focus_context_present={f('focus_context_present'):.0f} "
            f"affect_latents_present={f('affect_latents_present'):.0f} "
            f"uncertainty={f('self_uncertainty'):.3f} "
            f"curiosity={f('self_curiosity'):.3f} | "
            f"{sc.get('self_experience_text', '')}"
        )
