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

    def _ensure_affect_packet_for_self_core(self, obs: dict, out: dict) -> None:
        """
        Stage-4/5 bridge.

        M13/M15, M10 and M9 need affect_latents before self binding. Create the same
        affect packet here when it is not already present. EmotionalDrive caches
        the packet on out["emotion"], so LifeRuntime can reuse it later without a
        second EMA/progress update.
        """
        affect = out.get("affect")
        if isinstance(affect, dict) and torch.is_tensor(affect.get("affect_latents")):
            return
        if not hasattr(self, "emotional_drive") or self.emotional_drive is None:
            return
        try:
            emotion = self.emotional_drive.compute(out, obs)
            out["emotion"] = emotion
            if isinstance(emotion.get("affect"), dict):
                out["affect"] = emotion["affect"]
        except Exception as e:
            if not hasattr(self, "_self_core_affect_warned"):
                print(f"[self_core] affect precompute skipped: {e}")
                self._self_core_affect_warned = True

    def _run_pre_self_autobiographical_retrieval(self, obs: dict, out: dict) -> None:
        """
        M13 pre-self retrieval.

        Runs after M11 affect packet is available and before M15. It may blend a
        retrieved autobiographical context into M5 focus_context, so M15 searches
        chains with past self-relevant episodes available.
        """
        if not hasattr(self, "compute_autobiographical_retrieval"):
            return
        try:
            self.compute_autobiographical_retrieval(obs, out)
        except Exception as e:
            if not hasattr(self, "_autobiographical_retrieval_warned"):
                print(f"[autobiographical_memory] pre-self retrieval skipped: {e}")
                self._autobiographical_retrieval_warned = True

    def _run_pre_self_thought_chain(self, obs: dict, out: dict) -> None:
        """
        Correct architecture order:
            M5 focus_context + M11 affect_latents + M13 memory -> M15 chain search
            M15 writes best chain back into M5 focus_context
            M10 selects/broadcasts conscious-access material
            M9 self-binds the broadcast focus_context
        """
        if not hasattr(self, "compute_thought_chain"):
            return
        try:
            self.compute_thought_chain(obs, out, pre_self_binding=True)
        except TypeError:
            # Compatibility with older ThoughtChainRuntimeMixin signature.
            try:
                self.compute_thought_chain(obs, out)
            except Exception as e:
                if not hasattr(self, "_thought_chain_warned"):
                    print(f"[thought_chain] pre-self compute skipped: {e}")
                    self._thought_chain_warned = True
        except Exception as e:
            if not hasattr(self, "_thought_chain_warned"):
                print(f"[thought_chain] pre-self compute skipped: {e}")
                self._thought_chain_warned = True

    def _run_pre_self_global_broadcast(self, obs: dict, out: dict) -> None:
        """
        M10 global conscious broadcast gate.

        Runs after M15 chain search and before M9 self-binding. It writes
        out["broadcast"] and replaces out["focus_context"] with the selected
        broadcast_latent used by M9.
        """
        if not hasattr(self, "compute_global_broadcast"):
            return
        try:
            self.compute_global_broadcast(obs, out)
        except Exception as e:
            if not hasattr(self, "_global_broadcast_warned"):
                print(f"[global_broadcast] pre-self compute skipped: {e}")
                self._global_broadcast_warned = True

    def _build_self_core_focus_context(self, out: dict, workspace: torch.Tensor) -> torch.Tensor:
        """
        Read the M5/M13/M15/M10-owned focus_context.

        M5 creates the focus_context. M13 may blend retrieved memory into it.
        M15 may enhance it with the best chain. M10 may replace it with
        broadcast_latent. M9 must not manually reconstruct focus from lower-level
        internals.
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
        Read the M11 affect packet.

        M9 binds M10 broadcast/focus_context with M11 affect_latents.
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
        self._ensure_affect_packet_for_self_core(obs, out)
        self._run_pre_self_autobiographical_retrieval(obs, out)
        self._run_pre_self_thought_chain(obs, out)
        self._run_pre_self_global_broadcast(obs, out)

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
        bc = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}
        sc["broadcast_present"] = torch.tensor(
            [1.0 if torch.is_tensor(bc.get("broadcast_latent")) else 0.0],
            device=self.device,
            dtype=sc["self_state"].dtype,
        )
        memory = out.get("autobiographical_memory", {}) if isinstance(out.get("autobiographical_memory"), dict) else {}
        sc["autobiographical_memory_present"] = torch.tensor(
            [1.0 if torch.is_tensor(memory.get("retrieved_context")) else 0.0],
            device=self.device,
            dtype=sc["self_state"].dtype,
        )
        out["self_core"] = sc
        out["self_experience_text"] = sc["self_experience_text"]

        # Stage-6 bridge: M7 verbalizes self-bound focus after M9.
        if hasattr(self, "compute_inner_speech"):
            try:
                self.compute_inner_speech(obs, out)
            except Exception as e:
                if not hasattr(self, "_inner_speech_warned"):
                    print(f"[inner_speech] compute skipped: {e}")
                    self._inner_speech_warned = True

        # Stage-7 bridge: M12 monitors confidence/doubt after M9+M7.
        if hasattr(self, "compute_metacognition"):
            try:
                self.compute_metacognition(obs, out)
            except Exception as e:
                if not hasattr(self, "_metacognition_warned"):
                    print(f"[metacognition] compute skipped: {e}")
                    self._metacognition_warned = True
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
            f"broadcast_present={f('broadcast_present'):.0f} "
            f"memory_present={f('autobiographical_memory_present'):.0f} "
            f"uncertainty={f('self_uncertainty'):.3f} "
            f"curiosity={f('self_curiosity'):.3f} | "
            f"{sc.get('self_experience_text', '')}"
        )
