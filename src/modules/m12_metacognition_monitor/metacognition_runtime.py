from __future__ import annotations

import torch

from src.modules.m12_metacognition_monitor.metacognition_monitor import (
    MetacognitionMonitor,
    MetacognitionMonitorConfig,
    render_metacognition_text,
)


class MetacognitionRuntimeMixin:
    def ensure_metacognition_ready(self) -> None:
        if hasattr(self, "metacognition_monitor") and self.metacognition_monitor is not None:
            return

        self_bound_dim = (
            int(getattr(self.cfg.self_core, "self_dim", 128))
            + int(getattr(self.cfg.self_core, "focus_context_dim", 256))
            + int(getattr(self.cfg.self_core, "affect_latent_dim", 12))
        )
        # M12 only needs the self_state portion directly; self_bound_context can
        # still be represented by focus/affect/report/plan inputs.
        cfg = MetacognitionMonitorConfig(
            enabled=bool(getattr(getattr(self.cfg, "metacognition", None), "enabled", True)),
            self_dim=int(getattr(self.cfg.self_core, "self_dim", 128)),
            focus_context_dim=int(getattr(self.cfg.self_core, "focus_context_dim", 256)),
            affect_latent_dim=int(getattr(self.cfg.self_core, "affect_latent_dim", 12)),
            report_latent_dim=int(getattr(getattr(self.cfg, "inner_speech", None), "report_latent_dim", 128)),
            plan_context_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "plan_context_dim", 256)),
            hidden_dim=int(getattr(getattr(self.cfg, "metacognition", None), "hidden_dim", 256)),
            verification_threshold=float(getattr(getattr(self.cfg, "metacognition", None), "verification_threshold", 0.55)),
            doubt_threshold=float(getattr(getattr(self.cfg, "metacognition", None), "doubt_threshold", 0.50)),
            action_hold_threshold=float(getattr(getattr(self.cfg, "metacognition", None), "action_hold_threshold", 0.70)),
        )
        del self_bound_dim
        self.metacognition_monitor = MetacognitionMonitor(cfg).to(self.device)
        try:
            if hasattr(self, "module_training_gate"):
                self.module_training_gate.apply()
                self.rebuild_optimizer_from_trainable_modules()
            else:
                self.optimizer.add_param_group({"params": self.metacognition_monitor.parameters()})
        except Exception as e:
            print(f"[metacognition] lazy optimizer attach skipped: {e}")
        print("[metacognition] lazy initialized")

    def _metacog_scalar(self, value, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().float().reshape(-1)[0].cpu().item())
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _metacog_scalar_tensor(self, out: dict) -> torch.Tensor:
        sc = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        bc = out.get("broadcast", {}) if isinstance(out.get("broadcast"), dict) else {}
        report = out.get("inner_speech", {}) if isinstance(out.get("inner_speech"), dict) else {}
        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        metrics = tc.get("thought_chain_metrics", {}) if isinstance(tc.get("thought_chain_metrics"), dict) else {}

        values = [
            self._metacog_scalar(sc.get("agency_score"), 0.0),
            self._metacog_scalar(sc.get("self_continuity_score"), 0.0),
            self._metacog_scalar(report.get("confidence"), 0.0),
            self._metacog_scalar(sc.get("focus_binding_score"), 0.0),
            self._metacog_scalar(sc.get("affect_binding_score"), 0.0),
            self._metacog_scalar(metrics.get("planning_readiness"), 0.0),
            self._metacog_scalar(tc.get("best_chain_score"), 0.0),
            self._metacog_scalar(tc.get("predicted_affect_delta"), 0.0),
            self._metacog_scalar(affect.get("panic_latent"), 0.0),
            self._metacog_scalar(tc.get("no_viable_chain"), 0.0),
            self._metacog_scalar(bc.get("priority"), 0.0),
            self._metacog_scalar(bc.get("broadcast_gate"), 0.0),
        ]
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def compute_metacognition(self, obs: dict, out: dict):
        del obs
        if not bool(getattr(getattr(self.cfg, "metacognition", None), "enabled", True)):
            return None

        self.ensure_metacognition_ready()
        sc = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}
        report = out.get("inner_speech", {}) if isinstance(out.get("inner_speech"), dict) else {}
        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}

        meta = self.metacognition_monitor(
            self_state=sc.get("self_state"),
            focus_context=out.get("focus_context"),
            affect_latents=affect.get("affect_latents"),
            report_latent=report.get("report_latent"),
            plan_context=tc.get("plan_context") if isinstance(tc, dict) else out.get("plan_context"),
            scalar_features=self._metacog_scalar_tensor(out),
        )
        meta["text"] = render_metacognition_text(meta)
        meta["source"] = "m12_after_m9_m7"
        out["metacognition"] = meta
        out["metacognition_text"] = meta["text"]
        return meta

    def maybe_print_metacognition_trace(self, out: dict) -> None:
        cfg = getattr(self.cfg, "metacognition", None)
        every = int(getattr(cfg, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        meta = out.get("metacognition")
        if not isinstance(meta, dict):
            print(f"[metacognition step={self.global_step}] no metacognition output")
            return

        def f(x):
            try:
                if torch.is_tensor(x):
                    return float(x.detach().cpu().reshape(-1)[0].item())
                return float(x)
            except Exception:
                return 0.0

        print(
            f"[metacognition step={self.global_step}] "
            f"conf={f(meta.get('metacognitive_confidence')):.3f} "
            f"doubt={f(meta.get('doubt')):.3f} "
            f"verify={f(meta.get('verification_need')):.3f} "
            f"hold={f(meta.get('action_hold')):.3f} | "
            f"{meta.get('text', '')}"
        )


__all__ = ["MetacognitionRuntimeMixin"]
