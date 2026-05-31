from __future__ import annotations

import torch

from src.modules.m07_inner_speech_thoughts.inner_speech_decoder import (
    InnerSpeechDecoder,
    InnerSpeechDecoderConfig,
    render_inner_speech_text,
)


class InnerSpeechRuntimeMixin:
    def ensure_inner_speech_ready(self) -> None:
        if hasattr(self, "inner_speech_decoder") and self.inner_speech_decoder is not None:
            return

        self_bound_dim = (
            int(getattr(self.cfg.self_core, "self_dim", 128))
            + int(getattr(self.cfg.self_core, "focus_context_dim", 256))
            + int(getattr(self.cfg.self_core, "affect_latent_dim", 12))
        )
        cfg = InnerSpeechDecoderConfig(
            enabled=bool(getattr(getattr(self.cfg, "inner_speech", None), "enabled", True)),
            active_thought_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "thought_dim", 128)),
            plan_context_dim=int(getattr(getattr(self.cfg, "thought_chain", None), "plan_context_dim", 256)),
            self_bound_context_dim=self_bound_dim,
            subjective_affect_dim=16,
            affect_latent_dim=int(getattr(self.cfg.self_core, "affect_latent_dim", 12)),
            hidden_dim=int(getattr(getattr(self.cfg, "inner_speech", None), "hidden_dim", 256)),
            report_latent_dim=int(getattr(getattr(self.cfg, "inner_speech", None), "report_latent_dim", 128)),
            vocab_size=int(getattr(getattr(self.cfg, "inner_speech", None), "vocab_size", 2048)),
            max_tokens=int(getattr(getattr(self.cfg, "inner_speech", None), "max_tokens", 24)),
        )
        self.inner_speech_decoder = InnerSpeechDecoder(cfg).to(self.device)
        try:
            if hasattr(self, "module_training_gate"):
                self.module_training_gate.apply()
                self.rebuild_optimizer_from_trainable_modules()
            else:
                self.optimizer.add_param_group({"params": self.inner_speech_decoder.parameters()})
        except Exception as e:
            print(f"[inner_speech] lazy optimizer attach skipped: {e}")
        print("[inner_speech] lazy initialized")

    def _inner_speech_scalar(self, value, default: float = 0.0) -> float:
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

    def _life_runtime_inner_report(self, obs: dict, out: dict) -> tuple[str, str, float]:
        """
        Runtime report bridge used by LifeRuntimeMixin.life_step().

        Because InnerSpeechRuntimeMixin appears before LifeRuntimeMixin in the
        UnifiedSystem MRO, this method overrides the legacy helper. Prefer
        self-bound M7 outputs and keep symbolic_report only as a final fallback.
        """
        del obs
        report = {}
        report_key = ""
        for key in ("inner_speech", "conscious_report", "m7_inner_speech", "symbolic_report"):
            value = out.get(key)
            if isinstance(value, dict):
                report = value
                report_key = key
                break

        scalar = getattr(self, "_life_runtime_scalar", None)
        if callable(scalar):
            confidence = scalar(report.get("confidence"), 0.0)
        else:
            confidence = self._inner_speech_scalar(report.get("confidence"), 0.0)

        decoded_report = ""
        for key in ("text", "decoded_text", "report_text"):
            value = report.get(key)
            if value:
                decoded_report = str(value)
                break

        token_ids = report.get("text_token_ids")
        if not decoded_report and token_ids is not None and hasattr(self, "speech_vocab") and self.speech_vocab is not None:
            try:
                ids = token_ids
                if torch.is_tensor(ids) and ids.ndim > 1:
                    ids = ids[0]
                decoded_report = str(self.speech_vocab.decode(ids, skip_special=True))
            except Exception:
                decoded_report = ""

        tc = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        metrics = tc.get("thought_chain_metrics", {}) if isinstance(tc.get("thought_chain_metrics"), dict) else {}
        self._latest_inner_speech_diagnostics = {
            "inner_speech_source": str(report.get("source", report_key)),
            "uses_thought_chain": bool(self._inner_speech_scalar(report.get("uses_thought_chain"), 0.0) > 0.5),
            "uses_self_bound_context": bool(self._inner_speech_scalar(report.get("uses_self_bound_context"), 0.0) > 0.5),
            "uses_affect_latents": bool(self._inner_speech_scalar(report.get("uses_affect_latents"), 0.0) > 0.5),
            "thought_chain_planning_readiness": self._inner_speech_scalar(metrics.get("planning_readiness"), 0.0),
        }

        target_report = ""
        if hasattr(self, "speech_teacher") and self.speech_teacher is not None:
            try:
                target_report = str(self.speech_teacher.build_report({}, out))
            except Exception:
                target_report = ""

        return decoded_report, target_report, confidence

    def compute_inner_speech(self, obs: dict, out: dict):
        del obs
        if not bool(getattr(getattr(self.cfg, "inner_speech", None), "enabled", True)):
            return None

        thought_chain = out.get("thought_chain", {}) if isinstance(out.get("thought_chain"), dict) else {}
        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        affect = out.get("affect", {}) if isinstance(out.get("affect"), dict) else {}

        active_thought = thought_chain.get("active_thought")
        # Compatibility fallback for partial runs before M15 is active.
        if not torch.is_tensor(active_thought):
            pre = out.get("preconscious_thoughts", {}) if isinstance(out.get("preconscious_thoughts"), dict) else {}
            active_thought = pre.get("thought_candidate")
        if not torch.is_tensor(active_thought):
            return None

        plan_context = thought_chain.get("plan_context")
        if not torch.is_tensor(plan_context):
            plan_context = out.get("plan_context")

        self.ensure_inner_speech_ready()
        report = self.inner_speech_decoder(
            active_thought=active_thought,
            plan_context=plan_context,
            self_bound_context=self_core.get("self_bound_context"),
            subjective_affect_state=self_core.get("subjective_affect_state"),
            affect_latents=affect.get("affect_latents"),
        )
        confidence = 0.0
        try:
            confidence = float(report["confidence"].detach().cpu().reshape(-1)[0].item())
        except Exception:
            confidence = 0.0

        text = render_inner_speech_text(
            self_core=self_core,
            thought_chain=thought_chain,
            affect=affect,
            confidence=confidence,
        )
        report["text"] = text
        report["decoded_text"] = text
        report["report_text"] = text
        report["source"] = "m7_self_bound_thought_chain" if isinstance(thought_chain, dict) and torch.is_tensor(thought_chain.get("active_thought")) else "m7_compat_preconscious_fallback"
        report["uses_self_bound_context"] = torch.tensor([1.0 if torch.is_tensor(self_core.get("self_bound_context")) else 0.0], device=self.device)
        report["uses_affect_latents"] = torch.tensor([1.0 if torch.is_tensor(affect.get("affect_latents")) else 0.0], device=self.device)
        report["uses_thought_chain"] = torch.tensor([1.0 if torch.is_tensor(thought_chain.get("active_thought")) else 0.0], device=self.device)

        metrics = thought_chain.get("thought_chain_metrics", {}) if isinstance(thought_chain.get("thought_chain_metrics"), dict) else {}
        self._latest_inner_speech_diagnostics = {
            "inner_speech_source": str(report["source"]),
            "uses_thought_chain": bool(self._inner_speech_scalar(report.get("uses_thought_chain"), 0.0) > 0.5),
            "uses_self_bound_context": bool(self._inner_speech_scalar(report.get("uses_self_bound_context"), 0.0) > 0.5),
            "uses_affect_latents": bool(self._inner_speech_scalar(report.get("uses_affect_latents"), 0.0) > 0.5),
            "thought_chain_planning_readiness": self._inner_speech_scalar(metrics.get("planning_readiness"), 0.0),
        }

        out["inner_speech"] = report
        out["conscious_report"] = report
        # Compatibility output only: old visualizers/life_runtime may still read
        # symbolic_report, but M7 no longer uses symbolic_report as input.
        out["symbolic_report"] = report
        out["decoded_report"] = text
        return report

    def maybe_print_inner_speech_trace(self, out: dict) -> None:
        cfg = getattr(self.cfg, "inner_speech", None)
        every = int(getattr(cfg, "print_every_steps", 30))
        if every <= 0 or self.global_step % every != 0:
            return
        report = out.get("inner_speech")
        if not isinstance(report, dict):
            print(f"[inner_speech step={self.global_step}] no inner_speech output")
            return
        conf = report.get("confidence")
        try:
            conf_v = float(conf.detach().cpu().reshape(-1)[0].item()) if torch.is_tensor(conf) else float(conf or 0.0)
        except Exception:
            conf_v = 0.0
        print(
            f"[inner_speech step={self.global_step}] "
            f"source={report.get('source', '')} "
            f"confidence={conf_v:.3f} | {report.get('text', '')}"
        )


__all__ = ["InnerSpeechRuntimeMixin"]
