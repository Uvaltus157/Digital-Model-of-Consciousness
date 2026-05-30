from __future__ import annotations

import numpy as np
import torch


class LifeReportHelpersMixin:
    def _life_runtime_scalar(self, value, default: float = 0.0) -> float:
        """Safely convert optional model/runtime outputs to a Python float."""
        if value is None:
            return float(default)
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float(default)
                return float(value.detach().float().reshape(-1)[0].cpu().item())
            return float(value)
        except Exception:
            return float(default)

    def _life_runtime_nested_get(self, data: dict, *keys: str):
        cur = data
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def _life_runtime_self_confidence(self, out: dict) -> float:
        """
        Runtime fallback for confidence while M7/M12/self-core boundaries are
        being split. Prefer the old reflection field when present, then try
        explicit self-core confidence, then derive a conservative self-core
        score from agency/ownership/continuity.
        """
        direct = self._life_runtime_nested_get(out, "reflection_out", "self_confidence")
        if direct is not None:
            return self._life_runtime_scalar(direct, 0.0)

        self_core = out.get("self_core", {}) if isinstance(out.get("self_core"), dict) else {}
        for key in ("self_confidence", "confidence", "reflective_confidence"):
            if key in self_core:
                return self._life_runtime_scalar(self_core.get(key), 0.0)

        parts = [
            self_core.get("agency_score"),
            self_core.get("body_ownership_score"),
            self_core.get("self_continuity_score"),
        ]
        vals = [self._life_runtime_scalar(v, float("nan")) for v in parts if v is not None]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            return float(np.clip(np.mean(vals), 0.0, 1.0))
        return 0.0

    def _life_runtime_inner_report(self, obs: dict, out: dict) -> tuple[str, str, float]:
        """
        Build human-readable inner speech without assuming M7 has already added
        out["symbolic_report"]. Missing reports become empty strings/confidence 0.
        """
        symbolic = out.get("symbolic_report", {}) if isinstance(out.get("symbolic_report"), dict) else {}
        confidence = self._life_runtime_scalar(symbolic.get("confidence"), 0.0)

        decoded_report = ""
        token_ids = symbolic.get("text_token_ids")
        if token_ids is not None and hasattr(self, "speech_vocab") and self.speech_vocab is not None:
            try:
                ids = token_ids
                if torch.is_tensor(ids) and ids.ndim > 1:
                    ids = ids[0]
                decoded_report = str(self.speech_vocab.decode(ids, skip_special=True))
            except Exception:
                decoded_report = ""

        if not decoded_report:
            for key in ("text", "decoded_text", "report_text"):
                value = symbolic.get(key)
                if value:
                    decoded_report = str(value)
                    break

        target_report = ""
        if hasattr(self, "speech_teacher") and self.speech_teacher is not None:
            try:
                target_report = str(self.speech_teacher.build_report(obs, out))
            except Exception:
                target_report = ""

        return decoded_report, target_report, confidence


__all__ = ["LifeReportHelpersMixin"]
