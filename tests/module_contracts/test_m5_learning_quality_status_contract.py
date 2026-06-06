from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import torch


class DummyM5System:
    def __init__(self):
        self.global_step = 1
        self.train_steps = 0
        self.training_enabled = False
        self.cfg = SimpleNamespace(train=SimpleNamespace(enabled=False))
        self.last_train_loss = 1.0
        self.last_train_reason = "dummy"
        self.last_train_error = ""
        self.latest_object_decoder_stats = {"recon": 0.9}
        self.latest_long_dynamic_memory_stats = {"reward_proxy": 0.2}
        self.latest_out = {
            "focus_context": torch.ones(1, 256) * 0.1,
            "workspace_out": torch.ones(1, 256) * 0.2,
            "obs_embed": torch.ones(1, 128) * 0.3,
            "event_dream_replay": {
                "next_focus_context_seed": torch.ones(1, 256),
                "next_focus_context_seed_gate": torch.tensor([[0.5]]),
            },
            "focus_feedback": {"total_gate": torch.tensor([[0.4]])},
            "long_dynamic_memory": {
                "identity_stability": torch.tensor([[0.3]]),
                "identity_novelty": torch.tensor([[0.7]]),
            },
            "prediction_error": torch.tensor([[0.8]]),
            "reconstruction_error": torch.tensor([[0.9]]),
            "latent_coherence": torch.tensor([[0.2]]),
        }

    def is_full_sleep_mode(self):
        return False


def test_m5_learning_quality_baseline_payload_and_trends():
    from src.modules.m08_debug_visual_control.m5_learning_quality_status import build_m5_learning_quality_status

    system = DummyM5System()
    first = build_m5_learning_quality_status(system)

    for key in (
        "global_step",
        "train_steps",
        "training_enabled",
        "cfg_train_enabled",
        "full_sleep",
        "verdict",
        "learning_quality",
        "learning_quality_ema",
        "last_train_reason",
        "last_train_error",
        "m5_loss",
        "m5_latent",
        "m5_seed_response",
        "object_identity_proxy",
        "baseline",
        "current",
        "delta",
        "history",
        "samples",
        "note",
    ):
        assert key in first

    assert first["verdict"] in ("idle", "tracking", "improving", "training_error", "seed_reactive_untrained", "untrained_or_no_data")
    assert 0.0 <= first["learning_quality"] <= 1.0
    assert 0.0 <= first["learning_quality_ema"] <= 1.0
    assert first["m5_loss"]["train_loss"] == 1.0
    assert first["m5_seed_response"]["seed_gate"] == 0.5
    assert first["m5_seed_response"]["seed_norm"] > 0
    assert "baseline" in first

    system.global_step = 2
    system.train_steps = 1
    system.training_enabled = True
    system.cfg.train.enabled = True
    system.last_train_loss = 0.7
    system.latest_out["prediction_error"] = torch.tensor([[0.6]])
    system.latest_out["reconstruction_error"] = torch.tensor([[0.7]])
    system.latest_out["latent_coherence"] = torch.tensor([[0.35]])

    second = build_m5_learning_quality_status(system)

    assert second["m5_loss"]["train_loss_delta"] < 0
    assert second["m5_loss"]["prediction_error_delta"] < 0
    assert second["m5_loss"]["reconstruction_error_delta"] < 0
    assert second["m5_latent"]["latent_coherence_delta"] > 0
    assert second["learning_quality"] > 0
    assert len(second["history"]) >= 2

    repeated = build_m5_learning_quality_status(system)
    assert repeated["m5_loss"]["train_loss_delta"] == second["m5_loss"]["train_loss_delta"]
    assert repeated["m5_loss"]["prediction_error_delta"] == second["m5_loss"]["prediction_error_delta"]
    assert repeated["m5_loss"]["reconstruction_error_delta"] == second["m5_loss"]["reconstruction_error_delta"]
    assert repeated["m5_latent"]["latent_coherence_delta"] == second["m5_latent"]["latent_coherence_delta"]


def test_m5_learning_quality_verdict_precedence():
    from src.modules.m08_debug_visual_control.m5_learning_quality_status import build_m5_learning_quality_status

    system = DummyM5System()
    system.last_train_loss = 0.0
    payload = build_m5_learning_quality_status(system)
    assert payload["verdict"] == "seed_reactive_untrained"

    system.last_train_error = "boom"
    payload = build_m5_learning_quality_status(system)
    assert payload["verdict"] == "training_error"


def test_m5_learning_quality_control_panel_button_contract():
    source = Path("src/modules/m08_debug_visual_control/control_panel.py").read_text(encoding="utf-8")

    assert 'self.btn_m5_learning_quality = QtWidgets.QPushButton("M5 Learning Quality")' in source
    assert '"btn_m5_learning_quality"' in source
    assert "self.btn_m5_learning_quality.clicked.connect(self.open_m5_learning_quality_window)" in source
    assert "def open_m5_learning_quality_window(self):" in source
    assert "def refresh_m5_learning_quality_window(self):" in source
    assert "self.refresh_m5_learning_quality_window()" in source
    assert '"m5": ["btn_m5_learning_quality", "btn_m5_latent_prototype"]' in source
    assert '"m8": ["btn_module_debug", "btn_module_lab", "btn_sleep_replay_monitor", "btn_replay_quality_monitor"]' in source
    assert "self._style_plain_status_button(\n                self.btn_m5_learning_quality," in source
    assert 'self._window_visible(getattr(self, "m5_learning_quality_window", None))' in source
    assert "self.btn_replay_quality_monitor,\n                self.btn_m5_learning_quality," not in source
