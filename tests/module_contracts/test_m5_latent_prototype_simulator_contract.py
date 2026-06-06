from __future__ import annotations

from pathlib import Path

import torch


class DummySystem:
    from src.modules.m05_world_model_attention_workspace.imit.m5_latent_prototype_runtime import M5LatentPrototypeRuntimeMixin

    _m5_latent_prototype_dim = M5LatentPrototypeRuntimeMixin._m5_latent_prototype_dim
    make_m5_latent_prototype = M5LatentPrototypeRuntimeMixin.make_m5_latent_prototype
    request_m5_latent_prototype = M5LatentPrototypeRuntimeMixin.request_m5_latent_prototype
    get_m5_latent_prototype_focus_seed = M5LatentPrototypeRuntimeMixin.get_m5_latent_prototype_focus_seed
    m5_latent_prototype_status = M5LatentPrototypeRuntimeMixin.m5_latent_prototype_status

    def __init__(self):
        self.global_step = 7
        self.latest_out = {"focus_context": torch.zeros(1, 256)}
        self.status_writes = 0

    def write_module_debug_status(self):
        self.status_writes += 1


def test_m5_latent_prototype_lives_only_in_imit_layout():
    assert Path("src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py").exists()
    assert not Path("src/modules/m05_world_model_attention_workspace/m5_latent_prototype_runtime.py").exists()


def test_cube_and_tetra_prototypes_are_stable_and_separated_from_imit():
    system = DummySystem()
    cube, cube_desc = system.make_m5_latent_prototype("cube")
    tetra, tetra_desc = system.make_m5_latent_prototype("tetrahedron")

    assert tuple(cube.shape) == (1, 256)
    assert tuple(tetra.shape) == (1, 256)
    assert cube_desc["faces"] == 6.0
    assert tetra_desc["faces"] == 4.0
    sim = torch.nn.functional.cosine_similarity(cube, tetra, dim=-1).item()
    assert sim < 0.98


def test_request_m5_latent_prototype_from_imit_returns_seed_and_status():
    system = DummySystem()
    state = system.request_m5_latent_prototype({"kind": "cube", "gate": 0.9, "duration": 3})

    assert state["active"] is True
    assert state["kind"] == "cube"
    assert state["layout"] == "imit"
    assert state["cube_similarity"] > state["tetra_similarity"]
    assert system.status_writes == 1

    seed, gate = system.get_m5_latent_prototype_focus_seed(stage="pre_observe")
    assert torch.is_tensor(seed)
    assert torch.is_tensor(gate)
    assert tuple(seed.shape) == (1, 256)
    assert abs(float(gate.item()) - 0.9) < 1e-6
    assert system._m5_latent_prototype_state["remaining"] == 2


def test_status_payload_marks_simulated_latent_as_imit():
    system = DummySystem()
    system.request_m5_latent_prototype({"kind": "tetrahedron", "gate": 0.75, "duration": 2})

    from src.modules.m08_debug_visual_control.m5_latent_prototype_status import build_m5_latent_prototype_status

    status = build_m5_latent_prototype_status(system)
    assert status["active"] is True
    assert status["kind"] == "tetrahedron"
    assert status["layout"] == "imit"
    assert status["is_simulated_learned_latent"] is True
    assert status["seed_norm"] > 0


def test_m5_learning_quality_sees_latent_prototype_seed_response():
    system = DummySystem()
    system.request_m5_latent_prototype({"kind": "cube", "gate": 0.8, "duration": 3})
    seed, gate = system.get_m5_latent_prototype_focus_seed(stage="pre_observe")
    assert torch.is_tensor(seed)
    assert torch.is_tensor(gate)

    from src.modules.m08_debug_visual_control.m5_learning_quality_status import build_m5_learning_quality_status

    status = build_m5_learning_quality_status(system)
    assert status["m5_seed_response"]["seed_gate"] > 0
    assert status["m5_seed_response"]["seed_norm"] > 0
    assert status["m5_seed_response"]["seed_response"] > 0


def test_m5_latent_prototype_runner_uses_imit_mixin():
    source = Path("src/apps/runner.py").read_text(encoding="utf-8")

    assert (
        "from src.modules.m05_world_model_attention_workspace.imit."
        "m5_latent_prototype_runtime import M5LatentPrototypeRuntimeMixin"
    ) in source
    assert "M5LatentPrototypeRuntimeMixin," in source


def test_m5_latent_prototype_control_panel_contract():
    source = Path("src/modules/m08_debug_visual_control/control_panel.py").read_text(encoding="utf-8")

    assert 'self.btn_m5_latent_prototype = QtWidgets.QPushButton("M5 Latent Prototypes")' in source
    assert '"btn_m5_latent_prototype"' in source
    assert "self.btn_m5_latent_prototype.clicked.connect(self.open_m5_latent_prototype_window)" in source
    assert "def open_m5_latent_prototype_window(self):" in source
    assert "def refresh_m5_latent_prototype_window(self):" in source
    assert "self.refresh_m5_latent_prototype_window()" in source
    assert "m5_latent_prototype_inject" in source
    assert 'source="m8_m5_latent_prototype_window"' in source
    assert '"m5": ["btn_m5_learning_quality", "btn_m5_latent_prototype"]' in source
    assert '"m8": ["btn_module_debug", "btn_module_lab", "btn_sleep_replay_monitor", "btn_replay_quality_monitor"]' in source
    assert "Inject cube latent" in source
    assert "Inject tetrahedron latent" in source
    assert "Inject cube↔tetra morph" in source
    assert "Clear" in source
    assert "self._style_plain_status_button(\n                self.btn_m5_latent_prototype," in source
    assert 'self._window_visible(getattr(self, "m5_latent_prototype_window", None))' in source
    assert "self.btn_m5_learning_quality,\n                self.btn_m5_latent_prototype," not in source
