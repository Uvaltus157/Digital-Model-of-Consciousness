from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_optimizer import (
    collect_trainable_named_parameters,
    count_parameter_elements,
    rebuild_optimizer_from_trainable_modules_for_system,
)


class DummyParam:
    def __init__(self, n: int):
        self.n = n

    def numel(self) -> int:
        return self.n


class DummyGate:
    flags = {"world_model": True}

    def __init__(self, params):
        self.params = params

    def trainable_named_parameters(self):
        return list(self.params)

    def count_trainable(self):
        return {"total": sum(p.numel() for _name, p in self.params)}


class DummyOptimizer:
    def __init__(self, params, lr, weight_decay):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay


def test_collect_trainable_named_parameters() -> None:
    gate = DummyGate([("a", DummyParam(3)), ("b", DummyParam(4))])
    items = collect_trainable_named_parameters(gate)
    assert [name for name, _p in items] == ["a", "b"]


def test_count_parameter_elements() -> None:
    assert count_parameter_elements([DummyParam(3), DummyParam(4), object()]) == 7


def test_rebuild_optimizer_no_params_keeps_existing_optimizer() -> None:
    system = SimpleNamespace(
        module_training_gate=DummyGate([]),
        cfg=SimpleNamespace(train=SimpleNamespace(lr=0.001, weight_decay=0.01)),
        optimizer="existing",
    )
    snapshot = rebuild_optimizer_from_trainable_modules_for_system(system, optimizer_factory=DummyOptimizer)

    assert snapshot.rebuilt is False
    assert snapshot.reason == "no_trainable_parameters"
    assert system.optimizer == "existing"


def test_rebuild_optimizer_sets_optimizer_when_params_exist() -> None:
    system = SimpleNamespace(
        module_training_gate=DummyGate([("a", DummyParam(3)), ("b", DummyParam(4))]),
        cfg=SimpleNamespace(train=SimpleNamespace(lr=0.002, weight_decay=0.03)),
        optimizer=None,
    )
    snapshot = rebuild_optimizer_from_trainable_modules_for_system(system, optimizer_factory=DummyOptimizer)

    assert snapshot.rebuilt is True
    assert snapshot.selected_params == 7
    assert snapshot.selected_tensors == 2
    assert isinstance(system.optimizer, DummyOptimizer)
    assert system.optimizer.lr == 0.002
    assert system.optimizer.weight_decay == 0.03
