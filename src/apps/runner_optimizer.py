from __future__ import annotations

"""Optimizer helpers for the V5.10 runner.

This is a safe extraction boundary for optimizer rebuild logic. The helper keeps
behavior equivalent to the previous `rebuild_optimizer_from_trainable_modules()`
implementation while making the logic testable and independent from the heavy
runner file.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


NamedParameter = Tuple[str, Any]


@dataclass(frozen=True)
class OptimizerRebuildSnapshot:
    selected_params: int
    selected_tensors: int
    lr: float
    weight_decay: float
    rebuilt: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def collect_trainable_named_parameters(module_training_gate: Any) -> List[NamedParameter]:
    """Collect trainable named parameters from the module training gate."""
    if module_training_gate is None or not hasattr(module_training_gate, "trainable_named_parameters"):
        return []
    return list(module_training_gate.trainable_named_parameters())


def count_parameter_elements(params: Iterable[Any]) -> int:
    """Count parameter elements without importing torch at module import time."""
    total = 0
    for param in params:
        try:
            total += int(param.numel())
        except Exception:
            total += 0
    return total


def build_adamw_optimizer(
    params: Sequence[Any],
    lr: float,
    weight_decay: float,
    optimizer_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Build an AdamW optimizer for selected parameters."""
    if optimizer_factory is None:
        import torch.optim as optim

        optimizer_factory = optim.AdamW
    return optimizer_factory(params, lr=float(lr), weight_decay=float(weight_decay))


def rebuild_optimizer_from_trainable_modules_for_system(
    system: Any,
    optimizer_factory: Optional[Callable[..., Any]] = None,
) -> OptimizerRebuildSnapshot:
    """Method-compatible optimizer rebuild for `UnifiedSystem` instances."""
    named_params = collect_trainable_named_parameters(getattr(system, "module_training_gate", None))
    params = [param for _name, param in named_params]
    lr = float(getattr(system.cfg.train, "lr", 0.0))
    weight_decay = float(getattr(system.cfg.train, "weight_decay", 0.0))

    if not params:
        print("[module_debug] no trainable parameters selected; optimizer keeps one frozen dummy-free state")
        return OptimizerRebuildSnapshot(
            selected_params=0,
            selected_tensors=0,
            lr=lr,
            weight_decay=weight_decay,
            rebuilt=False,
            reason="no_trainable_parameters",
        )

    system.optimizer = build_adamw_optimizer(params, lr=lr, weight_decay=weight_decay, optimizer_factory=optimizer_factory)
    counts = system.module_training_gate.count_trainable() if hasattr(system.module_training_gate, "count_trainable") else {}
    selected_params = int(counts.get("total", count_parameter_elements(params))) if isinstance(counts, dict) else count_parameter_elements(params)
    flags = getattr(system.module_training_gate, "flags", {})
    print(f"[module_debug] optimizer rebuilt | trainable_total={selected_params:,} | flags={flags}")
    return OptimizerRebuildSnapshot(
        selected_params=selected_params,
        selected_tensors=len(params),
        lr=lr,
        weight_decay=weight_decay,
        rebuilt=True,
        reason="rebuilt",
    )
