from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch.nn as nn


@dataclass
class ModuleTrainingDebugConfig:
    enabled: bool = True
    window_name: str = "module debug / отладка модулей"
    width: int = 760
    height: int = 520
    show_every_steps: int = 2

    # Default: train all modules.
    train_core_model: bool = True
    train_action_heads: bool = True
    train_world_model: bool = True
    train_object_imagery: bool = True
    train_long_dynamic_memory: bool = True
    train_inner_speech: bool = True
    train_leg_control: bool = True
    train_self_core: bool = True


DEFAULT_MODULE_FLAGS = {
    "core_model": True,
    "action_heads": True,
    "world_model": True,
    "object_imagery": True,
    "long_dynamic_memory": True,
    "inner_speech": True,
    "leg_control": True,
    "self_core": True,
}


def set_requires_grad(module: nn.Module | None, flag: bool):
    if module is None:
        return
    for p in module.parameters(recurse=True):
        p.requires_grad = bool(flag)


def count_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(int(p.numel()) for p in module.parameters() if getattr(p, "requires_grad", False))


class ModuleTrainingGate:
    """
    Controls which module groups are trainable.

    It does not remove losses. It gates gradients by requires_grad, then the
    runner can rebuild the optimizer using only trainable parameters.
    """

    def __init__(self, system):
        self.system = system
        self.flags = dict(DEFAULT_MODULE_FLAGS)

    def set_flags(self, flags: Dict[str, bool]):
        for k, v in flags.items():
            if k in self.flags:
                self.flags[k] = bool(v)
        self.apply()

    def toggle(self, key: str):
        if key in self.flags:
            self.flags[key] = not self.flags[key]
        self.apply()

    def apply(self):
        s = self.system

        # Start from everything off, then selectively enable.
        if hasattr(s, "model"):
            set_requires_grad(s.model, False)

        # External modules.
        set_requires_grad(getattr(s, "inner_object_system", None), self.flags.get("object_imagery", True))
        set_requires_grad(getattr(s, "long_dynamic_object_memory", None), self.flags.get("long_dynamic_memory", True))
        set_requires_grad(getattr(s, "leg_control_head", None), self.flags.get("leg_control", True))
        set_requires_grad(getattr(s, "self_core", None), self.flags.get("self_core", True))

        model = getattr(s, "model", None)
        if model is not None:
            if self.flags.get("core_model", True):
                # Enable whole core model first.
                set_requires_grad(model, True)

            # More selective controls override broad core_model flag.
            # Uses fuzzy module-name matching so it survives small architecture changes.
            for name, module in model.named_modules():
                lname = name.lower()

                if any(k in lname for k in ("action", "embodied", "hand", "policy", "motor")):
                    set_requires_grad(module, self.flags.get("action_heads", True))

                if any(k in lname for k in ("world", "dreamer", "transition", "rssm", "dynamics", "decoder", "encoder")):
                    set_requires_grad(module, self.flags.get("world_model", True))

                if any(k in lname for k in ("symbolic", "speech", "report", "text", "phoneme")):
                    set_requires_grad(module, self.flags.get("inner_speech", True))

        return self.flags

    def trainable_named_parameters(self) -> Iterable[Tuple[str, object]]:
        s = self.system
        modules = [
            ("model", getattr(s, "model", None)),
            ("inner_object_system", getattr(s, "inner_object_system", None)),
            ("long_dynamic_object_memory", getattr(s, "long_dynamic_object_memory", None)),
            ("leg_control_head", getattr(s, "leg_control_head", None)),
            ("self_core", getattr(s, "self_core", None)),
        ]
        seen = set()
        for prefix, module in modules:
            if module is None:
                continue
            for name, p in module.named_parameters(recurse=True):
                if id(p) in seen:
                    continue
                seen.add(id(p))
                if getattr(p, "requires_grad", False):
                    yield f"{prefix}.{name}", p

    def _count_model_matching(self, words: tuple[str, ...]) -> int:
        model = getattr(self.system, "model", None)
        if model is None:
            return 0

        seen = set()
        total = 0
        for name, module in model.named_modules():
            lname = name.lower()
            if any(w in lname for w in words):
                for p in module.parameters(recurse=True):
                    if id(p) not in seen and getattr(p, "requires_grad", False):
                        seen.add(id(p))
                        total += int(p.numel())
        return total

    def count_trainable(self) -> Dict[str, int]:
        s = self.system

        model = getattr(s, "model", None)
        inner_object_system = getattr(s, "inner_object_system", None)
        long_dynamic_object_memory = getattr(s, "long_dynamic_object_memory", None)
        leg_control_head = getattr(s, "leg_control_head", None)
        self_core = getattr(s, "self_core", None)

        counts = {
            # UI keys used by pyqt_module_debug_ipc_status.py cards.
            "core_model": count_params(model),
            "world_model": self._count_model_matching(("world", "dreamer", "transition", "rssm", "dynamics", "decoder", "encoder")),
            "action_heads": self._count_model_matching(("action", "embodied", "hand", "policy", "motor")),
            "object_imagery": count_params(inner_object_system),
            "long_dynamic_memory": count_params(long_dynamic_object_memory),
            "inner_speech": self._count_model_matching(("symbolic", "speech", "report", "text", "phoneme")),
            "leg_control": count_params(leg_control_head),
            "self_core": count_params(self_core),

            # Backward-compatible/internal keys.
            "model": count_params(model),
            "inner_object_system": count_params(inner_object_system),
            "long_dynamic_object_memory": count_params(long_dynamic_object_memory),
            "leg_control_head": count_params(leg_control_head),
        }

        # Total is computed from real parameter stream to avoid double-counting
        # fuzzy UI subgroup overlaps inside model.
        seen = set()
        total = 0
        for _name, p in self.trainable_named_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += int(p.numel())
        counts["total"] = total

        return counts
