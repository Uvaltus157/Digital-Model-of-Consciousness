from __future__ import annotations

"""JSON-friendly status schema for M8 Debug Visual Control.

This module intentionally stays dependency-light. It can be imported by GUI,
IPC and tests without importing MuJoCo, Torch, Open3D or PyQt.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ModuleDebugStatus:
    module_id: str
    title: str
    active: bool = False
    trainable: bool = False
    training_enabled: bool = False
    mode: str = "unknown"
    confidence: float = 0.0
    error: str = ""
    updated_at: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["active"] = bool(self.active)
        data["trainable"] = bool(self.trainable)
        data["training_enabled"] = bool(self.training_enabled)
        data["confidence"] = float(self.confidence)
        data["updated_at"] = float(self.updated_at)
        return data


@dataclass
class ModuleDebugStatusPayload:
    ready: bool = False
    updated_at: float = 0.0
    modules: Dict[str, ModuleDebugStatus] = field(default_factory=dict)
    trainable_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": bool(self.ready),
            "updated_at": float(self.updated_at),
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "trainable_counts": dict(self.trainable_counts),
        }
