from __future__ import annotations

"""Runtime wiring boundary for cross-module integration.

This file is intentionally dependency-light. It should not import MuJoCo,
Torch, Open3D, PyQt, or the heavy runner.

The active V5.10 runtime is still composed in `src/apps/runner.py`. This module
holds small helpers and a documented wiring plan so future refactors can move
cross-module integration out of individual M-modules without changing behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class RuntimeWiringContext:
    """Shared app-level context for future module wiring."""

    modules: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def register_module(self, name: str, module: Any) -> None:
        self.modules[str(name)] = module

    def register_service(self, name: str, service: Any) -> None:
        self.services[str(name)] = service

    def get_module(self, name: str, default: Any = None) -> Any:
        return self.modules.get(str(name), default)

    def get_service(self, name: str, default: Any = None) -> Any:
        return self.services.get(str(name), default)


@dataclass(frozen=True)
class RuntimeMixinSpec:
    module_id: str
    mixin_name: str
    import_path: str
    reason: str


@dataclass
class RuntimeWiringPlan:
    mixins: List[RuntimeMixinSpec] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_mixin(self, module_id: str, mixin_name: str, import_path: str, reason: str) -> None:
        self.mixins.append(RuntimeMixinSpec(module_id, mixin_name, import_path, reason))

    def by_module(self, module_id: str) -> List[RuntimeMixinSpec]:
        return [spec for spec in self.mixins if spec.module_id == module_id]


def build_current_object_imagery_wiring_plan() -> RuntimeWiringPlan:
    """Document mixins that are currently composed through M1 runtime.

    This does not change runtime behavior. It only gives the next refactor a
    single source of truth for moving cross-module wiring upward into `apps/`.
    """
    plan = RuntimeWiringPlan()
    plan.notes.append("M1 runtime is currently an integration point; move these links to apps/runtime_wiring.py gradually.")
    plan.add_mixin("M8", "StaticDynamicCodeDebugRuntimeMixin", "src.modules.m08_debug_visual_control.static_dynamic_code_debug_runtime", "debug visualization state")
    plan.add_mixin("M4", "DynamicObjectPassportRuntimeMixin", "src.modules.m04_long_dynamic_memory.dynamic_object_passport_runtime", "object passport state")
    plan.add_mixin("M4", "PassportDebugRuntimeMixin", "src.modules.m04_long_dynamic_memory.passport_debug_runtime", "object passport debug")
    plan.add_mixin("M15", "InnerScenarioMindRuntimeMixin", "src.modules.m15_counterfactual_imagination_planning.inner_scenario_mind_runtime", "scenario diagnostics")
    plan.add_mixin("M3", "InnerActionDecoderRuntimeMixin", "src.modules.m03_self_action_causality.inner_action_decoder_runtime", "inner action decoding")
    plan.add_mixin("M3", "InnerOutcomeEvaluatorRuntimeMixin", "src.modules.m03_self_action_causality.inner_outcome_evaluator_runtime", "outcome evaluation")
    plan.add_mixin("M12", "InnerTrustGateRuntimeMixin", "src.modules.m12_metacognition_monitor.inner_trust_gate_runtime", "trust gate")
    plan.add_mixin("M3", "InnerRealActionTraceRuntimeMixin", "src.modules.m03_self_action_causality.inner_real_action_trace_runtime", "real action trace")
    return plan


def call_if_present(target: Any, method_name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    method: Optional[Callable[..., Any]] = getattr(target, method_name, None)
    if method is None:
        return default
    return method(*args, **kwargs)


def collect_available_hooks(target: Any, names: Iterable[str]) -> Dict[str, Callable[..., Any]]:
    hooks: Dict[str, Callable[..., Any]] = {}
    for name in names:
        value = getattr(target, str(name), None)
        if callable(value):
            hooks[str(name)] = value
    return hooks
