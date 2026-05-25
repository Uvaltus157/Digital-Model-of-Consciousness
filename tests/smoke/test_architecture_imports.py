from __future__ import annotations

import importlib


LIGHTWEIGHT_MODULES = [
    "src.apps.bootstrap",
    "src.apps.runtime_wiring",
    "src.apps.system_factory",
    "src.shared.config",
    "src.modules.m08_debug_visual_control.module_registry",
    "src.modules.m08_debug_visual_control.module_debug_graph",
    "src.modules.m08_debug_visual_control.status_schema",
    "src.modules.m10_global_conscious_broadcast.state",
    "src.modules.m10_global_conscious_broadcast.runtime",
    "src.modules.m10_global_conscious_broadcast.debug",
    "src.modules.m13_autobiographical_memory.state",
    "src.modules.m13_autobiographical_memory.memory",
    "src.modules.m13_autobiographical_memory.runtime",
    "src.modules.m13_autobiographical_memory.debug",
]


def test_lightweight_architecture_imports() -> None:
    """Import dependency-light architecture modules.

    This test intentionally avoids the heavy runner, MuJoCo, Open3D and PyQt
    entrypoints. Its purpose is to catch accidental import cycles in the new
    architecture/scaffold layer.
    """
    for module_name in LIGHTWEIGHT_MODULES:
        importlib.import_module(module_name)
