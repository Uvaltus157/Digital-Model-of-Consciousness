from __future__ import annotations

"""Registry-backed launcher for the PyQt Module Debug window.

This wrapper keeps the existing UI/runtime behavior but replaces the hard-coded
module list, edges, presets and positions with definitions derived from the
canonical M8 module registry.

Use this while the original `pyqt_module_debug_ipc_status.py` is gradually
refactored. It avoids a risky full rewrite of the large PyQt file.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "runner.yaml").exists():
            sys.path.insert(0, str(parent))
            break

from src.modules.m08_debug_visual_control import module_debug_graph as graph
from src.modules.m08_debug_visual_control import pyqt_module_debug_ipc_status as legacy


def apply_registry_graph() -> None:
    """Patch the legacy PyQt module-debug globals from the registry bridge."""
    legacy.MODULES = list(graph.LEGACY_TRAINING_MODULES)
    legacy.EDGES = list(graph.LEGACY_TRAINING_EDGES)
    legacy.DEFAULT_FLAGS = dict(graph.DEFAULT_FLAGS)
    legacy.COLLECTIVE_PRESETS = list(graph.COLLECTIVE_PRESETS)

    original_init = legacy.DiagramCanvas.__init__

    def registry_init(self):
        original_init(self)
        self.positions = dict(graph.LEGACY_POSITIONS)

    if getattr(legacy.DiagramCanvas.__init__, "__name__", "") != "registry_init":
        legacy.DiagramCanvas.__init__ = registry_init


def main() -> None:
    apply_registry_graph()
    legacy.main()


if __name__ == "__main__":
    main()
