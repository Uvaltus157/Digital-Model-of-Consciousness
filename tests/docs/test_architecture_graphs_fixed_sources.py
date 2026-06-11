from pathlib import Path

GRAPH_ROOT = Path("docs/architecture/graphs")

EXPECTED_GRAPHS = [
    "unconscious_contour_architecture.dot",
    "unconscious_contour_runtime_life_step.dot",
    "module_m1_architecture.dot",
    "module_m1_code.dot",
    "module_m2_event_dream_replay.dot",
    "module_m4_long_dynamic_memory.dot",
    "module_m5_seed_bus.dot",
]

def test_architecture_graphs_are_fixed_sources():
    assert GRAPH_ROOT.exists(), "Architecture graph folder must exist"
    for name in EXPECTED_GRAPHS:
        path = GRAPH_ROOT / name
        assert path.exists(), f"Missing architecture graph: {path}"
        text = path.read_text(encoding="utf-8")
        assert "digraph" in text

def test_old_m1_object_imagery_name_is_retired():
    assert not (GRAPH_ROOT / "module_m1_object_imagery.dot").exists(), (
        "module_m1_object_imagery.dot was renamed to module_m1_code.dot"
    )
