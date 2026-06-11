from pathlib import Path
import re

GRAPH_ROOT = Path("docs/architecture/graphs")


def _cluster_blocks(text: str):
    """Simple brace counter for subgraph cluster blocks."""
    out = []
    for m in re.finditer(r"\bsubgraph\s+cluster_[A-Za-z0-9_]*\s*\{", text):
        start = m.start()
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out.append(text[start:i])
    return out


def test_dot_graphs_keep_edges_top_level():
    for path in GRAPH_ROOT.glob("*.dot"):
        text = path.read_text(encoding="utf-8")
        for block in _cluster_blocks(text):
            assert "->" not in block, f"Edges must be top-level, not inside cluster: {path}"


def test_dot_graphs_do_not_use_compound_cluster_edges():
    for path in GRAPH_ROOT.glob("*.dot"):
        text = path.read_text(encoding="utf-8")
        assert "ltail=" not in text, f"Do not use ltail in architecture DOT: {path}"
        assert "lhead=" not in text, f"Do not use lhead in architecture DOT: {path}"
        assert "compound=true" not in text.replace(" ", ""), f"Do not use compound=true in architecture DOT: {path}"


def test_nn_nodes_have_light_red_fill():
    for path in GRAPH_ROOT.glob("*.dot"):
        text = path.read_text(encoding="utf-8")
        # Find node blocks whose visible label starts with NN:
        for m in re.finditer(r'\[[^\]]*label\s*=\s*"NN:[\s\S]*?\]', text):
            block = m.group(0)
            assert 'fillcolor="#ffd6d6"' in block, f"NN node must be light-red: {path}\n{block[:200]}"
