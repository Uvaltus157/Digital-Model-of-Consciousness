from pathlib import Path


def test_conscious_overview_has_m9_before_inner_speech():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture/dna/conscious_loop_dna.md").read_text(encoding="utf-8")

    overview_start = text.index("## 1. Conscious contour overview")
    m10_pos = text.index("M10 Global Conscious Broadcast", overview_start)
    m9_pos = text.index("M9 Self Core", overview_start)
    m7_pos = text.index("M7 Inner Speech", overview_start)

    assert m10_pos < m9_pos < m7_pos
    window = text[m9_pos:m9_pos + 160].lower()
    assert "body" in window or "agency" in window or "ownership" in window


def test_genome_conscious_loop_has_m9_explicitly():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture/dna/architecture_genome.md").read_text(encoding="utf-8")
    assert "M9 self core" in text or "M9 Self Core" in text
    assert "M3 controlled action proposal / ActionGuard" in text
