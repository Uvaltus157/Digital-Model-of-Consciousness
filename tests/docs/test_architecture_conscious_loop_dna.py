from pathlib import Path
import json


def test_architecture_dna_has_both_contours():
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "docs/architecture/dna/architecture_dna_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "unconscious" in manifest["contours"]
    assert "conscious" in manifest["contours"]
    assert "M2" in manifest["contours"]["unconscious"]
    assert "M10" in manifest["contours"]["conscious"]
    assert "M15" in manifest["contours"]["conscious"]

    for rel in manifest["required_docs"]:
        assert (root / rel).exists(), f"missing DNA doc: {rel}"


def test_conscious_loop_doc_mentions_seed_boundary_and_m3_guard():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture/dna/conscious_loop_dna.md").read_text(encoding="utf-8")

    assert "M10" in text
    assert "M7" in text
    assert "M15" in text
    assert "FocusFeedbackBoundary" in text
    assert "M3" in text and "blocked" in text


def test_genome_mentions_two_contours():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture/dna/architecture_genome.md").read_text(encoding="utf-8")
    assert "Strict unconscious loop" in text
    assert "Strict conscious loop" in text
