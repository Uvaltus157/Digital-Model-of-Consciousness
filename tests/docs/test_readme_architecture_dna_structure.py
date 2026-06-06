from pathlib import Path


def test_readme_mentions_architecture_dna_and_codegen():
    root = Path(__file__).resolve().parents[2]
    text = (root / "README.md").read_text(encoding="utf-8")

    assert "docs/architecture/dna/architecture_genome.md" in text
    assert "docs/architecture/dna/conscious_loop_dna.md" in text
    assert "docs/architecture/dna/codegen/module_api_specs.md" in text
    assert "Code DNA" in text
    assert "Architecture DNA" in text


def test_readme_mentions_two_contours_and_levels():
    root = Path(__file__).resolve().parents[2]
    text = (root / "README.md").read_text(encoding="utf-8")

    assert "Unconscious contour" in text
    assert "Conscious contour" in text
    assert "M10" in text and "M9" in text and "M15" in text
    assert "Level 1: wiring and contracts" in text
    assert "Level 5: compare real latents/content" in text
