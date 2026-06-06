from pathlib import Path
import json


def test_architecture_dna_docs_live_under_dna_folder():
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "docs/architecture/dna/architecture_dna_manifest.json"
    assert manifest_path.exists(), "DNA manifest must live under docs/architecture/dna/"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["root"] == "docs/architecture/dna"

    for rel in manifest["required_docs"]:
        assert (root / rel).exists(), f"missing DNA doc: {rel}"

    for rel in manifest["legacy_root_docs_to_remove"]:
        assert not (root / rel).exists(), f"legacy root-level DNA doc should be removed: {rel}"

    assert "dna_docs_live_under_docs_architecture_dna" in manifest["rules"]
    assert "imitators_live_under_module_imit_dir" in manifest["rules"]


def test_architecture_root_readme_points_to_dna_folder():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture/README.md").read_text(encoding="utf-8")
    assert "docs/architecture/dna/" in text
    assert "current_structure.md" in text
