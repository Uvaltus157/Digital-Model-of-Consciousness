from pathlib import Path
import json


def test_codegen_dna_manifest_and_docs_exist():
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "docs/architecture/dna/codegen/codegen_manifest.json"
    assert manifest_path.exists(), "codegen_manifest.json is missing"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["root"] == "docs/architecture/dna/codegen"

    for rel in manifest["required_docs"]:
        assert (root / rel).exists(), f"missing Code DNA doc: {rel}"

    assert "M1 object slot path" in manifest["minimum_regeneration_targets"]
    assert "conscious contour" in manifest["minimum_regeneration_targets"]


def test_module_api_specs_mentions_core_contracts():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/architecture/dna/codegen/module_api_specs.md").read_text(encoding="utf-8")

    assert "M1ObjectSlotLatentImitRuntimeMixin" in text
    assert "M5LatentPrototypeRuntimeMixin" in text
    assert "FocusFeedbackBoundary" in text
    assert "GlobalConsciousBroadcastRuntimeMixin" in text
    assert "CounterfactualPlanningRuntimeMixin" in text


def test_file_inventory_template_is_valid_json():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "docs/architecture/dna/codegen/file_inventory_dna_template.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["files"]
    assert "public_methods" in data["required_fields_per_file"]
