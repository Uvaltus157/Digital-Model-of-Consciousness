# Architecture DNA

This folder is the “DNA” of the project.

It is not only a file map. It describes how to reconstruct the whole system:

```text
modules
contracts
buses
guards
unconscious contour
conscious contour
imitators
monitors
validation levels
transition from imitators to real training
```

## Main read order

```text
1. architecture_genome.md
2. unconscious_loop_dna.md
3. conscious_loop_dna.md
4. module_contracts_dna.md
5. validation_levels_1_5_dna.md
6. imitators_dna.md
7. monitors_and_windows_dna.md
8. rebuild_from_dna.md
9. architecture_dna_manifest.json
```

## Code DNA layer

Implementation-level reconstruction docs live here:

```text
docs/architecture/dna/codegen/
```

Read:

```text
docs/architecture/dna/codegen/README.md
docs/architecture/dna/codegen/module_api_specs.md
docs/architecture/dna/codegen/runtime_wiring_dna.md
docs/architecture/dna/codegen/file_inventory_dna_template.json
docs/architecture/dna/codegen/config_dna.md
docs/architecture/dna/codegen/ui_dna.md
docs/architecture/dna/codegen/training_dna.md
docs/architecture/dna/codegen/reference_traces_dna.md
docs/architecture/dna/codegen/code_generation_order.md
```

## Main principle

```text
First imitate trained module outputs.
Then verify wiring/contracts/downstream behavior.
Then replace imitators with real trained modules.
Then compare trained latents against prototype/imit latents.
```

## Two contours

```text
Unconscious contour:
    M1 → M5 → M11 → M13/M4 → M2 → M5 → M3 guard

Conscious contour:
    M1/M5 → M10 → M9 → M7 → M15 → M12/M14/M13/M4 → M3 / M5 seed bus
```

## Mandatory imitator rule

All module-specific imitators must live under the module’s own `imit/` directory:

```text
src/modules/m01_object_imagery/imit/
src/modules/m05_world_model_attention_workspace/imit/
...
```

Never put imitator runtime files directly into the module root.
