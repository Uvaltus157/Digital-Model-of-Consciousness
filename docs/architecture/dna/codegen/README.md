# Code DNA

This folder describes how to regenerate implementation code from Architecture DNA.

Architecture DNA answers:

```text
what the organism is
what modules exist
how contours are connected
what must be verified
```

Code DNA answers:

```text
what files to create
what classes to expose
what methods each module must implement
what status payload keys exist
what tensor shapes are expected
what UI actions exist
what training loops replace imitators
what reference traces prove the organism works
```

## Read order

```text
1. module_api_specs.md
2. runtime_wiring_dna.md
3. file_inventory_dna_template.json
4. config_dna.md
5. ui_dna.md
6. training_dna.md
7. reference_traces_dna.md
8. code_generation_order.md
9. codegen_manifest.json
```

## Goal

After this layer is complete, a developer or Codex should be able to rebuild a compatible project implementation:

```text
same modules
same public methods
same payload contracts
same debug windows
same tests
same validation levels
```

Not necessarily byte-identical old code, but architecturally compatible code.
