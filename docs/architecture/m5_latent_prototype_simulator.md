# M5 Latent Prototype Simulator

This is an **imitator**, so the runtime lives under:

```text
src/modules/m05_world_model_attention_workspace/imit/m5_latent_prototype_runtime.py
```

It simulates internal learned M5 object latents:

```text
cube
tetrahedron
cube↔tetrahedron morph
```

It does not train or overwrite M5 weights.

The simulated latent enters M5 through:

```text
focus_context_seed
focus_context_seed_gate
↓
FocusFeedbackBoundary
```

Use it to test downstream wiring before the real M5 world model is trained:

```text
M5 seed/boundary path
M5 Learning Quality seed response
Replay Quality response
Sleep Replay Monitor reaction
M11/M4/M2 downstream effects
```

Interpretation:

```text
Circuit response = wiring works.
It does NOT mean M5 has actually learned cube/tetrahedron.
```
