# Validation Levels 1–5 DNA

## Level 1 — wiring and contracts

```text
Verify IPC, module contracts, tensor shapes, status payloads, M8 monitors, guards.
```

Includes both contours:

```text
unconscious: M1/M5/M11/M13/M4/M2/M3
conscious: M1/M5/M10/M7/M15/M12/M9/M14/M3
```

## Level 2 — simulated learned latents/content

```text
Inject learned-like latents and content as if neural modules were already trained.
```

Examples:

```text
M1 imit: cube z_obj → slot 1, tetra z_obj → slot 2
M5 imit: cube/tetra/morph focus_context_seed
M11 probe: stress/relief/curiosity
M13 imit: selected episode
M2 imit: replay material
M10 imit: conscious broadcast packet
M10 conscious broadcast probe
M7 imit: inner speech text
M7 inner speech probe
M15 imit: imagined rollout/candidate plan
M15 counterfactual planning probe
M12 imit: confidence/uncertainty
M12 confidence/uncertainty probe
```

## Level 3 — downstream behavior as if trained

```text
Verify the whole system reacts to imit/prototype outputs as if they came from trained modules.
```

Pass examples:

```text
Inner Object 3D shows selected slot
M5 Learning Quality sees seed_response
Replay Quality changes verdict/quality
M11/M4/M13/M2 update status
M10 broadcast visible
M7 subjective stream visible
M15 plan reaches action candidate path
M12 uncertainty visible
M3 remains blocked in sleep
```

## Level 4 — replace imit with real training

```text
Keep the same contracts, but replace imit source with trained module output.
```

Pass:

```text
train_steps increase
loss decreases
real output has same contract as imit output
downstream behavior remains stable
```

## Level 5 — compare real latents vs prototype latents/content

Metrics:

```text
cosine_similarity
L2/MSE distance
slot match
confidence gap
cluster overlap
downstream equivalence
text/content equivalence for conscious stream
plan equivalence for M15
```

Pass:

```text
real latent/content ≈ prototype/imit latent/content
same slot/content selected
same downstream reaction
imit can be disabled for that module
```
