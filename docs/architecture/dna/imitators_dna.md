# Imitators DNA

Imitators behave as if a neural module was already trained.

## Rule

All imitators live under:

```text
src/modules/<module>/imit/
```

## Known imitators / probes

| Imitator | Module | Purpose | Injection point |
|---|---|---|---|
| M1 Object Slot Latent Imit | M1 | fill slots with cube/tetra z_obj | build_inner_object_vision_proposals |
| M5 Latent Prototype Imit | M5 | seed world/workspace with prototypes | common M5 seed bus |
| Dream Probe | M2/M11 | stimulate stress/replay seed | pre-M11/M2/M5 |
| Replay Material Imit | M2 | force selected replay material | M2 replay payload |
| Memory Episode Imit | M13 | force selected episode summary | M13 memory payload |
| Emotion Probe | M11 | force stress/relief/curiosity | M11 affect path |

## Required status fields

```text
active
kind
remaining
duration
source
layout = imit
target
selected_slot / gate / seed_norm
details
note
```

## Safety

```text
does not train
does not call optimizer.step()
does not mutate real weights
does not pretend to be learned
publishes source/layout clearly
uses normal bus/path where possible
```
