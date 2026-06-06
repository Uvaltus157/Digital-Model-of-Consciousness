# Training DNA

This document describes how imitators are replaced by real training.

## Rule

Do not remove an imitator until the trained module matches the same contract.

```text
imit output contract
↓
trained output contract
↓
same downstream behavior
↓
latent/content comparison passes
↓
imit can be disabled
```

Required phrase: imit output contract → trained output contract.

## M1 training target

Goal:

```text
camera/depth/contact → z_obj_slots
```

Compare against:

```text
M1 cube/tetra/morph imit z_obj
```

Metrics:

```text
slot confidence
slot match
latent cosine similarity
decoded shape quality
Inner Object 3D nonempty
```

## M5 training target

Goal:

```text
observation/history/replay → stable focus/world latents
```

Compare against:

```text
M5 cube/tetra/morph latent prototypes
```

Metrics:

```text
prediction_error
reconstruction_error
seed_response
latent_coherence
cosine similarity to prototype
downstream response equivalence
```

## M2 training target

Goal:

```text
M11 + M13 + M4 + M5 state → replay seed that reduces pressure
```

Metrics:

```text
dream_pressure_delta < 0
stress_delta < 0
relief_delta > 0
coherence_delta > 0
identity_stability_delta > 0
```

## M10/M7/M15/M12 conscious training target

Alias: Conscious training target.

### M10

```text
select useful conscious content from candidates
```

### M7

```text
turn broadcast packet into useful inner speech / subjective stream
```

### M15

```text
produce imagined rollouts and candidate plan
```

### M12

```text
predict confidence/uncertainty/recheck need
```

Metrics:

```text
plan usefulness
confidence calibration
subjective stream consistency
action outcome prediction
```

## Guard

Required phrase: guards must not be disabled.

Training must never disable:

```text
M3 sleep/action guard
M5 FocusFeedbackBoundary
status visibility
source labels
```
