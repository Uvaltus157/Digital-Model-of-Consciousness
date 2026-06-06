# Config DNA

This document defines configuration fields needed to regenerate compatible code.

## Core dimensions

```yaml
object_image:
  latent_dim: 128
  num_slots: 10

self_core:
  focus_context_dim: 256

world_model:
  rssm_dim: TBD
  obs_embed_dim: TBD

training:
  enabled: true
  online: false
```

## Sleep / replay

```yaml
sleep:
  replay_mode: false
  weaken_m1_sensors: true
  block_m3_motor: true
  keep_m11_active: true
  keep_m13_active: true
  keep_m4_active: true
  keep_m2_active: true
```

## Sensor mask

```yaml
sleep_sensor_mask:
  video: false
  contact: false
  imu: false
```

## Imitators

```yaml
imitators:
  enabled: true

  m1_object_slot:
    enabled: true
    default_cube_slot: 1
    default_tetra_slot: 2
    default_morph_slot: 3

  m5_latent_prototype:
    enabled: true
    default_gate: 0.85
    default_duration: 160
```

## Debug UI

```yaml
debug_ui:
  m8_enabled: true
  show_sleep_replay_monitor: true
  show_replay_quality_monitor: true
  show_m5_learning_quality: true
  show_m1_object_slot_imit: true
  show_m5_latent_prototypes: true
```

## Codegen rule

Every config field used by code must be documented here or in module-specific config DNA.
