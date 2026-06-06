
# M1 Object Slot Latent Imitator

Imitator path:

```text
src/modules/m01_object_imagery/imit/m1_object_slot_latent_runtime.py
```

It injects deterministic z_obj-like latents into:

```text
ObjectImageryRuntimeMixin.build_inner_object_vision_proposals()
```

It fills:

```text
_inner_object_proposal_target_slots
_inner_object_proposal_kinds
_inner_object_proposal_target_names
```

Then the existing path writes selected slots:

```text
_run_progressive_inner_object_system()
↓
_memory_update_forced_slot(... force_slot_index=slot)
↓
inner_object_system.decode_z(...)
↓
inner_object_viz.requested_dream_slot_index
```

This makes the inner-object 3D window show the selected simulated cube/tetrahedron slot.
