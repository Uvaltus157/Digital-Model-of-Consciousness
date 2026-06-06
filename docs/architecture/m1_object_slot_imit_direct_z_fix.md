# M1 Object Slot Imit direct z_update fix

The M1 imit proposal is already a `z_obj`-like latent.

Correct path:

```text
m1_imit proposal v_i
↓
source == "m1_imit_dynamic_object"
↓
z_update = v_i
↓
_memory_update_forced_slot(... force_slot_index=target_slot)
↓
ObjectSlotMemory writes slot
```

Incorrect path:

```text
m1_imit proposal v_i
↓
inner_object_system.fusion(v_i, tactile, body, hand, leg)
```

because fusion expects raw sensory summaries, not ready object latents.
