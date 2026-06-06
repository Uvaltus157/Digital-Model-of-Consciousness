# Energy Resonator integration

## Meaning

This patch adds an experimental two-agent social-affect process.

The process is based on the user's "энергетический резонатор" scheme:

- animal mode: he acts for his own positive emotion;
- her positive emotion is initially a side-effect;
- conscious mode: he tolerates negative cost and keeps her positive emotion as the main goal;
- resonance appears when he receives positive emotion from her positive emotion.

## Files

```text
src/modules/m11_motivational_homeostasis/imit/energy_resonator.py
src/modules/m11_motivational_homeostasis/imit/energy_resonator_runtime.py
scripts/run_energy_resonator_demo.py
patches/energy_resonator_integration.diff
```

## Runtime insertion

Insert the bridge after M1/M4 and before M9 self binding:

```python
out["inner_object"] = self.compute_inner_object_image(obs, out)
self._compute_long_dynamic_memory(obs, out)
self.compute_energy_resonator(obs, out)
out["self_core"] = self.compute_self_core(obs, out)
```

This is the intended position because:

- M1 has current object/agent image;
- M4 has dynamic identity;
- M5 `focus_context` already exists;
- M9 can self-bind the resulting social-affect packet;
- M7/M12 can verbalize/check it after self binding.

## Output contract

```python
out["energy_resonator"]
out["other_agent_focus"]
out["energy_resonator_latent"]
out["social_affect"]
```

`out["other_agent_focus"]` is the M5-like working image of the other agent:

```python
{
    "self_agent": "он",
    "other_agent": "она",
    "current_focus": "self_positive_emotion" | "other_agent_positive_emotion",
    "other_observed_positive": float,
    "self_tolerating_negative": bool,
    "conscious_alignment": float,
    "resonance_index": float,
}
```

## Safe default

By default the bridge does not modify `out["focus_context"]`.

To make it actively influence M5 focus:

```python
cfg.energy_resonator.blend_into_focus_context = True
cfg.energy_resonator.focus_blend_weight = 0.015
```

If the config section does not exist yet, the runtime uses safe defaults.
