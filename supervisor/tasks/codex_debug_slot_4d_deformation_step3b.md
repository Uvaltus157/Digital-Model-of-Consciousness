# Codex Debug Prompt: Step 3B — Neural 4D Deformation Field

You are Codex CLI working inside the repository:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System
```

Branch:

```text
work_2
```

## Goal

Debug and verify **Step 3B: neural 4D deformation field**.

Step 3A already provides a per-slot Gaussian timeline:

```text
slot-0/tetrahedron -> Gaussian state at t0, t1, t2...
slot-1/cube        -> Gaussian state at t0, t1, t2...
```

Step 3B must train a small deformation field from adjacent timeline frames:

```text
xyz(t0) + Slot4DDeformationModel([xyz(t0), time_code]) -> predicted xyz(t1)
```

Then compare:

```text
predicted xyz(t1) ~= real xyz(t1)
```

This is the first real 4D deformation-learning step. It must not replace object slots, Gaussian reconstruction, or LongDynamicObjectMemory.

## Strict Output Policy

Do not print git diff.

Do not print full file contents.

Do not print long changed-file summaries.

Use only short commented stages:

```text
[STAGE] <stage_id> status=<status> comment=<short explanation>
```

Examples:

```text
[STAGE] repo_discovery status=running comment=проверяю что Step 3A timeline уже подключен после Gaussian reconstruction
[STAGE] code_check status=running comment=ищу Slot4DDeformationModel и trainer
[STAGE] patch_needed status=planned comment=нет deformation trainer, добавляю обучение по двум соседним timeline frames
[STAGE] verify_run status=running comment=запускаю tetra->cube и проверяю slot_4d_deformation_train
[STAGE] success status=ok comment=slot-0 и slot-1 обучают отдельные deformation fields
```

## Files to Inspect First

```text
runtime/object_imagery_runtime.py
runtime/slot_4d_reconstruction.py
runtime/slot_4d_deformation.py
runtime/slot_gaussian_reconstruction.py
runtime/tetra_dynamic_slot_diagnostic.py
runtime/config.py
conf/runner.yaml
```

## Dependencies

Step 3B depends on Step 3A.

Required Step 3A evidence:

```text
event=slot_4d_frame
event=slot_4d_timeline
event=SUCCESS_SLOT_4D_TIMELINE_STEP3A
```

Required runtime sequence before Step 3B:

```python
self._slot_observation_reconstruction_step(target_slot, target_name_for_recon, source)
self._slot_gaussian_reconstruction_step(target_slot, target_name_for_recon, source)
self._slot_4d_timeline_step(target_slot, target_name_for_recon, source)
self._slot_4d_deformation_step(target_slot, target_name_for_recon, source)
```

If Step 3A is missing, repair Step 3A first or stop with:

```text
[STAGE] blocked status=failed comment=Step 3A timeline отсутствует, Step 3B не может обучать deformation field
```

## Required New Module

Add or repair:

```text
runtime/slot_4d_deformation.py
```

It must define:

```python
Slot4DDeformationModel
Slot4DDeformationTrainer
Slot4DDeformationMetrics
```

## Required Model Behavior

`Slot4DDeformationModel` must be a real `torch.nn.Module`.

Minimum acceptable model:

```python
class Slot4DDeformationModel(nn.Module):
    def forward(self, xyz, time_code):
        # input: [xyz, time_code]
        # output: delta_xyz
```

Expected input/output:

```text
input:
  xyz: [N, 3]
  time_code: scalar or [N, 1]

output:
  delta_xyz: [N, 3]
```

Expected architecture:

```text
Linear(4 -> hidden)
activation
Linear(hidden -> hidden)
activation
Linear(hidden -> 3)
```

## Required Trainer Behavior

`Slot4DDeformationTrainer` must store a separate model per slot:

```text
slot_id -> Slot4DDeformationModel
slot_id -> optimizer
```

Training must use the last two `Slot4DFrame` snapshots from Step 3A:

```text
prev = timeline.frames[slot_id][-2]
curr = timeline.frames[slot_id][-1]
```

Training target:

```text
x0 = prev.xyz_sample
x1 = curr.xyz_sample
pred_delta = model(x0, dt)
pred_x1 = x0 + pred_delta
loss = MSE(pred_x1, x1) + delta_regularization
```

The trainer must not mix slots:

```text
slot-0 model trains only from tetrahedron timeline
slot-1 model trains only from cube timeline
```

## Required Object Runtime Integration

In:

```text
runtime/object_imagery_runtime.py
```

Required methods or equivalent:

```python
_ensure_slot_4d_deformation
_slot_4d_deformation_step
_maybe_log_slot_4d_deformation_success
```

Step 3B must run after Step 3A:

```python
self._slot_4d_timeline_step(target_slot, target_name_for_recon, source)
self._slot_4d_deformation_step(target_slot, target_name_for_recon, source)
```

It must only run for dynamic sources:

```python
if "dynamic" not in str(source):
    return
```

## Required Config

In `runtime/config.py`, inside `InnerObjectImageConfig`, these fields must exist:

```python
slot_4d_deformation_enabled: bool = True
slot_4d_deformation_hidden_dim: int = 96
slot_4d_deformation_lr: float = 0.002
slot_4d_deformation_train_steps_per_update: int = 1
slot_4d_deformation_min_frames: int = 2
slot_4d_deformation_delta_reg_weight: float = 0.0001
```

In `conf/runner.yaml`, under `object_image`, these keys must exist:

```yaml
slot_4d_deformation_enabled: true
slot_4d_deformation_hidden_dim: 96
slot_4d_deformation_lr: 0.002
slot_4d_deformation_train_steps_per_update: 1
slot_4d_deformation_min_frames: 2
slot_4d_deformation_delta_reg_weight: 0.0001
```

## Required Diagnostics

Main diagnostic log:

```text
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

Required events:

```text
event=slot_4d_deformation_model
event=slot_4d_deformation_train
event=SUCCESS_SLOT_4D_DEFORMATION_STEP3B
```

Expected fields for `slot_4d_deformation_model`:

```text
slot_id
target_name
model_type=Slot4DDeformationModel
trainable_params
enabled
trainable
```

Expected fields for `slot_4d_deformation_train`:

```text
slot_id
target_name
enabled
trainable
valid
updates
loss
motion_norm
pred_delta_norm
sample_count
temporal_dt
model_type=Slot4DDeformationModel
```

Success marker:

```text
event=SUCCESS_SLOT_4D_DEFORMATION_STEP3B
```

Required fields:

```text
slot_0_target=tetrahedron
slot_0_deformation_updates
slot_0_deformation_loss
slot_0_motion_norm
slot_0_sample_count

slot_1_target=cube
slot_1_deformation_updates
slot_1_deformation_loss
slot_1_motion_norm
slot_1_sample_count

trainable_params
slot_0_overwritten=0
slot_1_allocated=1
reason=per_slot_4d_deformation_field_training_active
```

## Inner Object Visualizer Metrics

Attach or expose these metrics so Inner Object can display them:

```text
slot_4d_deformation_updates
slot_4d_deformation_loss
slot_4d_deformation_pred_delta_norm
slot_4d_deformation_sample_count
```

No 4D playback UI is required in this step.

## Runtime Verification

Run or reuse the normal tetra -> cube sequence:

```text
attention -> tetrahedron
tetrahedron forms slot-0
slot-0 gets Gaussian timeline frames
slot-0 trains deformation model

attention -> cube
cube forms slot-1
slot-1 gets Gaussian timeline frames
slot-1 trains deformation model
```

Then verify:

```text
slot-0/tetrahedron deformation_updates > 0
slot-0/tetrahedron sample_count > 0
slot-0/tetrahedron trainable=true

slot-1/cube deformation_updates > 0
slot-1/cube sample_count > 0
slot-1/cube trainable=true

trainable_params > 0
slot-0 is not overwritten by cube
slot-1 is not mixed with tetrahedron
```

Useful grep:

```bash
grep "event=slot_4d_deformation_model" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 40
grep "event=slot_4d_deformation_train" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 40
grep "SUCCESS_SLOT_4D_DEFORMATION_STEP3B" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

## Forbidden

Do not replace Step 3A timeline.

Do not replace Gaussian reconstruction.

Do not replace LongDynamicObjectMemory.

Do not bypass object slots.

Do not mix tetrahedron and cube deformation models.

Do not train cube in slot-0.

Do not fake `updates`, `loss`, `sample_count`, or `trainable_params`.

Do not hard-code success.

Do not implement full 4D playback UI here.

Do not print diff or full file contents.

## Repair Rules

If something is missing:

1. Make the smallest targeted repair.
2. If Step 3A exists but Step 3B call is missing, insert `_slot_4d_deformation_step(...)` immediately after `_slot_4d_timeline_step(...)`.
3. If the deformation model exists but does not inherit `nn.Module`, repair it.
4. Log the reason with:

```text
event=DIAGNOSIS_FAILED
event=repair_applied
```

5. Re-run verification.
6. Stop after success marker is written.

## Final Output

Only print short stage comments and final values:

```text
slot_0_target
slot_0_deformation_updates
slot_0_deformation_loss
slot_0_motion_norm
slot_0_sample_count

slot_1_target
slot_1_deformation_updates
slot_1_deformation_loss
slot_1_motion_norm
slot_1_sample_count

trainable_params
SUCCESS_SLOT_4D_DEFORMATION_STEP3B written/not written
```
