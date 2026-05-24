# Codex Debug Prompt: Step 2A — Per-Slot Low-Res Gaussian 3DGS

You are Codex CLI working inside the repository:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System
```

Branch:

```text
work_2
```

## Goal

Debug and verify **Step 2A: per-slot low-res Gaussian 3DGS reconstructor**.

Step 1 should already be successful:

```text
object slot
-> RGB-D observation buffer
-> lightweight point-cloud accumulator
```

Step 2A must verify:

```text
slot point cloud
-> initialize Gaussian primitives
-> low-res differentiable Gaussian renderer
-> RGB/depth reconstruction loss
-> Adam update
-> diagnostics per slot
```

This is **not Step 2B**.

Step 2A is allowed to use learnable Gaussian parameters directly. It does **not** need `SlotGaussianModel(nn.Module)` or `SlotGaussianPredictorNet`. Those belong to Step 2B.

## Strict Output Policy

Do not print git diff.

Do not print full file contents.

Do not print long changed-file summaries.

Use short commented stages:

```text
[STAGE] <stage_id> status=<status> comment=<short explanation>
```

Examples:

```text
[STAGE] repo_discovery status=running comment=проверяю что Step 1 уже пишет RGB-D наблюдения и point cloud по слотам
[STAGE] code_check status=running comment=ищу low-res Gaussian reconstructor и его подключение после pointcloud
[STAGE] patch_needed status=planned comment=не хватает события slot_gaussian_train, добавляю минимальную диагностику
[STAGE] verify_run status=running comment=запускаю tetra->cube и проверяю gaussian_count и updates
[STAGE] success status=ok comment=slot-0 и slot-1 обучают отдельные low-res Gaussian модели
```

## Files to Inspect First

```text
runtime/slot_observation_reconstruction.py
runtime/slot_gaussian_reconstruction.py
runtime/object_imagery_runtime.py
runtime/tetra_dynamic_slot_diagnostic.py
visualizer/inner_object_visualizer.py
runner.py
conf/runner.yaml
```

## Required Step 1 Dependency

Before checking Step 2A, verify Step 1 exists and works.

Required module:

```text
runtime/slot_observation_reconstruction.py
```

Required classes:

```text
SlotObservation
SlotObservationBuffer
SlotPointCloudReconstructor
```

Required Step 1 events:

```text
event=slot_observation_buffer
event=slot_pointcloud_reconstruction
event=SUCCESS_SLOT_OBSERVATION_POINTCLOUD_STEP1
```

If Step 1 is missing, repair Step 1 first or stop with:

```text
[STAGE] blocked status=failed comment=Step 1 не установлен, Step 2A не может стартовать без point cloud
```

## Required Step 2A File

Add or repair:

```text
runtime/slot_gaussian_reconstruction.py
```

It should define:

```text
SlotGaussianState
SimpleGaussianRenderer
SlotGaussianReconstructor
```

Acceptable Step 2A design:

```text
SlotGaussianState:
  learnable torch Parameters:
    xyz
    log_scale
    opacity_logit
    color_logit
  Adam optimizer
  updates counter

SimpleGaussianRenderer:
  low-res differentiable splat renderer
  renders rgb/depth/alpha

SlotGaussianReconstructor:
  slot_id -> SlotGaussianState
  initialize from SlotPointCloudReconstructor points/colors
  train on latest SlotObservation RGB/depth
```

Important: this Step 2A does not require a full neural model class. That is Step 2B.

## Required Integration

In:

```text
runtime/object_imagery_runtime.py
```

verify or add:

```text
_ensure_slot_gaussian_reconstruction
_slot_gaussian_reconstruction_step
_maybe_log_slot_gaussian_step2_success
```

Step 2A must run **after** Step 1 pointcloud update:

```text
_slot_observation_reconstruction_step(...)
_slot_gaussian_reconstruction_step(...)
```

The input source must be dynamic:

```text
source contains "dynamic"
z_source=z_dynamic_object
```

Do not train 3DGS from static-only slot writes.

## Required Diagnostics

Main diagnostic log:

```text
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

Required events:

```text
event=slot_gaussian_init
event=slot_gaussian_train
event=slot_gaussian_render
event=SUCCESS_SLOT_GAUSSIAN_3DGS_STEP2
```

Expected fields for `slot_gaussian_init`:

```text
slot_id
target_name
gaussian_count
source_points
```

Expected fields for `slot_gaussian_train`:

```text
slot_id
target_name
initialized
gaussian_count
updates
rgb_loss
depth_loss
total_loss
render_valid
```

Expected fields for `slot_gaussian_render`:

```text
slot_id
target_name
gaussian_count
updates
render_valid
rgb_loss
depth_loss
```

Success marker:

```text
event=SUCCESS_SLOT_GAUSSIAN_3DGS_STEP2
```

Required fields:

```text
slot_0_target=tetrahedron
slot_0_gaussian_count
slot_0_recon_loss
slot_0_updates

slot_1_target=cube
slot_1_gaussian_count
slot_1_recon_loss
slot_1_updates

slot_0_overwritten=0
slot_1_allocated=1
reason=per_slot_low_res_gaussian_training_active
```

## Inner Object Visualizer Metrics

Attach or expose these metrics so the Inner Object Visualizer can show them:

```text
slot_gaussian_count
slot_gaussian_updates
slot_gaussian_rgb_loss
slot_gaussian_depth_loss
slot_gaussian_total_loss
```

No render preview is required for Step 2A. Metrics are enough.

## Runtime Verification

Run or reuse the normal tetra -> cube sequence:

```text
attention -> tetrahedron
tetrahedron forms slot-0
slot-0 gets point cloud
slot-0 initializes/train low-res Gaussians

attention -> cube
cube forms slot-1
slot-1 gets point cloud
slot-1 initializes/train low-res Gaussians
```

Then verify:

```text
slot-0/tetrahedron gaussian_count > 0
slot-0/tetrahedron updates > 0
slot-0/tetrahedron total_loss > 0

slot-1/cube gaussian_count > 0
slot-1/cube updates > 0
slot-1/cube total_loss > 0

slot-0 is not overwritten by cube
slot-1 is not mixed with tetrahedron
```

Useful grep:

```bash
grep "event=slot_gaussian" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 60
grep "SUCCESS_SLOT_GAUSSIAN_3DGS_STEP2" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

## Forbidden

Do not implement Step 2B here.

Do not add `SlotGaussianModel(nn.Module)` here unless it already exists and is harmless.

Do not add `SlotGaussianPredictorNet` here.

Do not implement CUDA rasterization.

Do not implement 4D deformation fields.

Do not replace LongDynamicObjectMemory.

Do not bypass object slots.

Do not mix tetrahedron and cube Gaussian states.

Do not train cube in slot-0.

Do not fake `updates`, `loss`, or `gaussian_count`.

Do not hard-code success.

Do not print diff or full file contents.

## Repair Rules

If something is missing:

1. Make the smallest targeted repair.
2. Log the reason with:

```text
event=DIAGNOSIS_FAILED
event=repair_applied
```

3. Re-run verification.
4. Stop after success marker is written.

## Final Output

Only print short stage comments and final values:

```text
slot_0_target
slot_0_gaussian_count
slot_0_updates
slot_0_recon_loss

slot_1_target
slot_1_gaussian_count
slot_1_updates
slot_1_recon_loss

SUCCESS_SLOT_GAUSSIAN_3DGS_STEP2 written/not written
```
