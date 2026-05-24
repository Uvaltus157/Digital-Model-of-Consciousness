# Codex Debug Prompt: Slot Observation Buffer + PointCloud Step 1

You are Codex CLI working inside the repository:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System
```

Branch:

```text
work_2
```

## Goal

Debug and verify the first step toward 3DGS / 4D reconstruction:

```text
object slot
-> RGB-D observation buffer
-> lightweight point-cloud accumulator
-> diagnostics per slot
```

Expected behavior:

```text
slot-0 = tetrahedron -> own RGB-D observation buffer -> own point cloud
slot-1 = cube        -> own RGB-D observation buffer -> own point cloud
```

Do not implement full 3DGS yet. This task is only about the first bridge layer.

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
[STAGE] repo_discovery status=running comment=смотрю куда встроен object imagery runtime
[STAGE] code_check status=running comment=проверяю наличие SlotObservationBuffer и pointcloud reconstructor
[STAGE] patch_needed status=planned comment=не хватает события slot_pointcloud_reconstruction
[STAGE] verify_run status=running comment=запускаю tetra->cube проверку и смотрю новые события в логе
[STAGE] success status=ok comment=slot-0 и slot-1 копят отдельные RGB-D наблюдения и point cloud
```

## Files to inspect first

```text
runtime/slot_observation_reconstruction.py
runtime/object_imagery_runtime.py
runtime/tetra_dynamic_slot_diagnostic.py
visualizer/inner_object_visualizer.py
runner.py
conf/runner.yaml
```

## Required Code Objects

There should be a new module:

```text
runtime/slot_observation_reconstruction.py
```

It should define:

```text
SlotObservation
SlotObservationBuffer
SlotPointCloudReconstructor
```

Expected responsibilities:

```text
SlotObservation:
  slot_id
  target_name
  live_step
  rgb
  depth
  camera_pose
  formed_conf
  z_dynamic_norm

SlotObservationBuffer:
  stores separate observation history per slot_id

SlotPointCloudReconstructor:
  converts depth/rgb into a lightweight point cloud per slot_id
  keeps points separate for slot-0 and slot-1
```

## Required Integration

In:

```text
runtime/object_imagery_runtime.py
```

verify or add minimal code so that when a dynamic slot write happens:

```text
slot_id
target_name
source
rgb/depth observation
formed_conf
z_dynamic_norm
```

are passed into the slot observation buffer and pointcloud reconstructor.

Required helper methods, or equivalent logic:

```text
_ensure_slot_observation_reconstruction
_slot_observation_reconstruction_step
```

The runtime must preserve target identity:

```text
slot-0 / tetrahedron observations do not go into slot-1
slot-1 / cube observations do not go into slot-0
```

## Required Diagnostics

Main diagnostic log:

```text
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

Add or verify these events:

```text
event=slot_observation_buffer
event=slot_pointcloud_reconstruction
```

Expected fields for `slot_observation_buffer`:

```text
slot_id
target_name
frame_count
depth_valid
live_step
```

Expected fields for `slot_pointcloud_reconstruction`:

```text
slot_id
target_name
depth_valid
points_added
points_total
frame_count
formed_conf
z_dynamic_norm
```

## Required Runtime Verification

Run or reuse the normal tetra -> cube sequence:

```text
attention -> tetrahedron
tetrahedron forms slot-0
attention -> cube
cube forms slot-1
```

Then verify:

```text
slot-0/tetrahedron frame_count > 0
slot-0/tetrahedron points_total > 0
slot-1/cube frame_count > 0
slot-1/cube points_total > 0
slot-0 target_name does not become cube
slot-1 target_name does not become tetrahedron
```

Use:

```bash
grep "event=slot_observation_buffer" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 30
grep "event=slot_pointcloud_reconstruction" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 30
```

## Success Criteria

Write this diagnostic marker only after all checks are true:

```text
event=SUCCESS_SLOT_OBSERVATION_POINTCLOUD_STEP1
```

Fields:

```text
slot_0_target=tetrahedron
slot_0_frame_count
slot_0_points_total
slot_1_target=cube
slot_1_frame_count
slot_1_points_total
slot_0_overwritten=0
slot_1_allocated=1
reason=rgbd_observation_buffers_and_pointclouds_separate
```

## Forbidden

Do not implement full Gaussian Splatting yet.

Do not replace LongDynamicObjectMemory.

Do not bypass object slots.

Do not mix tetrahedron and cube observations.

Do not fake `points_total`.

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

Only print short stage comments:

```text
[STAGE] success status=ok comment=slot observation buffer и pointcloud step-1 подтверждены
```

Include final values only:

```text
slot_0_target
slot_0_frame_count
slot_0_points_total
slot_1_target
slot_1_frame_count
slot_1_points_total
SUCCESS_SLOT_OBSERVATION_POINTCLOUD_STEP1 written/not written
```
