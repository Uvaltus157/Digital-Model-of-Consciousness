# Codex Debug Prompt: Step 3A — Per-Slot 4D Gaussian Timeline

You are Codex CLI working inside the repository:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System
```

Branch:

```text
work_2
```

## Goal

Debug and verify **Step 3A: per-slot 4D Gaussian timeline**.

Step 3A is the first 4D reconstruction layer. It must not implement deformation yet.

Current pipeline should be:

```text
object attention
-> dynamic object slot
-> RGB-D observation buffer
-> point cloud
-> per-slot Gaussian reconstruction
-> Step 2B backend preview
-> Step 3A per-slot Gaussian timeline
```

Step 3A must add:

```text
slot-0/tetrahedron Gaussian state at t0, t1, t2...
slot-1/cube        Gaussian state at t0, t1, t2...
```

This timeline is the data foundation for the future Step 3B deformation field.

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
[STAGE] repo_discovery status=running comment=проверяю что Step 2B уже пишет gaussian state по слотам
[STAGE] code_check status=running comment=ищу Slot4DTimelineBuffer и подключение после gaussian reconstruction
[STAGE] patch_needed status=planned comment=Step 3A отсутствует, добавляю timeline buffer и diagnostics
[STAGE] verify_run status=running comment=запускаю tetra->cube и проверяю slot_4d_frame события
[STAGE] success status=ok comment=slot-0 и slot-1 имеют отдельные 4D timelines
```

## Files to Inspect First

```text
runtime/object_imagery_runtime.py
runtime/slot_4d_reconstruction.py
runtime/slot_gaussian_reconstruction.py
runtime/tetra_dynamic_slot_diagnostic.py
runtime/config.py
conf/runner.yaml
```

## Dependencies

Step 3A depends on Step 1 and Step 2A/2B.

Required Step 1 evidence:

```text
event=slot_observation_buffer
event=slot_pointcloud_reconstruction
SUCCESS_SLOT_OBSERVATION_POINTCLOUD_STEP1
```

Required Step 2A/2B evidence:

```text
event=slot_gaussian_train
event=slot_gaussian_render
event=slot_gaussian_preview_frame
SUCCESS_SLOT_GAUSSIAN_3DGS_STEP2
or
SUCCESS_SLOT_GAUSSIAN_CUDA_PREVIEW_STEP2B
```

If Step 2 Gaussian states are unavailable, Step 3A cannot succeed. Repair Step 2 first or stop with a clear stage message.

## Required New Module

Add or repair:

```text
runtime/slot_4d_reconstruction.py
```

It should define:

```python
Slot4DFrame
Slot4DTimelineBuffer
Slot4DReconstructor
```

## Required Behavior

`Slot4DFrame` stores a compact snapshot of one slot at one time:

```text
slot_id
target_name
live_step
gaussian_count
updates
recon_loss
formed_conf
z_dynamic_norm
xyz_mean
xyz_std
xyz_sample
backend
```

`Slot4DTimelineBuffer` stores separate frame histories per slot:

```text
slot_id -> deque[Slot4DFrame]
```

`Slot4DReconstructor` reads from the existing Gaussian reconstructor:

```text
gaussian_reconstructor.states[slot_id]
gaussian_reconstructor.last_metrics[slot_id]
```

and writes one timeline frame per dynamic slot update.

Important:

```text
slot-0/tetrahedron timeline must not receive cube frames
slot-1/cube timeline must not receive tetrahedron frames
```

## Required Object Runtime Integration

In:

```text
runtime/object_imagery_runtime.py
```

Step 3A must run after Gaussian reconstruction:

```python
self._slot_observation_reconstruction_step(target_slot, target_name_for_recon, source)
self._slot_gaussian_reconstruction_step(target_slot, target_name_for_recon, source)
self._slot_4d_timeline_step(target_slot, target_name_for_recon, source)
```

Required methods or equivalent:

```python
_ensure_slot_4d_reconstruction
_slot_4d_timeline_step
_maybe_log_slot_4d_timeline_success
```

`_slot_4d_timeline_step` must only run for dynamic sources:

```python
if "dynamic" not in str(source):
    return
```

## Required Config

In `runtime/config.py`, inside `InnerObjectImageConfig`, these fields must exist:

```python
slot_4d_timeline_enabled: bool = True
slot_4d_timeline_max_frames: int = 256
slot_4d_sample_points: int = 128
```

In `conf/runner.yaml`, under `object_image`, these keys must exist:

```yaml
slot_4d_timeline_enabled: true
slot_4d_timeline_max_frames: 256
slot_4d_sample_points: 128
```

## Required Diagnostics

Main diagnostic log:

```text
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

Required events:

```text
event=slot_4d_frame
event=slot_4d_timeline
event=SUCCESS_SLOT_4D_TIMELINE_STEP3A
```

Expected fields for `slot_4d_frame`:

```text
slot_id
target_name
live_step
frame_count
gaussian_count
updates
recon_loss
formed_conf
z_dynamic_norm
motion_norm
backend
valid
```

Expected fields for `slot_4d_timeline`:

```text
slot_id
target_name
timeline_frames
temporal_span
gaussian_count
updates
motion_norm
mean_delta_x
mean_delta_y
mean_delta_z
backend
valid
```

Success marker:

```text
event=SUCCESS_SLOT_4D_TIMELINE_STEP3A
```

Required fields:

```text
slot_0_target=tetrahedron
slot_0_timeline_frames
slot_0_gaussian_count
slot_0_temporal_span

slot_1_target=cube
slot_1_timeline_frames
slot_1_gaussian_count
slot_1_temporal_span

slot_0_overwritten=0
slot_1_allocated=1
reason=per_slot_gaussian_timeline_ready_for_4d_deformation
```

## Inner Object Visualizer Metrics

Attach or expose these metrics so Inner Object can display them:

```text
slot_4d_timeline_frames
slot_4d_temporal_span
slot_4d_motion_norm
slot_4d_gaussian_count
```

No timeline playback UI is required yet. Metrics are enough for Step 3A.

## Runtime Verification

Run or reuse the normal tetra -> cube sequence:

```text
attention -> tetrahedron
tetrahedron forms slot-0
slot-0 gets Gaussian state
slot-0 gets 4D timeline frames

attention -> cube
cube forms slot-1
slot-1 gets Gaussian state
slot-1 gets 4D timeline frames
```

Then verify:

```text
slot-0/tetrahedron timeline_frames > 0
slot-0/tetrahedron gaussian_count > 0

slot-1/cube timeline_frames > 0
slot-1/cube gaussian_count > 0

slot-0 target remains tetrahedron
slot-1 target remains cube
slot-0 is not overwritten by cube
```

Useful grep:

```bash
grep "event=slot_4d_frame" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 40
grep "event=slot_4d_timeline" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 40
grep "SUCCESS_SLOT_4D_TIMELINE_STEP3A" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

## Forbidden

Do not implement Step 3B deformation field in this task.

Do not implement neural 4D predictor yet.

Do not replace the Gaussian reconstructor.

Do not replace LongDynamicObjectMemory.

Do not bypass object slots.

Do not mix tetrahedron and cube timelines.

Do not write cube frames into slot-0.

Do not fake `timeline_frames`, `gaussian_count`, or `motion_norm`.

Do not hard-code success.

Do not print diff or full file contents.

## Repair Rules

If something is missing:

1. Make the smallest targeted repair.
2. If Step 2B methods already exist, insert Step 3A after `_slot_gaussian_reconstruction_step(...)`.
3. Log the reason with:

```text
event=DIAGNOSIS_FAILED
event=repair_applied
```

4. Re-run verification.
5. Stop after success marker is written.

## Final Output

Only print short stage comments and final values:

```text
slot_0_target
slot_0_timeline_frames
slot_0_gaussian_count
slot_0_temporal_span

slot_1_target
slot_1_timeline_frames
slot_1_gaussian_count
slot_1_temporal_span

SUCCESS_SLOT_4D_TIMELINE_STEP3A written/not written
```
