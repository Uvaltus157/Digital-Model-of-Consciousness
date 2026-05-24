# Codex Debug Prompt: Step 2B — CUDA 3DGS Backend Switch + Real-Time Preview

You are Codex CLI working inside the repository:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System
```

Branch:

```text
work_2
```

## Goal

Debug and verify **Step 2B: configurable Gaussian renderer backend + CUDA 3DGS preview path**.

Step 2A already verifies the safe low-res PyTorch Gaussian renderer:

```text
slot point cloud
-> Gaussian primitives
-> low-res PyTorch render
-> RGB/depth loss
-> Adam update
```

Step 2B must add a backend switch:

```text
object_image.slot_gaussian_renderer_backend: torch_lowres | cuda_3dgs | auto
```

Expected behavior:

```text
torch_lowres:
  always use Step 2A PyTorch renderer

cuda_3dgs:
  request CUDA 3DGS rasterizer
  if unavailable and fallback is allowed -> use torch_lowres fallback

auto:
  use CUDA if available
  otherwise use torch_lowres fallback
```

Important: Step 2B must not break Step 2A. If CUDA packages are missing, the runtime must continue with fallback.

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
[STAGE] repo_discovery status=running comment=проверяю текущий Step 2A и места подключения backend switch
[STAGE] code_check status=running comment=проверяю что renderer backend читается из object_image config
[STAGE] patch_needed status=planned comment=метод уже был из Step 2A, заменяю его на backend-aware версию
[STAGE] verify_config status=running comment=проверяю переключение torch_lowres/cuda_3dgs/auto в конфиге
[STAGE] verify_run status=running comment=запускаю tetra->cube и проверяю preview/backend события
[STAGE] success status=ok comment=Step 2B backend switch работает, fallback не ломает runtime
```

## Files to Inspect First

```text
runtime/slot_gaussian_cuda_adapter.py
runtime/slot_gaussian_reconstruction.py
runtime/object_imagery_runtime.py
runtime/tetra_dynamic_slot_diagnostic.py
runtime/config.py
conf/runner.yaml
```

## Required Files

Add or repair:

```text
runtime/slot_gaussian_cuda_adapter.py
runtime/slot_gaussian_reconstruction.py
```

Patch if needed:

```text
runtime/object_imagery_runtime.py
runtime/tetra_dynamic_slot_diagnostic.py
runtime/config.py
conf/runner.yaml
```

## Required Config

In `runtime/config.py`, inside `InnerObjectImageConfig`, these fields must exist:

```python
slot_gaussian_renderer_backend: str = "auto"
slot_gaussian_cuda_allow_fallback: bool = True
slot_gaussian_image_size: int = 64
slot_gaussian_max_gaussians: int = 768
slot_gaussian_max_render_gaussians: int = 256
slot_gaussian_lr: float = 0.003
slot_gaussian_train_steps_per_update: int = 1
slot_gaussian_depth_weight: float = 0.35
slot_gaussian_preview_every_steps: int = 1
```

In `conf/runner.yaml`, under `object_image`, these keys must exist:

```yaml
slot_gaussian_renderer_backend: auto   # torch_lowres | cuda_3dgs | auto
slot_gaussian_cuda_allow_fallback: true
slot_gaussian_image_size: 64
slot_gaussian_max_gaussians: 768
slot_gaussian_max_render_gaussians: 256
slot_gaussian_lr: 0.003
slot_gaussian_train_steps_per_update: 1
slot_gaussian_depth_weight: 0.35
slot_gaussian_preview_every_steps: 1
```

## Required Backend Adapter

`runtime/slot_gaussian_cuda_adapter.py` must define:

```text
SlotGaussianCUDAAdapter
CUDABackendStatus
```

Required behavior:

```text
requested_backend = torch_lowres | cuda_3dgs | auto
detect torch.cuda.is_available()
try importing common CUDA rasterizer packages:
  gsplat
  diff_gaussian_rasterization
if CUDA/rasterizer unavailable:
  fallback to torch_lowres if allow_fallback=true
never crash main runtime because CUDA import/build fails
```

Required methods or equivalent:

```python
status()
render_preview(state, fallback_renderer)
```

`render_preview()` must return backend metrics:

```text
backend
requested_backend
cuda_available
rasterizer_available
fallback_used
fps
import_error
rgb/depth/alpha preview tensors if available
```

## Required Reconstructor Integration

`runtime/slot_gaussian_reconstruction.py` must support:

```text
renderer_backend
allow_fallback
preview_every_steps
SlotGaussianCUDAAdapter
```

Expected constructor args:

```python
SlotGaussianReconstructor(
    renderer_backend="auto",
    allow_fallback=True,
    preview_every_steps=1,
    ...
)
```

Training path:

```text
SlotGaussianReconstructor.train_step(...)
-> cuda_adapter.render_preview(state, fallback_renderer)
-> compute RGB/depth loss
-> optimizer.step()
-> return metrics including backend/fallback/fps
```

Required metric fields:

```text
backend
requested_backend
cuda_available
rasterizer_available
fallback_used
preview_fps
gaussian_count
updates
rgb_loss
depth_loss
total_loss
render_valid
```

## Required Object Runtime Integration

In `runtime/object_imagery_runtime.py`, the existing Step 2A method may already exist:

```python
_ensure_slot_gaussian_reconstruction
```

If it already exists, do not skip it. Replace/repair it so it reads:

```python
slot_gaussian_renderer_backend = getattr(cfg_obj, "slot_gaussian_renderer_backend", "auto")
slot_gaussian_cuda_allow_fallback = getattr(cfg_obj, "slot_gaussian_cuda_allow_fallback", True)
```

and passes them into `SlotGaussianReconstructor`.

This exact issue has happened before:

```text
AssertionError: runtime/object_imagery_runtime.py missing: ['slot_gaussian_renderer_backend']
```

So explicitly verify `runtime/object_imagery_runtime.py` contains:

```text
slot_gaussian_renderer_backend
slot_gaussian_cuda_allow_fallback
slot_gaussian_preview_fps
slot_gaussian_backend_is_cuda
slot_gaussian_fallback_used
```

Step 2B must run after Step 1 pointcloud update:

```text
_slot_observation_reconstruction_step(...)
_slot_gaussian_reconstruction_step(...)
```

## Required Diagnostics

Main diagnostic log:

```text
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

Required events:

```text
event=slot_gaussian_cuda_backend
event=slot_gaussian_preview_frame
event=SUCCESS_SLOT_GAUSSIAN_CUDA_PREVIEW_STEP2B
```

Expected fields for `slot_gaussian_cuda_backend`:

```text
requested_backend
active_backend
cuda_available
rasterizer_available
fallback_used
fps
import_error
```

Expected fields for `slot_gaussian_preview_frame`:

```text
slot_id
target_name
backend
render_valid
gaussian_count
updates
preview_fps
fallback_used
```

Success marker:

```text
event=SUCCESS_SLOT_GAUSSIAN_CUDA_PREVIEW_STEP2B
```

Required fields:

```text
slot_0_target=tetrahedron
slot_0_gaussian_count
slot_0_recon_loss
slot_0_updates
slot_0_backend
slot_0_preview_fps

slot_1_target=cube
slot_1_gaussian_count
slot_1_recon_loss
slot_1_updates
slot_1_backend
slot_1_preview_fps

fallback_used
slot_0_overwritten=0
slot_1_allocated=1
reason=configurable_gaussian_renderer_backend_with_preview
```

## Required Config Verification

Verify all three config modes:

### 1. torch_lowres

```yaml
object_image:
  slot_gaussian_renderer_backend: torch_lowres
```

Expected:

```text
active_backend=torch_lowres
fallback_used=0 or false
runtime does not crash
slot_gaussian_preview_frame appears
```

### 2. auto

```yaml
object_image:
  slot_gaussian_renderer_backend: auto
```

Expected:

```text
if CUDA rasterizer available:
  active_backend=cuda_3dgs
else:
  active_backend=torch_lowres
  fallback_used=1
runtime does not crash
```

### 3. cuda_3dgs

```yaml
object_image:
  slot_gaussian_renderer_backend: cuda_3dgs
slot_gaussian_cuda_allow_fallback: true
```

Expected:

```text
if CUDA/rasterizer available:
  active_backend=cuda_3dgs
else:
  active_backend=torch_lowres
  fallback_used=1
runtime does not crash
```

Do not require CUDA to be installed for the task to pass. The safe fallback behavior is part of success.

## Runtime Verification

Run or reuse the normal tetra -> cube sequence:

```text
attention -> tetrahedron
tetrahedron forms slot-0
slot-0 gets Gaussian preview frame

attention -> cube
cube forms slot-1
slot-1 gets Gaussian preview frame
```

Then verify:

```text
slot-0/tetrahedron gaussian_count > 0
slot-0/tetrahedron updates > 0
slot-0 preview_fps > 0

slot-1/cube gaussian_count > 0
slot-1/cube updates > 0
slot-1 preview_fps > 0

backend metrics are present
slot-0 is not overwritten by cube
```

Useful grep:

```bash
grep "event=slot_gaussian_cuda_backend" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 30
grep "event=slot_gaussian_preview_frame" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log | tail -n 30
grep "SUCCESS_SLOT_GAUSSIAN_CUDA_PREVIEW_STEP2B" /home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

## Inner Object Visualizer Metrics

Attach or expose these metrics so Inner Object can display them:

```text
slot_gaussian_count
slot_gaussian_updates
slot_gaussian_rgb_loss
slot_gaussian_depth_loss
slot_gaussian_total_loss
slot_gaussian_preview_fps
slot_gaussian_backend_is_cuda
slot_gaussian_fallback_used
```

No full image preview is required if not already wired. Metrics are enough for this debug step.

## Forbidden

Do not break Step 2A.

Do not require CUDA to be installed.

Do not crash when `gsplat` or `diff_gaussian_rasterization` is missing.

Do not implement Step 2C neural predictor here.

Do not replace LongDynamicObjectMemory.

Do not bypass object slots.

Do not mix tetrahedron and cube Gaussian states.

Do not train cube in slot-0.

Do not fake `updates`, `loss`, `gaussian_count`, or `preview_fps`.

Do not hard-code success.

Do not print diff or full file contents.

## Repair Rules

If something is missing:

1. Make the smallest targeted repair.
2. If an existing Step 2A method blocks insertion, replace that method with backend-aware Step 2B logic.
3. Log the reason with:

```text
event=DIAGNOSIS_FAILED
event=repair_applied
```

4. Re-run verification.
5. Stop after success marker is written or after clear fallback success is proven.

## Final Output

Only print short stage comments and final values:

```text
slot_0_target
slot_0_backend
slot_0_gaussian_count
slot_0_updates
slot_0_preview_fps

slot_1_target
slot_1_backend
slot_1_gaussian_count
slot_1_updates
slot_1_preview_fps

requested_backend
cuda_available
rasterizer_available
fallback_used
SUCCESS_SLOT_GAUSSIAN_CUDA_PREVIEW_STEP2B written/not written
```
