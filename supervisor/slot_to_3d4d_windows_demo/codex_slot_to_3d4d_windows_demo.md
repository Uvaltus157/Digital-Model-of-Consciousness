# Codex Task: ready slot → 3D decode → 4D decode → Inner Object + Inner Object Open3D RPC

Repository:

```text
/home/user/_project_d/GitHub/Conscious-World-Model-System
```

Branch:

```text
work_2
```

Logs:

```text
/home/user/_project/temp/logs
/home/user/_project/temp/logs/tetra_dynamic_slot_diagnostic.log
```

## Goal

Demonstrate and verify the full path:

```text
ready dynamic slot
-> 3D decoding / 3DGS Gaussian reconstruction
-> 4D timeline
-> 4D neural deformation
-> 4D playback preview
-> Inner Object window
-> Inner Object Open3D JSON-RPC window
```

Use the existing slot-0/tetrahedron and slot-1/cube if already formed. Train only if required to make the pipeline real.

## Output policy

Do not print git diff, full file contents, long file lists, raw tensors, or full logs.

Print only short stages:

```text
[STAGE] <stage_id> status=<running|ok|failed|repair_applied> comment=<short explanation>
```

Example:

```text
[STAGE] slot_ready_check status=ok comment=slot-0/tetrahedron and slot-1/cube are present
[STAGE] decode_3d status=running comment=checking Gaussian xyz point counts
[STAGE] decode_4d status=running comment=checking timeline, deformation and playback
[STAGE] windows status=running comment=opening Inner Object and Inner Object Open3D RPC
```

## Required windows

Show or launch:

```text
Inner Object
Inner Object Open3D RPC
```

Open3D RPC command:

```bash
python visualizer/open3d_slot_viewer_rpc.py --host 127.0.0.1 --port 8771 --slot both --mode deformed
```

If the control panel is used, the button should be:

```text
Open Inner Object Open3D RPC
```

## Required code presence

Check these files:

```text
runtime/object_imagery_runtime.py
runtime/slot_4d_reconstruction.py
runtime/slot_4d_deformation.py
runtime/slot_4d_playback.py
runtime/slot_4d_jsonrpc_stream.py
visualizer/inner_object_visualizer_v2.py
visualizer/open3d_slot_viewer_rpc.py
ctrl_panel/pyqt_module_debug_ipc_status.py
```

## Pipeline verification

### 1. Ready slots

Verify:

```text
slot-0 target=tetrahedron
slot-1 target=cube
slot-0 is not overwritten by cube
slot-1 is separately allocated
```

### 2. 3D decode / 3DGS

Verify:

```text
slot_gaussian_reconstructor.states[0] exists
slot_gaussian_reconstructor.states[1] exists
raw Gaussian xyz point count > 0 for both slots
```

Useful events:

```text
event=slot_observation_reconstruction
event=slot_gaussian_reconstruction
event=slot_gaussian_preview
```

### 3. 4D timeline

Verify Step 3A:

```text
event=slot_4d_frame
event=slot_4d_timeline
event=SUCCESS_SLOT_4D_TIMELINE_STEP3A
```

Both slots should have timeline frames > 1.

### 4. 4D deformation

Verify Step 3B:

```text
event=slot_4d_deformation_model
event=slot_4d_deformation_train
event=SUCCESS_SLOT_4D_DEFORMATION_STEP3B
```

Both slots should have deformation_updates > 0.

### 5. 4D playback

Verify Step 3C:

```text
event=slot_4d_playback_frame
event=slot_4d_deformed_render
event=SUCCESS_SLOT_4D_PLAYBACK_STEP3C
```

Both slots should have:

```text
render_valid=1
deformation_used=1
playback_frames > 0
```

### 6. Inner Object embedded preview

Verify Step 3E:

```text
_four_d_preview_image_panel exists
slot_4d_playback_rgb/depth/alpha are attached to decoded object
panel title: 4D / 3DGS PREVIEW IMAGE
panel is in row 2, right of decoder panels
```

Expected row:

```python
top_row = np.concatenate([rgb, depth, mask, long_dynamic_panel, preview_image_panel], axis=1)
```

### 7. Inner Object Open3D JSON-RPC

Verify Step 3H:

```text
runtime/slot_4d_jsonrpc_stream.py
visualizer/open3d_slot_viewer_rpc.py
event=slot_4d_jsonrpc_stream
event=SUCCESS_SLOT_4D_JSONRPC_STREAM_STEP3H
```

JSON-RPC methods:

```text
slot_viewer.ping
slot_viewer.get_status
slot_viewer.get_slot_frame
slot_viewer.get_both_slots
```

Manual status test:

```bash
python - <<'PY'
import json, urllib.request
url = 'http://127.0.0.1:8771/'
req = urllib.request.Request(url, data=json.dumps({'jsonrpc':'2.0','id':1,'method':'slot_viewer.get_status','params':{}}).encode(), headers={'Content-Type':'application/json'})
print(urllib.request.urlopen(req, timeout=2).read().decode())
PY
```

Expected: slot-0 and slot-1 point counts > 0.

## If something is missing

Make the smallest targeted repair. Do not rewrite the architecture. Do not fake tensors, point counts, window success, or success markers. Do not require CUDA; torch fallback is acceptable.

After each repair, run a concrete check and comment the result with `[STAGE]`.

## Runtime sequence

1. Check code for Step 3A/3B/3C/3E/3H.
2. Start `runner_v5_10.py` if not running.
3. Wait for live_step.
4. Verify slot-0/tetrahedron and slot-1/cube.
5. Verify 3D Gaussian xyz for both slots.
6. Verify 4D timeline/deformation/playback for both slots.
7. Open Inner Object.
8. Open Inner Object Open3D RPC.
9. Query JSON-RPC status.
10. Print final evidence.

If slots are missing, run tetra → cube scenario. If training occurs, save weights/checkpoint via existing IPC before final success.

## Success marker

Write or verify:

```text
event=SUCCESS_SLOT_TO_3D4D_WINDOWS_DEMO
```

Required fields:

```text
slot_0_target=tetrahedron
slot_0_raw_points>0
slot_0_deformed_points>0
slot_0_render_valid=1
slot_0_jsonrpc_points>0
slot_1_target=cube
slot_1_raw_points>0
slot_1_deformed_points>0
slot_1_render_valid=1
slot_1_jsonrpc_points>0
inner_object_window=shown
inner_object_open3d_rpc_window=shown
jsonrpc_host=127.0.0.1
jsonrpc_port=8771
weights_saved_via_ipc=0_or_1
```

## Final output

Print only concise values:

```text
slot_0_target
slot_0_raw_points
slot_0_deformed_points
slot_0_render_valid
slot_0_jsonrpc_points
slot_1_target
slot_1_raw_points
slot_1_deformed_points
slot_1_render_valid
slot_1_jsonrpc_points
inner_object_window
inner_object_open3d_rpc_window
jsonrpc_status
weights_saved_via_ipc
checkpoint_path_if_saved
SUCCESS_SLOT_TO_3D4D_WINDOWS_DEMO written/not written
```
