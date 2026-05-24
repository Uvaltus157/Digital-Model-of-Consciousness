#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-/home/user/_project_d/GitHub/Conscious-World-Model-System}"
SUP_DIR="$REPO_ROOT/supervisor"
TASK_DIR="$SUP_DIR/slot_to_3d4d_windows_demo"
python "$SUP_DIR/codex_interact_universal.py" \
  --repo-root "$REPO_ROOT" \
  --task-name "slot_to_3d4d_windows_demo" \
  --prompt-file "$TASK_DIR/codex_slot_to_3d4d_windows_demo.md" \
  --task-file "$TASK_DIR/codex_slot_to_3d4d_windows_demo.json" \
  --start-demo
