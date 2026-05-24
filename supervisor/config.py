from pathlib import Path

SUPERVISOR_DIR = Path(__file__).resolve().parent

NAME_TASK = "codex_debug_open3d_rpc_window_output"

PROMPT_FILE = SUPERVISOR_DIR / "tasks" / f"{NAME_TASK}.md"
TASK_FILE   = SUPERVISOR_DIR / "tasks" / f"{NAME_TASK}.json"



