from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SUPERVISOR_DIR = Path(__file__).resolve().parent
ROOT = SUPERVISOR_DIR.parent

import config

CODEX_SANDBOX = "danger-full-access"
PROMPT_FILE = config.PROMPT_FILE
TASK_FILE = config.TASK_FILE

TEMP_LOG_DIR = Path("/home/user/_project/temp/logs")
DIAG_LOG = TEMP_LOG_DIR / "tetra_dynamic_slot_diagnostic.log"
SUPERVISOR_LOG = TEMP_LOG_DIR / "supervisor_run.log"


def stage(name: str) -> None:
    line = f"[SUPERVISOR_STAGE] {name}"
    print(line)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=str(PROMPT_FILE))
    parser.add_argument("--task", default=str(TASK_FILE))
    parser.add_argument("--sandbox", default=CODEX_SANDBOX)
    args = parser.parse_args()

    prompt_path = Path(args.prompt)
    task_path = Path(args.task)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")
    if not task_path.exists():
        raise SystemExit(f"Task json not found: {task_path}")

    TEMP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_LOG.touch(exist_ok=True)
    SUPERVISOR_LOG.touch(exist_ok=True)

    stage("launcher_prepare")
    prompt = prompt_path.read_text(encoding="utf-8")
    task_json = task_path.read_text(encoding="utf-8")
    prompt = (
        prompt
        + "\n\nFull task JSON:\n"
        + task_json
        + "\n\nRuntime paths prepared by launcher:\n"
        + f"- REPO_ROOT={ROOT}\n"
        + f"- REPO_NAME=Conscious-World-Model-System\n"
        + f"- SUPERVISOR_DIR={SUPERVISOR_DIR}\n"
        + f"- TASK_FILE={task_path}\n"
        + f"- TEMP_LOG_DIR={TEMP_LOG_DIR}\n"
        + f"- DIAG_LOG={DIAG_LOG}\n"
        + f"- SUPERVISOR_LOG={SUPERVISOR_LOG}\n"
        + "\nImportant: run all repository commands from REPO_ROOT, not from the supervisor folder.\n"
    )

    with SUPERVISOR_LOG.open("a", encoding="utf-8") as f:
        f.write("\n=== launching codex supervisor ===\n")
        f.write(f"repo_root={ROOT}\n")
        f.write(f"repo_name=Conscious-World-Model-System\n")
        f.write(f"supervisor_dir={SUPERVISOR_DIR}\n")
        f.write(f"task_file={task_path}\n")
        f.write(f"diagnostic_log={DIAG_LOG}\n")

    stage("codex_exec_start")
    subprocess.run(
        [
            "codex",
            "exec",
            "--sandbox",
            args.sandbox,
            prompt,
        ],
        cwd=ROOT,
        text=True,
        check=False,
    )
    stage("codex_exec_finished")


if __name__ == "__main__":
    main()
