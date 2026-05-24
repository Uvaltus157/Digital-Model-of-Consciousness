from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

SUPERVISOR_DIR = Path(__file__).resolve().parent
ROOT = SUPERVISOR_DIR.parent

import config

CODEX_SANDBOX = "danger-full-access"
PROMPT_FILE = config.PROMPT_FILE
TASK_FILE = config.TASK_FILE

TEMP_LOG_DIR = Path("/home/user/_project/temp/logs")
DIAG_LOG = TEMP_LOG_DIR / "interactive_diagnostic.log"
SUPERVISOR_LOG = TEMP_LOG_DIR / "interactive_supervisor.log"
TRANSCRIPT = TEMP_LOG_DIR / "interactive_transcript.md"


HELP = """
Commands:
  :demo       Ask Codex to demonstrate/explain the current tetra->cube two-slot result.
  :status     Ask Codex to read current logs and summarize slot-0/slot-1 status.
  :fix <text> Ask Codex to make a targeted repair, then verify.
  :ask <text> Ask a question without forcing code changes.
  :log        Show diagnostic log path.
  :quit       Exit interactive supervisor.

You can also type a normal question directly.
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_line(text: str) -> None:
    TEMP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def append_transcript(role: str, text: str) -> None:
    TEMP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TRANSCRIPT.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {role} @ {now()}\n\n{text.strip()}\n")


def build_prompt(user_request: str, mode: str) -> str:
    base = PROMPT_FILE.read_text(encoding="utf-8")
    task_json = TASK_FILE.read_text(encoding="utf-8")
    previous = ""
    if TRANSCRIPT.exists():
        txt = TRANSCRIPT.read_text(encoding="utf-8")
        # Keep enough context without making prompts enormous.
        previous = txt[-24000:]

    user_prompt = (
        base
        + "\n\nFull task JSON:\n"
        + task_json
        + "\n\nRuntime paths prepared by launcher:\n"
        + f"- REPO_ROOT={ROOT}\n"
        + f"- REPO_NAME=Conscious-World-Model-System\n"
        + f"- SUPERVISOR_DIR={SUPERVISOR_DIR}\n"
        + f"- TEMP_LOG_DIR={TEMP_LOG_DIR}\n"
        + f"- DIAG_LOG={DIAG_LOG}\n"
        + f"- SUPERVISOR_LOG={SUPERVISOR_LOG}\n"
        + f"- TRANSCRIPT={TRANSCRIPT}\n"
        + "\nRecent interactive transcript:\n"
        + previous
        + "\n\nCurrent interactive user request:\n"
        + f"mode={mode}\n"
        + user_request.strip()
        + "\n\nImportant: run all repository commands from REPO_ROOT. "
          "Do not print git diff or file lists. Explain current work in commented stage lines.\n"
    )
    
    return user_prompt


def run_codex(user_request: str, mode: str, sandbox: str) -> int:
    TEMP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_LOG.touch(exist_ok=True)
    SUPERVISOR_LOG.touch(exist_ok=True)

    prompt = build_prompt(user_request, mode)
    append_transcript("USER", f"mode={mode}\n{user_request}")

    print(f"[INTERACTIVE] sending_to_codex mode={mode}")
    log_line(f"[{now()}] sending_to_codex mode={mode} request={user_request[:200]!r}")

    proc = subprocess.run(
        ["codex", "exec", "--sandbox", sandbox, prompt],
        cwd=ROOT,
        text=True,
        check=False,
    )
    log_line(f"[{now()}] codex_returncode={proc.returncode} mode={mode}")
    append_transcript("SYSTEM", f"Codex return code: {proc.returncode}")
    return int(proc.returncode)


def normalize_input(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text:
        return None
    if text in (":q", ":quit", "quit", "exit"):
        return ("quit", "")
    if text == ":help":
        return ("help", "")
    if text == ":demo":
        return ("demo", "Demonstrate what was done: read current diagnostics/logs, explain the two-slot pipeline, show slot-0 tetrahedron and slot-1 cube evidence, and explain what is still editable. Do not modify code unless verification is broken.")
    if text == ":status":
        return ("status", "Read current logs and summarize the current status of slot-0, slot-1, success markers, inner object window, and whether any issue remains. Do not modify code.")
    if text == ":log":
        return ("log", "")
    if text.startswith(":fix "):
        return ("fix", text[len(":fix "):].strip())
    if text.startswith(":ask "):
        return ("ask", text[len(":ask "):].strip())
    return ("ask", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", default=CODEX_SANDBOX)
    parser.add_argument("--start-demo", action="store_true", help="Immediately ask Codex to demonstrate current result.")
    args = parser.parse_args()

    TEMP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("[INTERACTIVE] Codex two-slot supervisor")
    print(f"[INTERACTIVE] repo={ROOT}")
    print(f"[INTERACTIVE] diag_log={DIAG_LOG}")
    print(f"[INTERACTIVE] transcript={TRANSCRIPT}")
    print(HELP)

    log_line(f"\n=== interactive two-slot supervisor started {now()} ===")
    log_line(f"repo_root={ROOT}")
    log_line(f"diag_log={DIAG_LOG}")

    if args.start_demo:
        run_codex(
            "Start with a demonstration of the current tetrahedron->cube two-slot result. Explain what was done, read the diagnostic log, and show evidence in short commented stages. Do not print diffs/files.",
            "demo",
            args.sandbox,
        )

    while True:
        try:
            line = input("supervisor> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[INTERACTIVE] exit")
            break

        parsed = normalize_input(line)
        if parsed is None:
            continue

        mode, request = parsed
        if mode == "quit":
            print("[INTERACTIVE] stopped by user")
            log_line(f"=== interactive two-slot supervisor stopped {now()} ===")
            break
        if mode == "help":
            print(HELP)
            continue
        if mode == "log":
            print(f"Diagnostic log: {DIAG_LOG}")
            print(f"Supervisor log:  {SUPERVISOR_LOG}")
            print(f"Transcript:      {TRANSCRIPT}")
            continue

        if mode == "fix" and not request:
            print("Write: :fix <what to fix>")
            continue
        if mode == "ask" and not request:
            print("Write a question, or :ask <question>")
            continue

        run_codex(request, mode, args.sandbox)


if __name__ == "__main__":
    main()
