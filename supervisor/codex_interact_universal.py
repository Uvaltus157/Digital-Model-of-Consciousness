from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

DEFAULT_REPO_ROOT = Path('/home/user/_project_d/GitHub/Conscious-World-Model-System')
DEFAULT_TEMP_LOG_DIR = Path('/home/user/_project/temp/logs')
DEFAULT_SANDBOX = 'danger-full-access'

HELP = '''
Commands:
  :demo              Ask Codex to demonstrate the configured task.
  :verify            Verify current result without code changes unless critical check is broken.
  :status            Read current logs and summarize status.
  :fix <text>        Make a targeted repair, then verify.
  :ask <text>        Ask a question without forcing code changes.
  :run <stage>       Ask Codex to run a named stage from the task.
  :log               Show log/transcript paths.
  :help              Show this help.
  :quit              Exit.

Normal text is treated as :ask.
'''


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


class UniversalCodexSupervisor:
    def __init__(self, repo_root: Path, prompt_file: Path, task_file: Path, task_name: str, sandbox: str, temp_log_dir: Path, max_transcript_chars: int = 24000) -> None:
        self.repo_root = repo_root.resolve()
        self.prompt_file = prompt_file.resolve()
        self.task_file = task_file.resolve()
        self.task_name = task_name
        self.sandbox = sandbox
        self.temp_log_dir = temp_log_dir
        self.max_transcript_chars = max_transcript_chars
        self.temp_log_dir.mkdir(parents=True, exist_ok=True)
        safe = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in task_name)
        self.diag_log = self.temp_log_dir / f'{safe}_interactive_diagnostic.log'
        self.supervisor_log = self.temp_log_dir / f'{safe}_interactive_supervisor.log'
        self.transcript = self.temp_log_dir / f'{safe}_interactive_transcript.md'

    def log_line(self, text: str) -> None:
        with self.supervisor_log.open('a', encoding='utf-8') as f:
            f.write(text.rstrip() + '\n')

    def append_transcript(self, role: str, text: str) -> None:
        with self.transcript.open('a', encoding='utf-8') as f:
            f.write(f'\n\n## {role} @ {now()}\n\n{text.strip()}\n')

    def recent_transcript(self) -> str:
        if not self.transcript.exists():
            return ''
        return self.transcript.read_text(encoding='utf-8')[-self.max_transcript_chars:]

    def build_prompt(self, user_request: str, mode: str) -> str:
        base = self.prompt_file.read_text(encoding='utf-8')
        task_json = self.task_file.read_text(encoding='utf-8')
        return (
            base
            + '\n\nFull task JSON:\n' + task_json
            + '\n\nRuntime paths prepared by universal supervisor:\n'
            + f'- TASK_NAME={self.task_name}\n'
            + f'- REPO_ROOT={self.repo_root}\n'
            + f'- TEMP_LOG_DIR={self.temp_log_dir}\n'
            + f'- DIAG_LOG={self.diag_log}\n'
            + f'- SUPERVISOR_LOG={self.supervisor_log}\n'
            + f'- TRANSCRIPT={self.transcript}\n'
            + '\nRecent interactive transcript:\n' + self.recent_transcript()
            + '\n\nCurrent interactive user request:\n'
            + f'mode={mode}\n{user_request.strip()}\n'
            + '\nImportant execution policy:\n'
            + '- Run repository commands from REPO_ROOT.\n'
            + '- Do not print git diff or full file contents.\n'
            + '- Use short [STAGE] lines to explain what you are doing.\n'
            + '- If you modify code, apply the patch and verify with concrete evidence.\n'
            + '- If training is performed, save weights/checkpoint via IPC if the task requires it.\n'
        )

    def run_codex(self, user_request: str, mode: str) -> int:
        self.diag_log.touch(exist_ok=True)
        self.supervisor_log.touch(exist_ok=True)
        prompt = self.build_prompt(user_request, mode)
        self.append_transcript('USER', f'mode={mode}\n{user_request}')
        print(f'[INTERACTIVE] sending_to_codex task={self.task_name} mode={mode}')
        self.log_line(f'[{now()}] sending_to_codex task={self.task_name} mode={mode} request={user_request[:300]!r}')
        proc = subprocess.run(['codex', 'exec', '--sandbox', self.sandbox, prompt], cwd=self.repo_root, text=True, check=False)
        self.log_line(f'[{now()}] codex_returncode={proc.returncode} task={self.task_name} mode={mode}')
        self.append_transcript('SYSTEM', f'Codex return code: {proc.returncode}')
        return int(proc.returncode)

    def normalize_input(self, line: str) -> tuple[str, str] | None:
        text = line.strip()
        if not text:
            return None
        if text in (':q', ':quit', 'quit', 'exit'):
            return ('quit', '')
        if text == ':help':
            return ('help', '')
        if text == ':log':
            return ('log', '')
        if text == ':demo':
            return ('demo', 'Demonstrate the configured task end-to-end. Read current diagnostics, open/show required windows if possible, and provide concrete evidence in short [STAGE] comments. Do not modify code unless verification is broken.')
        if text == ':verify':
            return ('verify', 'Verify the configured task end-to-end. Do not modify code unless a critical check is broken. Use concrete logs/status/window evidence.')
        if text == ':status':
            return ('status', 'Read current logs and runtime state. Summarize the status of the configured task, including success markers and remaining issues. Do not modify code.')
        if text.startswith(':fix '):
            return ('fix', text[len(':fix '):].strip())
        if text.startswith(':ask '):
            return ('ask', text[len(':ask '):].strip())
        if text.startswith(':run '):
            return ('run', 'Run or verify this named stage from the configured task: ' + text[len(':run '):].strip())
        return ('ask', text)

    def print_banner(self) -> None:
        print('[INTERACTIVE] Universal Codex Supervisor')
        print(f'[INTERACTIVE] task={self.task_name}')
        print(f'[INTERACTIVE] repo={self.repo_root}')
        print(f'[INTERACTIVE] prompt={self.prompt_file}')
        print(f'[INTERACTIVE] task_json={self.task_file}')
        print(f'[INTERACTIVE] diag_log={self.diag_log}')
        print(f'[INTERACTIVE] transcript={self.transcript}')
        print(HELP)

    def loop(self, start_demo: bool = False, start_verify: bool = False) -> None:
        self.print_banner()
        self.log_line(f'\n=== universal supervisor started {now()} task={self.task_name} ===')
        if start_demo:
            self.run_codex('Start with an end-to-end demonstration of the configured task.', 'demo')
        if start_verify:
            self.run_codex('Start with verification of the configured task.', 'verify')
        while True:
            try:
                line = input('supervisor> ')
            except (EOFError, KeyboardInterrupt):
                print('\n[INTERACTIVE] exit')
                break
            parsed = self.normalize_input(line)
            if parsed is None:
                continue
            mode, request = parsed
            if mode == 'quit':
                print('[INTERACTIVE] stopped by user')
                self.log_line(f'=== universal supervisor stopped {now()} ===')
                break
            if mode == 'help':
                print(HELP)
                continue
            if mode == 'log':
                print(f'Diagnostic log: {self.diag_log}')
                print(f'Supervisor log:  {self.supervisor_log}')
                print(f'Transcript:      {self.transcript}')
                continue
            if mode in ('fix', 'ask', 'run') and not request:
                print(f'Write: :{mode} <text>')
                continue
            self.run_codex(request, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description='Universal interactive Codex supervisor')
    parser.add_argument('--repo-root', default=str(DEFAULT_REPO_ROOT))
    parser.add_argument('--prompt-file', required=True)
    parser.add_argument('--task-file', required=True)
    parser.add_argument('--task-name', default='codex_task')
    parser.add_argument('--sandbox', default=DEFAULT_SANDBOX)
    parser.add_argument('--temp-log-dir', default=str(DEFAULT_TEMP_LOG_DIR))
    parser.add_argument('--max-transcript-chars', type=int, default=24000)
    parser.add_argument('--start-demo', action='store_true')
    parser.add_argument('--start-verify', action='store_true')
    args = parser.parse_args()
    sup = UniversalCodexSupervisor(
        repo_root=Path(args.repo_root),
        prompt_file=Path(args.prompt_file),
        task_file=Path(args.task_file),
        task_name=args.task_name,
        sandbox=args.sandbox,
        temp_log_dir=Path(args.temp_log_dir),
        max_transcript_chars=args.max_transcript_chars,
    )
    sup.loop(start_demo=args.start_demo, start_verify=args.start_verify)


if __name__ == '__main__':
    main()
