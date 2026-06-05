#!/usr/bin/env python3
from __future__ import annotations

"""
Anchor fix for unconscious_sleep_loop_patch_v1.

Use when the previous patch failed with:

    [ERROR] Anchor not found in src/apps/life_runtime.py

This script only fixes life_runtime.py and is safe after a partial patch.
It does not require the exact old anchor block.
"""

from pathlib import Path
import re
import sys


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").exists() and ((candidate / ".git").exists() or (candidate / "config").exists()):
            return candidate
    return Path.cwd().resolve()


ROOT = find_repo_root()
PATH = ROOT / "src/apps/life_runtime.py"


UNCONSCIOUS_BLOCK = """
        # Strict unconscious sleep/replay loop:
        #   M11 affect -> M13 retrieval + M4 identity -> M2 replay selector
        #   -> next M5 FocusFeedbackBoundary seed.
        if hasattr(self, "compute_autobiographical_retrieval"):
            try:
                self.compute_autobiographical_retrieval(obs, out)
            except Exception as e:
                if not hasattr(self, "_autobiographical_retrieval_warned"):
                    print(f"[autobiographical_memory] retrieval skipped: {e}")
                    self._autobiographical_retrieval_warned = True
        if hasattr(self, "compute_event_dream_replay"):
            try:
                self.compute_event_dream_replay(obs, out)
            except Exception as e:
                if not hasattr(self, "_event_dream_replay_warned"):
                    print(f"[event_dream_replay] compute skipped: {e}")
                    self._event_dream_replay_warned = True
        if hasattr(self, "maybe_print_event_dream_replay_trace"):
            self.maybe_print_event_dream_replay_trace(out)
        if hasattr(self, "maybe_print_autobiographical_memory_trace"):
            self.maybe_print_autobiographical_memory_trace(out)

"""


def main() -> int:
    if not PATH.exists():
        raise FileNotFoundError(f"missing file: {PATH}")

    text = PATH.read_text(encoding="utf-8")
    original = text

    # 1) Add stage labels to the two M5 calls, but do not fail if already changed.
    if 'model_stage="pre_observe"' not in text:
        text, n = re.subn(
            r'out0\s*=\s*self\.model_step\(\s*obs0\s*,\s*self\.state\s*\)',
            'out0 = self.model_step(obs0, self.state, model_stage="pre_observe")',
            text,
            count=1,
        )
        if n:
            print('[ok] life_runtime.py: added model_stage="pre_observe"')
        else:
            print('[warn] life_runtime.py: pre_observe model_step anchor not found; maybe already changed')
    else:
        print('[skip] life_runtime.py: model_stage="pre_observe" already present')

    if 'model_stage="main"' not in text:
        text, n = re.subn(
            r'out\s*=\s*self\.model_step\(\s*obs\s*,\s*self\.state\s*\)',
            'out = self.model_step(obs, self.state, model_stage="main")',
            text,
            count=1,
        )
        if n:
            print('[ok] life_runtime.py: added model_stage="main"')
        else:
            print('[warn] life_runtime.py: main model_step anchor not found; maybe already changed')
    else:
        print('[skip] life_runtime.py: model_stage="main" already present')

    # 2) Insert the strict unconscious block after affect is attached.
    if "Strict unconscious sleep/replay loop" in text:
        print("[skip] life_runtime.py: unconscious loop block already present")
    else:
        anchor = '            out["affect"] = emotion["affect"]\n'
        idx = text.find(anchor)
        if idx < 0:
            raise RuntimeError(
                'Could not find affect anchor: out["affect"] = emotion["affect"]. '
                'Insert the unconscious block manually after affect is attached.'
            )
        insert_at = idx + len(anchor)
        text = text[:insert_at] + UNCONSCIOUS_BLOCK + text[insert_at:]
        print("[ok] life_runtime.py: inserted strict unconscious sleep/replay block after affect")

    if text != original:
        PATH.write_text(text, encoding="utf-8")
        print("[ok] life_runtime.py written")
    else:
        print("[skip] no changes")

    print("\nNow run:")
    print("  python -m py_compile src/apps/life_runtime.py")
    print("  python -m py_compile src/modules/m02_event_dream_replay/event_dream_replay.py")
    print("  python -m py_compile src/modules/m02_event_dream_replay/event_dream_runtime.py")
    print("  python -m py_compile src/apps/unified_conscious_viewer.py")
    print("  python -m py_compile src/shared/config.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
