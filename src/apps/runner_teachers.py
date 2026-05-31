from __future__ import annotations

"""Teacher loading helpers for the V5.10 runner.

This module is dependency-light and is part of the gradual runner unload.
The runtime module owns `UnifiedSystem`, but the Hydra
entrypoint patches the runtime to use this extracted helper.
"""

import importlib
from pathlib import Path
from typing import Any, Tuple


def resolve_inner_speech_teacher_module_name(cfg: Any) -> str:
    """Resolve the configured inner-speech teacher module name."""
    train_cfg = getattr(cfg, "train", None)
    teacher_file = str(getattr(train_cfg, "inner_speech_teacher_file", "english_inner_speech_teacher.py"))
    if bool(getattr(train_cfg, "russian_inner_speech_teacher_enabled", False)):
        teacher_file = "russian_inner_speech_teacher.py"
    elif not bool(getattr(train_cfg, "english_inner_speech_teacher_enabled", True)):
        print("[inner_speech_teacher] english teacher disabled; falling back to russian teacher")
        teacher_file = "russian_inner_speech_teacher.py"

    module_name = Path(teacher_file).stem if teacher_file.endswith(".py") else teacher_file
    if "." not in module_name:
        module_name = f"src.modules.m07_inner_speech_thoughts.{module_name}"
    return module_name


def load_inner_speech_teacher_from_config(cfg: Any) -> Tuple[Any, Any]:
    """Load and instantiate the configured inner-speech vocab and teacher."""
    module_name = resolve_inner_speech_teacher_module_name(cfg)
    module = importlib.import_module(module_name)

    vocab_cls = (
        getattr(module, "InnerSpeechVocab", None)
        or getattr(module, "AnglishInnerSpeechVocab", None)
        or getattr(module, "EnglishInnerSpeechVocab", None)
        or getattr(module, "RussianInnerSpeechVocab", None)
    )
    teacher_cls = (
        getattr(module, "InnerSpeechTeacher", None)
        or getattr(module, "AnglishInnerSpeechTeacher", None)
        or getattr(module, "EnglishInnerSpeechTeacher", None)
        or getattr(module, "RussianInnerSpeechTeacher", None)
    )
    if vocab_cls is None or teacher_cls is None:
        raise RuntimeError(f"inner speech teacher module {module_name!r} must expose vocab and teacher classes")

    vocab = vocab_cls()
    teacher = teacher_cls(vocab)
    print(f"[inner_speech_teacher] loaded {module_name} | vocab={vocab.size}")
    return vocab, teacher
