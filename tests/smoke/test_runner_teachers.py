from __future__ import annotations

from types import SimpleNamespace

from src.apps.runner_teachers import resolve_inner_speech_teacher_module_name


def _cfg(**train_kwargs):
    return SimpleNamespace(train=SimpleNamespace(**train_kwargs))


def test_resolve_default_inner_speech_teacher_module() -> None:
    name = resolve_inner_speech_teacher_module_name(_cfg())
    assert name == "src.modules.m07_inner_speech_thoughts.english_inner_speech_teacher"


def test_resolve_russian_inner_speech_teacher_module() -> None:
    name = resolve_inner_speech_teacher_module_name(
        _cfg(russian_inner_speech_teacher_enabled=True)
    )
    assert name == "src.modules.m07_inner_speech_thoughts.russian_inner_speech_teacher"


def test_resolve_custom_fully_qualified_teacher_module() -> None:
    name = resolve_inner_speech_teacher_module_name(
        _cfg(inner_speech_teacher_file="custom.package.teacher")
    )
    assert name == "custom.package.teacher"


def test_resolve_custom_short_py_teacher_module() -> None:
    name = resolve_inner_speech_teacher_module_name(
        _cfg(inner_speech_teacher_file="my_teacher.py")
    )
    assert name == "src.modules.m07_inner_speech_thoughts.my_teacher"
