"""Runtime helpers for runner refactor."""

try:
    from src.shared.console_colors import install_colored_errors

    install_colored_errors()
except Exception:
    pass
