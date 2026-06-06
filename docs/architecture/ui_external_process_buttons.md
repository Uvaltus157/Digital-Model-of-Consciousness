# UI external process buttons

This rule is architectural and applies to every control-panel button that starts
or shows a separate process/window outside the main runner lifecycle.

It does not apply to command-only buttons that only send IPC/state/action
messages and do not own a detached process/window. Command-only buttons should
use normal command styling unless their own feature contract explicitly says
otherwise.

## Standard behavior

Buttons that launch a separate process or a separate monitor window must:

- remain enabled regardless of runner connection, runner activity, or module
  runtime state;
- use the same purple base styling as `Module Debug` while their process/window
  is not visible;
- turn green only while the corresponding process/window is currently open or
  visible;
- close or hide their own process/window on the second click, then return to the
  purple base styling;
- never be treated as runner-dependent action buttons.

The green state is an active visibility/process indicator, not an "available"
indicator. A button must not be green merely because the feature exists or
because the runner is connected.

## Current examples

These M8 buttons follow this contract:

- `Module Debug`
- `Module Lab`
- `Sleep Replay Monitor`
- `Replay Quality Monitor`
- `M5 Learning Quality`

Any new button that opens a detached diagnostics process, detached monitor, or
standalone tool window must follow the same contract by default.

`Сон / replay mode` is not part of this contract: it sends a state command to the
runner and does not launch a separate process/window.
