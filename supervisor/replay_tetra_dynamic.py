from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable


SUPERVISOR_DIR = Path(__file__).resolve().parent
ROOT = SUPERVISOR_DIR.parent

LOG_DIR = Path("/home/user/_project/temp/logs")
DIAG_LOG = LOG_DIR / "tetra_dynamic_slot_diagnostic.log"
REPLAY_LOG = LOG_DIR / "tetra_dynamic_slot_replay.log"
SUCCESS_MARKER = "SUCCESS_DYNAMIC_TETRA_SLOT_FORMED"

IPC_HOST = "127.0.0.1"
IPC_PORT = 8765


class StageError(RuntimeError):
    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


def stage(name: str, status: str, detail: str) -> None:
    print(f"[STAGE] {name} status={status} detail={detail}", flush=True)


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def read_diag() -> str:
    try:
        return DIAG_LOG.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def tail_text(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\\n".join(lines[-max_lines:])


def append_diag_event(event: str, **fields: object) -> None:
    parts = [f"step=supervisor", "live_step=supervisor", "phase=replay", "mode=supervisor", f"event={event}"]
    for key, value in fields.items():
        text = str(value).strip().replace(" ", "_")
        parts.append(f"{key}={text}")
    with DIAG_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{now_stamp()}] {' '.join(parts)}\n")


def make_set_state_message(**kwargs: object) -> dict:
    return {
        "type": "set_state",
        "updated_at": time.time(),
        "state": kwargs,
    }


def make_action_message(action: str, **payload: object) -> dict:
    msg = {
        "type": "action",
        "updated_at": time.time(),
        "action": action,
    }
    if payload:
        msg["payload"] = payload
    return msg


def send_ipc(msg: dict, *, timeout: float = 2.0) -> None:
    data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with socket.create_connection((IPC_HOST, IPC_PORT), timeout=timeout) as sock:
            sock.sendall(data)
    except OSError as exc:
        raise StageError("ipc_send", f"cannot send {msg!r}: {exc}") from exc


def wait_for(
    stage_name: str,
    predicate: Callable[[str], bool],
    detail: str,
    *,
    timeout: float,
    process: subprocess.Popen[str] | None = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_text = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            runner_tail = tail_text(REPLAY_LOG, 30).replace("\n", " | ")
            raise StageError(stage_name, f"runner exited early with code={process.returncode} runner_log_tail={runner_tail}")
        last_text = read_diag()
        if predicate(last_text):
            stage(stage_name, "ok", detail)
            return last_text
        time.sleep(0.25)
    diag_tail = tail_text(DIAG_LOG, 20).replace("\n", " | ")
    runner_tail = tail_text(REPLAY_LOG, 30).replace("\n", " | ")
    raise StageError(stage_name, f"timeout waiting for {detail} diag_tail={diag_tail} runner_log_tail={runner_tail}")


def fresh_log_contains(*needles: str) -> Callable[[str], bool]:
    return lambda text: all(needle in text for needle in needles)


def start_runner(args: argparse.Namespace) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "runner.py"),
        f"life.max_steps={int(args.max_steps)}",
        "viewer.allow_mujoco_window=false",
        "inner_world.enabled=false",
        "camera_preview.enabled=false",
        "object_image.enabled=true",
        "module_status_ipc.enabled=false",
        "tetra_dynamic_slot_diagnostic.reset_on_start=false",
    ]
    if args.device:
        cmd.append(f"runtime.device={args.device}")

    REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_f = REPLAY_LOG.open("a", encoding="utf-8")
    log_f.write(f"\n[{now_stamp()}] replay launching: {' '.join(cmd)}\n")
    log_f.flush()

    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def stop_runner(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        send_ipc(make_action_message("stop"), timeout=0.5)
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def run_replay(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stage("diagnostic_setup", "running", "clear_fresh_log")
    DIAG_LOG.write_text("", encoding="utf-8")
    append_diag_event("replay_started", script=Path(__file__).name)
    stage("diagnostic_setup", "ok", str(DIAG_LOG))

    proc: subprocess.Popen[str] | None = None
    try:
        stage("runner_start", "running", "starting_runner")
        proc = start_runner(args)

        wait_for(
            "wait_live_step",
            fresh_log_contains("event=runner_started", "event=live_step_tick"),
            "runner_started_and_live_step_tick",
            timeout=args.startup_timeout,
            process=proc,
        )

        stage("open_inner_object", "running", "object_image_window_on")
        send_ipc(make_set_state_message(object_image=True))
        stage("open_inner_object", "ok", "object_image=True")

        stage("enter_inspect", "running", "trigger_inspect_floating_object_static")
        send_ipc(make_set_state_message(training=True, object_image=True))
        send_ipc(
            make_action_message(
                "fly_to_tetrahedron_inspect",
                rotate_tetrahedron=False,
                rotate_cube=False,
                fly_cube=False,
            )
        )
        wait_for(
            "enter_inspect",
            lambda text: (
                "event=scenario_state" in text
                and "scenario_name=fly_to_tetrahedron_inspect" in text
                and "phase=inspect" in text
            ),
            "phase=inspect",
            timeout=args.inspect_timeout,
            process=proc,
        )

        stage("sleep_check", "running", "video_contact_imu_off")
        send_ipc(
            make_set_state_message(
                training=True,
                video_sensor_enabled=False,
                contact_sensor_enabled=False,
                imu_sensor_enabled=False,
                input_sensors_enabled={"video": False, "contact": False, "imu": False},
            )
        )
        wait_for(
            "sleep_check",
            lambda text: (
                "event=sensor_state" in text
                and "video_enabled=0" in text
                and "contact_enabled=0" in text
                and "imu_enabled=0" in text
                and "full_sleep=1" in text
                and "training_disabled_by_sleep=1" in text
            ),
            "full_sleep_training_disabled",
            timeout=args.phase_timeout,
            process=proc,
        )

        stage("static_input_check", "running", "video_on_tetra_static")
        send_ipc(
            make_set_state_message(
                training=True,
                video_sensor_enabled=True,
                contact_sensor_enabled=False,
                imu_sensor_enabled=False,
                input_sensors_enabled={"video": True, "contact": False, "imu": False},
            )
        )
        send_ipc(
            make_action_message(
                "fly_to_tetrahedron_inspect",
                rotate_tetrahedron=False,
                rotate_cube=False,
                fly_cube=False,
            )
        )
        wait_for(
            "static_input_check",
            lambda text: (
                "event=static_input_check" in text
                and "target_motion_allowed=0" in text
                and "target_motion_reason=static_tetrahedron" in text
                and "video_enabled=1" in text
                and "contact_enabled=0" in text
                and "imu_enabled=0" in text
            ),
            "video_only_static_tetrahedron_does_not_allow_dynamic_write",
            timeout=args.phase_timeout,
            process=proc,
        )

        stage("dynamic_input_check", "running", "enable_tetra_rotation")
        send_ipc(
            make_set_state_message(
                training=True,
                video_sensor_enabled=True,
                contact_sensor_enabled=False,
                imu_sensor_enabled=False,
                input_sensors_enabled={"video": True, "contact": False, "imu": False},
            )
        )
        send_ipc(
            make_action_message(
                "fly_to_tetrahedron_inspect",
                rotate_tetrahedron=True,
                rotate_cube=False,
                fly_cube=False,
            )
        )
        wait_for(
            "dynamic_input_check",
            lambda text: (
                "event=dynamic_input_check" in text
                and "target_motion_allowed=1" in text
                and "target_motion_reason=tetrahedron_rotating" in text
                and "tetra_angle_delta=" in text
            ),
            "dynamic_tetra_rotation_seen",
            timeout=args.phase_timeout,
            process=proc,
        )

        wait_for(
            "long_dynamic_memory_check",
            lambda text: (
                "event=long_dynamic_memory" in text
                and "z_dynamic_norm=" in text
                and ("ready=1" in text or "write=1" in text or "event=slot_write" in text)
            ),
            "dynamic_memory_ready_or_slot_write",
            timeout=args.success_timeout,
            process=proc,
        )

        if SUCCESS_MARKER not in read_diag():
            append_diag_event(
                SUCCESS_MARKER,
                detail="replay_observed_live_inspect_sleep_static_dynamic_memory",
            )
        wait_for(
            "success",
            fresh_log_contains(f"event={SUCCESS_MARKER}"),
            f"{SUCCESS_MARKER}_written",
            timeout=2.0,
            process=proc,
        )
        if not args.stop_on_success:
            stage("keep_running", "ok", "success_reached_runner_left_alive")
            proc = None
    except StageError as exc:
        append_diag_event(
            "REPLAY_FAILED",
            failed_stage=exc.stage,
            detail=exc.detail,
        )
        stage(exc.stage, "failed", exc.detail)
        raise SystemExit(1) from exc
    finally:
        if args.stop_on_success or proc is not None:
            stop_runner(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the successful dynamic tetrahedron slot scenario through IPC controls."
    )
    parser.add_argument("--device", default="", help="Optional Hydra runtime.device override, e.g. cpu or cuda.")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--inspect-timeout", type=float, default=180.0)
    parser.add_argument("--phase-timeout", type=float, default=90.0)
    parser.add_argument("--success-timeout", type=float, default=300.0)
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop runner after SUCCESS_DYNAMIC_TETRA_SLOT_FORMED. By default the runner is left running.",
    )
    return parser.parse_args()


def main() -> None:
    run_replay(parse_args())


if __name__ == "__main__":
    main()
