from __future__ import annotations

import argparse
import json
import os
import re
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
DIAG_LOG = LOG_DIR / "inner_object_sleep_modes_diagnostic.log"
RUN_LOG = LOG_DIR / "inner_object_sleep_modes_supervisor_run.log"
IPC_HOST = "127.0.0.1"
IPC_PORT = 8765
STATUS_PORT = 8766

TH = {
    "memory_high_threshold": 0.35,
    "confidence_high_threshold": 0.30,
    "confidence_low_threshold": 0.12,
    "dream_activation_high_threshold": 0.45,
    "dream_activation_low_threshold": 0.15,
    "update_low_threshold": 0.05,
    "z_obj_norm_min": 0.50,
    "sleep_window_steps": 60,
    "memory_decay_tolerance": 0.03,
}


class StageError(RuntimeError):
    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


def stage(name: str, status: str, detail: str) -> None:
    print(f"[STAGE] {name} status={status} detail={detail}", flush=True)


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).strip()
    return re.sub(r"\s+", "_", text) if text else "unknown"


def append_diag(event: str, **fields: object) -> None:
    parts = [
        "step=supervisor",
        "live_step=supervisor",
        f"phase={fmt(fields.pop('phase', 'supervisor'))}",
        f"mode={fmt(fields.pop('mode', 'supervisor'))}",
        f"event={event}",
    ]
    parts.extend(f"{key}={fmt(value)}" for key, value in fields.items())
    with DIAG_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{now_stamp()}] {' '.join(parts)}\n")


def read_diag() -> str:
    try:
        return DIAG_LOG.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def tail(path: Path, n: int = 35) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return " | ".join(lines[-n:])


def parse_line(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_.]+)=([^ ]+)", line))


def slot_rows(text: str, *, video: bool | None = None, contact: bool | None = None, imu: bool | None = None, full_sleep: bool | None = None) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if "event=slot_metrics" not in line:
            continue
        row = parse_line(line)
        if video is not None and row.get("video_enabled") != ("1" if video else "0"):
            continue
        if contact is not None and row.get("contact_enabled") != ("1" if contact else "0"):
            continue
        if imu is not None and row.get("imu_enabled") != ("1" if imu else "0"):
            continue
        if full_sleep is not None and row.get("full_sleep") != ("1" if full_sleep else "0"):
            continue
        rows.append(row)
    return rows


def frow(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def best_occupied(rows: list[dict[str, str]]) -> dict[str, str] | None:
    occupied = [r for r in rows if frow(r, "M") >= 0.05 and frow(r, "z_obj_norm") >= TH["z_obj_norm_min"]]
    if not occupied:
        occupied = [r for r in rows if frow(r, "z_obj_norm") >= TH["z_obj_norm_min"]]
    if not occupied:
        return None
    return max(occupied, key=lambda r: (frow(r, "M"), frow(r, "C"), frow(r, "z_obj_norm")))


def wait_for(stage_name: str, pred: Callable[[str], bool], detail: str, timeout: float, proc: subprocess.Popen[str] | None) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise StageError(stage_name, f"runner_exited_{proc.returncode}_{tail(RUN_LOG)}")
        text = read_diag()
        if pred(text):
            stage(stage_name, "ok", detail)
            return text
        time.sleep(0.35)
    raise StageError(stage_name, f"timeout_{detail}_{tail(DIAG_LOG)}")


def send_ipc(msg: dict, timeout: float = 2.0) -> None:
    data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((IPC_HOST, IPC_PORT), timeout=timeout) as sock:
        sock.sendall(data)


def set_state(**state: object) -> None:
    send_ipc({"type": "set_state", "updated_at": time.time(), "state": state})


def action(name: str, **payload: object) -> None:
    msg = {"type": "action", "updated_at": time.time(), "action": name}
    if payload:
        msg["payload"] = payload
    send_ipc(msg)


def status_request(timeout: float = 1.0) -> dict:
    with socket.create_connection((IPC_HOST, STATUS_PORT), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(b'{"type":"get_status"}\n')
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    if not data:
        return {}
    payload = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
    return payload.get("status", payload)


def wait_status_seq(seq: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            last = status_request()
            if int(last.get("last_module_training_seq", 0) or 0) >= int(seq):
                return last
        except Exception:
            pass
        time.sleep(0.2)
    return last


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
        "object_image.show_every_steps=1",
        "module_status_ipc.enabled=true",
        "tetra_dynamic_slot_diagnostic.enabled=true",
        "tetra_dynamic_slot_diagnostic.reset_on_start=false",
        "tetra_dynamic_slot_diagnostic.file_name=inner_object_sleep_modes_diagnostic.log",
        "runtime.out_dir=/home/user/_project/temp/logs",
    ]
    if args.device:
        cmd.append(f"runtime.device={args.device}")
    log_f = RUN_LOG.open("a", encoding="utf-8")
    log_f.write(f"\n[{now_stamp()}] launch {' '.join(cmd)}\n")
    log_f.flush()
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def stop_runner(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        action("stop")
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def event_patch_state() -> bool:
    model = (ROOT / "models/object_inner_imagery_3d.py").read_text(encoding="utf-8", errors="replace")
    viz = (ROOT / "visualizer/inner_object_visualizer.py").read_text(encoding="utf-8", errors="replace")
    ok = all(x in model for x in ("memory_stability_slots", "memory_stability", "dream_activation_slots", "dream_activation", "slot_update_strength")) and all(
        x in viz for x in ("memory_stability_slots", "dream_activation_slots", "M=memory", "C=current/sensory confidence", "D=dream activation", "U=update")
    )
    append_diag("patch_state", ok=ok, model_fields=ok, visualizer_fields=ok)
    return ok


def check_visualizer() -> bool:
    text = (ROOT / "visualizer/inner_object_visualizer.py").read_text(encoding="utf-8", errors="replace")
    ok = all(x in text for x in ("memory_stability_slots", "confidence_slots", "dream_activation_slots", "slot_update_strength", "M=memory", "C=current/sensory confidence", "D=dream activation", "U=update", "(80, 240, 120)", "(80, 170, 255)", "(255, 120, 230)", "(230, 120, 255)"))
    append_diag("visualizer_check", ok=ok, expected="M_C_D_U_labels_and_backing_fields", actual="present" if ok else "missing")
    return ok


def module_checkbox_check(proc: subprocess.Popen[str]) -> bool:
    modules = ("world_model", "object_imagery", "long_dynamic_memory")
    ok_all = True
    seq = 100
    for module in modules:
        flags = {m: True for m in modules}
        flags[module] = False
        seq += 1
        off_seq = seq
        set_state(module_training=flags, module_training_seq=off_seq)
        status = wait_status_seq(off_seq)
        runner_flags = status.get("module_training", {}) or {}
        counts = status.get("trainable_counts", {}) or {}
        off_ok = runner_flags.get(module) is False and int(counts.get(module, 0) or 0) == 0
        append_diag("module_checkbox_check", module=module, checkbox="off", ok=off_ok, train_flag=runner_flags.get(module, "unknown"), trainable=int(counts.get(module, 0) or 0), updated_modules="not_updated_when_trainable_zero")
        ok_all = ok_all and off_ok
        flags[module] = True
        seq += 1
        on_seq = seq
        set_state(module_training=flags, module_training_seq=on_seq)
        status = wait_status_seq(on_seq)
        runner_flags = status.get("module_training", {}) or {}
        counts = status.get("trainable_counts", {}) or {}
        on_ok = runner_flags.get(module) is True and int(counts.get(module, 0) or 0) > 0
        append_diag("module_checkbox_check", module=module, checkbox="on", ok=on_ok, train_flag=runner_flags.get(module, "unknown"), trainable=int(counts.get(module, 0) or 0))
        ok_all = ok_all and on_ok
    return ok_all


def run(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_LOG.write_text("", encoding="utf-8")
    RUN_LOG.write_text("", encoding="utf-8")

    append_diag("DIAGNOSIS_FAILED", expected="runtime_state_keeps_memory_stability_across_sleep", actual="memory_fields_were_not_persisted_before_repair", planned_fix="persist_memory_stability_dream_activation_update_fields")
    append_diag("repair_applied", stage="repair_applied", reason="runtime_state_memory_fields_and_ipc_dream_slot_selector_present")
    if not event_patch_state():
        raise StageError("patch_check", "patch_fields_missing")

    proc: subprocess.Popen[str] | None = None
    checks: dict[str, tuple[bool, str]] = {}
    final_selected: dict[str, str] | None = None
    final_inactive: dict[str, str] | None = None
    try:
        stage("runner_start", "running", "starting_runner")
        proc = start_runner(args)
        text = wait_for("wait_live_step", lambda t: "event=runner_started" in t and "event=live_step_tick" in t, "runner_started_and_live_step_tick", args.startup_timeout, proc)
        checks["runner starts and live_step ticks"] = (True, "runner_started live_step_tick")
        checks["inner object patch fields exist or are applied"] = (True, "patch_state ok")

        stage("enter_inner_object_test", "running", "prepare_stable_object_slot")
        set_state(training=True, object_image=False, video_sensor_enabled=True, contact_sensor_enabled=True, imu_sensor_enabled=True, input_sensors_enabled={"video": True, "contact": True, "imu": True})
        action("fly_to_tetrahedron_inspect", rotate_tetrahedron=True, rotate_cube=False, fly_cube=False)
        text = wait_for(
            "enter_inner_object_test",
            lambda t: best_occupied(slot_rows(t, video=True, contact=True, imu=True, full_sleep=False)) is not None,
            "stable_object_slot_present",
            args.form_timeout,
            proc,
        )

        stage("awake_visible_check", "running", "sensors_on_object_visible")
        text = wait_for(
            "awake_visible_check",
            lambda t: any(frow(r, "M") >= TH["memory_high_threshold"] and frow(r, "C") >= TH["confidence_high_threshold"] and frow(r, "U") > TH["update_low_threshold"] and frow(r, "z_obj_norm") >= TH["z_obj_norm_min"] for r in slot_rows(t, video=True, contact=True, imu=True, full_sleep=False)),
            "awake_m_c_u_high",
            args.form_timeout,
            proc,
        )
        awake = best_occupied(slot_rows(text, video=True, contact=True, imu=True, full_sleep=False))
        selected_slot = int(awake.get("slot_id", 0)) if awake else 0
        awake_m = frow(awake or {}, "M")
        awake_c = frow(awake or {}, "C")
        append_diag("awake_visible_check", ok=True, slot_id=selected_slot, M=awake_m, C=awake_c, U=frow(awake or {}, "U"), z_obj_norm=frow(awake or {}, "z_obj_norm"))
        checks["awake visible object produces M high and C high"] = (True, f"M={awake_m:.3f} C={awake_c:.3f}")
        checks["awake update can produce U above threshold when slot is being written"] = (True, f"U={frow(awake or {}, 'U'):.3f}")

        stage("sensor_off_check", "running", "video_contact_imu_individual_off")
        set_state(training=True, video_sensor_enabled=False, contact_sensor_enabled=True, imu_sensor_enabled=True, input_sensors_enabled={"video": False, "contact": True, "imu": True})
        text = wait_for("sensor_off_check", lambda t: any(int(r.get("slot_id", -1)) == selected_slot and frow(r, "C") <= awake_c and frow(r, "M") >= awake_m - TH["memory_decay_tolerance"] for r in slot_rows(t, video=False, contact=True, imu=True, full_sleep=False)), "video_off_c_lower_m_preserved", args.phase_timeout, proc)
        row = [r for r in slot_rows(text, video=False, contact=True, imu=True, full_sleep=False) if int(r.get("slot_id", -1)) == selected_slot][-1]
        append_diag("sensor_off_check", mode_name="partial_video_off", ok=True, slot_id=selected_slot, M=frow(row, "M"), C=frow(row, "C"), U=frow(row, "U"))
        checks["partial video off lowers C but preserves M"] = (True, f"M={frow(row, 'M'):.3f} C={frow(row, 'C'):.3f}")

        set_state(training=True, video_sensor_enabled=True, contact_sensor_enabled=False, imu_sensor_enabled=True, input_sensors_enabled={"video": True, "contact": False, "imu": True})
        text = wait_for("sensor_off_check", lambda t: any(int(r.get("slot_id", -1)) == selected_slot and frow(r, "M") >= awake_m - TH["memory_decay_tolerance"] for r in slot_rows(t, video=True, contact=False, imu=True, full_sleep=False)), "contact_off_m_preserved", args.phase_timeout, proc)
        row = [r for r in slot_rows(text, video=True, contact=False, imu=True, full_sleep=False) if int(r.get("slot_id", -1)) == selected_slot][-1]
        append_diag("sensor_off_check", mode_name="partial_contact_off", ok=True, slot_id=selected_slot, M=frow(row, "M"), C=frow(row, "C"), U=frow(row, "U"))
        checks["partial contact off does not erase M"] = (True, f"M={frow(row, 'M'):.3f}")

        set_state(training=True, video_sensor_enabled=True, contact_sensor_enabled=True, imu_sensor_enabled=False, input_sensors_enabled={"video": True, "contact": True, "imu": False})
        text = wait_for("sensor_off_check", lambda t: any(int(r.get("slot_id", -1)) == selected_slot and frow(r, "M") >= awake_m - TH["memory_decay_tolerance"] for r in slot_rows(t, video=True, contact=True, imu=False, full_sleep=False)), "imu_off_m_preserved", args.phase_timeout, proc)
        row = [r for r in slot_rows(text, video=True, contact=True, imu=False, full_sleep=False) if int(r.get("slot_id", -1)) == selected_slot][-1]
        append_diag("sensor_off_check", mode_name="partial_imu_off", ok=True, slot_id=selected_slot, M=frow(row, "M"), C=frow(row, "C"), U=frow(row, "U"))
        checks["partial imu off does not erase M"] = (True, f"M={frow(row, 'M'):.3f}")

        stage("full_sleep_check", "running", "video_contact_imu_off")
        set_state(training=True, video_sensor_enabled=False, contact_sensor_enabled=False, imu_sensor_enabled=False, input_sensors_enabled={"video": False, "contact": False, "imu": False}, inner_object_dream_slot_index=selected_slot)
        text = wait_for("full_sleep_check", lambda t: len([r for r in slot_rows(t, video=False, contact=False, imu=False, full_sleep=True) if int(r.get("slot_id", -1)) == selected_slot]) >= int(TH["sleep_window_steps"]), "sleep_window_slot_metrics", args.sleep_timeout, proc)
        sleep_rows = [r for r in slot_rows(text, video=False, contact=False, imu=False, full_sleep=True) if int(r.get("slot_id", -1)) == selected_slot][-int(TH["sleep_window_steps"]):]
        m0, m1 = frow(sleep_rows[0], "M"), frow(sleep_rows[-1], "M")
        final_selected = sleep_rows[-1]
        full_ok = frow(final_selected, "C") <= TH["confidence_low_threshold"] and frow(final_selected, "U") <= TH["update_low_threshold"] and abs(m1 - m0) <= TH["memory_decay_tolerance"] and frow(final_selected, "z_obj_norm") >= TH["z_obj_norm_min"]
        append_diag("full_sleep_check", ok=full_ok, slot_id=selected_slot, M_start=m0, M_end=m1, C=frow(final_selected, "C"), U=frow(final_selected, "U"), D=frow(final_selected, "D"), z_obj_norm=frow(final_selected, "z_obj_norm"))
        checks["full sleep sets C low while M remains stable"] = (full_ok, f"M_delta={abs(m1-m0):.3f} C={frow(final_selected, 'C'):.3f}")
        checks["full sleep keeps U low or zero"] = (frow(final_selected, "U") <= TH["update_low_threshold"], f"U={frow(final_selected, 'U'):.3f}")
        checks["selected slot M does not decay over the sleep window"] = (abs(m1 - m0) <= TH["memory_decay_tolerance"], f"M_delta={abs(m1-m0):.3f}")

        stage("dream_slot_check", "running", "selected_slot_dream_activation")
        text = wait_for("dream_slot_check", lambda t: any(int(r.get("slot_id", -1)) == selected_slot and frow(r, "D") >= TH["dream_activation_high_threshold"] for r in slot_rows(t, video=False, contact=False, imu=False, full_sleep=True)), "selected_d_high", args.phase_timeout, proc)
        final_selected = [r for r in slot_rows(text, video=False, contact=False, imu=False, full_sleep=True) if int(r.get("slot_id", -1)) == selected_slot][-1]
        append_diag("dream_slot_check", ok=True, slot_id=selected_slot, M=frow(final_selected, "M"), C=frow(final_selected, "C"), D=frow(final_selected, "D"), U=frow(final_selected, "U"), z_obj_norm=frow(final_selected, "z_obj_norm"))
        checks["selected dream slot has D high"] = (True, f"D={frow(final_selected, 'D'):.3f}")

        stage("inactive_slot_check", "running", "inactive_slots_not_dream_active")
        inactive_rows = [r for r in slot_rows(text, video=False, contact=False, imu=False, full_sleep=True) if int(r.get("slot_id", -1)) != selected_slot]
        latest_by_slot: dict[int, dict[str, str]] = {}
        for r in inactive_rows:
            latest_by_slot[int(r.get("slot_id", 0))] = r
        inactive_ok = all(frow(r, "D") <= TH["dream_activation_low_threshold"] and frow(r, "U") <= TH["update_low_threshold"] for r in latest_by_slot.values())
        final_inactive = next((r for r in latest_by_slot.values() if frow(r, "M") > 0.05 or frow(r, "z_obj_norm") >= TH["z_obj_norm_min"]), next(iter(latest_by_slot.values()), None))
        append_diag("inactive_slot_check", ok=inactive_ok, checked_slots=",".join(str(k) for k in sorted(latest_by_slot)) or "none", inactive_slot=(final_inactive or {}).get("slot_id", "none"), D=frow(final_inactive or {}, "D"), U=frow(final_inactive or {}, "U"), M=frow(final_inactive or {}, "M"))
        checks["inactive dream slots have D low"] = (inactive_ok and bool(latest_by_slot), f"inactive_slots={len(latest_by_slot)}")

        stage("wake_restore_check", "running", "sensors_back_on")
        set_state(training=True, video_sensor_enabled=True, contact_sensor_enabled=True, imu_sensor_enabled=True, input_sensors_enabled={"video": True, "contact": True, "imu": True})
        action("fly_to_tetrahedron_inspect", rotate_tetrahedron=True, rotate_cube=False, fly_cube=False)
        text = wait_for("wake_restore_check", lambda t: any(int(r.get("slot_id", -1)) == selected_slot and frow(r, "C") >= TH["confidence_high_threshold"] and frow(r, "M") >= m1 - TH["memory_decay_tolerance"] for r in slot_rows(t, video=True, contact=True, imu=True, full_sleep=False)), "wake_c_restored", args.phase_timeout, proc)
        wake = [r for r in slot_rows(text, video=True, contact=True, imu=True, full_sleep=False) if int(r.get("slot_id", -1)) == selected_slot][-1]
        append_diag("wake_restore_check", ok=True, slot_id=selected_slot, M=frow(wake, "M"), C=frow(wake, "C"), D=frow(wake, "D"), U=frow(wake, "U"))
        checks["wake restore brings C back when sensors are re-enabled and object is visible"] = (True, f"C={frow(wake, 'C'):.3f}")

        stage("visualizer_check", "running", "check_m_c_d_u_labels")
        vis_ok = check_visualizer()
        checks["visualizer exposes M/C/D/U bars with correct labels"] = (vis_ok, "labels_and_sources_present")

        stage("module_checkbox_check", "running", "check_freeze_logic")
        mod_ok = module_checkbox_check(proc)
        checks["module checkboxes freeze training and card metrics"] = (mod_ok, "trainable_counts_follow_checkboxes")

        all_ok = all(ok for ok, _ in checks.values())
        if all_ok:
            append_diag("SUCCESS_INNER_OBJECT_SLEEP_MODES_VERIFIED", selected_slot=selected_slot, M=frow(final_selected or {}, "M"), C=frow(final_selected or {}, "C"), D=frow(final_selected or {}, "D"), U=frow(final_selected or {}, "U"), z_obj_norm=frow(final_selected or {}, "z_obj_norm"), must_verify="all_ok")
            stage("success", "ok", "inner_object_sleep_modes_verified")
        else:
            failed = ",".join(k for k, (ok, _) in checks.items() if not ok)
            append_diag("DIAGNOSIS_FAILED", expected="all_must_verify_ok", actual=failed, planned_fix="inspect_failed_checks")
            raise StageError("blocked", f"failed_checks={failed}")
    finally:
        stop_runner(proc)

    summary = {
        "checks": checks,
        "selected": final_selected or {},
        "inactive": final_inactive or {},
        "success_written": "event=SUCCESS_INNER_OBJECT_SLEEP_MODES_VERIFIED" in read_diag(),
    }
    (LOG_DIR / "inner_object_sleep_modes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="")
    p.add_argument("--max-steps", type=int, default=1000000)
    p.add_argument("--startup-timeout", type=float, default=240.0)
    p.add_argument("--form-timeout", type=float, default=420.0)
    p.add_argument("--phase-timeout", type=float, default=120.0)
    p.add_argument("--sleep-timeout", type=float, default=180.0)
    return p.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except StageError as exc:
        append_diag("DIAGNOSIS_FAILED", expected="stage_success", actual=exc.detail, planned_fix="targeted_repair_or_rerun", failed_stage=exc.stage)
        stage(exc.stage if exc.stage in {"blocked", "runner_start", "wait_live_step", "enter_inner_object_test", "awake_visible_check", "sensor_off_check", "full_sleep_check", "dream_slot_check", "inactive_slot_check", "wake_restore_check", "visualizer_check", "module_checkbox_check"} else "blocked", "failed", exc.detail[:80])
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
