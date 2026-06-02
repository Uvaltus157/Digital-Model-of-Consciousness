from __future__ import annotations

"""Life-loop helper for the V5.10 unified system.

This module extracts the `UnifiedSystem.run()` loop from the runner
without importing MuJoCo/OpenCV at module import time. The slim entrypoint
patches the heavy runtime class to use this function.
"""

import time
from threading import Thread
from typing import Any

from src.apps.runner_thread_affinity import apply_thread_affinity


def run_unified_life_loop(system: Any) -> None:
    """Run the V5.10 life loop for an already constructed system.

    This is a behavior-preserving extraction of the previous
    `UnifiedSystem.run()` implementation. Heavy viewer imports are kept
    inside the function so smoke tests can import this module without a display
    or MuJoCo runtime.
    """
    from src.platform.gui.opencv_gui_thread import shutdown_cv2_gui_thread
    from src.platform.mujoco_world.mujoco_viewer_thread import MujocoViewerThread

    system.world.reset()
    system.log_tetra_runner_started()

    train_thread = Thread(target=system.train_loop, name="RunnerTrainLoop", daemon=True)
    train_thread.start()
    apply_thread_affinity(system.cfg, "train", train_thread, label="train loop")

    period = 1.0 / max(system.cfg.life.fps, 1e-6)

    viewer = None
    last_viewer_sync_time = 0.0
    threaded_viewer = None
    if bool(getattr(system.cfg.viewer, "mujoco_threaded", False)):
        threaded_viewer = MujocoViewerThread(
            system.world.model,
            sync_fps=float(getattr(system.cfg.viewer, "mujoco_sync_fps", 8.0)),
            show_left_ui=False,
            show_right_ui=False,
        )
        threaded_viewer.start()
        apply_thread_affinity(
            system.cfg,
            "mujoco_viewer",
            getattr(threaded_viewer, "_thread", None),
            label="MuJoCo viewer",
        )

    try:
        while not system.shutdown and system.global_step < system.cfg.life.max_steps:
            t0 = time.time()
            system.life_step()
            system.tick_slot_4d_jsonrpc_streamer()

            if threaded_viewer is not None:
                threaded_viewer.set_enabled(bool(system.cfg.viewer.allow_mujoco_window))
                sync_every_steps = max(1, int(getattr(system.cfg.viewer, "mujoco_sync_every_steps", 1)))
                if system.cfg.viewer.allow_mujoco_window and (system.global_step % sync_every_steps) == 0:
                    threaded_viewer.update_from(system.world.data)
            else:
                if system.cfg.viewer.allow_mujoco_window is True and viewer is None:
                    import mujoco.viewer

                    viewer = mujoco.viewer.launch_passive(
                        system.world.model,
                        system.world.data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                elif system.cfg.viewer.allow_mujoco_window is False and viewer is not None:
                    viewer.close()
                    viewer = None

                if viewer is not None:
                    if not viewer.is_running():
                        try:
                            viewer.close()
                        except Exception:
                            pass
                        viewer = None
                        system.cfg.viewer.allow_mujoco_window = False
                    else:
                        sync_fps = float(getattr(system.cfg.viewer, "mujoco_sync_fps", 8.0))
                        sync_every_steps = max(1, int(getattr(system.cfg.viewer, "mujoco_sync_every_steps", 1)))
                        sync_interval = 0.0 if sync_fps <= 0.0 else 1.0 / max(sync_fps, 1e-6)
                        now = time.time()
                        if (system.global_step % sync_every_steps) == 0 and (now - last_viewer_sync_time) >= sync_interval:
                            try:
                                viewer.sync()
                                last_viewer_sync_time = now
                            except Exception as e:
                                print(f"[mujoco.viewer] sync failed, closing viewer: {e}")
                                try:
                                    viewer.close()
                                except Exception:
                                    pass
                                viewer = None
                                system.cfg.viewer.allow_mujoco_window = False

            system.maybe_print_status()

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
    finally:
        system.shutdown = True
        train_thread.join(timeout=2.0)
        if threaded_viewer is not None:
            threaded_viewer.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        try:
            shutdown_cv2_gui_thread(timeout=3.0)
        except Exception as e:
            print(f"[opencv_gui_thread] shutdown skipped: {e}")
        system.shutdown_slot_4d_jsonrpc_streamer()
        system.maybe_save_checkpoint(force=True, owner="life")
        system.world.close()
