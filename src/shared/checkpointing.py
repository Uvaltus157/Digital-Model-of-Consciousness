from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import threading
import time

import torch


class CheckpointingMixin:
    def _checkpoint_load_enabled(self) -> bool:
        ckpt_cfg = self.cfg.checkpoint_load
        return bool(getattr(ckpt_cfg, "enabled_load", getattr(ckpt_cfg, "enabled", True)))

    def _checkpoint_save_enabled(self) -> bool:
        ckpt_cfg = self.cfg.checkpoint_load
        return bool(getattr(ckpt_cfg, "enabled_save", getattr(ckpt_cfg, "enabled", True)))

    def _get_checkpoint_save_lock(self):
        """
        One in-process lock for checkpoint writes.

        Only life_step() should own periodic saving, but this lock prevents file
        corruption if some old code path/manual command accidentally calls
        save_checkpoint() at the same time.
        """
        lock = getattr(self, "_checkpoint_save_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._checkpoint_save_lock = lock
        return lock

    def checkpoint_path_for_load(self):
        ckpt_cfg = self.cfg.checkpoint_load
        load_path = getattr(ckpt_cfg, "load_path", "")
        if load_path:
            return Path(load_path)
        return self.out_dir / "last.pt"

    def checkpoint_path_for_save(self):
        """
        Save path for runtime checkpoint.

        Priority:
            cfg.checkpoint_load.save_path  -> where new last.pt is written
            cfg.checkpoint_load.load_path  -> fallback when save_path is empty
            runtime.out_dir / "last.pt"    -> default fallback

        This lets loading and saving use different files when needed.
        """
        ckpt_cfg = self.cfg.checkpoint_load
        save_path = getattr(ckpt_cfg, "save_path", "")
        if save_path:
            return Path(save_path)
        load_path = getattr(ckpt_cfg, "load_path", "")
        if load_path:
            return Path(load_path)
        return self.out_dir / "last.pt"

    def _detach_to_cpu_tree(self, value: Any):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._detach_to_cpu_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._detach_to_cpu_tree(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._detach_to_cpu_tree(v) for v in value)
        return value

    def _move_tree_to_device(self, value: Any):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._move_tree_to_device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_tree_to_device(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_tree_to_device(v) for v in value)
        return value

    def build_checkpoint_payload(self) -> dict:
        """
        Full checkpoint: weights + runtime object-slot memory.

        A stable slot is runtime memory, not only model weights.
        The important fields are inside self.inner_object_state:
            z_obj_slots, confidence_slots, slot_age, active_slot_index, dream_tick
        """
        payload = {
            "version": "v5_10_inner_object_memory_checkpoint",
            "model": self.model.state_dict() if hasattr(self, "model") else {},
            "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "global_step": int(getattr(self, "global_step", 0)),
            "train_steps": int(getattr(self, "train_steps", 0)),
            "quality": float(self.quality.get()) if hasattr(self, "quality") else None,
        }

        if hasattr(self, "inner_object_system") and self.inner_object_system is not None:
            payload["inner_object_system"] = self.inner_object_system.state_dict()
        if hasattr(self, "inner_object_state"):
            payload["inner_object_state"] = self._detach_to_cpu_tree(self.inner_object_state)
        if hasattr(self, "inner_object_slot_snapshots"):
            payload["inner_object_slot_snapshots"] = self._detach_to_cpu_tree(self.inner_object_slot_snapshots)

        if hasattr(self, "event_latent_memory") and self.event_latent_memory is not None:
            try:
                payload["event_latent_memory"] = self._detach_to_cpu_tree(self.event_latent_memory.state_dict())
            except Exception as e:
                print(f"[checkpoint] event_latent_memory not saved: {e}")

        if hasattr(self, "dynamic_object_passports") and self.dynamic_object_passports is not None:
            try:
                payload["dynamic_object_passports"] = self._detach_to_cpu_tree(self.dynamic_object_passports.state_dict())
            except Exception as e:
                print(f"[checkpoint] dynamic_object_passports not saved: {e}")

        if hasattr(self, "neural_event_decoder") and self.neural_event_decoder is not None:
            try:
                payload["neural_event_decoder"] = self._detach_to_cpu_tree(self.neural_event_decoder.state_dict())
            except Exception as e:
                print(f"[checkpoint] neural_event_decoder not saved: {e}")

        if hasattr(self, "inner_scenario_mind") and self.inner_scenario_mind is not None:
            try:
                payload["inner_scenario_mind"] = self._detach_to_cpu_tree(self.inner_scenario_mind.state_dict())
            except Exception as e:
                print(f"[checkpoint] inner_scenario_mind not saved: {e}")

        if hasattr(self, "inner_action_decoder") and self.inner_action_decoder is not None:
            try:
                payload["inner_action_decoder"] = self._detach_to_cpu_tree(self.inner_action_decoder.state_dict())
            except Exception as e:
                print(f"[checkpoint] inner_action_decoder not saved: {e}")

        if hasattr(self, "inner_outcome_evaluator") and self.inner_outcome_evaluator is not None:
            try:
                payload["inner_outcome_evaluator"] = self._detach_to_cpu_tree(self.inner_outcome_evaluator.state_dict())
            except Exception as e:
                print(f"[checkpoint] inner_outcome_evaluator not saved: {e}")

        if hasattr(self, "inner_trust_gate") and self.inner_trust_gate is not None:
            try:
                payload["inner_trust_gate"] = self._detach_to_cpu_tree(self.inner_trust_gate.state_dict())
            except Exception as e:
                print(f"[checkpoint] inner_trust_gate not saved: {e}")

        if hasattr(self, "self_core") and self.self_core is not None:
            payload["self_core"] = self.self_core.state_dict()
        if hasattr(self, "self_core_state"):
            payload["self_core_state"] = self._detach_to_cpu_tree(self.self_core_state)
        if hasattr(self, "leg_control_head") and self.leg_control_head is not None:
            payload["leg_control_head"] = self.leg_control_head.state_dict()

        if hasattr(self, "state"):
            payload["runtime_state"] = self._detach_to_cpu_tree(self.state)
        if hasattr(self, "prev_embodied_action"):
            payload["prev_embodied_action"] = self._detach_to_cpu_tree(self.prev_embodied_action)
        if hasattr(self, "prev_hand_motor"):
            payload["prev_hand_motor"] = self._detach_to_cpu_tree(self.prev_hand_motor)
        if hasattr(self, "prev_leg_motor"):
            payload["prev_leg_motor"] = self._detach_to_cpu_tree(self.prev_leg_motor)

        return payload

    def save_checkpoint(self, path=None) -> bool:
        """
        Save checkpoint with model weights + object-slot memory.

        Safety:
            - periodic saving is owned by life_step(), not train_loop();
            - this method is still protected by a lock;
            - file write is atomic: write *.tmp first, then os.replace(tmp, final).

        Atomic replace prevents a half-written last.pt if the process is killed
        during torch.save().
        """
        lock = self._get_checkpoint_save_lock()
        with lock:
            tmp_path = None
            try:
                path = Path(path) if path is not None else self.checkpoint_path_for_save()
                path.parent.mkdir(parents=True, exist_ok=True)

                payload = self.build_checkpoint_payload()

                # Unique temp name avoids collision even if a stale temp exists.
                tmp_path = path.with_name(
                    f"{path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
                )

                torch.save(payload, tmp_path)
                os.replace(tmp_path, path)

                print(
                    f"[checkpoint] saved: {path} | "
                    f"global_step={payload.get('global_step', 0)} "
                    f"train_steps={payload.get('train_steps', 0)} "
                    f"inner_object_state={'yes' if 'inner_object_state' in payload else 'no'}"
                )
                return True

            except Exception as e:
                print(f"[checkpoint] save failed: {e}")
                try:
                    if tmp_path is not None and Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                except Exception:
                    pass
                return False


    def maybe_save_checkpoint(self, force: bool = False, owner: str = "life") -> bool:
        """
        Save periodically according to cfg.life.checkpoint_every_steps.

        Single-owner policy:
            - owner="life" is allowed.
            - owner="manual" is allowed only when force=True.
            - owner="train" is ignored.

        This prevents the parallel train thread from racing with life_step()
        while writing last.pt.
        """
        ckpt_cfg = self.cfg.checkpoint_load
        if not self._checkpoint_save_enabled():
            #print("[checkpoint] save disabled")
            return False
        
        try:
            owner = str(owner or "life").lower().strip()
            if owner == "train":
                return False
            if owner not in ("life", "manual", "external", "force"):
                if not force:
                    return False

            every = int(getattr(self.cfg.life, "checkpoint_every_steps", 0))
            if not force and every > 0:
                step = int(getattr(self, "global_step", 0))
                if step <= 0 or (step % every) != 0:
                    return False
            elif not force and every <= 0:
                return False

            return self.save_checkpoint()
        except Exception as e:
            print(f"[checkpoint] maybe_save failed: {e}")
            return False


    def _load_state_dict_safe(self, module, state, name: str, strict: bool = False):
        if module is None or state is None:
            return False
        try:
            result = module.load_state_dict(state, strict=strict)
            missing = getattr(result, "missing_keys", [])
            unexpected = getattr(result, "unexpected_keys", [])
            if missing:
                print(f"[checkpoint] missing {name} keys: {len(missing)}")
            if unexpected:
                print(f"[checkpoint] unexpected {name} keys: {len(unexpected)}")
            print(f"[checkpoint] {name} loaded")
            return True
        except Exception as e:
            print(f"[checkpoint] {name} not loaded: {e}")
            return False

    def maybe_load_checkpoint(self) -> bool:
        ckpt_cfg = self.cfg.checkpoint_load
        if not self._checkpoint_load_enabled():
            print("[checkpoint] autoload disabled")
            return False

        path = self.checkpoint_path_for_load()
        if not path.exists():
            print(f"[checkpoint] no checkpoint found: {path}")
            return False

        try:
            print(f"[checkpoint] loading: {path}")
            ckpt = torch.load(path, map_location=self.device)

            # Old checkpoints can be raw model state_dict. New checkpoints are dict payloads.
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            result = self.model.load_state_dict(state, strict=ckpt_cfg.strict)

            missing = getattr(result, "missing_keys", [])
            unexpected = getattr(result, "unexpected_keys", [])
            if missing:
                print(f"[checkpoint] missing model keys: {len(missing)}")
            if unexpected:
                print(f"[checkpoint] unexpected model keys: {len(unexpected)}")

            if isinstance(ckpt, dict):
                if "inner_object_system" in ckpt and hasattr(self, "inner_object_system"):
                    self._load_state_dict_safe(
                        self.inner_object_system,
                        ckpt.get("inner_object_system"),
                        "inner_object_system",
                        strict=False,
                    )
                else:
                    print("[checkpoint] inner_object_system not found in checkpoint")

                if "inner_object_state" in ckpt:
                    try:
                        loaded_state = self._move_tree_to_device(ckpt["inner_object_state"])
                        if isinstance(loaded_state, dict):
                            self.inner_object_state = loaded_state
                            slot_shape = None
                            if torch.is_tensor(self.inner_object_state.get("z_obj_slots")):
                                slot_shape = tuple(self.inner_object_state["z_obj_slots"].shape)
                            print(f"[checkpoint] inner_object_state loaded | z_obj_slots={slot_shape}")
                    except Exception as e:
                        print(f"[checkpoint] inner_object_state not loaded: {e}")
                else:
                    print("[checkpoint] inner_object_state not found in checkpoint")

                if "inner_object_slot_snapshots" in ckpt:
                    try:
                        self.inner_object_slot_snapshots = self._move_tree_to_device(ckpt["inner_object_slot_snapshots"])
                        print(f"[checkpoint] inner_object_slot_snapshots loaded: {len(self.inner_object_slot_snapshots)}")
                    except Exception as e:
                        print(f"[checkpoint] inner_object_slot_snapshots not loaded: {e}")

                if "event_latent_memory" in ckpt:
                    try:
                        if hasattr(self, "_ensure_event_latent_memory"):
                            self._ensure_event_latent_memory()
                        if hasattr(self, "event_latent_memory") and self.event_latent_memory is not None:
                            self.event_latent_memory.load_state_dict(self._move_tree_to_device(ckpt["event_latent_memory"]))
                            print(f"[checkpoint] event_latent_memory loaded: {len(self.event_latent_memory)} events")
                    except Exception as e:
                        print(f"[checkpoint] event_latent_memory not loaded: {e}")

                if "dynamic_object_passports" in ckpt:
                    try:
                        if hasattr(self, "_ensure_dynamic_object_passports"):
                            self._ensure_dynamic_object_passports()
                        if hasattr(self, "dynamic_object_passports") and self.dynamic_object_passports is not None:
                            self.dynamic_object_passports.load_state_dict(self._move_tree_to_device(ckpt["dynamic_object_passports"]))
                            print(f"[checkpoint] dynamic_object_passports loaded: {len(self.dynamic_object_passports)} passports")
                    except Exception as e:
                        print(f"[checkpoint] dynamic_object_passports not loaded: {e}")

                if "neural_event_decoder" in ckpt:
                    try:
                        if hasattr(self, "_ensure_neural_event_decoder"):
                            self._ensure_neural_event_decoder()
                        if hasattr(self, "neural_event_decoder") and self.neural_event_decoder is not None:
                            self._load_state_dict_safe(self.neural_event_decoder, ckpt.get("neural_event_decoder"), "neural_event_decoder", strict=False)
                    except Exception as e:
                        print(f"[checkpoint] neural_event_decoder not loaded: {e}")

                if "inner_scenario_mind" in ckpt:
                    try:
                        if hasattr(self, "_ensure_inner_scenario_mind"):
                            self._ensure_inner_scenario_mind()
                        if hasattr(self, "inner_scenario_mind") and self.inner_scenario_mind is not None:
                            self.inner_scenario_mind.load_state_dict(self._move_tree_to_device(ckpt["inner_scenario_mind"]))
                            print("[checkpoint] inner_scenario_mind loaded")
                    except Exception as e:
                        print(f"[checkpoint] inner_scenario_mind not loaded: {e}")

                if "inner_action_decoder" in ckpt:
                    try:
                        if hasattr(self, "_ensure_inner_action_decoder"):
                            self._ensure_inner_action_decoder()
                        if hasattr(self, "inner_action_decoder") and self.inner_action_decoder is not None:
                            self._load_state_dict_safe(self.inner_action_decoder, ckpt.get("inner_action_decoder"), "inner_action_decoder", strict=False)
                    except Exception as e:
                        print(f"[checkpoint] inner_action_decoder not loaded: {e}")

                if "inner_outcome_evaluator" in ckpt:
                    try:
                        if hasattr(self, "_ensure_inner_outcome_evaluator"):
                            self._ensure_inner_outcome_evaluator()
                        if hasattr(self, "inner_outcome_evaluator") and self.inner_outcome_evaluator is not None:
                            self.inner_outcome_evaluator.load_state_dict(self._move_tree_to_device(ckpt["inner_outcome_evaluator"]))
                            print("[checkpoint] inner_outcome_evaluator loaded")
                    except Exception as e:
                        print(f"[checkpoint] inner_outcome_evaluator not loaded: {e}")

                if "inner_trust_gate" in ckpt:
                    try:
                        if hasattr(self, "_ensure_inner_trust_gate"):
                            self._ensure_inner_trust_gate()
                        if hasattr(self, "inner_trust_gate") and self.inner_trust_gate is not None:
                            self.inner_trust_gate.load_state_dict(self._move_tree_to_device(ckpt["inner_trust_gate"]))
                            print("[checkpoint] inner_trust_gate loaded")
                    except Exception as e:
                        print(f"[checkpoint] inner_trust_gate not loaded: {e}")

                if "self_core" in ckpt and hasattr(self, "self_core"):
                    self._load_state_dict_safe(self.self_core, ckpt.get("self_core"), "self_core", strict=False)
                if "self_core_state" in ckpt:
                    try:
                        self.self_core_state = self._move_tree_to_device(ckpt["self_core_state"])
                        print("[checkpoint] self_core_state loaded")
                    except Exception as e:
                        print(f"[checkpoint] self_core_state not loaded: {e}")

                if "leg_control_head" in ckpt and hasattr(self, "leg_control_head"):
                    self._load_state_dict_safe(self.leg_control_head, ckpt.get("leg_control_head"), "leg_control_head", strict=False)

                if "runtime_state" in ckpt:
                    try:
                        self.state = self._move_tree_to_device(ckpt["runtime_state"])
                        print("[checkpoint] runtime_state loaded")
                    except Exception as e:
                        print(f"[checkpoint] runtime_state not loaded: {e}")

                for key in ("prev_embodied_action", "prev_hand_motor", "prev_leg_motor"):
                    if key in ckpt:
                        try:
                            setattr(self, key, self._move_tree_to_device(ckpt[key]))
                            print(f"[checkpoint] {key} loaded")
                        except Exception as e:
                            print(f"[checkpoint] {key} not loaded: {e}")

            if ckpt_cfg.load_optimizer and isinstance(ckpt, dict) and "optimizer" in ckpt and ckpt["optimizer"] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    print("[checkpoint] optimizer loaded")
                except Exception as e:
                    print(f"[checkpoint] optimizer not loaded: {e}")

            if ckpt_cfg.load_counters and isinstance(ckpt, dict):
                self.global_step = int(ckpt.get("global_step", self.global_step))
                self.train_steps = int(ckpt.get("train_steps", self.train_steps))
                q = ckpt.get("quality", None)
                if q is not None and hasattr(self.quality, "value"):
                    try:
                        self.quality.value = float(q)
                    except Exception:
                        pass

            print(f"[checkpoint] loaded OK | global_step={self.global_step} train_steps={self.train_steps}")
            return True

        except Exception as e:
            print(f"[checkpoint] load failed: {e}")
            return False
