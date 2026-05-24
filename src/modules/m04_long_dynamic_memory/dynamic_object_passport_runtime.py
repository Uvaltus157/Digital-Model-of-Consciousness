from __future__ import annotations

import torch

from src.modules.m04_long_dynamic_memory.dynamic_object_passport import DynamicObjectPassportConfig, DynamicObjectPassportManager


class DynamicObjectPassportRuntimeMixin:
    """
    Runtime glue for DynamicObjectPassport.

    It adds the semantic identity layer above ObjectSlotMemory:
        SLOT_N is storage;
        OBJ_NNN / passport is identity over time.

    It can also reproduce the selected passport into the first-order inner world
    and optionally decode it to second-order human-visible RGB/depth/3D.
    """

    def _ensure_dynamic_object_passports(self) -> None:
        if hasattr(self, "dynamic_object_passports") and self.dynamic_object_passports is not None:
            return

        cfg = getattr(self.cfg, "dynamic_object_passport", None)
        self.dynamic_object_passports = DynamicObjectPassportManager(DynamicObjectPassportConfig(
            enabled=bool(getattr(cfg, "enabled", True)),
            max_passports=int(getattr(cfg, "max_passports", 32)),
            latent_dim=int(getattr(self.cfg.object_image, "latent_dim", 128)),
            token_prefix=str(getattr(cfg, "token_prefix", "OBJ")),
            similarity_threshold=float(getattr(cfg, "similarity_threshold", 0.72)),
            confidence_ema_decay=float(getattr(cfg, "confidence_ema_decay", 0.94)),
            signature_ema_decay=float(getattr(cfg, "signature_ema_decay", 0.97)),
            min_dynamic_score=float(getattr(cfg, "min_dynamic_score", 0.010)),
            min_confidence_to_create=float(getattr(cfg, "min_confidence_to_create", 0.02)),
            create_scene_passport=bool(getattr(cfg, "create_scene_passport", True)),
            replay_enabled=bool(getattr(cfg, "replay_enabled", True)),
            replay_in_sleep=bool(getattr(cfg, "replay_in_sleep", True)),
            decode_to_second_order=bool(getattr(cfg, "decode_to_second_order", True)),
        ))

    def update_dynamic_object_passport(self, obj: dict, obs: dict | None = None, out: dict | None = None, dream_mode: bool = False) -> dict:
        """
        Observe current z_obj/event and update or create a dynamic passport.

        The passport decides whether this is:
            - continuation of existing OBJ_NNN
            - or a newly formed dynamic object identity.
        """
        try:
            cfg = getattr(self.cfg, "dynamic_object_passport", None)
            if not bool(getattr(cfg, "enabled", True)):
                return obj
            if not isinstance(obj, dict):
                return obj

            self._ensure_dynamic_object_passports()

            # Make global_step visible to manager without binding manager to runtime.
            obj["global_step"] = int(getattr(self, "global_step", 0))

            info = self.dynamic_object_passports.observe(
                obj,
                event_memory=getattr(self, "event_latent_memory", None),
                dream_mode=bool(dream_mode),
                global_step=int(getattr(self, "global_step", 0)),
            )

            if isinstance(info, dict) and info:
                ref = obj.get("z_obj")
                device = ref.device if torch.is_tensor(ref) else getattr(self, "device", "cpu")
                dtype = ref.dtype if torch.is_tensor(ref) else torch.float32

                for k, v in info.items():
                    if isinstance(v, (float, int, bool)):
                        obj[k] = torch.tensor([[float(v)]], device=device, dtype=dtype)
                    else:
                        obj[k] = v

            return obj
        except Exception as e:
            if not hasattr(self, "_dynamic_object_passport_warned"):
                print(f"[dynamic_object_passport] update failed: {e}")
                self._dynamic_object_passport_warned = True
            return obj

    def reproduce_dynamic_object_from_passport(self, obj: dict, dream_mode: bool = False) -> dict:
        """
        Reproduce object from DynamicObjectPassport.

        First-order:
            passport -> passport_inner_world_z

        Second-order:
            passport_inner_world_z -> ObjectImaginationHead2D/Object3DHead
        """
        try:
            cfg = getattr(self.cfg, "dynamic_object_passport", None)
            if not bool(getattr(cfg, "enabled", True)):
                return obj
            if not bool(getattr(cfg, "replay_enabled", True)):
                return obj
            if dream_mode and not bool(getattr(cfg, "replay_in_sleep", True)):
                return obj
            if not isinstance(obj, dict):
                return obj

            self._ensure_dynamic_object_passports()

            ref = obj.get("z_obj")
            if not torch.is_tensor(ref):
                return obj

            token = str(obj.get("passport_token", "") or obj.get("slot_token", "") or "")
            rep = self.dynamic_object_passports.reproduce_inner_world(
                token=token,
                device=ref.device,
                dtype=ref.dtype,
            )
            if not rep:
                return obj

            z = rep.get("passport_inner_world_z")
            if not torch.is_tensor(z):
                return obj

            # Export first-order coded-world reproduction fields.
            for k, v in rep.items():
                if isinstance(v, (float, int, bool)):
                    obj[k] = torch.tensor([[float(v)]], device=ref.device, dtype=ref.dtype)
                else:
                    obj[k] = v

            # Only in sleep/dream or explicit config do we replace visible decode
            # with the passport replay. In active mode, keep live sensory object
            # while still exposing passport z.
            decode_to_second = bool(getattr(cfg, "decode_to_second_order", True))
            should_replace_decode = decode_to_second and (bool(dream_mode) or bool(getattr(cfg, "decode_also_when_active", False)))
            if not should_replace_decode:
                return obj

            extra = {k: v for k, v in obj.items() if k != "z_obj"}
            decoded = self.inner_object_system.decode_z(z, extra)

            # Preserve runtime semantic fields and replace decoded visual fields.
            for k, v in obj.items():
                if k not in decoded:
                    decoded[k] = v

            decoded["z_obj"] = z
            decoded["passport_second_order_decoded"] = torch.tensor([[1.0]], device=z.device, dtype=z.dtype)
            decoded["passport_replay_active"] = torch.tensor([[1.0]], device=z.device, dtype=z.dtype)
            return decoded

        except Exception as e:
            if not hasattr(self, "_dynamic_object_passport_replay_warned"):
                print(f"[dynamic_object_passport] replay failed: {e}")
                self._dynamic_object_passport_replay_warned = True
            return obj
