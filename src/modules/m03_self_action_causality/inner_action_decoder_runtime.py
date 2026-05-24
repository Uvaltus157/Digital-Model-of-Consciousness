from __future__ import annotations

import torch

from src.modules.m03_self_action_causality.models.inner_action_decoder import InnerActionDecoder, InnerActionDecoderConfig


class InnerActionDecoderRuntimeMixin:
    """
    Runtime glue for decoding selected internal scenario_z into an action intention.

    Safe policy:
        - by default, this does not override the main model action;
        - it adds intention fields into inner_object/debug state;
        - optional blending can be enabled later with config.
    """

    def _ensure_inner_action_decoder(self) -> None:
        if hasattr(self, "inner_action_decoder") and self.inner_action_decoder is not None:
            return

        cfg_act = getattr(self.cfg, "inner_action_decoder", None)
        self.inner_action_decoder = InnerActionDecoder(InnerActionDecoderConfig(
            enabled=bool(getattr(cfg_act, "enabled", True)),
            latent_dim=int(getattr(self.cfg.object_image, "latent_dim", 128)),
            embodied_dim=int(getattr(self.cfg, "embodied_dim", 15)),
            hand_dim=int(getattr(self.cfg, "hand_motor_dim", 34)),
            hidden_dim=int(getattr(cfg_act, "hidden_dim", 256)),
            max_intent_norm=float(getattr(cfg_act, "max_intent_norm", 0.25)),
            confidence_threshold=float(getattr(cfg_act, "confidence_threshold", 0.10)),
            blend_to_policy=bool(getattr(cfg_act, "blend_to_policy", False)),
            blend_alpha=float(getattr(cfg_act, "blend_alpha", 0.10)),
        )).to(self.device)

        # Register trainable params if optimizer already exists.
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                existing = set()
                for group in self.optimizer.param_groups:
                    for p in group.get("params", []):
                        existing.add(id(p))
                params = [p for p in self.inner_action_decoder.parameters() if id(p) not in existing]
                if params:
                    self.optimizer.add_param_group({"params": params})
                    print(f"[inner_action_decoder] added params to optimizer: {sum(p.numel() for p in params):,}")
        except Exception as e:
            if not hasattr(self, "_inner_action_decoder_opt_warned"):
                print(f"[inner_action_decoder] optimizer add skipped: {e}")
                self._inner_action_decoder_opt_warned = True

    def update_inner_action_decoder(self, obj: dict, out: dict | None = None) -> dict:
        """
        Decode selected inner_mind_z into intention vectors.
        """
        try:
            cfg_act = getattr(self.cfg, "inner_action_decoder", None)
            if not bool(getattr(cfg_act, "enabled", True)):
                return obj

            if not isinstance(obj, dict) or not obj.get("inner_mind_active", False):
                return obj

            self._ensure_inner_action_decoder()

            ref = obj.get("z_obj")
            device = ref.device if torch.is_tensor(ref) else getattr(self, "device", "cpu")
            dtype = ref.dtype if torch.is_tensor(ref) else torch.float32

            intent = self.inner_action_decoder.decode_intention(obj, device=device, dtype=dtype)
            if isinstance(intent, dict) and intent:
                obj.update(intent)

            # Optional safe blending into model output; off by default.
            if out is not None and bool(getattr(cfg_act, "blend_to_policy", False)):
                conf = intent.get("inner_action_confidence")
                if torch.is_tensor(conf):
                    c = float(conf.detach().reshape(-1)[0].cpu().item())
                else:
                    c = 0.0

                if c >= float(getattr(cfg_act, "confidence_threshold", 0.10)):
                    alpha = float(getattr(cfg_act, "blend_alpha", 0.10))
                    body = intent.get("inner_action_body")
                    hand = intent.get("inner_action_hand")

                    if torch.is_tensor(body) and "embodied_targets" in out and torch.is_tensor(out["embodied_targets"]):
                        out["embodied_targets"] = (1.0 - alpha) * out["embodied_targets"] + alpha * body.to(out["embodied_targets"].device, out["embodied_targets"].dtype)
                    if torch.is_tensor(hand) and "hand_ctrl" in out and torch.is_tensor(out["hand_ctrl"]):
                        out["hand_ctrl"] = (1.0 - alpha) * out["hand_ctrl"] + alpha * hand.to(out["hand_ctrl"].device, out["hand_ctrl"].dtype)

                    obj["inner_action_blended_to_policy"] = torch.tensor([[1.0]], device=device, dtype=dtype)
                else:
                    obj["inner_action_blended_to_policy"] = torch.tensor([[0.0]], device=device, dtype=dtype)

            return obj

        except Exception as e:
            if not hasattr(self, "_inner_action_decoder_warned"):
                print(f"[inner_action_decoder] update failed: {e}")
                self._inner_action_decoder_warned = True
            return obj
