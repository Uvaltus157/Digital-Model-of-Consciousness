from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectInnerImagery3DConfig:
    enabled: bool = True
    latent_dim: int = 128
    hidden_dim: int = 256
    image_size: int = 64
    vision_dim: int = 12
    tactile_dim: int = 50
    body_dim: int = 12
    hand_dim: int = 34
    leg_dim: int = 18
    point_count: int = 128
    voxel_res: int = 16
    slot_decay: float = 0.92
    confidence_decay: float = 0.94

    # Multi-slot object memory. Public z_obj/confidence stay compatible and
    # represent the currently active slot.
    num_slots: int = 10
    slot_binding_temperature: float = 0.20
    empty_slot_bonus: float = 0.35
    slot_activity_threshold: float = 0.05

    # Sleep / dream mode. When eyes are closed, keep slots frozen and
    # decode from the internal latent code instead of learning black frames.
    sleep_freeze_memory_update: bool = False  # deprecated: memory is live; dream_step handles full sleep

    # Slot routing.
    # Hard binding prevents the same z_update from being written into every slot.
    hard_slot_binding: bool = True

    # Sticky object identity routing.
    # A slot keeps owning the object while it remains reasonably similar.
    # Switching requires a clearly better candidate, not a tiny score fluctuation.
    same_slot_similarity_threshold: float = 0.55
    sticky_confidence_threshold: float = 0.12
    sticky_similarity_threshold: float = 0.30
    switch_margin: float = 0.18
    new_object_similarity_threshold: float = 0.22

    # Multi-object proposals.
    # If vision has shape [B, P, vision_dim], each proposal is routed sequentially.
    # With proposal_slot_lock=True, proposal p writes to slot p, giving stable
    # slot identity for cube/sphere/ground-style scenes.
    max_object_proposals: int = 10
    proposal_slot_lock: bool = True

    # Sleep / dream stream.
    # When all external sensors are disabled, do not just decode a static slot.
    # Run a tiny autonomous latent transition so the internal image keeps updating.
    dream_latent_dynamics: bool = True
    dream_strength: float = 0.025
    dream_cycle_slots: bool = False
    dream_slot_cycle_steps: int = 90
    dream_empty_confidence_threshold: float = 0.05


def pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def summarize_vision_tensors(left: torch.Tensor, right: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
    def stats(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.float().reshape(x.shape[0], -1)
        return torch.cat([
            x.mean(dim=-1, keepdim=True),
            x.std(dim=-1, keepdim=True),
            x.min(dim=-1, keepdim=True).values,
            x.max(dim=-1, keepdim=True).values,
        ], dim=-1)

    if depth is None:
        depth = torch.zeros_like(left[:, :1]) if left.ndim == 4 else torch.zeros_like(left[:1])
    return torch.cat([stats(left), stats(right), stats(depth)], dim=-1)


class VisualTactileObjectFusion(nn.Module):
    def __init__(self, cfg: ObjectInnerImagery3DConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.vision_dim + cfg.tactile_dim + cfg.body_dim + cfg.hand_dim + cfg.leg_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )
        self.vision_gate = nn.Sequential(nn.Linear(cfg.vision_dim, 1), nn.Sigmoid())
        self.touch_gate = nn.Sequential(nn.Linear(cfg.tactile_dim, 1), nn.Sigmoid())

    def forward(self, vision, tactile, body, hand, leg) -> Dict[str, torch.Tensor]:
        x = torch.cat([vision, tactile, body, hand, leg], dim=-1)
        z = self.net(x)
        vision_activity = vision.abs().mean(dim=-1, keepdim=True)
        touch_activity = tactile.abs().mean(dim=-1, keepdim=True)
        return {
            "z_update": z,
            "vision_strength": self.vision_gate(vision) * torch.clamp(vision_activity * 2.0, 0.0, 1.0),
            "touch_strength": self.touch_gate(tactile) * torch.clamp(touch_activity * 2.0, 0.0, 1.0),
            "touch_activity": touch_activity,
            "vision_activity": vision_activity,
        }


class ObjectSlotMemory(nn.Module):
    """Multi-slot latent object memory with backward-compatible z_obj output."""

    def __init__(self, cfg: ObjectInnerImagery3DConfig):
        super().__init__()
        self.cfg = cfg
        self.update_gate = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2 + 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.Sigmoid(),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(cfg.latent_dim + 2, cfg.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    @property
    def num_slots(self) -> int:
        return max(1, int(getattr(self.cfg, "num_slots", 1)))

    def initial_state(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        s = self.num_slots
        d = self.cfg.latent_dim
        z_slots = torch.zeros(batch_size, s, d, device=device)
        c_slots = torch.zeros(batch_size, s, 1, device=device)
        age = torch.zeros(batch_size, s, 1, device=device)
        active_idx = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        return {
            "z_obj": z_slots[:, 0],
            "confidence": c_slots[:, 0],
            "z_obj_slots": z_slots,
            "confidence_slots": c_slots,
            "memory_stability_slots": c_slots.clone(),
            "memory_stability": c_slots[:, 0].clone(),
            "dream_activation_slots": c_slots.clone() * 0.0,
            "dream_activation": c_slots[:, 0].clone() * 0.0,
            "slot_age": age,
            "active_slot_index": active_idx,
        }

    def _coerce_prev_slots(self, prev: Dict[str, torch.Tensor], z_update: torch.Tensor):
        b, d = z_update.shape
        s = self.num_slots
        device = z_update.device
        dtype = z_update.dtype
        prev = prev or {}

        z_slots = prev.get("z_obj_slots")
        c_slots = prev.get("confidence_slots")
        age = prev.get("slot_age")

        if z_slots is None:
            old_z = prev.get("z_obj")
            if old_z is None:
                old_z = torch.zeros(b, d, device=device, dtype=dtype)
            if old_z.ndim == 1:
                old_z = old_z.unsqueeze(0)
            z_slots = torch.zeros(b, s, d, device=device, dtype=dtype)
            z_slots[:, 0, :] = old_z[:, :d].to(device=device, dtype=dtype)
        else:
            z_slots = z_slots.to(device=device, dtype=dtype)
            if z_slots.ndim == 2:
                z_slots = z_slots.unsqueeze(1)
            if z_slots.shape[1] < s:
                pad = torch.zeros(b, s - z_slots.shape[1], z_slots.shape[-1], device=device, dtype=dtype)
                z_slots = torch.cat([z_slots, pad], dim=1)
            elif z_slots.shape[1] > s:
                z_slots = z_slots[:, :s, :]
            if z_slots.shape[-1] > d:
                z_slots = z_slots[..., :d]
            elif z_slots.shape[-1] < d:
                pad = torch.zeros(b, s, d - z_slots.shape[-1], device=device, dtype=dtype)
                z_slots = torch.cat([z_slots, pad], dim=-1)

        if c_slots is None:
            old_c = prev.get("confidence")
            if old_c is None:
                old_c = torch.zeros(b, 1, device=device, dtype=dtype)
            if old_c.ndim == 1:
                old_c = old_c.unsqueeze(-1)
            c_slots = torch.zeros(b, s, 1, device=device, dtype=dtype)
            c_slots[:, 0, :] = old_c[:, :1].to(device=device, dtype=dtype)
        else:
            c_slots = c_slots.to(device=device, dtype=dtype)
            if c_slots.ndim == 2:
                c_slots = c_slots.unsqueeze(1)
            if c_slots.shape[1] < s:
                pad = torch.zeros(b, s - c_slots.shape[1], 1, device=device, dtype=dtype)
                c_slots = torch.cat([c_slots[:, :, :1], pad], dim=1)
            elif c_slots.shape[1] > s:
                c_slots = c_slots[:, :s, :]

        if age is None:
            age = torch.zeros(b, s, 1, device=device, dtype=dtype)
        else:
            age = age.to(device=device, dtype=dtype)
            if age.ndim == 2:
                age = age.unsqueeze(-1)
            if age.shape[1] < s:
                pad = torch.zeros(b, s - age.shape[1], 1, device=device, dtype=dtype)
                age = torch.cat([age[:, :, :1], pad], dim=1)
            elif age.shape[1] > s:
                age = age[:, :s, :]

        return z_slots, c_slots, age

    def _read_active_from_slots(
        self,
        z_slots: torch.Tensor,
        c_slots: torch.Tensor,
        age: torch.Tensor,
        explicit_active_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = z_slots.shape[0]
        if explicit_active_idx is not None:
            active_idx = explicit_active_idx.to(device=z_slots.device)
            if active_idx.ndim == 2:
                active_idx = active_idx[:, 0]
            active_idx = active_idx.long().clamp(0, z_slots.shape[1] - 1)
        else:
            active_idx = torch.argmax(c_slots.squeeze(-1), dim=-1)

        batch_idx = torch.arange(b, device=z_slots.device)
        return {
            "z_obj": z_slots[batch_idx, active_idx],
            "confidence": c_slots[batch_idx, active_idx],
            "active_slot_age": age[batch_idx, active_idx],
            "active_slot_index": active_idx.view(b, 1),
        }

    def read_state(self, prev: Dict[str, torch.Tensor], z_template: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Read previous slots without applying sensory update. Used in sleep/dream mode."""
        z_slots, c_slots, age = self._coerce_prev_slots(prev, z_template)
        mem_slots = prev.get("memory_stability_slots") if isinstance(prev, dict) else None
        if mem_slots is None:
            mem_slots = c_slots.clone()
        else:
            mem_slots = mem_slots.to(device=z_slots.device, dtype=z_slots.dtype)
            if mem_slots.ndim == 2:
                mem_slots = mem_slots.unsqueeze(-1)
            if mem_slots.shape[1] < z_slots.shape[1]:
                pad = torch.zeros(z_slots.shape[0], z_slots.shape[1] - mem_slots.shape[1], 1, device=z_slots.device, dtype=z_slots.dtype)
                mem_slots = torch.cat([mem_slots[:, :, :1], pad], dim=1)
            elif mem_slots.shape[1] > z_slots.shape[1]:
                mem_slots = mem_slots[:, :z_slots.shape[1], :1]
        active_idx = prev.get("active_slot_index") if isinstance(prev, dict) else None
        active = self._read_active_from_slots(z_slots, c_slots, age, explicit_active_idx=active_idx)
        bidx = torch.arange(z_slots.shape[0], device=z_slots.device)
        aidx = active["active_slot_index"][:, 0].long()
        zeros = torch.zeros(z_slots.shape[0], z_slots.shape[1], 1, device=z_slots.device, dtype=z_slots.dtype)
        return {
            **active,
            "update_gate_mean": torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype),
            "z_obj_slots": z_slots,
            "confidence_slots": c_slots,
            "memory_stability_slots": mem_slots,
            "memory_stability": mem_slots[bidx, aidx],
            "dream_activation_slots": zeros,
            "dream_activation": zeros[bidx, aidx],
            "slot_age": age,
            "slot_binding": zeros,
            "slot_similarity": zeros,
            "slot_update_strength": zeros,
            "active_slot_similarity": torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype),
            "active_slot_binding": torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype),
            "sleep_dream_mode": torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype),
            "dream_empty_mode": torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype),
        }

    def forward(
        self,
        prev: Dict[str, torch.Tensor],
        z_update,
        vision_strength,
        touch_strength,
        freeze_update: bool = False,
        force_slot_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if freeze_update:
            return self.read_state(prev, z_update)

        b, d = z_update.shape
        s = self.num_slots
        z_prev, c_prev, age_prev = self._coerce_prev_slots(prev, z_update)
        mem_prev = prev.get("memory_stability_slots") if isinstance(prev, dict) else None
        if mem_prev is None:
            mem_prev = c_prev.clone()
        else:
            mem_prev = mem_prev.to(device=z_update.device, dtype=z_update.dtype)
            if mem_prev.ndim == 2:
                mem_prev = mem_prev.unsqueeze(-1)
            if mem_prev.shape[1] < s:
                pad = torch.zeros(b, s - mem_prev.shape[1], 1, device=z_update.device, dtype=z_update.dtype)
                mem_prev = torch.cat([mem_prev[:, :, :1], pad], dim=1)
            elif mem_prev.shape[1] > s:
                mem_prev = mem_prev[:, :s, :1]

        z_u = z_update.unsqueeze(1).expand(-1, s, -1)
        v = vision_strength.unsqueeze(1).expand(-1, s, -1)
        t = touch_strength.unsqueeze(1).expand(-1, s, -1)

        z_prev_norm = F.normalize(z_prev, dim=-1, eps=1e-6)
        z_u_norm = F.normalize(z_u, dim=-1, eps=1e-6)
        similarity = (z_prev_norm * z_u_norm).sum(dim=-1, keepdim=True)
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # Sticky object identity routing.
        #
        # Problem fixed:
        #   Without sticky routing, a partially learned object can jump between
        #   slots because small similarity fluctuations change the selected slot.
        #
        # Policy:
        #   1) Prefer the previous active slot while it is still plausible.
        #   2) Switch only if another occupied slot is better by switch_margin.
        #   3) If no occupied slot is similar enough, allocate the emptiest slot.
        #   4) Update only the selected slot; all other slots are left alone
        #      except for gentle confidence decay.
        # ------------------------------------------------------------------
        mem_flat = torch.clamp(mem_prev.squeeze(-1), 0.0, 1.0)
        conf_flat = mem_flat
        sim_flat = similarity.squeeze(-1)

        if isinstance(prev, dict) and prev.get("active_slot_index") is not None:
            active_prev = prev["active_slot_index"].to(device=z_update.device)
            if active_prev.ndim == 2:
                active_prev = active_prev[:, 0]
            active_prev = active_prev.long().clamp(0, s - 1)
        else:
            active_prev = torch.argmax(conf_flat, dim=-1)

        batch_idx = torch.arange(b, device=z_update.device)
        active_conf = conf_flat[batch_idx, active_prev]
        active_sim = sim_flat[batch_idx, active_prev]

        sticky_conf_thr = float(getattr(self.cfg, "sticky_confidence_threshold", 0.12))
        sticky_sim_thr = float(getattr(self.cfg, "sticky_similarity_threshold", 0.30))
        switch_margin = float(getattr(self.cfg, "switch_margin", 0.18))
        new_obj_thr = float(getattr(self.cfg, "new_object_similarity_threshold", 0.22))
        slot_active_thr = float(getattr(self.cfg, "slot_activity_threshold", 0.05))

        occupied = conf_flat > slot_active_thr
        sim_for_existing = sim_flat.masked_fill(~occupied, -1e6)
        best_existing_sim, best_existing_idx = torch.max(sim_for_existing, dim=-1)

        emptiest_idx = torch.argmin(conf_flat, dim=-1)
        any_occupied = occupied.any(dim=-1)

        other_is_much_better = (
            any_occupied
            & (best_existing_idx != active_prev)
            & (best_existing_sim > active_sim + switch_margin)
            & (best_existing_sim > new_obj_thr)
        )

        active_is_bad = (
            (active_conf <= sticky_conf_thr)
            | (active_sim <= new_obj_thr)
        )

        target_idx = active_prev
        target_idx = torch.where(other_is_much_better, best_existing_idx, target_idx)

        allocate_new = active_is_bad & (best_existing_sim <= new_obj_thr)
        target_idx = torch.where(allocate_new, emptiest_idx, target_idx)

        target_idx = torch.where(~any_occupied, emptiest_idx, target_idx)

        # Explicit proposal->slot lock. Used when the runner provides several
        # object proposals in one frame. This is the stable path for scenes with
        # known proposal order: slot 0 = proposal 0, slot 1 = proposal 1, etc.
        if force_slot_index is not None:
            forced = int(max(0, min(int(force_slot_index), s - 1)))
            target_idx = torch.full_like(target_idx, forced)

        binding = torch.zeros(b, s, 1, device=z_update.device, dtype=z_update.dtype)
        binding.scatter_(1, target_idx.view(b, 1, 1), 1.0)

        gate_in = torch.cat([z_prev, z_u, v, t], dim=-1).reshape(b * s, d * 2 + 2)
        gate = self.update_gate(gate_in).view(b, s, d)
        evidence_strength = torch.clamp(vision_strength + touch_strength, 0.0, 1.0)
        effective_gate = gate * evidence_strength.unsqueeze(1)
        update_strength = binding * effective_gate

        selected_update = (1.0 - effective_gate) * z_prev + effective_gate * z_u

        # Do not rewrite non-selected slots with the current percept.
        # This is what gives the slot a persistent object identity.
        z_new = (1.0 - binding) * z_prev + binding * selected_update

        conf_in = torch.cat([z_new, v, t], dim=-1).reshape(b * s, d + 2)
        conf_raw = self.conf_head(conf_in).view(b, s, 1)

        conf_candidate = conf_raw * evidence_strength.unsqueeze(1)
        # Current confidence is a sensory-confirmation signal, not memory.
        # Give the selected slot a direct evidence contribution so a clearly
        # visible/touched object crosses the awake-confidence threshold, while
        # disabled sensors still drive C to zero through evidence_strength.
        sensory_boost = 0.35 * evidence_strength.unsqueeze(1) * binding

        conf_selected = torch.clamp(conf_candidate + sensory_boost, 0.0, 1.0)
        conf_non_selected = torch.zeros_like(c_prev)
        conf = (1.0 - binding) * conf_non_selected + binding * conf_selected
        mem_candidate = torch.clamp(self.cfg.confidence_decay * mem_prev + (1.0 - self.cfg.confidence_decay) * torch.maximum(conf_selected, mem_prev), 0.0, 1.0)
        mem = (1.0 - binding) * mem_prev + binding * mem_candidate

        active_mask = mem > slot_active_thr
        age = torch.where(active_mask, age_prev + 1.0, torch.zeros_like(age_prev))

        active_idx = target_idx
        z_obj = z_new[batch_idx, active_idx]
        confidence = conf[batch_idx, active_idx]
        memory_stability = mem[batch_idx, active_idx]

        active_similarity = similarity[batch_idx, active_idx]
        active_binding = binding[batch_idx, active_idx]
        active_age = age[batch_idx, active_idx]

        route_reason = torch.zeros(b, 1, device=z_update.device, dtype=z_update.dtype)
        route_reason = torch.where(other_is_much_better.view(b, 1), torch.ones_like(route_reason), route_reason)
        route_reason = torch.where(allocate_new.view(b, 1), torch.full_like(route_reason, 2.0), route_reason)
        route_reason = torch.where((~any_occupied).view(b, 1), torch.full_like(route_reason, 3.0), route_reason)
        if force_slot_index is not None:
            route_reason = torch.full_like(route_reason, 4.0)

        return {
            "z_obj": z_obj,
            "confidence": confidence,
            "update_gate_mean": update_strength.mean(dim=(1, 2), keepdim=False).unsqueeze(-1),
            "z_obj_slots": z_new,
            "confidence_slots": conf,
            "memory_stability_slots": mem,
            "memory_stability": memory_stability,
            "dream_activation_slots": torch.zeros(b, s, 1, device=z_update.device, dtype=z_update.dtype),
            "dream_activation": torch.zeros(b, 1, device=z_update.device, dtype=z_update.dtype),
            "slot_age": age,
            "active_slot_index": active_idx.view(b, 1),
            "previous_active_slot_index": active_prev.view(b, 1),
            "slot_binding": binding,
            "slot_similarity": similarity,
            "slot_update_strength": update_strength.mean(dim=-1, keepdim=True),
            "active_slot_similarity": active_similarity,
            "active_slot_binding": active_binding,
            "active_slot_age": active_age,
            "route_reason": route_reason,
            "best_existing_similarity": best_existing_sim.view(b, 1),
            "active_previous_similarity": active_sim.view(b, 1),
            "sleep_dream_mode": torch.zeros(b, 1, device=z_update.device, dtype=z_update.dtype),
        }


class ObjectImaginationHead2D(nn.Module):
    def __init__(self, cfg: ObjectInnerImagery3DConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 8 * 8 * 96),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 5, 3, padding=1),
        )
        self.attr_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 16),
        )

    def forward(self, z_obj) -> Dict[str, torch.Tensor]:
        b = z_obj.shape[0]
        h = self.fc(z_obj).view(b, 96, 8, 8)
        img = self.decoder(h)
        if img.shape[-1] != self.cfg.image_size:
            img = F.interpolate(img, size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False)
        attrs = self.attr_head(z_obj)
        return {
            "rgb": torch.sigmoid(img[:, 0:3]),
            "depth": torch.sigmoid(img[:, 3:4]),
            "mask": torch.sigmoid(img[:, 4:5]),
            "shape_logits": attrs[:, 0:4],
            "color_rgb": torch.sigmoid(attrs[:, 4:7]),
            "size": torch.sigmoid(attrs[:, 7:8]),
            "hardness": torch.sigmoid(attrs[:, 8:9]),
            "stability": torch.sigmoid(attrs[:, 9:10]),
            "novelty": torch.sigmoid(attrs[:, 10:11]),
            "affordance": torch.sigmoid(attrs[:, 11:16]),
        }


class Object3DHead(nn.Module):
    def __init__(self, cfg: ObjectInnerImagery3DConfig):
        super().__init__()
        self.cfg = cfg
        pc_dim = cfg.point_count * 3
        conf_dim = cfg.point_count
        vox_dim = cfg.voxel_res ** 3
        self.point_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, pc_dim),
        )
        self.point_conf_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, conf_dim),
        )
        self.voxel_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, vox_dim),
        )

    def forward(self, z_obj: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = z_obj.shape[0]
        pts = torch.tanh(self.point_net(z_obj)).view(b, self.cfg.point_count, 3)
        point_conf = torch.sigmoid(self.point_conf_net(z_obj)).view(b, self.cfg.point_count, 1)
        vox = torch.sigmoid(self.voxel_net(z_obj)).view(b, 1, self.cfg.voxel_res, self.cfg.voxel_res, self.cfg.voxel_res)
        return {"point_cloud": pts, "point_conf": point_conf, "voxel_occ": vox}


class InnerObjectRepresentation3DSystem(nn.Module):
    def __init__(self, cfg: Optional[ObjectInnerImagery3DConfig] = None):
        super().__init__()
        self.cfg = cfg or ObjectInnerImagery3DConfig()
        self.fusion = VisualTactileObjectFusion(self.cfg)
        self.memory = ObjectSlotMemory(self.cfg)
        self.decoder_2d = ObjectImaginationHead2D(self.cfg)
        self.decoder_3d = Object3DHead(self.cfg)

    def initial_state(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        return self.memory.initial_state(batch_size, device)

    def decode_z(self, z_obj: torch.Tensor, extra: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        decoded_2d = self.decoder_2d(z_obj)
        decoded_3d = self.decoder_3d(z_obj)
        out = {"z_obj": z_obj, **decoded_2d, **decoded_3d}
        if extra:
            out.update(extra)
        return out

    def decode_slot(self, state: Dict[str, torch.Tensor], slot_index: int = 0) -> Dict[str, torch.Tensor]:
        z_slots = state.get("z_obj_slots")
        if z_slots is None:
            conf = state.get("confidence")
            if conf is None:
                conf = torch.zeros(state["z_obj"].shape[0], 1, device=state["z_obj"].device, dtype=state["z_obj"].dtype)
            return self.decode_z(
                state["z_obj"],
                {
                    "confidence": conf,
                    "sleep_dream_mode": torch.ones_like(conf),
                },
            )

        s = z_slots.shape[1]
        slot_index = max(0, min(int(slot_index), s - 1))
        z_obj = z_slots[:, slot_index]
        conf_slots = state.get("confidence_slots")
        mem_slots = state.get("memory_stability_slots")
        age = state.get("slot_age")
        confidence = conf_slots[:, slot_index] if conf_slots is not None else torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype)
        if mem_slots is None:
            mem_slots = conf_slots
        memory_stability = mem_slots[:, slot_index] if mem_slots is not None else confidence
        dream_slots = state.get("dream_activation_slots")
        if dream_slots is None:
            dream_slots = torch.zeros(z_slots.shape[0], z_slots.shape[1], 1, device=z_slots.device, dtype=z_slots.dtype)
        extra = {
            "z_obj_slots": z_slots,
            "confidence_slots": conf_slots,
            "memory_stability_slots": mem_slots,
            "memory_stability": memory_stability,
            "dream_activation_slots": dream_slots,
            "dream_activation": dream_slots[:, slot_index],
            "slot_age": age,
            "active_slot_index": torch.full((z_slots.shape[0], 1), slot_index, device=z_slots.device, dtype=torch.long),
            "confidence": confidence,
            "active_slot_age": age[:, slot_index] if age is not None else torch.zeros_like(confidence),
            "sleep_dream_mode": torch.ones(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype),
        }
        return self.decode_z(z_obj, extra)

    def decode_active_slot(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_slots = state.get("z_obj_slots")
        if z_slots is None:
            return self.decode_slot(state, 0)
        active_idx = state.get("active_slot_index")
        if active_idx is None:
            conf = state.get("confidence_slots")
            idx = int(torch.argmax(conf[0, :, 0]).item()) if conf is not None else 0
        else:
            idx = int(active_idx.reshape(-1)[0].item())
        return self.decode_slot(state, idx)

    def _state_from_slot_output(self, slot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "z_obj": slot.get("z_obj"),
            "confidence": slot.get("confidence"),
            "z_obj_slots": slot.get("z_obj_slots"),
            "confidence_slots": slot.get("confidence_slots"),
            "memory_stability_slots": slot.get("memory_stability_slots"),
            "memory_stability": slot.get("memory_stability"),
            "dream_activation_slots": slot.get("dream_activation_slots"),
            "dream_activation": slot.get("dream_activation"),
            "slot_age": slot.get("slot_age"),
            "active_slot_index": slot.get("active_slot_index"),
            "dream_tick": slot.get("dream_tick"),
        }

    def dream_step(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Internal dream stream.

        Full sleep means all external sensors are off, so sensory memory writes
        must stay frozen. But the inner world should not freeze visually: this
        method applies a very small deterministic latent transition to the active
        slot and then decodes it.

        This is intentionally gentle:
        - it does not erase object identity;
        - it does not learn from black frames;
        - it can cycle between occupied slots over time;
        - it exposes dream_tick and sleep_dream_mode for visualizers.
        """
        z_slots = state.get("z_obj_slots")
        if z_slots is None:
            # Old single-slot fallback.
            z_obj = state.get("z_obj")
            if z_obj is None:
                raise ValueError("dream_step requires z_obj or z_obj_slots")
            conf = state.get("confidence")
            if conf is None:
                conf = torch.zeros(z_obj.shape[0], 1, device=z_obj.device, dtype=z_obj.dtype)
            tick = state.get("dream_tick")
            if tick is None:
                tick = torch.zeros(z_obj.shape[0], 1, device=z_obj.device, dtype=z_obj.dtype)
            tick = tick + 1.0

            dream_empty_thr = float(getattr(self.cfg, "dream_empty_confidence_threshold", 0.05))
            is_empty = conf <= dream_empty_thr

            d = z_obj.shape[-1]
            phase = tick / 17.0
            idx = torch.arange(d, device=z_obj.device, dtype=z_obj.dtype).view(1, -1)
            wave = torch.sin(phase + idx * 0.173)
            strength = float(getattr(self.cfg, "dream_strength", 0.025))
            z_dream = z_obj + strength * wave * torch.tanh(z_obj.abs() + 0.25)

            return self.decode_z(z_dream, {
                "confidence": conf,
                "dream_tick": tick,
                "sleep_dream_mode": torch.ones_like(conf),
                "dream_empty_mode": is_empty.to(conf.dtype),
                "dream_latent_delta": (z_dream - z_obj).norm(dim=-1, keepdim=True),
            })

        conf_slots = state.get("confidence_slots")
        if conf_slots is None:
            conf_slots = torch.zeros(z_slots.shape[0], z_slots.shape[1], 1, device=z_slots.device, dtype=z_slots.dtype)
        mem_slots = state.get("memory_stability_slots")
        if mem_slots is None:
            mem_slots = conf_slots.clone()
        else:
            mem_slots = mem_slots.to(device=z_slots.device, dtype=z_slots.dtype)
            if mem_slots.ndim == 2:
                mem_slots = mem_slots.unsqueeze(-1)
            if mem_slots.shape[1] < z_slots.shape[1]:
                pad = torch.zeros(z_slots.shape[0], z_slots.shape[1] - mem_slots.shape[1], 1, device=z_slots.device, dtype=z_slots.dtype)
                mem_slots = torch.cat([mem_slots[:, :, :1], pad], dim=1)
            elif mem_slots.shape[1] > z_slots.shape[1]:
                mem_slots = mem_slots[:, :z_slots.shape[1], :1]
        age = state.get("slot_age")
        if age is None:
            age = torch.zeros_like(conf_slots)

        dream_empty_thr = float(getattr(self.cfg, "dream_empty_confidence_threshold", 0.05))
        any_dream_object = bool((mem_slots > dream_empty_thr).any().detach().cpu().item())

        tick = state.get("dream_tick")
        if tick is None:
            tick = torch.zeros(z_slots.shape[0], 1, device=z_slots.device, dtype=z_slots.dtype)
        else:
            tick = tick.to(device=z_slots.device, dtype=z_slots.dtype)
            if tick.ndim == 0:
                tick = tick.view(1, 1)
            elif tick.ndim == 1:
                tick = tick.view(-1, 1)
        tick = tick + 1.0

        b, s, d = z_slots.shape
        active_idx = state.get("active_slot_index")
        if active_idx is None:
            active_idx = torch.argmax(mem_slots.squeeze(-1), dim=-1)
        else:
            active_idx = active_idx.to(device=z_slots.device)
            if active_idx.ndim == 2:
                active_idx = active_idx[:, 0]
            active_idx = active_idx.long().clamp(0, s - 1)

        # Optional slow cycling through occupied slots: dream visits memories.
        cycle_steps = max(1, int(getattr(self.cfg, "dream_slot_cycle_steps", 90)))
        occupied = mem_slots.squeeze(-1) > float(getattr(self.cfg, "slot_activity_threshold", 0.05))
        if bool(getattr(self.cfg, "dream_latent_dynamics", True)) and bool(getattr(self.cfg, "dream_cycle_slots", False)) and s > 1:
            phase_slot = ((tick[:, 0].long() // cycle_steps) % s).long()
            # Use cycled slot only if it is occupied; otherwise keep active.
            cyc_occ = occupied.gather(1, phase_slot.view(b, 1)).squeeze(1)
            active_idx = torch.where(cyc_occ, phase_slot, active_idx)

        batch_idx = torch.arange(b, device=z_slots.device)
        z_active = z_slots[batch_idx, active_idx]

        strength = float(getattr(self.cfg, "dream_strength", 0.025))
        dim_idx = torch.arange(d, device=z_slots.device, dtype=z_slots.dtype).view(1, -1)
        phase = tick / 19.0
        wave_a = torch.sin(phase + dim_idx * 0.113)
        wave_b = torch.cos(phase * 0.73 + dim_idx * 0.071)

        # A tiny deterministic latent motion. The tanh envelope keeps it bounded
        # and identity-preserving.
        delta = strength * (0.65 * wave_a + 0.35 * wave_b) * torch.tanh(z_active.abs() + 0.20)
        if not any_dream_object:
            delta = torch.zeros_like(delta)
        z_dream = z_active + delta

        z_new = z_slots.clone()
        z_new[batch_idx, active_idx] = z_dream

        conf_new = torch.zeros_like(conf_slots)

        # Full sleep has no negative sensory evidence.
        #
        # Important semantic split:
        #   confidence_slots        = current/sensory confidence carried by the slot API
        #   memory_stability_slots  = how stable the remembered object is internally
        #   dream_activation_slots  = which memory is currently active in the dream stream
        #
        # The previous formula:
        #     conf * 0.995 + 0.002
        # has an equilibrium around 0.4, so high-confidence memories slowly shrink
        # in full sleep even when there is no external negative evidence. In dream,
        # the object memory should stay stable; only external sensory confirmation
        # is absent.
        memory_stability = mem_slots.clone()
        dream_activation = torch.zeros(b, s, 1, device=z_slots.device, dtype=z_slots.dtype)
        if any_dream_object:
            dream_activation[batch_idx, active_idx] = torch.clamp(
                0.35 + 0.65 * memory_stability[batch_idx, active_idx],
                0.0,
                1.0,
            )

        age_new = age + (memory_stability > float(getattr(self.cfg, "slot_activity_threshold", 0.05))).to(age.dtype)

        decoded = self.decode_z(z_dream, {
            "z_obj_slots": z_new,
            "confidence_slots": conf_new,
            "memory_stability_slots": memory_stability,
            "memory_stability": memory_stability[batch_idx, active_idx],
            "dream_activation_slots": dream_activation,
            "dream_activation": dream_activation[batch_idx, active_idx],
            "slot_age": age_new,
            "active_slot_index": active_idx.view(b, 1),
            "confidence": conf_new[batch_idx, active_idx],
            "active_slot_age": age_new[batch_idx, active_idx],
            "sleep_dream_mode": torch.ones(b, 1, device=z_slots.device, dtype=z_slots.dtype),
            "dream_empty_mode": torch.zeros(b, 1, device=z_slots.device, dtype=z_slots.dtype) if any_dream_object else torch.ones(b, 1, device=z_slots.device, dtype=z_slots.dtype),
            "dream_tick": tick,
            "dream_latent_delta": delta.norm(dim=-1, keepdim=True) if any_dream_object else torch.zeros(b, 1, device=z_slots.device, dtype=z_slots.dtype),
            "slot_binding": torch.zeros(b, s, 1, device=z_slots.device, dtype=z_slots.dtype),
            "slot_similarity": torch.zeros(b, s, 1, device=z_slots.device, dtype=z_slots.dtype),
            "slot_update_strength": torch.zeros(b, s, 1, device=z_slots.device, dtype=z_slots.dtype),
            "active_slot_binding": torch.zeros(b, 1, device=z_slots.device, dtype=z_slots.dtype),
            "active_slot_similarity": torch.zeros(b, 1, device=z_slots.device, dtype=z_slots.dtype),
        })
        return decoded

    def forward(
        self,
        prev_state,
        vision,
        tactile,
        body,
        hand,
        leg,
        freeze_memory_update: bool = False,
        dream_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if dream_mode:
            # Full sleep: no sensory fusion/write, but internal latent dynamics continue.
            return self.dream_step(prev_state)

        # Multi-object mode:
        # vision [B, P, vision_dim] means the runner has extracted P object
        # proposals from the current frame. Each proposal updates one slot.
        if vision.ndim == 3:
            b, p, _ = vision.shape
            max_p = min(int(p), int(getattr(self.cfg, "max_object_proposals", p)), self.memory.num_slots)
            state = prev_state
            last_out = None
            for pi in range(max_p):
                fused = self.fusion(vision[:, pi, :], tactile, body, hand, leg)
                force_idx = pi if bool(getattr(self.cfg, "proposal_slot_lock", True)) else None
                slot = self.memory(
                    state,
                    fused["z_update"],
                    fused["vision_strength"],
                    fused["touch_strength"],
                    freeze_update=bool(freeze_memory_update),
                    force_slot_index=force_idx,
                )
                state = self._state_from_slot_output(slot)
                last_out = {**fused, **slot, "proposal_index": torch.full((b, 1), pi, device=vision.device, dtype=vision.dtype)}

            if last_out is None:
                return self.decode_active_slot(prev_state)

            decoded_2d = self.decoder_2d(last_out["z_obj"])
            decoded_3d = self.decoder_3d(last_out["z_obj"])
            return {**last_out, **decoded_2d, **decoded_3d}

        fused = self.fusion(vision, tactile, body, hand, leg)
        slot = self.memory(
            prev_state,
            fused["z_update"],
            fused["vision_strength"],
            fused["touch_strength"],
            freeze_update=bool(freeze_memory_update),
        )

        decoded_2d = self.decoder_2d(slot["z_obj"])
        decoded_3d = self.decoder_3d(slot["z_obj"])

        return {**fused, **slot, **decoded_2d, **decoded_3d}
