from __future__ import annotations

"""Factory helpers for the V5.10 simulation world.

Heavy simulation imports stay inside factory functions so this module remains
safe for lightweight smoke tests.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SimulationWorldSnapshot:
    kwargs: Dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kwargs": dict(self.kwargs)}


def simulation_world_kwargs(cfg: Any) -> Dict[str, Any]:
    return {
        "embodied_dim": cfg.embodied_dim,
        "hand_motor_dim": cfg.hand_motor_dim,
        "add_vestibular_to_body_state": cfg.vestibular.add_to_body_state,
        "balance_reward_weight": cfg.vestibular.balance_reward_weight,
        "balance_gyro_penalty": cfg.vestibular.balance_gyro_penalty,
        "balance_diff_penalty": cfg.vestibular.balance_diff_penalty,
    }


def create_simulation_world(cfg: Any, device: Any) -> Any:
    from src.platform.mujoco_world.mujoco_live_world_mocap_contacts import (
        MujocoLiveWorldMocapContacts as MujocoLiveWorldDynamicRig,
    )

    return MujocoLiveWorldDynamicRig(device, cfg.mujoco_world, **simulation_world_kwargs(cfg))


def simulation_world_snapshot(cfg: Any) -> SimulationWorldSnapshot:
    return SimulationWorldSnapshot(kwargs=simulation_world_kwargs(cfg))
