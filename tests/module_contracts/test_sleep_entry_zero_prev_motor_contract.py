from __future__ import annotations

from types import SimpleNamespace

import torch


class DummySleepSystem:
    from src.modules.m06_learning_sleep_consolidation.sleep_sensors import SleepSensorsMixin

    apply_startup_state = SleepSensorsMixin.apply_startup_state
    input_sensors_enabled_dict_no_startup_apply = SleepSensorsMixin.input_sensors_enabled_dict_no_startup_apply
    sensor_state_label_no_startup_apply = SleepSensorsMixin.sensor_state_label_no_startup_apply
    is_full_sleep_mode = SleepSensorsMixin.is_full_sleep_mode
    sensor_state_label = SleepSensorsMixin.sensor_state_label
    input_sensors_enabled_dict = SleepSensorsMixin.input_sensors_enabled_dict
    sleep_sensor_mask_dict = SleepSensorsMixin.sleep_sensor_mask_dict
    apply_sleep_sensor_state = SleepSensorsMixin.apply_sleep_sensor_state
    _zero_prev_motor_state_on_sleep_entry = SleepSensorsMixin._zero_prev_motor_state_on_sleep_entry

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            sleep_sensors=SimpleNamespace(
                startup_state="awake",
                enabled=True,
            )
        )
        self.video_sensor_enabled = True
        self.contact_sensor_enabled = True
        self.imu_sensor_enabled = True
        self.prev_embodied_action = torch.ones(1, 15)
        self.prev_hand_motor = torch.ones(1, 8) * 0.5
        self._status_writes = 0

    def write_module_debug_status(self):
        self._status_writes += 1


def test_sleep_entry_zeros_prev_embodied_and_hand_once():
    system = DummySleepSystem()

    changed = system.apply_sleep_sensor_state({
        "input_sensors_enabled": {
            "video": False,
            "contact": False,
            "imu": False,
        }
    })

    assert changed is True
    assert system.is_full_sleep_mode() is True
    assert torch.allclose(system.prev_embodied_action, torch.zeros_like(system.prev_embodied_action))
    assert torch.allclose(system.prev_hand_motor, torch.zeros_like(system.prev_hand_motor))
    assert system._status_writes == 1
    assert system._sleep_replay_prev_motor_reset["reason"] == "entered_full_sleep"


def test_partial_cut_does_not_zero_prev_motor():
    system = DummySleepSystem()

    changed = system.apply_sleep_sensor_state({
        "input_sensors_enabled": {
            "video": False,
            "contact": True,
            "imu": True,
        }
    })

    assert changed is True
    assert system.is_full_sleep_mode() is False
    assert torch.allclose(system.prev_embodied_action, torch.ones_like(system.prev_embodied_action))
    assert torch.allclose(system.prev_hand_motor, torch.ones_like(system.prev_hand_motor) * 0.5)


def test_runner_startup_snapshot_preserves_semantic_full_sleep():
    from src.apps.runner_startup_state import build_startup_state_snapshot

    cfg = SimpleNamespace(
        control_startup=SimpleNamespace(),
        sleep_sensors=SimpleNamespace(
            startup_state="sleep",
            video_sensor_enabled=True,
            contact_sensor_enabled=True,
            imu_sensor_enabled=True,
        ),
    )

    snapshot = build_startup_state_snapshot(cfg)
    assert snapshot.video_sensor_enabled is False
    assert snapshot.contact_sensor_enabled is False
    assert snapshot.imu_sensor_enabled is False
