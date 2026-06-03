#!/usr/bin/env python3
from __future__ import annotations

"""
Headless demo for EnergyResonator.

Run from repository root:

    python scripts/run_energy_resonator_demo.py
"""

from src.modules.m11_motivational_homeostasis.energy_resonator import run_headless_demo


if __name__ == "__main__":
    run_headless_demo(steps=100, switch_at=50)
