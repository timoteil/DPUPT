"""Flight loop for Webots.

Each stage file implements step(state, position, measured_velocity, yaw, dt,
gains, waypoints) and returns the command dict the Webots supervisor expects.
Real-hardware scripts use runner.fly instead, which composes the keyword step()
functions from the laboratory handout.
"""

import importlib

from runner import FlightState  # noqa: F401 — re-export for the Webots controller

_STAGE_MODULES = {
    1: "step1_position",
    2: "step2_small_yaw",
    3: "step3_cruise",
    4: "step4_acceleration",
}


def fly(stage, state, position, velocity, yaw, dt, gains, waypoints):
    module = importlib.import_module(_STAGE_MODULES[stage])
    return module.step(state, position, velocity, yaw, dt, gains, waypoints)
