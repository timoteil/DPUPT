"""Call the stage the student is flying.

step() in each stage file is empty until they write it. The arguments and
the return values are the names in the laboratory notes. This file keeps
the waypoint index, the arrival timer, the yaw setpoint, and the last
velocity, and it turns what they return into the command the drone tracks.
"""

import numpy as np

from step1_position import step as step_position
from step2_small_yaw import step as step_yaw
from step3_cruise import step as step_cruise
from step4_acceleration import step as step_acceleration


class FlightState:
    """Memory that has to survive from one control step to the next."""

    def __init__(self):
        self.index = 1
        self.hold = 0.0
        self.yaw_set = 0.0
        self.velocity_sent = np.zeros(2)
        self.finished = False
        self._begun = False

    def begin(self, yaw):
        """Start the square from a hover. yaw is the heading at that moment."""
        self.index = 1
        self.hold = 0.0
        self.yaw_set = float(yaw)
        self.velocity_sent = np.zeros(2)
        self.finished = False
        self._begun = True


def fly(stage, state, position, velocity, yaw, dt, gains, waypoints):
    """One control cycle. The body-frame rotation stays outside the stage files."""
    if stage not in (1, 2, 3, 4):
        raise ValueError("STAGE must be 1, 2, 3, or 4")
    if not state._begun:
        state.begin(yaw)
    dt = float(dt)
    if dt <= 1e-6:
        dt = 1e-6

    waypoints = np.asarray(waypoints, dtype=float)
    position = np.asarray(position, dtype=float)
    measured = np.asarray(velocity, dtype=float)
    # Distance and Stage 2 both use the corner this cycle started on.
    # Stage 1 and Stage 3 advance the index only by returning a new one.
    index = int(state.index)
    hold = float(state.hold)
    goal = waypoints[index]
    distance = float(np.hypot(goal[0] - position[0], goal[1] - position[1]))
    v_x_measured = float(measured[0])
    v_y_measured = float(measured[1])

    if stage < 3:
        velocity_xy, yaw_rate, height, index, hold = _values(
            step_position(
                WAYPOINTS=waypoints,
                index=index,
                position=position,
                v_x_measured=v_x_measured,
                v_y_measured=v_y_measured,
                dt=dt,
                hold=hold,
                KP=gains.KP,
                V_MAX=gains.V_MAX,
                R_ACCEPT=gains.R_ACCEPT,
                V_ARRIVE=gains.V_ARRIVE,
                ARRIVE_HOLD=gains.ARRIVE_HOLD,
            ),
            5,
            "step1_position.step",
        )
        sent = _velocity(velocity_xy, "step1_position.step velocity")
        if stage == 2:
            yaw_rate, psi_set = _values(
                step_yaw(
                    WAYPOINTS=waypoints,
                    n=state.index,
                    distance=distance,
                    yaw=float(yaw),
                    dt=dt,
                    psi_set=float(state.yaw_set),
                    R_DISK=gains.R_DISK,
                    BETA=gains.BETA,
                    OMEGA_SET=gains.OMEGA_SET,
                    KY=gains.KY,
                    R_MAX=gains.R_MAX,
                ),
                2,
                "step2_small_yaw.step",
            )
            state.yaw_set = float(psi_set)
    else:
        wanted, index, hold = _values(
            step_cruise(
                WAYPOINTS=waypoints,
                k=index,
                position=position,
                v_x_measured=v_x_measured,
                v_y_measured=v_y_measured,
                dt=dt,
                hold=hold,
                R_DISK=gains.R_DISK,
                V_CRUISE=gains.V_CRUISE,
                K_CROSS=gains.K_CROSS,
                R_PASS=gains.R_PASS,
                D_SLOW=gains.D_SLOW,
                R_ACCEPT=gains.R_ACCEPT,
                V_ARRIVE=gains.V_ARRIVE,
                ARRIVE_HOLD=gains.ARRIVE_HOLD,
            ),
            3,
            "step3_cruise.step",
        )
        wanted = _velocity(wanted, "step3_cruise.step v")
        height = float(goal[2])
        yaw_rate, psi_set = _values(
            step_yaw(
                WAYPOINTS=waypoints,
                n=state.index,
                distance=distance,
                yaw=float(yaw),
                dt=dt,
                psi_set=float(state.yaw_set),
                R_DISK=gains.R_DISK,
                BETA=gains.BETA,
                OMEGA_SET=gains.OMEGA_SET,
                KY=gains.KY,
                R_MAX=gains.R_MAX,
            ),
            2,
            "step2_small_yaw.step",
        )
        state.yaw_set = float(psi_set)
        if stage == 3:
            sent = wanted
        else:
            sent, acceleration_xy, tilt = _values(
                step_acceleration(
                    velocity_wanted=wanted,
                    velocity_sent=np.array(state.velocity_sent, dtype=float),
                    dt=dt,
                    A_MAX=gains.A_MAX,
                    G=gains.G,
                ),
                3,
                "step4_acceleration.step",
            )
            sent = _velocity(sent, "step4_acceleration.step velocity_sent")
            acceleration_xy = _velocity(acceleration_xy, "step4_acceleration.step acceleration")
            state.velocity_sent = np.array(sent, dtype=float)
            state.index = int(index)
            state.hold = float(hold)
            _finish(state, waypoints, gains)
            return _command(
                state,
                sent,
                float(yaw_rate),
                height,
                acceleration_xy,
                distance,
                float(tilt),
            )

    state.velocity_sent = np.array(sent, dtype=float)
    state.index = int(index)
    state.hold = float(hold)
    _finish(state, waypoints, gains)
    return _command(state, sent, float(yaw_rate), float(height), np.zeros(2), distance, 0.0)


def _finish(state, waypoints, gains):
    """The last corner is held once equation (4) has been true for t_hold."""
    last = len(waypoints) - 1
    if state.index == last and state.hold >= gains.ARRIVE_HOLD:
        state.finished = True


def _command(state, sent, yaw_rate, height, acceleration, distance, tilt):
    return {
        "vx": float(sent[0]),
        "vy": float(sent[1]),
        "yaw_rate": float(yaw_rate),
        "height": float(height),
        "tilt": float(tilt),
        "acceleration": float(np.hypot(acceleration[0], acceleration[1])),
        "index": int(state.index),
        "distance": float(distance),
        "yaw_set": float(state.yaw_set),
        "finished": bool(state.finished),
    }


def _values(result, count, name):
    if isinstance(result, (str, bytes)) or result is None:
        values = None
    else:
        try:
            values = tuple(result)
        except TypeError:
            values = None
    if values is None or len(values) != count:
        raise ValueError(
            "%s must return %d values, in the order in its docstring." % (name, count)
        )
    return values


def _velocity(value, name):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (2,):
        raise ValueError("%s must be v_x and v_y." % name)
    return array
