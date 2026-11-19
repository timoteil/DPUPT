"""Progressive navigation square in Webots using Lab_1_navigation.

Edit the stage scripts in Lab_1_navigation/, set STAGE below, save, and reload
the world:

    ./scripts/run-lab-world.sh
"""

import math
import sys
from pathlib import Path

import numpy as np
from controller import Supervisor

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAB_ROOT = _REPO_ROOT / "Lab_1_navigation"
_SHARED = _REPO_ROOT / "controllers_shared" / "python_based"

for path in (_LAB_ROOT, _SHARED):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pid_controller import pid_velocity_fixed_height_controller  # noqa: E402

from gains import LessonGains  # noqa: E402
from geometry import body_velocity  # noqa: E402
from runner import FlightState  # noqa: E402
from sim_runner import fly  # noqa: E402

# Write the matching step file in Lab_1_navigation/, then set this to 1, 2, 3, or 4.
STAGE = 4

STEP_FILES = {
    1: "step1_position.py",
    2: "step2_small_yaw.py",
    3: "step3_cruise.py",
    4: "step4_acceleration.py",
}

WAYPOINTS = np.array([
    [0.0, 0.0, 0.6],
    [1.0, 0.0, 0.6],
    [1.0, 1.0, 0.6],
    [0.0, 1.0, 0.6],
    [0.0, 0.0, 0.6],
])

TAKEOFF_ALTITUDE_TOLERANCE = 0.08
TAKEOFF_SETTLE_TIME = 0.8


def draw_square(robot, waypoints, disk_radius):
    """Draw the square, a marker on each corner, and the approach disks."""
    coordinates = " ".join(
        "%.4f %.4f %.4f" % (point[0], point[1], point[2]) for point in waypoints
    )
    index_line = " ".join(str(index) for index in range(len(waypoints))) + " -1"
    markers = []
    for point in waypoints[:-1]:
        markers.append(
            "Transform { translation %.4f %.4f %.4f children [ Shape { "
            "appearance Appearance { material Material { emissiveColor 1 0.45 0.1 } } "
            "geometry Sphere { radius 0.025 } } ] }" % (point[0], point[1], point[2])
        )
    rings = []
    samples = 48
    for point in waypoints[1:-1]:
        ring = []
        for sample in range(samples):
            angle = 2.0 * math.pi * sample / samples
            ring.append(
                "%.4f %.4f %.4f" % (
                    point[0] + disk_radius * math.cos(angle),
                    point[1] + disk_radius * math.sin(angle),
                    point[2],
                )
            )
        ring_index = " ".join(str(sample) for sample in range(samples)) + " 0 -1"
        rings.append(
            "Shape { appearance Appearance { material Material { emissiveColor 0.88 0.48 0.18 } } "
            "geometry IndexedLineSet { coord Coordinate { point [ %s ] } coordIndex [ %s ] } }" % (
                " ".join(ring), ring_index
            )
        )
    node = (
        "DEF ProgressiveSquare Group { children [ "
        "Shape { appearance Appearance { material Material { emissiveColor 0.12 0.45 0.75 } } "
        "geometry IndexedLineSet { coord Coordinate { point [ %s ] } coordIndex [ %s ] } } "
        "%s %s ] }" % (coordinates, index_line, " ".join(markers), " ".join(rings))
    )
    robot.getRoot().getField("children").importMFNodeFromString(-1, node)


def _hover(position):
    return {
        "vx": 0.0,
        "vy": 0.0,
        "yaw_rate": 0.0,
        "height": float(WAYPOINTS[0, 2]),
        "tilt": 0.0,
        "index": 0,
        "distance": float(np.hypot(position[0], position[1])),
        "finished": False,
    }


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    motors = []
    for name, sign in (("m1_motor", -1), ("m2_motor", 1), ("m3_motor", -1), ("m4_motor", 1)):
        motor = robot.getDevice(name)
        motor.setPosition(float("inf"))
        motor.setVelocity(sign)
        motors.append((motor, sign))

    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    gains = LessonGains()
    flight = FlightState()
    print("Lab 1 navigation, stage %d." % STAGE, flush=True)
    print("Edit Lab_1_navigation/%s" % STEP_FILES[STAGE], flush=True)
    print("  1 position   2 small yaw   3 velocity   4 acceleration", flush=True)
    print("Change STAGE in crazyflie_controller_lab_1_navigation.py.", flush=True)
    for index, point in enumerate(WAYPOINTS):
        print("  corner %d  %s" % (index, np.array2string(point, precision=2)), flush=True)

    try:
        draw_square(robot, WAYPOINTS, gains.R_DISK)
    except Exception as error:
        print("Could not draw the square: %s" % error, flush=True)

    pid = pid_velocity_fixed_height_controller()
    robot.step(timestep)
    past_position = np.array(gps.getValues())
    past_time = robot.getTime()
    phase = "takeoff"
    settle_time = 0.0
    reported_done = False
    previous_index = 1
    missing_reported = False

    while robot.step(timestep) != -1:
        time = robot.getTime()
        dt = time - past_time
        if dt <= 1e-6:
            continue

        position = np.array(gps.getValues())
        velocity = (position - past_position) / dt
        roll, pitch, yaw = imu.getRollPitchYaw()
        yaw_rate_measured = gyro.getValues()[2]

        if phase == "takeoff":
            command = _hover(position)
            command["height"] = float(WAYPOINTS[0, 2])
            if abs(position[2] - WAYPOINTS[0, 2]) < TAKEOFF_ALTITUDE_TOLERANCE and abs(velocity[2]) < 0.15:
                settle_time += dt
            else:
                settle_time = 0.0
            if settle_time >= TAKEOFF_SETTLE_TIME:
                phase = "track"
                flight.begin(yaw)
                print("Hover settled. Starting stage %d." % STAGE, flush=True)
        else:
            try:
                command = fly(STAGE, flight, position, velocity, yaw, dt, gains, WAYPOINTS)
            except (NotImplementedError, ValueError) as error:
                if not missing_reported:
                    missing_reported = True
                    print("This stage still has a blank function.", flush=True)
                    print(error, flush=True)
                    print("Fill it in, save, and reload the world.", flush=True)
                command = _hover(position)

            if command["index"] != previous_index and not command["finished"]:
                print("Reached corner %d. Next goal is corner %d." % (
                    previous_index, command["index"]), flush=True)
                previous_index = command["index"]
            if command["finished"] and not reported_done:
                reported_done = True
                print("Square complete. Holding the last corner.", flush=True)

        forward, sideways = body_velocity(command["vx"], command["vy"], yaw)
        motor_power = pid.pid(
            dt, forward, sideways, command["yaw_rate"], command["height"],
            roll, pitch, yaw_rate_measured, position[2],
            velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw),
            -velocity[0] * math.sin(yaw) + velocity[1] * math.cos(yaw),
        )
        for (motor, sign), power in zip(motors, motor_power):
            motor.setVelocity(sign * float(power))

        if int(time) != int(past_time) and phase == "track":
            message = "t=%5.2f  goal=%d  dist=%.2f m  speed=%.2f m/s  yaw=%5.1f deg" % (
                time,
                command["index"],
                command["distance"],
                float(np.hypot(velocity[0], velocity[1])),
                math.degrees(yaw),
            )
            if STAGE >= 4:
                message += "  tilt=%4.1f deg" % math.degrees(command["tilt"])
            print(message, flush=True)

        past_time = time
        past_position = position


if __name__ == "__main__":
    main()
