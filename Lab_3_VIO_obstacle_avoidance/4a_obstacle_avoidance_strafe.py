#!/usr/bin/env python3
"""Fly to a point and strafe around people on the way. Yaw once, then never again.

This is 4_obstacle_avoidance.py with a different way of moving. The detection,
the blocker test, the side choice, the detour planning, the fence, and the
safety limits are all reused from that file. What changes:

  * After the climb the drone turns its nose once toward the goal (for a goal
    to the right, that is a 90 degree right turn), holding its position while
    it turns. From then on the yaw is frozen at that heading. The camera
    always looks down the path to the goal.
  * When a person blocks the path the drone does not turn to a detour point.
    It translates sideways (strafes) with the nose still on the goal
    direction, passes, and slides back to the line to the goal.
  * Only a person in the middle band of the image counts. The sidestep stops
    as soon as the band is empty (plus a small margin), so the drone moves
    only as far to the side as it has to. People elsewhere in the frame are
    ignored.
  * At the goal the drone holds its position, turns 180 degrees, and flies the
    same line back to the takeoff point, dodging people the same way. Over the
    takeoff point it descends in place and disarms.
  * Position setpoints are sent every control tick, not once per YOLO frame.
    They pass through a velocity low-pass filter, so a new detour bends the
    path instead of kinking it, and the drone eases out of each move.
  * While someone is still in the middle band, the drone keeps strafing the
    same way and the sidestep grows with it, up to MAX_SIDESTEP_M. Past that
    it re-plans from where it is and picks a side again.

Pass --hover to fly. With no --hover the window shows the boxes and the
blocker, and the drone stays on the ground. Closing the window during flight
descends in place.

    python 4a_obstacle_avoidance_strafe.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
    python 4a_obstacle_avoidance_strafe.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
    python 4a_obstacle_avoidance_strafe.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover --goal-forward 2.5 --max-radius 3.5
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path


def _load_base():
    """Load 4_obstacle_avoidance.py. A leading digit is not a legal import name."""
    here = Path(__file__).resolve().parent
    path = here / "4_obstacle_avoidance.py"
    if not path.exists():
        raise SystemExit(f"Could not find {path}")
    spec = importlib.util.spec_from_file_location("obstacle_avoidance_base", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()
follow = base.follow
hover = base.hover
Pose = base.Pose
Box = base.Box
Fence = base.Fence

GOAL_FORWARD_M = 0.0
GOAL_RIGHT_M = 4.0

# The sidestep is only as wide as it needs to be. Other people in the frame do
# not matter, only a person in the middle band (see CORRIDOR_FRAC in the base
# file).
#   * While someone has been in the band for PUSH_HOLD_S, the drone keeps
#     strafing: the sidestep target stays PUSH_LEAD_M ahead of where it is.
#   * As soon as the band has been empty for TRIM_HOLD_S, the sidestep is cut
#     short and ends CLEAR_MARGIN_M beyond where the drone is.
#   * A sidestep that would pass MAX_SIDESTEP_M is re-planned from where the
#     drone is, with a fresh choice of side, instead of drifting on.
TRIM_HOLD_S = 0.15
CLEAR_MARGIN_M = 0.05
PUSH_HOLD_S = 0.2
PUSH_LEAD_M = 0.30
MAX_SIDESTEP_M = 1.5
# After it is back at the start, the setpoint gets this long to settle before
# the drone descends in place.
LAND_SETTLE_S = 1.0
# Position loop of the setpoint generator. The commanded speed is
# POS_GAIN_1_S times the remaining distance, capped at the safety speed.
POS_GAIN_1_S = 1.0
# First-order lag on the commanded velocity. Larger is smoother and slower to
# react. The path rounds corners over roughly this many seconds.
VEL_TAU_S = 0.4
# Hold position when no detection frame has arrived for this long.
DETECTION_STALE_S = 1.0
# Closer to the target than this, snap onto it so the setpoint stops cleanly.
SNAP_M = 0.002


class StrafeController(base.AvoidController):
    """AvoidController that turns to the goal once, then keeps the yaw locked.

    update_heading() and advance() run every control tick. observe() runs once
    per new YOLO frame and sets the target.
    """

    def __init__(self, start: Pose, goal: Pose, home: Pose, fence: Fence, by_width: bool, block_frac: float):
        super().__init__(start, goal, home, fence, by_width, block_frac)
        # The heading for the current leg: the nose along the path. The setpoint
        # yaw slews to it at the safety yaw rate. It only changes between legs.
        self.yaw_target = math.degrees(math.atan2(self.path[1], self.path[0]))
        self.turning = True
        self.turn_status = "turning to goal"
        # Leg 1 is takeoff to goal. Leg 2 is goal back to the takeoff point.
        self.leg = 1
        self.start_xy = (start.x, start.y)
        self.finished = False
        self.trimmed = False
        self.target = start
        self.vx = 0.0
        self.vy = 0.0
        # Once the turn is done the nose looks down the path, so the middle of
        # the image is always the way ahead.
        self.aligned = True
        self.status = self.turn_status
        self._tick: float | None = None

    def update_heading(self, measured: Pose):
        """End a turn once OptiTrack says the nose points down the path."""
        if not self.turning:
            return
        error = abs(hover.wrap_deg(self.yaw_target - measured.yaw_deg))
        if error <= base.YAW_GATE_DEG:
            self.turning = False
            where = "the goal" if self.leg == 1 else "the start point"
            print(
                f"Facing {where} (yaw {self.yaw_target:+.0f} deg). The nose stays locked on this leg.",
                flush=True,
            )

    def begin_return(self):
        """Goal reached: turn the nose around and fly the same line back to the start."""
        reached = self.goal
        self.origin = (reached.x, reached.y)
        self.goal = Pose(self.start_xy[0], self.start_xy[1], reached.z, reached.yaw_deg)
        dx = self.goal.x - reached.x
        dy = self.goal.y - reached.y
        length = math.hypot(dx, dy)
        self.path = (dx / length, dy / length)
        self.left = (-self.path[1], self.path[0])
        self.goal_s = length
        self.yaw_target = math.degrees(math.atan2(self.path[1], self.path[0]))
        self.leg = 2
        self.turn_status = "turning around"
        self.turning = True
        self.dodge = None
        self.arrived = False
        self.last_side = 0
        self.last_side_time = -math.inf
        self.trimmed = False
        self._last_note = ""
        self.status = self.turn_status

    # Planning, once per detection frame.

    def observe(self, boxes: list[Box], width: int, height: int, measured: Pose, stamp: float):
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return
        self._last_stamp = stamp
        command = self.plan(boxes, width, height, measured, stamp)
        if follow._finite_pose(command):
            self.target = command

    def plan(self, boxes: list[Box], width: int, height: int, measured: Pose, stamp: float) -> Pose:
        if self.finished:
            self.status = "back at start, landing"
            return Pose(self.goal.x, self.goal.y, self.goal.z, self.yaw_target)

        if self.turning:
            # Stay at the start of this leg while the nose swings around. The
            # image sweeps the room during the turn, so it is not checked.
            self.status = self.turn_status
            return Pose(self.origin[0], self.origin[1], self.goal.z, self.yaw_target)

        # The detour geometry uses the nose direction. Use the locked heading
        # so a little yaw noise in the measurement cannot bend the sidestep.
        held = Pose(measured.x, measured.y, measured.z, self.yaw_target)

        to_goal = math.hypot(self.goal.x - held.x, self.goal.y - held.y)
        if self.dodge is None and to_goal <= base.ARRIVE_M:
            if self.leg == 1:
                print("Goal reached. Turning around to fly back to the start.", flush=True)
                self.begin_return()
                self.status = self.turn_status
                return Pose(self.origin[0], self.origin[1], self.goal.z, self.yaw_target)
            print("Back at the start. Landing.", flush=True)
            self.finished = True
            self.status = "back at start, landing"
            return Pose(self.goal.x, self.goal.y, self.goal.z, self.yaw_target)

        blocker = base.find_blocker(boxes, width, height, self.block_frac)
        if blocker is not None:
            self.see_blocker(boxes, blocker, width, height, held, stamp)
        elif self.dodge is not None:
            self.see_clear(held, stamp)
        return self.drive(held)

    def note(self, message: str):
        # The parent says "turning back". This drone slides back, it does not turn.
        super().note(message.replace("Turning back to the goal.", "Sliding back toward the line."))

    def start_dodge(self, *args, **kwargs):
        super().start_dodge(*args, **kwargs)
        self.trimmed = False

    @staticmethod
    def moved_sideways(dodge, measured: Pose) -> float:
        """How far the drone has moved from the detour anchor, toward the detour side."""
        left_x, left_y = -dodge.fwd_y, dodge.fwd_x
        return dodge.side * (
            (measured.x - dodge.anchor_x) * left_x + (measured.y - dodge.anchor_y) * left_y
        )

    def see_clear(self, measured: Pose, stamp: float):
        """The middle band is empty. Cut the sidestep short, then let the parent finish the pass."""
        super().see_clear(measured, stamp)
        dodge = self.dodge
        if dodge is None or self.trimmed or dodge.blocked or dodge.clear_since is None:
            return
        if stamp - dodge.clear_since < TRIM_HOLD_S:
            return
        want = max(0.0, self.moved_sideways(dodge, measured) + CLEAR_MARGIN_M)
        if want < dodge.offset:
            dodge.offset = want
            self.note("Middle of the image is clear. Stopping the sidestep here.")
        self.trimmed = True

    def see_blocker(self, boxes, blocker: Box, width: int, height: int, measured: Pose, stamp: float):
        dodge = self.dodge
        if dodge is None:
            # New blocker. The parent applies side memory and plans the detour.
            super().see_blocker(boxes, blocker, width, height, measured, stamp)
            return

        dodge.clear_since = None
        if dodge.blocker_since is None:
            dodge.blocker_since = stamp

        if not dodge.blocked and stamp - dodge.blocker_since < PUSH_HOLD_S:
            return  # one frame is not enough to change the plan

        if not dodge.blocked:
            # Someone is still in the middle band: keep strafing. The sidestep
            # target stays PUSH_LEAD_M ahead of the drone until the band clears.
            want = self.moved_sideways(dodge, measured) + PUSH_LEAD_M
            if want <= dodge.offset:
                return  # the planned sidestep already reaches further than that
            if want <= MAX_SIDESTEP_M:
                previous = dodge.offset
                dodge.offset = want
                if self.inside_fence(*dodge.via()):
                    self.trimmed = False
                    self.note(
                        f"{blocker.class_name} still in the middle. Going {base.side_name(dodge.side)} "
                        "until it clears."
                    )
                    return
                dodge.offset = previous

        # No room, already blocked, or too wide: plan again from here, choosing a side afresh.
        side, left, right = base.choose_side(boxes, blocker, width, self.by_width)
        self.start_dodge(blocker, side, left, right, height, measured, "still ahead")

    def drive(self, measured: Pose) -> Pose:
        """Target the goal or the detour point. The yaw stays on the goal heading."""
        dodge = self.dodge
        if dodge is None:
            tx, ty = self.goal.x, self.goal.y
            name = "flying to goal" if self.leg == 1 else "flying back to start"
        else:
            tx, ty = dodge.via()
            name = f"strafing {base.side_name(dodge.side)}"
        if dodge is not None and dodge.blocked:
            self.status = "no room to pass, holding"
        else:
            self.status = name
        return Pose(tx, ty, self.goal.z, self.yaw_target)

    # Motion, every control tick.

    def hold(self, reason: str):
        """Stop where the setpoint is. The yaw keeps going to its heading."""
        self.target = self.setpoint
        self.vx = 0.0
        self.vy = 0.0
        self.status = self.turn_status if self.turning else reason

    def advance(self, now: float) -> Pose:
        """Move the setpoint toward the target with a smooth velocity."""
        if self._tick is None:
            self._tick = now
            return self.setpoint
        dt = min(follow.MAX_DT_S, max(0.0, now - self._tick))
        self._tick = now
        if dt <= 0.0:
            return self.setpoint

        aim = follow.clamp_to_fence(self.target, self.home, self.fence)
        sp = self.setpoint
        dx = aim.x - sp.x
        dy = aim.y - sp.y
        dist = math.hypot(dx, dy)

        if dist <= SNAP_M:
            x, y = aim.x, aim.y
            self.vx = 0.0
            self.vy = 0.0
        else:
            speed = min(follow.MAX_SPEED_M_S, POS_GAIN_1_S * dist)
            want_x = dx / dist * speed
            want_y = dy / dist * speed
            alpha = 1.0 - math.exp(-dt / VEL_TAU_S)
            self.vx += alpha * (want_x - self.vx)
            self.vy += alpha * (want_y - self.vy)
            actual = math.hypot(self.vx, self.vy)
            if actual > follow.MAX_SPEED_M_S:
                scale = follow.MAX_SPEED_M_S / actual
                self.vx *= scale
                self.vy *= scale
                actual = follow.MAX_SPEED_M_S
            if actual * dt >= dist:
                x, y = aim.x, aim.y
                self.vx = 0.0
                self.vy = 0.0
            else:
                x = sp.x + self.vx * dt
                y = sp.y + self.vy * dt

        z_step = follow.MAX_SPEED_M_S * dt
        z = sp.z + max(-z_step, min(z_step, aim.z - sp.z))
        # Yaw slews to the goal heading at the safety yaw rate. It is the same
        # target for the whole flight, so after the first turn it does not move.
        yaw_step = follow.MAX_YAW_RATE_DEG_S * dt
        yaw_error = hover.wrap_deg(self.yaw_target - sp.yaw_deg)
        yaw = hover.wrap_deg(sp.yaw_deg + max(-yaw_step, min(yaw_step, yaw_error)))
        self.setpoint = Pose(x, y, z, yaw)
        return self.setpoint


def avoid_until_land(cf, poses, home: Pose, start: Pose, goal: Pose, tracks, land,
                     fence: Fence, command, mission) -> Pose:
    """Fly to the goal, turn around, fly back to the start, then return so the caller lands.

    Each leg turns the nose once, then sends smoothed, yaw-locked setpoints.
    It also returns when the window closes or OptiTrack goes stale. avoid_loop()
    descends in place from the setpoint this returns, so landing happens over
    the start point.

    Replaces base.avoid_until_land, which avoid_loop() calls by name.
    """
    controller = StrafeController(start, goal, home, fence, mission.by_width, mission.block_frac)
    last_seq = 0
    period = 1.0 / hover.SETPOINT_HZ
    setpoint = start
    detector_lost = False
    stale_noted = False
    finished_at: float | None = None
    while not land.is_set():
        if not hover.pose_fresh(poses):
            print("OptiTrack update stopped.", flush=True)
            return setpoint
        seq, _image, boxes, _blocker, width, height, stamp, _infer_ms, error = tracks.snapshot()
        measured = follow.read_pose(poses)
        now = time.monotonic()
        if measured is not None:
            controller.update_heading(measured)

        if error is not None:
            if not detector_lost:
                print("Obstacle detector stopped. Holding position.", flush=True)
                detector_lost = True
            controller.hold("holding, detector failed")
        elif stamp is None or width <= 0 or height <= 0:
            # YOLO is still loading. Stay where we are.
            controller.hold("waiting for detector")
        elif now - stamp > DETECTION_STALE_S:
            if not stale_noted:
                print("No fresh detections. Holding position.", flush=True)
                stale_noted = True
            controller.hold("holding, no fresh detections")
        else:
            stale_noted = False
            if measured is not None and seq != last_seq:
                last_seq = seq
                controller.observe(boxes, width, height, measured, stamp)

        setpoint = controller.advance(now)
        cf.commander.send_position_setpoint(setpoint.x, setpoint.y, setpoint.z, setpoint.yaw_deg)
        command.publish(setpoint, controller.status)

        if controller.finished:
            if finished_at is None:
                finished_at = now
            elif now - finished_at >= LAND_SETTLE_S:
                return setpoint
        if land.wait(period):
            return setpoint
    return setpoint


# avoid_loop() in the base file looks this name up when it starts the flight.
base.avoid_until_land = avoid_until_land


def parse_args():
    parser = argparse.ArgumentParser(
        description="Turn to a goal once, fly to it, and strafe around people that YOLO sees on the way. The nose is locked after the first turn."
    )
    parser.add_argument("server", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This computer's IP on the OptiTrack network")
    parser.add_argument("--body-id", type=int, default=hover.BODY_ID, help="Motive rigid body id")
    parser.add_argument("--address", required=True, help="Crazyradio address, for example 0xE7E7E7E7E7")
    parser.add_argument("--hover", action="store_true", help="Climb, fly to the goal, then land when the window closes")
    parser.add_argument(
        "--climb",
        type=float,
        default=hover.HOVER_CLIMB_M,
        help=f"Meters above takeoff before the flight (default {hover.HOVER_CLIMB_M:.1f})",
    )
    parser.add_argument(
        "--goal-forward",
        type=float,
        default=GOAL_FORWARD_M,
        help=f"Goal distance ahead of the takeoff heading, meters (default {GOAL_FORWARD_M:.1f})",
    )
    parser.add_argument(
        "--goal-right",
        type=float,
        default=GOAL_RIGHT_M,
        help=f"Goal distance to the right of the takeoff heading, meters (default {GOAL_RIGHT_M:.1f})",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Treat every YOLO class as an obstacle and weigh sides by box width",
    )
    parser.add_argument(
        "--block-height",
        type=float,
        default=base.BLOCK_HEIGHT_FRAC,
        help=f"Smallest box height, as a share of the image, that counts as a blocker (default {base.BLOCK_HEIGHT_FRAC:.2f})",
    )
    parser.add_argument("--min-up", type=float, default=0.25, help="Lowest flying height above takeoff, meters")
    parser.add_argument("--max-up", type=float, default=1.5, help="Highest flying height above takeoff, meters")
    parser.add_argument("--max-radius", type=float, default=4.3, help="Horizontal fence around takeoff, meters")
    parser.add_argument("--conf", type=float, default=hover.DETECT_CONF, help="Minimum detection confidence")
    parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
    parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
    parser.add_argument("--scale", type=float, default=2.0, help="Display scale")
    parser.add_argument("--save", action="store_true", help="Save the frames shown in the window")
    parser.add_argument("--flip", action="store_true", help="Rotate the camera image 180 degrees")
    args = parser.parse_args()

    if args.climb <= 0.0 or args.min_up <= 0.0 or args.max_up <= 0.0 or args.max_radius <= 0.0:
        raise SystemExit("--climb, --min-up, --max-up, and --max-radius must be positive.")
    if not (args.min_up <= args.climb <= args.max_up):
        raise SystemExit("--climb must sit between --min-up and --max-up.")
    goal_dist = math.hypot(args.goal_right, args.goal_forward)
    if goal_dist < 2.0 * base.ARRIVE_M:
        raise SystemExit("The goal must be at least 0.25 m from takeoff.")
    if goal_dist > args.max_radius - base.FENCE_MARGIN_M:
        raise SystemExit(
            f"The goal is {goal_dist:.2f} m away. Raise --max-radius above {goal_dist + base.FENCE_MARGIN_M:.2f}."
        )
    if goal_dist + base.CLEAR_MAX_M > args.max_radius:
        print(
            f"Warning: --max-radius {args.max_radius:.2f} leaves little room to detour near the goal. "
            f"{goal_dist + base.CLEAR_MAX_M:.2f} m or more is safer.",
            flush=True,
        )
    if not 0.0 < args.block_height < 1.0:
        raise SystemExit("--block-height must be between 0 and 1.")
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf must be greater than 0 and at most 1.")
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
    return args


def main():
    args = parse_args()
    try:
        base.fly(args)
    except KeyboardInterrupt:
        print(flush=True)


if __name__ == "__main__":
    main()
