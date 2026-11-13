#!/usr/bin/env python3
"""Fly to a point on the drone's right and go around people on the way.

The camera, OptiTrack pose, radio, YOLO, and safety limits come from
6_person_follower_smooth.py. After the climb, the goal is latched
--goal-right meters to the right of the takeoff heading. The drone turns its
nose toward the goal first, then flies straight to it.

Every YOLO frame is checked for a blocker: a box that overlaps the middle
30% of the image and is at least --block-height of the image tall. When one
shows up, the drone picks the side with fewer other detections, or the wider
gap beside the blocker when the counts tie, and flies to a detour point on
that side. Once it has moved past and the middle of the image stays clear,
it turns back to the goal and finishes the line.

A blocker that shows up on the way to the detour point, for example a person
who steps into the detour, starts a new detour from where the drone is.
That happens at once when the box is close, otherwise after a second. When
the box fills the frame the new detour also backs away.

With --all-classes every YOLO class counts as an obstacle, and each side is
weighed by the width of its boxes instead of their count.

Pass --hover to fly. With no --hover the window shows the boxes and the
blocker, and the drone stays on the ground. Closing the window during flight
descends in place, so the drone never flies back through the person.

    python 11_obstacle_avoidance.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
    python 11_obstacle_avoidance.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
    python 11_obstacle_avoidance.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover --goal-right 2.0 --max-radius 3.0
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Set before cv2, as in 6_person_follower_smooth.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2


def _load_follow():
    """Load ../Lab_2_Follow_Me/6_person_follower_smooth.py. A leading digit is not a legal import name.

    That script finds 3_stream_hover_yolo.py, the axis calibration, the YOLO
    weights, and the viewer beside itself, so they all come from Lab_2_Follow_Me.
    """
    path = Path(__file__).resolve().parent.parent / "Lab_2_Follow_Me" / "6_person_follower_smooth.py"
    if not path.exists():
        raise SystemExit(f"Could not find {path}")
    spec = importlib.util.spec_from_file_location("person_follower_smooth", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


follow = _load_follow()
hover = follow.hover
Pose = follow.Pose
Box = follow.Box
Fence = follow.Fence

GOAL_RIGHT_M = 4
# The goal is reached when OptiTrack is this close to it.
ARRIVE_M = 0.12
# Move only once the nose is within YAW_GATE_DEG of the target. Stop and turn
# again when it drifts past YAW_RELEASE_DEG. The gap stops chatter.
YAW_GATE_DEG = 12.0
YAW_RELEASE_DEG = 25.0
# Closer than this, the bearing to the target swings wildly. Keep the yaw.
YAW_FREEZE_M = 0.25
# A blocker overlaps the middle CORRIDOR_FRAC of the image width and is at
# least BLOCK_HEIGHT_FRAC of the image tall. Box height is the only distance
# hint the camera gives: taller means closer.
CORRIDOR_FRAC = 0.30
BLOCK_HEIGHT_FRAC = 0.25
# Taller than this, the blocker is close. The detour becomes a pure sidestep.
NEAR_HEIGHT_FRAC = 0.50
# Taller than this, the box fills the frame. The sidestep also backs away.
VERY_NEAR_HEIGHT_FRAC = 0.90
BACKOFF_M = 0.3
# The detour point is this far ahead of the drone's nose, and the clearance
# to the side grows with the blocker height.
AHEAD_M = 0.6
CLEAR_MIN_M = 0.5
CLEAR_MAX_M = 0.9
# The detour ends after this much progress along the path, once the middle of
# the image has been empty for CLEAR_HOLD_S.
PASS_M = 0.8
CLEAR_HOLD_S = 0.4
# A blocker on the way to the detour point means the detour itself is
# blocked. Replan from where the drone is: at once when the blocker is close
# or the drone is already at the detour point, otherwise after REPLAN_S.
REPLAN_S = 1.0
# A new blocker soon after a detour keeps the same side, unless the other
# side has fewer detections by the margin. Equal counts fall back to the gap.
SIDE_MEMORY_S = 3.0
SWITCH_MARGIN_COUNT = 2
SWITCH_MARGIN_WIDTH = 0.25
FENCE_MARGIN_M = 0.05


def side_name(side: int) -> str:
    return "left" if side > 0 else "right"


def find_blocker(boxes: list[Box], width: int, height: int, block_frac: float) -> Box | None:
    """The tallest box in the middle of the image, if it is tall enough."""
    if width <= 0 or height <= 0:
        return None
    lo = width * (0.5 - CORRIDOR_FRAC / 2.0)
    hi = width * (0.5 + CORRIDOR_FRAC / 2.0)
    candidates = [
        box for box in boxes if box.x2 > lo and box.x1 < hi and box.height >= block_frac * height
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda box: box.height)


def choose_side(boxes: list[Box], blocker: Box, width: int, by_width: bool) -> tuple[int, float, float]:
    """Return (side, left_load, right_load). side is +1 for image left, -1 for right.

    The load of a side is the number of other boxes on it, or the sum of their
    widths as a share of the image when by_width is set. The lighter side wins.
    On a tie, the wider free gap beside the blocker wins, then the left.
    """
    blocker_u, _v = blocker.center
    left = 0.0
    right = 0.0
    for box in boxes:
        if box is blocker:
            continue
        load = box.width / float(width) if by_width else 1.0
        if box.center[0] < blocker_u:
            left += load
        else:
            right += load
    if left < right:
        return 1, left, right
    if right < left:
        return -1, left, right
    left_gap = blocker.x1
    right_gap = width - blocker.x2
    return (-1 if right_gap > left_gap else 1), left, right


def clearance_m(blocker: Box, height: int) -> float:
    """Sideways distance for the detour. A taller box is closer, so pass wider."""
    share = blocker.height / float(height)
    return min(CLEAR_MAX_M, max(CLEAR_MIN_M, 0.45 + 0.8 * share))


def goal_from(start: Pose, right_m: float, forward_m: float) -> Pose:
    """A point right_m to the right and forward_m ahead of the start heading.

    +Y is left, so the body right is (sin yaw, -cos yaw) in the world.
    """
    yaw = math.radians(start.yaw_deg)
    return Pose(
        start.x + forward_m * math.cos(yaw) + right_m * math.sin(yaw),
        start.y + forward_m * math.sin(yaw) - right_m * math.cos(yaw),
        start.z,
        start.yaw_deg,
    )


@dataclass
class Dodge:
    """A detour point, built from the drone's pose when the detour was planned.

    The point is ahead meters along the nose direction (fwd_x, fwd_y) from the
    anchor, and offset meters to the side given by side (+1 left, -1 right).
    start_s is the progress along the start-to-goal line at planning time.
    """

    side: int
    anchor_x: float
    anchor_y: float
    fwd_x: float
    fwd_y: float
    ahead: float
    offset: float
    start_s: float
    clear_since: float | None = None
    blocker_since: float | None = None
    blocked: bool = False

    def via(self) -> tuple[float, float]:
        left_x, left_y = -self.fwd_y, self.fwd_x
        return (
            self.anchor_x + self.ahead * self.fwd_x + self.side * self.offset * left_x,
            self.anchor_y + self.ahead * self.fwd_y + self.side * self.offset * left_y,
        )


class AvoidController:
    """Turn to the goal, fly straight, and detour around blockers.

    step() runs once per new YOLO frame and returns the next absolute
    position setpoint. The same approach() and clamp_to_fence() limits as the
    person follower are applied before the command is sent.
    """

    def __init__(self, start: Pose, goal: Pose, home: Pose, fence: Fence, by_width: bool, block_frac: float):
        self.setpoint = start
        self.goal = goal
        self.home = home
        self.fence = fence
        self.by_width = by_width
        self.block_frac = block_frac
        self.origin = (start.x, start.y)
        dx = goal.x - start.x
        dy = goal.y - start.y
        length = math.hypot(dx, dy)
        self.path = (dx / length, dy / length)
        self.left = (-self.path[1], self.path[0])
        self.goal_s = length
        self.dodge: Dodge | None = None
        self.aligned = False
        self.arrived = False
        self.last_side = 0
        self.last_side_time = -math.inf
        self.status = "turning to goal"
        self._last_note = ""
        self._last_stamp: float | None = None

    def to_path(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        return dx * self.path[0] + dy * self.path[1], dx * self.left[0] + dy * self.left[1]

    def inside_fence(self, x: float, y: float) -> bool:
        return math.hypot(x - self.home.x, y - self.home.y) <= self.fence.max_radius - FENCE_MARGIN_M

    def note(self, message: str):
        """Print a planner event once, not on every frame it stays true."""
        if message != self._last_note:
            print(message, flush=True)
            self._last_note = message

    def fit_detour(self, dodge: Dodge, want: float) -> bool:
        """Set the widest offset up to want that keeps the detour inside the fence.

        The side in dodge is tried first, then the other. False when neither
        side has CLEAR_MIN_M of room.
        """
        side = dodge.side
        for candidate in (side, -side):
            offset = want
            while offset >= CLEAR_MIN_M - 1e-6:
                dodge.side = candidate
                dodge.offset = offset
                if self.inside_fence(*dodge.via()):
                    return True
                offset -= 0.05
        dodge.side = side
        return False

    def step(self, boxes: list[Box], width: int, height: int, measured: Pose, stamp: float) -> Pose:
        if self._last_stamp is None:
            dt = 0.0
        else:
            dt = min(follow.MAX_DT_S, max(0.0, stamp - self._last_stamp))
        self._last_stamp = stamp
        if dt <= 0.0:
            return self.setpoint
        command = self.plan(boxes, width, height, measured, stamp)
        if not follow._finite_pose(command):
            return self.setpoint
        self.setpoint = follow.approach(self.setpoint, follow.clamp_to_fence(command, self.home, self.fence), dt)
        return self.setpoint

    def plan(self, boxes: list[Box], width: int, height: int, measured: Pose, stamp: float) -> Pose:
        to_goal = math.hypot(self.goal.x - measured.x, self.goal.y - measured.y)
        if self.arrived or (self.dodge is None and to_goal <= ARRIVE_M):
            if not self.arrived:
                print("Goal reached. Holding. Close the window to land.", flush=True)
            self.arrived = True
            self.status = "at goal"
            return Pose(self.goal.x, self.goal.y, self.goal.z, self.setpoint.yaw_deg)

        # The middle of the image is only the way ahead once the nose points at the target.
        blocker = find_blocker(boxes, width, height, self.block_frac) if self.aligned else None
        if blocker is not None:
            self.see_blocker(boxes, blocker, width, height, measured, stamp)
        elif self.dodge is not None:
            self.see_clear(measured, stamp)
        return self.drive(measured)

    def see_blocker(self, boxes, blocker: Box, width: int, height: int, measured: Pose, stamp: float):
        side, left, right = choose_side(boxes, blocker, width, self.by_width)
        near = blocker.height >= NEAR_HEIGHT_FRAC * height
        dodge = self.dodge

        if dodge is None:
            if (
                self.last_side
                and side != self.last_side
                and left != right
                and stamp - self.last_side_time <= SIDE_MEMORY_S
            ):
                margin = SWITCH_MARGIN_WIDTH if self.by_width else SWITCH_MARGIN_COUNT
                crowd = (left - right) if self.last_side > 0 else (right - left)
                if crowd < margin:
                    side = self.last_side
            self.start_dodge(blocker, side, left, right, height, measured, "ahead")
            return

        # The nose points at the detour point, so this blocker is on the detour itself.
        dodge.clear_since = None
        if dodge.blocker_since is None:
            dodge.blocker_since = stamp
        vx, vy = dodge.via()
        at_via = math.hypot(vx - measured.x, vy - measured.y) <= 2.0 * ARRIVE_M
        if near or at_via or dodge.blocked or stamp - dodge.blocker_since >= REPLAN_S:
            self.start_dodge(blocker, side, left, right, height, measured, "in the detour")

    def start_dodge(self, blocker: Box, side: int, left: float, right: float, height: int, measured: Pose, where: str):
        """Plan a new detour from the drone's current position and heading."""
        share = blocker.height / float(height)
        if share >= VERY_NEAR_HEIGHT_FRAC:
            ahead = -BACKOFF_M
        elif share >= NEAR_HEIGHT_FRAC:
            ahead = 0.0
        else:
            ahead = AHEAD_M
        yaw = math.radians(measured.yaw_deg)
        s, _l = self.to_path(measured.x, measured.y)
        dodge = Dodge(side, measured.x, measured.y, math.cos(yaw), math.sin(yaw), ahead, 0.0, s)
        if self.fit_detour(dodge, clearance_m(blocker, height)):
            self.note(
                f"{blocker.class_name} {where}. Going {side_name(dodge.side)} "
                f"(left {left:g}, right {right:g}, side {dodge.offset:.2f} m, ahead {ahead:+.2f} m)."
            )
        else:
            dodge.ahead = 0.0
            dodge.offset = 0.0
            dodge.blocked = True
            self.note(f"{blocker.class_name} {where} and no room inside the fence. Holding.")
        self.dodge = dodge
        self.aligned = False

    def see_clear(self, measured: Pose, stamp: float):
        dodge = self.dodge
        dodge.blocker_since = None
        if dodge.clear_since is None:
            dodge.clear_since = stamp
        vx, vy = dodge.via()
        at_via = math.hypot(vx - measured.x, vy - measured.y) <= 2.0 * ARRIVE_M
        stuck = False
        if at_via:
            # Walk the detour point forward, parallel to the path, until past the blocker.
            via_s, _via_l = self.to_path(vx, vy)
            step = min(AHEAD_M, self.goal_s - via_s)
            if step > 1e-3 and self.inside_fence(vx + step * self.path[0], vy + step * self.path[1]):
                dodge.anchor_x += step * self.path[0]
                dodge.anchor_y += step * self.path[1]
                dodge.blocked = False
            else:
                stuck = True
        s, _l = self.to_path(measured.x, measured.y)
        passed = s - dodge.start_s >= PASS_M or s >= self.goal_s - ARRIVE_M or stuck
        if passed and stamp - dodge.clear_since >= CLEAR_HOLD_S:
            self.last_side = dodge.side
            self.last_side_time = stamp
            self.dodge = None
            self.aligned = False
            self.note("Obstacle passed. Turning back to the goal.")

    def drive(self, measured: Pose) -> Pose:
        """Turn toward the active target. Move only once the nose points at it."""
        dodge = self.dodge
        if dodge is None:
            tx, ty = self.goal.x, self.goal.y
            name = "goal"
        else:
            tx, ty = dodge.via()
            name = f"detour {side_name(dodge.side)}"

        dist = math.hypot(tx - measured.x, ty - measured.y)
        if dist <= YAW_FREEZE_M:
            yaw = self.setpoint.yaw_deg
            self.aligned = True
        else:
            yaw = math.degrees(math.atan2(ty - measured.y, tx - measured.x))
            error = abs(hover.wrap_deg(yaw - measured.yaw_deg))
            if self.aligned and error > YAW_RELEASE_DEG:
                self.aligned = False
            elif not self.aligned and error <= YAW_GATE_DEG:
                self.aligned = True

        if dodge is not None and dodge.blocked:
            self.status = "no room to pass, holding"
        else:
            self.status = ("flying to " if self.aligned else "turning to ") + name
        if not self.aligned:
            return Pose(self.setpoint.x, self.setpoint.y, self.goal.z, yaw)
        return Pose(tx, ty, self.goal.z, yaw)


class ObstacleTracks:
    """Latest detections. The image already has the boxes and the corridor drawn."""

    def __init__(self):
        self._lock = threading.Lock()
        self.seq = 0
        self.image = None
        self.boxes: list[Box] = []
        self.blocker: Box | None = None
        self.width = 0
        self.height = 0
        self.stamp: float | None = None
        self.infer_ms = 0.0
        self.error: str | None = None

    def publish(self, seq, image, boxes, blocker, width, height, stamp, infer_ms):
        with self._lock:
            self.seq = seq
            self.image = image
            self.boxes = boxes
            self.blocker = blocker
            self.width = width
            self.height = height
            self.stamp = stamp
            self.infer_ms = infer_ms

    def fail(self, message):
        with self._lock:
            self.error = message

    def snapshot(self):
        with self._lock:
            return (
                self.seq,
                self.image,
                list(self.boxes),
                self.blocker,
                self.width,
                self.height,
                self.stamp,
                self.infer_ms,
                self.error,
            )


def draw_obstacles(image, boxes: list[Box], blocker: Box | None):
    """Draw the corridor, every box, and the blocker in red."""
    view = image.copy()
    height, width = view.shape[:2]
    lo = int(round(width * (0.5 - CORRIDOR_FRAC / 2.0)))
    hi = int(round(width * (0.5 + CORRIDOR_FRAC / 2.0)))
    cv2.line(view, (lo, 0), (lo, height), (200, 200, 200), 1)
    cv2.line(view, (hi, 0), (hi, height), (200, 200, 200), 1)
    for box in boxes:
        color = (0, 0, 255) if box is blocker else (80, 220, 80)
        cv2.rectangle(view, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        cv2.putText(
            view,
            f"{box.class_name} {box.confidence:.2f}",
            (box.x1, max(12, box.y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return view


def obstacle_line(boxes: list[Box], blocker: Box | None, width: int, height: int, by_width: bool) -> str:
    if blocker is None or width <= 0 or height <= 0:
        return f"boxes={len(boxes)}  path clear"
    side, left, right = choose_side(boxes, blocker, width, by_width)
    return (
        f"blocker {blocker.class_name} h={blocker.height / height * 100:.0f}%  "
        f"left={left:g} right={right:g}  -> {side_name(side)}"
    )


def obstacle_detect_loop(viewer, state, lock, tracks: ObstacleTracks, stop, conf: float, all_classes: bool,
                         block_frac: float, flip: bool = False):
    """Load YOLO nano and publish every obstacle box until stop is set."""
    try:
        model, device, _weights = hover.load_nano_model()
        names = model.names
        if all_classes:
            class_ids = None
            allowed = set(names.keys()) if isinstance(names, dict) else set(range(len(names)))
            labels = "all classes"
        else:
            class_ids, labels = follow.person_classes(model)
            allowed = set(class_ids)
    except SystemExit as exc:
        print(exc, flush=True)
        tracks.fail(str(exc))
        return
    except Exception as exc:
        print(f"YOLO nano failed: {exc}", flush=True)
        tracks.fail(str(exc))
        return
    print(f"YOLO classes treated as obstacles: {labels}.", flush=True)
    last_seq = 0
    smoother = follow.BoxSmoother()
    while not stop.is_set():
        with lock:
            seq = state["seq"]
            frame = state["frame"] if seq != last_seq else None
            error = state["error"]
        if frame is None:
            if error is not None or stop.wait(0.005):
                return
            continue
        last_seq = seq
        width, height, fmt, payload = frame
        try:
            color, _bayer = viewer.decode_frame(width, height, fmt, payload, flip)
        except ValueError as exc:
            print(exc, flush=True)
            continue
        if stop.is_set():
            return
        taken = time.monotonic()
        started = time.perf_counter()
        try:
            result = model.predict(
                color,
                conf=conf,
                classes=class_ids,
                device=device,
                verbose=False,
            )[0]
        except Exception as exc:
            print(f"YOLO nano failed: {exc}", flush=True)
            tracks.fail(str(exc))
            return
        frame_w, frame_h = color.shape[1], color.shape[0]
        boxes = follow.boxes_from_result(result, names, allowed, frame_w, frame_h)
        boxes = smoother.update(boxes, frame_w, frame_h, taken)
        blocker = find_blocker(boxes, frame_w, frame_h, block_frac)
        view = draw_obstacles(color, boxes, blocker)
        infer_ms = (time.perf_counter() - started) * 1000.0
        tracks.publish(seq, view, boxes, blocker, frame_w, frame_h, taken, infer_ms)


@dataclass(frozen=True)
class Mission:
    goal_right: float
    goal_forward: float
    by_width: bool
    block_frac: float
    climb: float


def avoid_until_land(cf, poses, home: Pose, start: Pose, goal: Pose, tracks: ObstacleTracks, land,
                     fence: Fence, command, mission: Mission) -> Pose:
    """Send AvoidController setpoints until the window closes or OptiTrack goes stale."""
    controller = AvoidController(start, goal, home, fence, mission.by_width, mission.block_frac)
    last_seq = 0
    period = 1.0 / hover.SETPOINT_HZ
    setpoint = start
    detector_lost = False
    while not land.is_set():
        if not hover.pose_fresh(poses):
            print("OptiTrack update stopped.", flush=True)
            return setpoint
        seq, _image, boxes, _blocker, width, height, stamp, _infer_ms, error = tracks.snapshot()
        measured = follow.read_pose(poses)
        if error is not None and not detector_lost:
            print("Obstacle detector stopped. Holding position.", flush=True)
            detector_lost = True
        if (
            error is None
            and measured is not None
            and stamp is not None
            and width > 0
            and height > 0
            and seq != last_seq
        ):
            last_seq = seq
            setpoint = controller.step(boxes, width, height, measured, stamp)
        status = "holding, detector failed" if error is not None else controller.status
        cf.commander.send_position_setpoint(setpoint.x, setpoint.y, setpoint.z, setpoint.yaw_deg)
        command.publish(setpoint, status)
        if land.wait(period):
            return setpoint
    return setpoint


def avoid_loop(cf, poses, estimates, land, landed, yaw_locked, tracks, command, fence, mission: Mission):
    """Arm, climb, fly to the goal around obstacles, then descend in place and disarm."""
    armed = False
    try:
        if tracks.snapshot()[-1] is not None:
            print("Obstacle detector failed. Staying on the ground.", flush=True)
            return
        if not hover.wait_until_tracking(poses, estimates, land, yaw_locked):
            print("Estimate did not lock onto OptiTrack. Staying on the ground.", flush=True)
            return

        home = follow.read_pose(poses)
        if home is None:
            print("OptiTrack pose was not fresh at arming. Staying on the ground.", flush=True)
            return
        if tracks.snapshot()[-1] is not None:
            print("Obstacle detector failed. Staying on the ground.", flush=True)
            return

        climb = mission.climb
        climb_s = max(1.0, climb / hover.climb_rate_m_s())
        hover_pose = Pose(home.x, home.y, home.z + climb, home.yaw_deg)
        goal = goal_from(hover_pose, mission.goal_right, mission.goal_forward)
        print(
            f"Climbing {climb:.2f} m in {climb_s:.0f} s "
            f"({hover.climb_rate_m_s():.2f} m/s) to z={hover_pose.z:+.2f}.",
            flush=True,
        )
        print(
            f"Then fly to the goal x={goal.x:+.2f} y={goal.y:+.2f} "
            f"({mission.goal_right:.2f} m right, {mission.goal_forward:.2f} m ahead). "
            "Close the window to land.",
            flush=True,
        )
        command.publish(home, "arming")
        cf.commander.send_notify_setpoint_stop()
        if land.wait(0.3):
            return
        cf.supervisor.send_arming_request(True)
        armed = True
        print("Armed. Holding on the ground for 3 seconds.", flush=True)
        if land.wait(3.0):
            print("Window closed before climb. Disarming.", flush=True)
            return

        command.publish(hover_pose, "climbing")
        finished, commanded_z = hover.move_z(
            cf, home.x, home.y, home.z, hover_pose.z, home.yaw_deg, climb_s, land, poses
        )
        sent = Pose(home.x, home.y, commanded_z, home.yaw_deg)
        command.publish(sent, "climbing")
        if finished and not land.is_set():
            print(
                f"In the air. Motion is limited to {follow.MAX_SPEED_M_S:.1f} m/s and "
                f"{follow.MAX_YAW_RATE_DEG_S:.0f} deg/s, inside {fence.max_radius:.1f} m of takeoff.",
                flush=True,
            )
            sent = avoid_until_land(cf, poses, home, sent, goal, tracks, land, fence, command, mission)

        if sent.z > home.z + 0.005:
            descend_s = max(1.0, abs(sent.z - home.z) / hover.climb_rate_m_s())
            print(f"Descending in place to z={home.z:+.2f} in {descend_s:.0f} s.", flush=True)
            command.publish(Pose(sent.x, sent.y, home.z, sent.yaw_deg), "landing")
            hover.move_z(cf, sent.x, sent.y, sent.z, home.z, sent.yaw_deg, descend_s, None, None)

        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.1)
        cf.supervisor.send_arming_request(False)
        armed = False
        print("Landed and disarmed.", flush=True)
    finally:
        if armed:
            try:
                cf.commander.send_notify_setpoint_stop()
                cf.supervisor.send_arming_request(False)
            except Exception:
                print("Could not disarm cleanly. Cut power if the motors are still on.", flush=True)
        landed.set()


def show_and_avoid(
    viewer,
    cf,
    poses,
    estimates,
    state,
    lock,
    land,
    landed,
    yaw_locked,
    scale,
    save,
    takeoff,
    started,
    tracks,
    command,
    fence,
    mission: Mission,
    flip=False,
):
    """Show the livestream. Obstacle boxes replace the raw frame once YOLO is ready."""
    cv2.namedWindow(viewer.WINDOW, cv2.WINDOW_NORMAL)
    shown = 0
    last_seq = 0
    last_det_seq = 0
    sized = False
    hz = 0.0
    last_frame_time = None
    announced = False
    det_announced = False

    while True:
        with lock:
            seq = state["seq"]
            frame = state["frame"] if seq != last_seq else None
            error = state["error"]

        det_seq, plotted, boxes, blocker, det_w, det_h, _stamp, infer_ms, det_error = tracks.snapshot()
        new_plot = det_error is None and plotted is not None and det_seq != last_det_seq

        image = None
        bayer = None
        kind = None
        if new_plot:
            image = plotted
            last_det_seq = det_seq
            if frame is not None:
                last_seq = seq
            kind = f"boxes={len(boxes)}  {infer_ms:.0f} ms"
        elif frame is not None and (plotted is None or det_error is not None):
            width, height, fmt, payload = frame
            last_seq = seq
            try:
                image, bayer = viewer.decode_frame(width, height, fmt, payload, flip)
            except ValueError as exc:
                print(exc, flush=True)
                continue
            kind = "bayer" if fmt == 0 else "jpeg"
        else:
            if error is not None:
                raise error
            if frame is not None:
                last_seq = seq
            if not viewer.pump_window(shown > 0):
                break
            continue

        now = time.perf_counter()
        if last_frame_time is not None and now > last_frame_time:
            instant = 1.0 / (now - last_frame_time)
            hz = instant if hz == 0.0 else (0.9 * hz + 0.1 * instant)
        last_frame_time = now
        shown += 1

        lines = [f"{hz:.1f} Hz  {image.shape[1]}x{image.shape[0]}  {kind}"]
        if det_error is not None:
            lines.append("obstacle detector failed")
        elif not new_plot:
            lines.append("loading obstacle detector")
        else:
            lines.append(obstacle_line(boxes, blocker, det_w, det_h, mission.by_width))
        pose, status = command.snapshot()
        if pose is not None and status:
            lines.append(
                f"{status}  x={pose.x:+.2f}  y={pose.y:+.2f}  z={pose.z:+.2f}  yaw={pose.yaw_deg:+.0f}"
            )
        elif not takeoff:
            lines.append("on the ground")
        if not viewer.present(follow.draw_status(image, lines), scale, sized):
            break
        sized = True
        if save:
            viewer.save_frame(shown, image, bayer)
        if new_plot and not det_announced:
            print("Obstacle boxes are on the live image.", flush=True)
            det_announced = True

        if takeoff and started["flight"] is None:
            print(
                f"Image is live. Climbing {mission.climb:.2f} m, then flying to the goal. "
                "Close the window to land.",
                flush=True,
            )
            flight = threading.Thread(
                target=avoid_loop,
                args=(cf, poses, estimates, land, landed, yaw_locked, tracks, command, fence, mission),
                name="avoid",
                daemon=True,
            )
            started["flight"] = flight
            flight.start()
        elif not takeoff and not announced:
            print("Image is live. Position is streaming. The drone stays on the ground.", flush=True)
            announced = True


def fly(args):
    """Stream the camera and OptiTrack position. Fly to the goal when --hover is set."""
    uri = hover.radio_uri(args.address)
    client_ip = args.client or hover.default_client_ip()
    matrix, path = hover.load_axis_calibration(args.body_id)
    if matrix is None:
        raise SystemExit(
            f"No axis calibration for rigid body {args.body_id}.\n"
            f"Looked for {path.name}."
        )
    print(f"Using {path.name} for position.")
    print(hover.format_axis_map(matrix))
    fence = Fence(args.min_up, args.max_up, args.max_radius)
    mission = Mission(args.goal_right, args.goal_forward, args.all_classes, args.block_height, args.climb)
    print(
        f"Limits: {follow.MAX_SPEED_M_S:.2f} m/s, {follow.MAX_YAW_RATE_DEG_S:.0f} deg/s, "
        f"{fence.max_radius:.2f} m around takeoff, "
        f"height {fence.min_up:.2f} to {fence.max_up:.2f} m above takeoff.",
        flush=True,
    )
    print(
        f"Goal {args.goal_right:.2f} m right and {args.goal_forward:.2f} m ahead of the takeoff heading. "
        f"Blocker: middle {CORRIDOR_FRAC * 100:.0f}% of the image, at least "
        f"{args.block_height * 100:.0f}% of the image tall.",
        flush=True,
    )

    viewer = hover.load_viewer()
    if args.save:
        os.makedirs("stream_out/raw", exist_ok=True)
        os.makedirs("stream_out/debayer", exist_ok=True)
        os.makedirs("stream_out/jpeg", exist_ok=True)

    print(f"Connecting to camera on {args.n}:{args.p}...", flush=True)
    deck = socket.create_connection((args.n, args.p), timeout=5)
    deck.settimeout(None)
    deck.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Camera connected.", flush=True)

    state = {"frame": None, "seq": 0, "received": 0, "error": None}
    frame_lock = threading.Lock()
    stop_stream = threading.Event()
    receiver = threading.Thread(
        target=viewer.receive_loop,
        args=(deck, state, frame_lock, stop_stream),
        name="aideck-rx",
        daemon=True,
    )
    receiver.start()

    tracks = ObstacleTracks()
    command = follow.CommandBuffer()
    detector = threading.Thread(
        target=obstacle_detect_loop,
        args=(viewer, state, frame_lock, tracks, stop_stream, args.conf, args.all_classes, args.block_height, args.flip),
        name="yolo-obstacles",
        daemon=True,
    )
    detector.start()
    print("YOLO nano is loading beside the camera. The window stays on the video until boxes are ready.", flush=True)

    poses = hover.PoseBuffer(args.body_id, matrix)
    estimates = hover.EstimateBuffer()
    print(f"Listening for rigid body {args.body_id} on {hover.MULTICAST_GROUP}:{hover.DATA_PORT}.")
    print(f"  Client {client_ip}    Server {args.server}")
    motive = hover.MotiveListener(client_ip, poses.on_rigid_body)
    started = {"flight": None}
    try:
        motive.start()
        hover.wait_for_motive(motive, poses, args.body_id, args.server)
        hover.lock_wall_heading(poses, matrix)

        hover.cflib.crtp.init_drivers()
        print(f"Connecting to {uri} ...", flush=True)
        land = threading.Event()
        landed = threading.Event()
        stop_tx = threading.Event()
        yaw_locked = threading.Event()
        sender = None
        with hover.SyncCrazyflie(uri, cf=hover.Crazyflie(rw_cache="./cache")) as scf:
            try:
                cf = scf.cf
                hover.ensure_kalman(cf)
                try:
                    cf.param.set_value("locSrv.extQuatStdDev", "0.05")
                except Exception as exc:
                    print(f"Could not set locSrv.extQuatStdDev ({exc}).", flush=True)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not hover.pose_fresh(poses):
                    time.sleep(0.05)
                if not hover.pose_fresh(poses):
                    raise SystemExit(f"No recent packets for rigid body {args.body_id}.")
                sender = threading.Thread(
                    target=hover.send_extpos_loop,
                    args=(cf, poses, estimates, stop_tx, yaw_locked),
                    name="extpos",
                    daemon=True,
                )
                sender.start()
                time.sleep(0.3)
                _motive, _cf_pos, _age, yaw = poses.snapshot()
                hover.reset_kalman(cf, None if yaw is None else math.radians(yaw))
                hover.start_state_log(cf, estimates)
                if args.hover:
                    print(
                        "OptiTrack position is going to the drone. "
                        "The window opens next, then a slow climb and the flight to the goal.",
                        flush=True,
                    )
                else:
                    print("OptiTrack position is going to the drone. Pass --hover to take off.", flush=True)
                show_and_avoid(
                    viewer,
                    cf,
                    poses,
                    estimates,
                    state,
                    frame_lock,
                    land,
                    landed,
                    yaw_locked,
                    args.scale,
                    args.save,
                    args.hover,
                    started,
                    tracks,
                    command,
                    fence,
                    mission,
                    args.flip,
                )
            finally:
                land.set()
                landing_s = 15.0 + args.climb / hover.climb_rate_m_s()
                if started["flight"] is not None and not landed.wait(timeout=landing_s):
                    print("Landing did not finish. Disconnecting.", flush=True)
                stop_tx.set()
                if sender is not None:
                    sender.join(timeout=1.0)
    finally:
        motive.shutdown()
        stop_stream.set()
        deck.close()
        receiver.join(timeout=1.0)
        detector.join(timeout=1.0)
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fly to a goal on the drone's right and go around people that YOLO sees on the way."
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
        "--goal-right",
        type=float,
        default=GOAL_RIGHT_M,
        help=f"Goal distance to the right of the takeoff heading, meters (default {GOAL_RIGHT_M:.1f})",
    )
    parser.add_argument("--goal-forward", type=float, default=0.0, help="Goal distance ahead of the takeoff heading, meters")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Treat every YOLO class as an obstacle and weigh sides by box width",
    )
    parser.add_argument(
        "--block-height",
        type=float,
        default=BLOCK_HEIGHT_FRAC,
        help=f"Smallest box height, as a share of the image, that counts as a blocker (default {BLOCK_HEIGHT_FRAC:.2f})",
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
    if goal_dist < 2.0 * ARRIVE_M:
        raise SystemExit("The goal must be at least 0.25 m from takeoff.")
    if goal_dist > args.max_radius - FENCE_MARGIN_M:
        raise SystemExit(f"The goal is {goal_dist:.2f} m away. Raise --max-radius above {goal_dist + FENCE_MARGIN_M:.2f}.")
    if goal_dist + CLEAR_MAX_M > args.max_radius:
        print(
            f"Warning: --max-radius {args.max_radius:.2f} leaves little room to detour near the goal. "
            f"{goal_dist + CLEAR_MAX_M:.2f} m or more is safer.",
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
        fly(args)
    except KeyboardInterrupt:
        print(flush=True)


if __name__ == "__main__":
    main()
