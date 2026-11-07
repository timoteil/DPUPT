#!/usr/bin/env python3
"""Follow a person with the AI-deck camera.

The camera, OptiTrack pose, and radio setup come from stream_hover_yolo.py.
This file adds the person follower. YOLO keeps only classes whose names
resemble a person (person, people, human, pedestrian, man, woman, child).
track_person() turns the chosen box into the next position setpoint. PIDs on
the horizontal image error slide and yaw to center the box, and a PID on
coverage holds the box near 15% of the image. The I and D gains start tiny.
The flight loop then limits that command to 0.5 m/s, 30 deg/s, and a fence
around the takeoff point.

Pass --hover to climb and follow. With no --hover the window shows the
person boxes and the drone stays on the ground. Closing the window during
flight returns to the takeoff point and lands.

    python follow_me_pid.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
    python follow_me_pid.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys

# Set before cv2. stream_hover_yolo also sets this, but this file imports cv2 first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


def _load_hover():
    """Load 3_stream_hover_yolo.py. A leading digit is not a legal import name."""
    path = Path(__file__).resolve().parent / "3_stream_hover_yolo.py"
    spec = importlib.util.spec_from_file_location("stream_hover_yolo", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hover = _load_hover()

# Same safety shell as mocap_extpos_follow_v3.py. track_person() can ask for
# more; the flight loop applies these limits before the command is sent.
MAX_DT_S = 0.2
MAX_SPEED_M_S = 0.5
MAX_YAW_RATE_DEG_S = 30.0
# Image-x error is about -1..1. Right of center is positive.
# x is sideways speed, m/s, positive to the right. yaw is deg/s.
# I and D are intentionally tiny so you can turn them up while tuning.
KP_X_M_S = 0.02
KI_X = 0.001
KD_X = 0.001
K_YAW_DEG_S = 80.0
KI_YAW = 2
KD_YAW = 0.00
TARGET_COVERAGE = 0.15
# Coverage error (target minus the box fraction) to forward speed, m/s.
K_DIST_M_S = 7.0
KI_DIST = 0.05
KD_DIST = 0.0

# Sums of error, and the previous error, for the three PIDs in track_person.
_i_x = 0.0
_i_yaw = 0.0
_i_dist = 0.0
_prev_x: float | None = None
_prev_yaw: float | None = None
_prev_dist: float | None = None


def _reset_track_pids() -> None:
    """Clear the three PID sums. Used when the person leaves the frame."""
    global _i_x, _i_yaw, _i_dist, _prev_x, _prev_yaw, _prev_dist
    _i_x = _i_yaw = _i_dist = 0.0
    _prev_x = _prev_yaw = _prev_dist = None


_PERSON_CLASS_NAME = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:persons?|people|humans?|pedestrians?|wom[ae]n|men|man|child(?:ren)?|boys?|girls?)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Pose:
    """Crazyflie world pose. x, y, z are meters. yaw_deg is degrees.

    +X is the direction the nose faced during axis calibration.
    +Y is left. +Z is up. Positive yaw turns left, counterclockwise from above.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class Box:
    """One person detection, in raw image pixels. The origin is the top left."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "person"

    @property
    def center(self) -> tuple[float, float]:
        """(u, v). u grows to the right. v grows downward."""
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class Fence:
    """Keep the flying setpoint near the takeoff point."""

    min_up: float
    max_up: float
    max_radius: float


def is_person_class(name: str) -> bool:
    """True when a detector class name is a person, human, or pedestrian."""
    return _PERSON_CLASS_NAME.search(str(name)) is not None


def track_person(
    box: Box | None,
    image_width: int,
    image_height: int,
    measured: Pose,
    setpoint: Pose,
    dt: float,
) -> Pose:
    """Keep the person box centered, and about 15% of the image.

    Sideways speed and yaw are PIDs on the horizontal image error. Forward
    speed is a PID on box coverage. Each integrator is the running sum of
    its error. Each D term is (error - previous error) / dt, so a sudden
    jump in the error kicks the command.

    box is the chosen person, or None when nobody is visible.
    image_width and image_height are the raw frame size in pixels.
    The image center is (image_width / 2, image_height / 2).
    measured is the drone's OptiTrack pose right now.
    setpoint is the pose this function returned on the previous frame.
    On the first frame, setpoint is the hover pose and dt is 0.
    dt is the time since the previous frame, in seconds, and it is at most 0.2.

    Return the next absolute position setpoint. The flight loop sends it with
    send_position_setpoint(x, y, z, yaw_deg). Returning setpoint unchanged
    holds the drone where it is. The caller then limits the command to
    0.5 m/s and 30 deg/s and keeps it inside the takeoff fence.
    """
    # Students implement this function. Until then, the drone holds position.
    # Declare the PID globals before assigning to them:
    #   global _i_x, _i_yaw, _i_dist, _prev_x, _prev_yaw, _prev_dist
    #
    # Inputs
    #   box: the chosen person, or None when nobody is visible.
    #        box.center is (u, v) in pixels. u grows to the right, v grows down.
    #        box.width and box.height are the box size in pixels.
    #   image_width, image_height: raw frame size in pixels.
    #        The image center is (image_width / 2, image_height / 2).
    #   measured: the drone's OptiTrack pose right now.
    #        x, y, z are meters. yaw_deg is degrees.
    #        +X is the nose direction from axis calibration. +Y is left. +Z is up.
    #        Positive yaw turns left, counterclockwise from above.
    #   setpoint: the Pose this function returned on the previous frame.
    #        On the first frame it is the hover pose.
    #   dt: seconds since the previous frame. 0 on the first frame, and at most 0.2.
    #
    # Output
    #   The next absolute Pose (x, y, z, yaw_deg).
    #   Return setpoint unchanged to hold. The caller then limits the command
    #   to 0.5 m/s and 30 deg/s and keeps it inside the takeoff fence.
    #
    # PID state, already declared above this function:
    #   _i_x, _i_yaw, _i_dist
    #       running sums of each error. This is the I term.
    #   _prev_x, _prev_yaw, _prev_dist
    #       the previous error, or None before the first sample.
    #       The D term is (error - previous error) / dt. Use 0 when previous is None.
    #   _reset_track_pids() clears all six. Call it when the person leaves.
    #
    # Constants already defined above this function:
    #   KP_X_M_S, KI_X, KD_X
    #       sideways speed, m/s. Positive slides to the drone's right.
    #   K_YAW_DEG_S, KI_YAW, KD_YAW
    #       yaw rate, deg/s.
    #   TARGET_COVERAGE
    #       desired box area as a fraction of the image (0.15).
    #   K_DIST_M_S, KI_DIST, KD_DIST
    #       forward speed, m/s. Positive flies toward the person.
    #
    # Each command is kp * error + ki * sum + kd * derivative.
    #
    # 1. No person, or an image size that is not positive: call
    #    _reset_track_pids() and return setpoint.
    #
    # 2. First frame, dt <= 0: return setpoint. Do not update the PID state.
    #
    # 3. Horizontal error. Subtract the image-center x from the box-center x
    #    and divide by the image-center x. Right of center is positive, and
    #    the result is about -1 to 1. The sideways and yaw PIDs both use it.
    #
    # 4. Sideways PID. Add the horizontal error into _i_x. D is 0 when
    #    _prev_x is None, otherwise (error - _prev_x) / dt. Then store the
    #    error in _prev_x.
    #    v_right = KP_X_M_S * error + KI_X * _i_x + KD_X * d_x.
    #    Positive v_right slides the drone to its right, which is the side
    #    the person is on.
    #
    # 5. Yaw PID on the same horizontal error, using _i_yaw and _prev_yaw.
    #    yaw_rate = K_YAW_DEG_S * error + KI_YAW * _i_yaw + KD_YAW * d_yaw.
    #    A person on the right must turn the nose right. Positive yaw is a
    #    left turn, so new_yaw = setpoint.yaw_deg - yaw_rate * dt.
    #
    # 6. Distance error. Coverage is box area divided by image area.
    #    err_dist = TARGET_COVERAGE - coverage. Positive means the box is
    #    smaller than 15% of the image. Same PID pattern with _i_dist and
    #    _prev_dist:
    #    v_forward = K_DIST_M_S * err_dist + KI_DIST * _i_dist + KD_DIST * d_dist.
    #
    # 7. World step. measured.yaw_deg is the nose heading. Convert it to
    #    radians. Body forward is the nose. Body right is -Y, because +Y is left.
    #      dx = (v_forward * cos(yaw) + v_right * sin(yaw)) * dt
    #      dy = (v_forward * sin(yaw) - v_right * cos(yaw)) * dt
    #    Leave z at setpoint.z.
    #
    # 8. Return Pose(setpoint.x + dx, setpoint.y + dy, setpoint.z, new_yaw).
    #    If x, y, z, or yaw is not finite, return setpoint instead.
    return setpoint


def _finite_pose(pose: Pose) -> bool:
    return all(math.isfinite(value) for value in (pose.x, pose.y, pose.z, pose.yaw_deg))


def clamp_to_fence(command: Pose, home: Pose, fence: Fence) -> Pose:
    """Keep a flying setpoint near the takeoff point."""
    dx = command.x - home.x
    dy = command.y - home.y
    dist = math.hypot(dx, dy)
    if dist > fence.max_radius > 0.0:
        scale = fence.max_radius / dist
        dx *= scale
        dy *= scale
    z = min(max(command.z, home.z + fence.min_up), home.z + fence.max_up)
    return Pose(home.x + dx, home.y + dy, z, hover.wrap_deg(command.yaw_deg))


def approach(previous: Pose, command: Pose, dt: float) -> Pose:
    """Move previous toward command, no faster than the safety speed.

    dt is the frame interval. A zero interval holds position so a duplicate
    frame cannot jump the full command.
    """
    step = max(0.0, dt)
    if step <= 0.0:
        return previous
    dist = math.hypot(command.x - previous.x, command.y - previous.y)
    max_step = MAX_SPEED_M_S * step
    if dist > max_step > 0.0:
        scale = max_step / dist
        x = previous.x + (command.x - previous.x) * scale
        y = previous.y + (command.y - previous.y) * scale
    else:
        x = command.x
        y = command.y
    z_step = MAX_SPEED_M_S * step
    z = previous.z + max(-z_step, min(z_step, command.z - previous.z))
    yaw_step = MAX_YAW_RATE_DEG_S * step
    yaw_error = hover.wrap_deg(command.yaw_deg - previous.yaw_deg)
    yaw = hover.wrap_deg(previous.yaw_deg + max(-yaw_step, min(yaw_step, yaw_error)))
    return Pose(x, y, z, yaw)


class FollowController:
    """Integrate track_person() once per new camera frame, then limit it."""

    def __init__(self, initial: Pose, home: Pose, fence: Fence):
        self.setpoint = initial
        self.home = home
        self.fence = fence
        self._last_stamp: float | None = None
        _reset_track_pids()

    def step(self, box: Box | None, width: int, height: int, measured: Pose, stamp: float) -> Pose:
        if self._last_stamp is None:
            dt = 0.0
        else:
            dt = min(MAX_DT_S, max(0.0, stamp - self._last_stamp))
        self._last_stamp = stamp
        command = track_person(box, width, height, measured, self.setpoint, dt)
        self.setpoint = approach(self.setpoint, clamp_to_fence(command, self.home, self.fence), dt)
        return self.setpoint


def read_pose(poses) -> Pose | None:
    _motive, cf_pos, age, yaw = poses.snapshot()
    if cf_pos is None or yaw is None or age is None or age > hover.STALE_S:
        return None
    pose = Pose(float(cf_pos[0]), float(cf_pos[1]), float(cf_pos[2]), float(yaw))
    if not _finite_pose(pose):
        return None
    return pose


def person_classes(model) -> tuple[list[int], str]:
    """Class ids whose names resemble a person. The rest of YOLO is dropped."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        pairs = [(int(cls_id), str(name)) for cls_id, name in names.items()]
    elif names:
        pairs = [(index, str(name)) for index, name in enumerate(names)]
    else:
        pairs = []
    kept = [(cls_id, name) for cls_id, name in pairs if is_person_class(name)]
    kept.sort(key=lambda item: item[0])
    if not kept:
        raise SystemExit(
            "This YOLO model has no class that resembles a person. "
            "Expected a name such as person, people, human, or pedestrian."
        )
    ids = [cls_id for cls_id, _name in kept]
    labels = ", ".join(f"{cls_id}:{name}" for cls_id, name in kept)
    return ids, labels


def class_label(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, "person"))
    try:
        return str(names[cls_id])
    except (IndexError, KeyError, TypeError):
        return "person"


def make_box(x1, y1, x2, y2, confidence: float, class_name: str, width: int, height: int) -> Box | None:
    if width <= 0 or height <= 0:
        return None
    left = max(0, min(width, int(round(float(x1)))))
    top = max(0, min(height, int(round(float(y1)))))
    right = max(0, min(width, int(round(float(x2)))))
    bottom = max(0, min(height, int(round(float(y2)))))
    if right <= left or bottom <= top:
        return None
    return Box(left, top, right, bottom, float(confidence), class_name)


def boxes_from_result(result, names, allowed: set[int], width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    raw = getattr(result, "boxes", None)
    if raw is None or len(raw) == 0:
        return boxes
    xyxy = raw.xyxy.cpu().numpy()
    confs = raw.conf.cpu().numpy()
    classes = raw.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
        cls_id = int(cls_id)
        if cls_id not in allowed:
            continue
        box = make_box(x1, y1, x2, y2, float(conf), class_label(names, cls_id), width, height)
        if box is not None:
            boxes.append(box)
    return boxes


def select_person(boxes: list[Box]) -> Box | None:
    """The person the controller should track. Highest confidence, then tallest box."""
    if not boxes:
        return None
    return max(boxes, key=lambda box: (box.confidence, box.height))


def draw_people(image, boxes: list[Box], selected: Box | None):
    """Draw person boxes on a copy of the camera frame."""
    view = image.copy()
    height, width = view.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for box in boxes:
        chosen = box is selected
        color = (0, 220, 255) if chosen else (80, 220, 80)
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
        if chosen:
            u, v = box.center
            cv2.line(view, center, (int(round(u)), int(round(v))), color, 1)
    return view


def draw_status(image, lines: list[str]):
    view = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, text in enumerate(lines):
        origin = (8, 16 + index * 16)
        cv2.putText(view, text, origin, font, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, origin, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def tracking_line(box: Box | None, width: int, height: int) -> str:
    if box is None or width <= 0 or height <= 0:
        return "no person, holding position"
    center_x = width / 2.0
    u, _v = box.center
    err_x = (u - center_x) / center_x
    coverage = (box.width * box.height) / float(width * height)
    return (
        f"{box.class_name} {box.confidence:.2f}  "
        f"cov={coverage * 100:.0f}%  target={TARGET_COVERAGE * 100:.0f}%  err_x={err_x:+.2f}"
    )


class PersonTracks:
    """Latest person detection. image has the boxes. clean is the camera frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self.seq = 0
        self.image = None
        self.clean = None
        self.box: Box | None = None
        self.count = 0
        self.width = 0
        self.height = 0
        self.stamp: float | None = None
        self.infer_ms = 0.0
        self.error: str | None = None

    def publish(self, seq, image, clean, box, count, width, height, stamp, infer_ms):
        with self._lock:
            self.seq = seq
            self.image = image
            self.clean = clean
            self.box = box
            self.count = count
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
                self.clean,
                self.box,
                self.count,
                self.width,
                self.height,
                self.stamp,
                self.infer_ms,
                self.error,
            )


class CommandBuffer:
    """Setpoint the flight thread is sending, for the camera window."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pose: Pose | None = None
        self.status = ""

    def publish(self, pose: Pose, status: str):
        with self._lock:
            self.pose = pose
            self.status = status

    def snapshot(self):
        with self._lock:
            return self.pose, self.status


def person_detect_loop(viewer, state, lock, tracks: PersonTracks, stop, conf: float, flip: bool = False):
    """Load YOLO nano and publish person boxes until stop is set."""
    try:
        model, device, _weights = hover.load_nano_model()
        class_ids, labels = person_classes(model)
    except SystemExit as exc:
        print(exc, flush=True)
        tracks.fail(str(exc))
        return
    except Exception as exc:
        print(f"YOLO nano failed: {exc}", flush=True)
        tracks.fail(str(exc))
        return
    print(f"YOLO classes kept for follow: {labels}.", flush=True)
    allowed = set(class_ids)
    names = model.names
    last_seq = 0
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
        boxes = boxes_from_result(result, names, allowed, color.shape[1], color.shape[0])
        selected = select_person(boxes)
        view = draw_people(color, boxes, selected)
        infer_ms = (time.perf_counter() - started) * 1000.0
        tracks.publish(
            seq, view, color, selected, len(boxes), color.shape[1], color.shape[0], taken, infer_ms
        )


def follow_until_land(cf, poses, home: Pose, initial: Pose, tracks: PersonTracks, land, fence: Fence, command: CommandBuffer) -> Pose:
    """Send track_person() until the window closes or OptiTrack goes stale."""
    controller = FollowController(initial, home, fence)
    last_seq = 0
    period = 1.0 / hover.SETPOINT_HZ
    setpoint = initial
    announced_hold = False
    detector_lost = False
    while not land.is_set():
        if not hover.pose_fresh(poses):
            print("OptiTrack update stopped.", flush=True)
            return setpoint
        seq, _image, _clean, box, _count, width, height, stamp, _infer_ms, error = tracks.snapshot()
        measured = read_pose(poses)
        if error is not None and not detector_lost:
            print("Person detector stopped. Holding position.", flush=True)
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
            setpoint = controller.step(box, width, height, measured, stamp)
            if box is None and not announced_hold:
                print("No person in view. Holding position.", flush=True)
                announced_hold = True
            elif box is not None and announced_hold:
                print("Person back in view.", flush=True)
                announced_hold = False
        status = "holding" if box is None or error is not None else "following"
        cf.commander.send_position_setpoint(setpoint.x, setpoint.y, setpoint.z, setpoint.yaw_deg)
        command.publish(setpoint, status)
        if land.wait(period):
            return setpoint
    return setpoint


def slide_home(cf, sent: Pose, home: Pose, command: CommandBuffer) -> Pose:
    """Translate back to the takeoff point at the safety speed. Altitude stays."""
    dist = math.hypot(sent.x - home.x, sent.y - home.y)
    if dist < 0.02:
        return sent
    duration = max(1.0, dist / MAX_SPEED_M_S)
    print(f"Returning to the takeoff point in {duration:.0f} s.", flush=True)
    period = 1.0 / hover.SETPOINT_HZ
    steps = max(1, int(round(duration * hover.SETPOINT_HZ)))
    for step in range(steps):
        alpha = (step + 1) / steps
        pose = Pose(
            sent.x + (home.x - sent.x) * alpha,
            sent.y + (home.y - sent.y) * alpha,
            sent.z,
            sent.yaw_deg,
        )
        cf.commander.send_position_setpoint(pose.x, pose.y, pose.z, pose.yaw_deg)
        command.publish(pose, "returning")
        time.sleep(period)
    return Pose(home.x, home.y, sent.z, sent.yaw_deg)


def follow_loop(cf, poses, estimates, land, landed, yaw_locked, tracks, command, fence, climb):
    """Arm, climb, follow a person, return home, then descend and disarm."""
    armed = False
    try:
        if tracks.snapshot()[-1] is not None:
            print("Person detector failed. Staying on the ground.", flush=True)
            return
        if not hover.wait_until_tracking(poses, estimates, land, yaw_locked):
            print("Estimate did not lock onto OptiTrack. Staying on the ground.", flush=True)
            return

        home = read_pose(poses)
        if home is None:
            print("OptiTrack pose was not fresh at arming. Staying on the ground.", flush=True)
            return
        if tracks.snapshot()[-1] is not None:
            print("Person detector failed. Staying on the ground.", flush=True)
            return

        climb_s = max(1.0, climb / hover.climb_rate_m_s())
        hover_pose = Pose(home.x, home.y, home.z + climb, home.yaw_deg)
        print(
            f"Climbing {climb:.2f} m in {climb_s:.0f} s "
            f"({hover.climb_rate_m_s():.2f} m/s) to z={hover_pose.z:+.2f}.",
            flush=True,
        )
        print(
            f"Then follow from x={hover_pose.x:+.2f} y={hover_pose.y:+.2f} "
            f"z={hover_pose.z:+.2f} yaw={hover_pose.yaw_deg:.0f} deg. "
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
                f"In the air. Motion is limited to {MAX_SPEED_M_S:.1f} m/s and "
                f"{MAX_YAW_RATE_DEG_S:.0f} deg/s, inside {fence.max_radius:.1f} m of takeoff. "
                "Close the window to land.",
                flush=True,
            )
            sent = follow_until_land(cf, poses, home, sent, tracks, land, fence, command)

        if sent.z > home.z + 0.005 or math.hypot(sent.x - home.x, sent.y - home.y) >= 0.02:
            sent = slide_home(cf, sent, home, command)
            descend_s = max(1.0, abs(sent.z - home.z) / hover.climb_rate_m_s())
            print(f"Descending to z={home.z:+.2f} in {descend_s:.0f} s.", flush=True)
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


def show_and_follow(
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
    climb,
    flip=False,
):
    """Show the livestream. Person boxes replace the raw frame once YOLO is ready."""
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

        det_seq, plotted, clean, box, det_count, det_w, det_h, _stamp, infer_ms, det_error = tracks.snapshot()
        new_plot = det_error is None and plotted is not None and det_seq != last_det_seq

        image = None
        bayer = None
        kind = None
        overlay_box = None
        overlay_size = (0, 0)
        if new_plot:
            image = plotted
            last_det_seq = det_seq
            if frame is not None:
                last_seq = seq
            kind = f"people={det_count}  {infer_ms:.0f} ms"
            overlay_box = box
            overlay_size = (det_w, det_h)
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
            lines.append("person detector failed")
        elif not new_plot:
            lines.append("loading person detector")
        else:
            lines.append(tracking_line(overlay_box, overlay_size[0], overlay_size[1]))
        pose, status = command.snapshot()
        if pose is not None and status:
            lines.append(
                f"{status}  x={pose.x:+.2f}  y={pose.y:+.2f}  z={pose.z:+.2f}  yaw={pose.yaw_deg:+.0f}"
            )
        elif not takeoff:
            lines.append("on the ground")
        if not viewer.present(draw_status(image, lines), scale, sized):
            break
        sized = True
        if save:
            # The window keeps the boxes and status text. The file is the camera frame.
            viewer.save_frame(shown, clean if new_plot and clean is not None else image, None)
        if new_plot and not det_announced:
            print("Person boxes are on the live image.", flush=True)
            det_announced = True

        if takeoff and started["flight"] is None:
            print(
                f"Image is live. Climbing {climb:.2f} m, then following the person. "
                "Close the window to land.",
                flush=True,
            )
            flight = threading.Thread(
                target=follow_loop,
                args=(cf, poses, estimates, land, landed, yaw_locked, tracks, command, fence, climb),
                name="follow",
                daemon=True,
            )
            started["flight"] = flight
            flight.start()
        elif not takeoff and not announced:
            print("Image is live. Position is streaming. The drone stays on the ground.", flush=True)
            announced = True


def fly(args):
    """Stream the camera and OptiTrack position. Follow a person when --hover is set."""
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
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
    print(
        f"Follow limits: {MAX_SPEED_M_S:.2f} m/s, {MAX_YAW_RATE_DEG_S:.0f} deg/s, "
        f"{fence.max_radius:.2f} m around takeoff, "
        f"height {fence.min_up:.2f} to {fence.max_up:.2f} m above takeoff.",
        flush=True,
    )
    print(
        f"Climb {args.climb:.2f} m at {hover.climb_rate_m_s():.2f} m/s, then keep the person near "
        f"{TARGET_COVERAGE * 100:.0f}% of the image.",
        flush=True,
    )
    print(
        f"PID  x kp={KP_X_M_S:g} ki={KI_X:g} kd={KD_X:g}   "
        f"yaw kp={K_YAW_DEG_S:g} ki={KI_YAW:g} kd={KD_YAW:g}   "
        f"forward kp={K_DIST_M_S:g} ki={KI_DIST:g} kd={KD_DIST:g}",
        flush=True,
    )

    viewer = hover.load_viewer()
    if args.save:
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

    tracks = PersonTracks()
    command = CommandBuffer()
    detector = threading.Thread(
        target=person_detect_loop,
        args=(viewer, state, frame_lock, tracks, stop_stream, args.conf, args.flip),
        name="yolo-person",
        daemon=True,
    )
    detector.start()
    print("YOLO nano is loading beside the camera. The window stays on the video until person boxes are ready.", flush=True)

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
                        "The window opens next, then a slow climb and person follow.",
                        flush=True,
                    )
                else:
                    print("OptiTrack position is going to the drone. Pass --hover to take off.", flush=True)
                show_and_follow(
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
                    args.climb,
                    args.flip,
                )
            finally:
                land.set()
                landing_s = 15.0 + args.climb / hover.climb_rate_m_s() + args.max_radius / MAX_SPEED_M_S
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
        description="Show the AI-deck stream, track a person with YOLO, and follow that person with --hover."
    )
    parser.add_argument("server", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This computer's IP on the OptiTrack network")
    parser.add_argument("--body-id", type=int, default=hover.BODY_ID, help="Motive rigid body id")
    parser.add_argument("--address", required=True, help="Crazyradio address, for example 0xE7E7E7E7E7")
    parser.add_argument(
        "--hover",
        action="store_true",
        help="Climb, follow the person, then land when the window closes",
    )
    parser.add_argument(
        "--climb",
        type=float,
        default=hover.HOVER_CLIMB_M,
        help=f"Meters above takeoff before following (default {hover.HOVER_CLIMB_M:.1f})",
    )
    parser.add_argument("--min-up", type=float, default=0.25, help="Lowest flying height above takeoff, meters")
    parser.add_argument("--max-up", type=float, default=1.5, help="Highest flying height above takeoff, meters")
    parser.add_argument("--max-radius", type=float, default=6.0, help="Horizontal fence around takeoff, meters")
    parser.add_argument("--conf", type=float, default=hover.DETECT_CONF, help="Minimum person detection confidence")
    parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
    parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
    parser.add_argument("--scale", type=float, default=2.0, help="Display scale")
    parser.add_argument("--save", action="store_true", help="Save the camera frames, without boxes or text")
    parser.add_argument("--flip", action="store_true", help="Rotate the camera image 180 degrees")
    args = parser.parse_args()
    if args.climb <= 0.0 or args.min_up <= 0.0 or args.max_up <= 0.0 or args.max_radius <= 0.0:
        raise SystemExit("--climb, --min-up, --max-up, and --max-radius must be positive.")
    if not (args.min_up <= args.climb <= args.max_up):
        raise SystemExit("--climb must sit between --min-up and --max-up.")
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
