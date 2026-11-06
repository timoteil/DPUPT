#!/usr/bin/env python3
"""Show the AI-deck camera, send OptiTrack position, and optionally hover.

This is the v2 hover. The hover loop is in this file. It does not import
mocap_extpos or mocap_extpos_lab.

The climb raises the altitude command from z0 to z0 + 0.4 m over 4 seconds.
x, y, and yaw stay at the OptiTrack pose captured at arming.
The hold then repeats (x0, y0, z0 + 0.4, yaw0) for 10 seconds. Descent uses
the same rate back to z0, then the drone disarms. Closing the window
cuts the climb or the hold short and still descends before disarming.

--detection starts a YOLO nano thread on the same frames. The window then
shows that thread's plotted boxes. Without --detection the window is the
video stream only.

    python stream_hover_v2.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
    python stream_hover_v2.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --detection
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket

# OpenCV and PyTorch each ship OpenMP. On Windows the second load aborts
# the process (OMP Error #15) unless this is set before either library loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import struct
import threading
import time
from pathlib import Path

import cflib.crtp
import cv2
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

HOVER_HOLD_S = 10.0
HOVER_CLIMB_M = 0.4
CLIMB_S = 4.0
SETPOINT_HZ = 20.0
SEND_HZ = 50.0
STALE_S = 0.2
TRACK_TOL_M = 0.20
YAW_TOL_DEG = 15.0
KALMAN_ESTIMATOR = 2
BODY_ID = 2
DETECT_CONF = 0.25

MULTICAST_GROUP = "239.255.42.99"
DATA_PORT = 1511
NAT_FRAMEOFDATA = 7
LABELED_MARKER_BYTES = 26

WALL_HOLD_S = 0.6
WALL_MIN_SAMPLES = 5
WALL_STILL_RESULTANT = 0.99

CALIBRATION_DIRS = (Path(__file__).resolve().parent,)


def hover_setpoint(home, yaw):
    """The single hover command. home is (x, y, z) meters, yaw is degrees."""
    x0, y0, z0 = home
    return x0, y0, z0 + HOVER_CLIMB_M, yaw


def ground_setpoint(home, yaw):
    x0, y0, z0 = home
    return x0, y0, z0, yaw


def pose_fresh(poses):
    _motive, cf_pos, age, _yaw = poses.snapshot()
    return cf_pos is not None and age is not None and age <= STALE_S


def climb_rate_m_s():
    return HOVER_CLIMB_M / CLIMB_S


def move_z(cf, x, y, z_from, z_to, yaw, duration, land, poses):
    """Move only the altitude command. Return (finished, last z sent)."""
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(round(duration * SETPOINT_HZ)))
    z = z_from
    for step in range(steps):
        if land is not None and land.is_set():
            return False, z
        if poses is not None and not pose_fresh(poses):
            print("OptiTrack update stopped.", flush=True)
            return False, z
        z = z_from + (z_to - z_from) * ((step + 1) / steps)
        cf.commander.send_position_setpoint(x, y, z, yaw)
        if land is not None and land.wait(period):
            return False, z
        elif land is None:
            time.sleep(period)
    return True, z_to


def repeat_setpoint(cf, x, y, z, yaw, duration, land, poses):
    """Send the same position at SETPOINT_HZ. Return False if the hold should end."""
    period = 1.0 / SETPOINT_HZ
    steps = max(1, int(round(duration * SETPOINT_HZ)))
    for _ in range(steps):
        if land is not None and land.is_set():
            return False
        if poses is not None and not pose_fresh(poses):
            print("OptiTrack update stopped.", flush=True)
            return False
        cf.commander.send_position_setpoint(x, y, z, yaw)
        if land is not None and land.wait(period):
            return False
        elif land is None:
            time.sleep(period)
    return True


def hover_loop(cf, poses, estimates, land, landed, yaw_locked):
    """Arm, climb slowly to z0 + 0.4, hold 10 seconds, then descend at the same rate."""
    armed = False
    try:
        if not wait_until_tracking(poses, estimates, land, yaw_locked):
            print("Estimate did not lock onto OptiTrack. Staying on the ground.", flush=True)
            return

        _motive, home, age, yaw = poses.snapshot()
        if home is None or yaw is None or age is None or age > STALE_S:
            print("OptiTrack pose was not fresh at arming. Staying on the ground.", flush=True)
            return

        hx, hy, hz, hyaw = hover_setpoint(home, yaw)
        gx, gy, gz, gyaw = ground_setpoint(home, yaw)
        print(
            f"Climbing {HOVER_CLIMB_M:.2f} m in {CLIMB_S:.0f} s "
            f"({climb_rate_m_s():.2f} m/s) to z={hz:+.2f}.",
            flush=True,
        )
        print(
            f"Then hold x={hx:+.2f} y={hy:+.2f} z={hz:+.2f} yaw={hyaw:.0f} deg "
            f"for {HOVER_HOLD_S:.0f} s.",
            flush=True,
        )
        cf.commander.send_notify_setpoint_stop()
        if land.wait(0.3):
            return
        cf.supervisor.send_arming_request(True)
        armed = True

        finished, commanded_z = move_z(cf, hx, hy, gz, hz, hyaw, CLIMB_S, land, poses)
        if finished and not land.is_set():
            repeat_setpoint(cf, hx, hy, hz, hyaw, HOVER_HOLD_S, land, poses)
        descend_s = max(1.0, abs(commanded_z - gz) / climb_rate_m_s())
        print(f"Descending to z={gz:+.2f} in {descend_s:.0f} s.", flush=True)
        move_z(cf, gx, gy, commanded_z, gz, gyaw, descend_s, None, None)

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


def tracking_error(poses, estimates):
    _motive, cf_pos, age, _yaw = poses.snapshot()
    estimate, _est_yaw, _roll, _pitch = estimates.snapshot()
    if cf_pos is None or estimate is None or age is None or age > STALE_S:
        return None
    return math.sqrt(
        (estimate[0] - cf_pos[0]) ** 2
        + (estimate[1] - cf_pos[1]) ** 2
        + (estimate[2] - cf_pos[2]) ** 2
    )


def wait_until_tracking(poses, estimates, land, yaw_locked):
    print(
        f"Waiting until position is within {TRACK_TOL_M:.2f} m "
        f"and yaw within {YAW_TOL_DEG:.0f} deg of OptiTrack.",
        flush=True,
    )
    matched_since = None
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and not land.is_set():
        error = tracking_error(poses, estimates)
        _motive, _cf, _age, mocap_yaw = poses.snapshot()
        _estimate, est_yaw, _roll, _pitch = estimates.snapshot()
        yaw_ok = mocap_yaw is not None and abs(wrap_deg(est_yaw - mocap_yaw)) <= YAW_TOL_DEG
        if error is not None and error <= TRACK_TOL_M and yaw_ok:
            if matched_since is None:
                matched_since = time.monotonic()
            elif time.monotonic() - matched_since >= 1.0:
                print(f"Estimate matched OptiTrack (error {error:.3f} m, yaw {est_yaw:.0f} deg).", flush=True)
                yaw_locked.set()
                return True
        else:
            matched_since = None
        land.wait(0.1)
    return False


def send_extpos_loop(cf, poses, estimates, stop, yaw_locked):
    """Send the latest OptiTrack position to the Crazyflie at 50 Hz."""
    period = 1.0 / SEND_HZ
    while not stop.is_set():
        _motive, cf_pos, age, yaw = poses.snapshot()
        if cf_pos is not None and age is not None and age <= STALE_S:
            if yaw is None or yaw_locked.is_set():
                cf.extpos.send_extpos(cf_pos[0], cf_pos[1], cf_pos[2])
            else:
                _estimate, est_yaw, _roll, _pitch = estimates.snapshot()
                if _estimate is None:
                    commanded = yaw
                else:
                    error = wrap_deg(yaw - est_yaw)
                    commanded = wrap_deg(est_yaw + max(-30.0, min(30.0, error)))
                qx, qy, qz, qw = rpy_deg_to_quat(0.0, 0.0, commanded)
                cf.extpos.send_extpose(cf_pos[0], cf_pos[1], cf_pos[2], qx, qy, qz, qw)
        stop.wait(period)


def ensure_kalman(cf):
    estimator = int(float(cf.param.get_value("stabilizer.estimator")))
    if estimator == KALMAN_ESTIMATOR:
        print("Kalman estimator is already active.", flush=True)
        return
    print(f"Switching stabilizer.estimator from {estimator} to Kalman.", flush=True)
    cf.param.set_value("stabilizer.estimator", str(KALMAN_ESTIMATOR))
    time.sleep(0.5)


def reset_kalman(cf, initial_yaw_rad):
    if initial_yaw_rad is not None:
        try:
            cf.param.set_value("kalman.initialYaw", f"{initial_yaw_rad:.6f}")
        except Exception as exc:
            print(f"Could not set kalman.initialYaw ({exc}).", flush=True)
        else:
            print(f"Kalman reset yaw set to {math.degrees(initial_yaw_rad):+.0f} deg.", flush=True)
        time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(0.1)


def start_state_log(cf, estimates):
    state = LogConfig(name="HoverState", period_in_ms=100)
    for name in (
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.yaw",
        "stateEstimate.roll",
        "stateEstimate.pitch",
    ):
        state.add_variable(name, "float")
    cf.log.add_config(state)
    state.data_received_cb.add_callback(estimates.on_log)
    state.start()

    received = LogConfig(name="HoverRx", period_in_ms=100)
    for name in ("locSrv.x", "locSrv.y", "locSrv.z"):
        received.add_variable(name, "float")
    try:
        cf.log.add_config(received)
    except KeyError:
        return
    received.data_received_cb.add_callback(estimates.on_log)
    received.start()


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def motive_to_crazyflie(x, y, z, matrix):
    row_x, row_y, row_z = matrix
    return (
        row_x[0] * x + row_x[1] * y + row_x[2] * z,
        row_y[0] * x + row_y[1] * y + row_y[2] * z,
        row_z[0] * x + row_z[1] * y + row_z[2] * z,
    )


def rpy_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def rotate_body_vector(rot, vector):
    qx, qy, qz, qw = (float(value) for value in rot)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    cx = qy * tz - qz * ty
    cy = qz * tx - qx * tz
    cz = qx * ty - qy * tx
    return (vx + qw * tx + cx, vy + qw * ty + cy, vz + qw * tz + cz)


def heading_of_body_axis(rot, body_axis, yaw_offset_deg, matrix):
    forward = rotate_body_vector(rot, body_axis)
    cf_x, cf_y, _cf_z = motive_to_crazyflie(*forward, matrix)
    return wrap_deg(math.degrees(math.atan2(cf_y, cf_x)) + yaw_offset_deg)


def average_quaternion(rotations):
    if not rotations:
        return None
    reference = tuple(float(value) for value in rotations[0])
    total = [0.0, 0.0, 0.0, 0.0]
    for rot in rotations:
        sample = tuple(float(value) for value in rot)
        if sum(left * right for left, right in zip(sample, reference)) < 0.0:
            sample = tuple(-value for value in sample)
        for index in range(4):
            total[index] += sample[index]
    length = math.sqrt(sum(value * value for value in total))
    if length < 1e-9:
        return None
    return tuple(value / length for value in total)


def _normalize_vector(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-9:
        return None
    return tuple(value / length for value in vector)


def wall_body_axis(rot, nose_motive):
    nose = _normalize_vector(nose_motive)
    if nose is None or rot is None or len(rot) != 4:
        return None
    return _normalize_vector(rotate_body_vector((-rot[0], -rot[1], -rot[2], rot[3]), nose))


def circular_mean_deg(angles):
    if not angles:
        return None, 0.0
    sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
    cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
    count = float(len(angles))
    return wrap_deg(math.degrees(math.atan2(sin_sum, cos_sum))), math.hypot(sin_sum, cos_sum) / count


class PoseBuffer:
    def __init__(self, body_id, matrix):
        self.body_id = body_id
        self.axis_matrix = matrix
        self.body_forward = (1.0, 0.0, 0.0)
        self.yaw_offset_deg = 0.0
        self._wall_heading = False
        self._lock = threading.Lock()
        self.cf = None
        self.yaw = None
        self.rot = None
        self.last_t = 0.0

    def set_wall_heading(self, body_axis, yaw_offset_deg):
        with self._lock:
            self.body_forward = tuple(float(value) for value in body_axis)
            self.yaw_offset_deg = float(yaw_offset_deg)
            self._wall_heading = True
            self.yaw = None

    def on_rigid_body(self, rb_id, pos, rot, markers=None):
        del markers
        if int(rb_id) != self.body_id:
            return
        cf = motive_to_crazyflie(float(pos[0]), float(pos[1]), float(pos[2]), self.axis_matrix)
        stored_rot = tuple(float(value) for value in rot) if rot is not None and len(rot) == 4 else None
        yaw = None
        if stored_rot is not None:
            axis = self.body_forward if self._wall_heading else (1.0, 0.0, 0.0)
            yaw = heading_of_body_axis(stored_rot, axis, self.yaw_offset_deg, self.axis_matrix)
        with self._lock:
            self.cf = cf
            self.yaw = yaw
            self.rot = stored_rot
            self.last_t = time.monotonic()

    def snapshot(self):
        with self._lock:
            age = None if self.cf is None else time.monotonic() - self.last_t
            return None, self.cf, age, self.yaw

    def rotation_snapshot(self):
        with self._lock:
            return self.rot, self.last_t


class EstimateBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self.estimate = None
        self.yaw = 0.0
        self.roll = None
        self.pitch = None

    def on_log(self, _timestamp, data, _logconf):
        with self._lock:
            if "stateEstimate.x" in data:
                self.estimate = (data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"])
                self.yaw = data["stateEstimate.yaw"]
                if "stateEstimate.roll" in data:
                    self.roll = data["stateEstimate.roll"]
                    self.pitch = data["stateEstimate.pitch"]

    def snapshot(self):
        with self._lock:
            return self.estimate, self.yaw, self.roll, self.pitch


class DetectionBuffer:
    """Latest YOLO nano plot. The image is the frame the boxes were drawn on."""

    def __init__(self):
        self._lock = threading.Lock()
        self.seq = 0
        self.image = None
        self.count = 0
        self.infer_ms = 0.0
        self.error = None

    def publish(self, seq, image, count, infer_ms):
        with self._lock:
            self.seq = seq
            self.image = image
            self.count = count
            self.infer_ms = infer_ms

    def fail(self, message):
        with self._lock:
            self.error = message

    def snapshot(self):
        with self._lock:
            return self.seq, self.image, self.count, self.infer_ms, self.error


def calibration_path(body_id):
    name = f"axis_calibration_{int(body_id)}.json"
    for folder in CALIBRATION_DIRS:
        path = folder / name
        if path.exists():
            return path
    legacy = CALIBRATION_DIRS[0] / "axis_calibration.json"
    if legacy.exists():
        return legacy
    return CALIBRATION_DIRS[0] / name


def load_axis_calibration(body_id):
    path = calibration_path(body_id)
    if not path.exists():
        return None, path
    with path.open() as handle:
        data = json.load(handle)
    matrix = tuple(tuple(float(value) for value in row) for row in data["matrix"])
    return matrix, path


def format_axis_map(matrix):
    names = ("X", "Y", "Z")
    lines = []
    for row, name in zip(matrix, names):
        dominant = max(range(3), key=lambda index: abs(row[index]))
        sign = "+" if row[dominant] >= 0.0 else "-"
        lines.append(f"Crazyflie {name} = {sign}OptiTrack {names[dominant]}")
    return "\n".join(lines)


def radio_uri(address):
    raw = address.strip()
    if raw.lower().startswith("0x"):
        raw = raw[2:]
    raw = raw.upper()
    if len(raw) != 10 or any(char not in "0123456789ABCDEF" for char in raw):
        raise SystemExit(f"Bad Crazyflie address {address!r}. Expected 0xE7E7E7E7E7.")
    return f"radio://0/80/2M/{raw}"


def default_client_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "192.168.8.249"
    finally:
        probe.close()


def _i32(data, offset):
    if offset + 4 > len(data):
        raise ValueError("short packet")
    return int.from_bytes(data[offset:offset + 4], byteorder="little", signed=True), offset + 4


def _skip(data, offset, nbytes):
    end = offset + nbytes
    if end > len(data):
        raise ValueError("short packet")
    return end


def _skip_size(data, offset, use_sizes):
    if not use_sizes:
        return offset
    _size, offset = _i32(data, offset)
    return offset


def _bodies_look_real(bodies):
    if not bodies:
        return False
    for body_id, pos, rot in bodies:
        if not 0 <= body_id < 100000:
            return False
        if any(not math.isfinite(value) or abs(value) > 100.0 for value in pos):
            return False
        norm = sum(value * value for value in rot)
        if not 0.5 <= norm <= 1.5:
            return False
    return True


def _markers_look_real(markers):
    for _full_id, model_id, marker_id, pos, _occluded in markers:
        if model_id < 0 or marker_id < 0:
            return False
        if any(not math.isfinite(value) or abs(value) > 100.0 for value in pos):
            return False
    return True


def _skip_marker_sets(data, offset, use_sizes):
    marker_sets, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= marker_sets <= 1000:
        raise ValueError("marker set count")
    for _ in range(marker_sets):
        end = data.index(0, offset)
        offset = end + 1
        count, offset = _i32(data, offset)
        if not 0 <= count <= 10000:
            raise ValueError("marker count")
        offset = _skip(data, offset, 12 * count)
    return offset


def _skip_other_markers(data, offset, use_sizes):
    others, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= others <= 100000:
        raise ValueError("other marker count")
    return _skip(data, offset, 12 * others)


def _read_rigid_bodies(data, offset, use_sizes):
    count, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= count <= 1000:
        raise ValueError("rigid body count")
    bodies = []
    for _ in range(count):
        body_id, offset = _i32(data, offset)
        pos = struct.unpack_from("<fff", data, offset)
        offset += 12
        rot = struct.unpack_from("<ffff", data, offset)
        offset += 16
        offset = _skip(data, offset, 6)
        bodies.append((body_id, pos, rot))
    return bodies, offset


def _skip_sized_block(data, offset, use_sizes):
    count, offset = _i32(data, offset)
    if use_sizes:
        size, offset = _i32(data, offset)
        if size < 0:
            raise ValueError("section size")
        return _skip(data, offset, size)
    if count != 0:
        raise ValueError("non-empty section without a size")
    return offset


def _read_labeled_markers(data, offset, use_sizes):
    offset = _skip_sized_block(data, offset, use_sizes)
    if use_sizes:
        offset = _skip_sized_block(data, offset, use_sizes)
    count, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= count <= 10000:
        raise ValueError("labeled marker count")
    markers = []
    for _ in range(count):
        if offset + LABELED_MARKER_BYTES > len(data):
            raise ValueError("short labeled marker")
        full_id, offset = _i32(data, offset)
        pos = struct.unpack_from("<fff", data, offset)
        offset += 12
        offset = _skip(data, offset, 4)
        param, = struct.unpack_from("<h", data, offset)
        offset += 2
        offset = _skip(data, offset, 4)
        model_id = (full_id >> 16) & 0xFFFF
        marker_id = full_id & 0xFFFF
        markers.append((full_id, model_id, marker_id, pos, (param & 0x01) != 0))
    if not _markers_look_real(markers):
        raise ValueError("labeled markers")
    return markers


def decode_frame(packet, locked_layout):
    if len(packet) < 8:
        return None
    message_id = int.from_bytes(packet[0:2], byteorder="little", signed=True)
    if message_id != NAT_FRAMEOFDATA:
        return None
    payload = packet[4:]
    layouts = (locked_layout,) if locked_layout is not None else (True, False)
    for use_sizes in layouts:
        try:
            bodies, markers = _walk_frame(payload, use_sizes)
        except (ValueError, struct.error, IndexError):
            continue
        if _bodies_look_real(bodies):
            return use_sizes, bodies, markers
    return None


def _walk_frame(data, use_sizes):
    offset = 0
    _frame, offset = _i32(data, offset)
    offset = _skip_marker_sets(data, offset, use_sizes)
    offset = _skip_other_markers(data, offset, use_sizes)
    bodies, offset = _read_rigid_bodies(data, offset, use_sizes)
    try:
        markers = _read_labeled_markers(data, offset, use_sizes)
    except (ValueError, struct.error, IndexError):
        markers = []
    return bodies, markers


class MotiveListener:
    def __init__(self, client_ip, on_rigid_body):
        self.client_ip = client_ip
        self.on_rigid_body = on_rigid_body
        self._stop = threading.Event()
        self._sock = None
        self._thread = None
        self._use_sizes = None
        self._announced = False
        self.packet_count = 0
        self.datagram_count = 0
        self.last_addr = None
        self.first_id = None
        self.first_len = 0
        self.first_hex = ""
        self.seen_ids = []

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", DATA_PORT))
            membership = struct.pack(
                "4s4s",
                socket.inet_aton(MULTICAST_GROUP),
                socket.inet_aton(self.client_ip),
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        except OSError as exc:
            sock.close()
            raise SystemExit(
                f"Could not listen on UDP {DATA_PORT} ({exc}). Close any other NatNet client first."
            ) from exc
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._receive, name="motive-listen", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _receive(self):
        while not self._stop.is_set():
            try:
                packet, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self.datagram_count += 1
            self.last_addr = addr[0]
            if self.first_id is None and len(packet) >= 2:
                self.first_id = int.from_bytes(packet[0:2], byteorder="little", signed=True)
                self.first_len = len(packet)
                self.first_hex = packet[:32].hex()
            decoded = decode_frame(packet, self._use_sizes)
            if decoded is None:
                continue
            layout, bodies, markers = decoded
            if self._use_sizes is None:
                self._use_sizes = layout
            if not self._announced:
                self._announced = True
                ids = ", ".join(str(body_id) for body_id, _pos, _rot in bodies) or "none"
                size_txt = "4.1+ size fields" if layout else "NatNet 3 / 4.0"
                print(
                    f"Motive stream from {addr[0]} ({size_txt}). "
                    f"Rigid bodies: {ids}. Labeled markers: {len(markers)}",
                    flush=True,
                )
            self.packet_count += 1
            for body_id, pos, rot in bodies:
                if body_id not in self.seen_ids:
                    self.seen_ids.append(body_id)
                self.on_rigid_body(body_id, pos, rot, markers)


def lock_wall_heading(poses, matrix):
    print()
    print("Face the nose at the wall you slid toward in calibrate_axes.py.")
    print("Hold the drone level and still. That heading is forward, and the drone flies that way.")
    while True:
        input("Press Enter to capture the heading. ")
        samples = _collect_rotations(poses)
        if samples is None:
            raise SystemExit("Motive sent no rigid-body orientation.")
        mean = average_quaternion(samples)
        body_axis = wall_body_axis(mean, matrix[0])
        if mean is None or body_axis is None:
            raise SystemExit("Could not read a heading from the rigid body.")
        yaws = [heading_of_body_axis(rot, body_axis, 0.0, matrix) for rot in samples]
        _mean_yaw, tightness = circular_mean_deg(yaws)
        if tightness < WALL_STILL_RESULTANT:
            print("The drone was moving. Hold it still and press Enter again.")
            continue
        offset = wrap_deg(-heading_of_body_axis(mean, body_axis, 0.0, matrix))
        poses.set_wall_heading(body_axis, offset)
        break

    deadline = time.monotonic() + 2.0
    yaw = None
    while time.monotonic() < deadline:
        _motive, _cf, _age, yaw = poses.snapshot()
        if yaw is not None:
            break
        time.sleep(0.05)
    if yaw is None:
        raise SystemExit("Heading stopped after the capture. Check the rigid body.")
    print(f"This pose is the wall. Heading {yaw:+.0f} deg.", flush=True)


def _collect_rotations(poses):
    started = time.monotonic()
    seen = set()
    samples = []
    while time.monotonic() < started + WALL_HOLD_S:
        rot, stamp = poses.rotation_snapshot()
        if rot is not None and stamp not in seen:
            seen.add(stamp)
            samples.append(rot)
        time.sleep(0.02)
    if len(samples) < WALL_MIN_SAMPLES:
        return None
    return samples


def wait_for_motive(motive, poses, body_id, server_ip):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and poses.snapshot()[1] is None:
        time.sleep(0.05)
    if poses.snapshot()[1] is not None:
        return
    motive.shutdown()
    if motive.datagram_count == 0:
        raise SystemExit(
            "No UDP packets arrived on port 1511.\n"
            "In Motive set Transmission Type to Multicast.\n"
            "Enable on, Local Interface "
            f"{server_ip}, data port 1511, multicast address {MULTICAST_GROUP}."
        )
    if motive.packet_count == 0:
        raise SystemExit(
            f"Received {motive.datagram_count} UDP packets from {motive.last_addr}, "
            "but none contained a rigid body."
        )
    ids = ", ".join(str(seen) for seen in motive.seen_ids)
    raise SystemExit(f"Motive is streaming rigid bodies {ids}, not {body_id}.")


def nano_weights_path():
    """YOLO11 nano weights shipped next to this file. Startup does not download."""
    return Path(__file__).resolve().parent / "yolo11n.pt"


def default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_nano_model():
    """Load YOLO nano. Import ultralytics only when --detection is set.

    Ultralytics checks DNS while it imports. On the OptiTrack network that
    lookup has no timeout, so the flight thread would sit still after the
    axis map. YOLO_OFFLINE skips the lookup. The weights file is already local.
    """
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path = nano_weights_path()
    if not path.is_file():
        raise SystemExit(f"YOLO weights not found: {path}")
    print(f"Loading YOLO nano {path}.", flush=True)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "YOLO nano needs ultralytics.\n"
            "From this folder, with the virtual environment activated:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    device = default_device()
    print(f"YOLO nano weights are open on {device}.", flush=True)
    return YOLO(str(path)), device, path


def detect_loop(model, viewer, state, lock, detections, stop, device):
    """Run YOLO nano on the newest camera frame and publish result.plot()."""
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
            color, _bayer = viewer.decode_frame(width, height, fmt, payload)
        except ValueError as exc:
            print(exc, flush=True)
            continue
        if stop.is_set():
            return
        started = time.perf_counter()
        try:
            result = model.predict(
                color,
                conf=DETECT_CONF,
                device=device,
                verbose=False,
            )[0]
            plotted = result.plot()
        except Exception as exc:
            print(f"YOLO nano failed: {exc}", flush=True)
            detections.fail(str(exc))
            return
        count = 0 if result.boxes is None else len(result.boxes)
        infer_ms = (time.perf_counter() - started) * 1000.0
        detections.publish(seq, plotted.copy(), count, infer_ms)


def load_viewer():
    path = Path(__file__).resolve().parent / "1_opencv-viewer.py"
    spec = importlib.util.spec_from_file_location("opencv_viewer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def show_and_maybe_hover(
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
    hover,
    started,
    detections=None,
):
    """Show the livestream. With a detection buffer, show YOLO plots instead of raw frames."""
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

        plotted = None
        det_count = 0
        infer_ms = 0.0
        det_error = None
        new_plot = False
        if detections is not None:
            det_seq, plotted, det_count, infer_ms, det_error = detections.snapshot()
            new_plot = det_error is None and plotted is not None and det_seq != last_det_seq

        image = None
        bayer = None
        kind = None
        if new_plot:
            image = plotted
            last_det_seq = det_seq
            if frame is not None:
                last_seq = seq
            kind = f"yolo {det_count}  {infer_ms:.0f} ms"
        elif frame is not None and (detections is None or plotted is None or det_error is not None):
            width, height, fmt, payload = frame
            last_seq = seq
            try:
                image, bayer = viewer.decode_frame(width, height, fmt, payload)
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
        label = f"{hz:.1f} Hz  {image.shape[1]}x{image.shape[0]} {kind}"
        if not viewer.present(viewer.draw_overlay(image, label), scale, sized):
            break
        sized = True
        if save:
            viewer.save_frame(shown, image, bayer)
        if new_plot and not det_announced:
            print("YOLO nano boxes are on the live image.", flush=True)
            det_announced = True

        if hover and started["flight"] is None:
            print(
                f"Image is live. Climbing {HOVER_CLIMB_M:.2f} m over {CLIMB_S:.0f} s. "
                "Close the window to land now.",
                flush=True,
            )
            flight = threading.Thread(
                target=hover_loop,
                args=(cf, poses, estimates, land, landed, yaw_locked),
                name="hover",
                daemon=True,
            )
            started["flight"] = flight
            flight.start()
        elif not hover and not announced:
            print("Image is live. Position is streaming. The drone stays on the ground.", flush=True)
            announced = True


def fly(
    address,
    hover,
    server,
    client=None,
    body_id=None,
    deck_ip="192.168.4.1",
    deck_port=5000,
    scale=2.0,
    save=False,
    detection=False,
):
    """Stream the camera and OptiTrack position. Hover when hover is true.

    address is the Crazyradio address, for example 0xE7E7E7E7E7.
    detection runs YOLO nano on the camera frames and draws those boxes.
    """
    if scale <= 0:
        raise SystemExit("--scale must be positive")
    if body_id is None:
        body_id = BODY_ID
    uri = radio_uri(address)
    client_ip = client or default_client_ip()
    matrix, path = load_axis_calibration(body_id)
    if matrix is None:
        raise SystemExit(
            f"No axis calibration for rigid body {body_id}.\n"
            f"Looked for {path.name} in {path.parent}."
        )
    print(f"Using {path.name} for position.")
    print(format_axis_map(matrix))

    viewer = load_viewer()
    nano = None
    if detection:
        nano = load_nano_model()
    if save:
        os.makedirs("stream_out/raw", exist_ok=True)
        os.makedirs("stream_out/debayer", exist_ok=True)
        os.makedirs("stream_out/jpeg", exist_ok=True)

    print(f"Connecting to camera on {deck_ip}:{deck_port}...", flush=True)
    deck = socket.create_connection((deck_ip, deck_port), timeout=5)
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

    detections = None
    detector = None
    if nano is not None:
        model, device, weights = nano
        detections = DetectionBuffer()
        detector = threading.Thread(
            target=detect_loop,
            args=(model, viewer, state, frame_lock, detections, stop_stream, device),
            name="yolo-nano",
            daemon=True,
        )
        detector.start()
        print(
            f"YOLO nano is reading the camera ({weights.name}, {device}). "
            "The window will show its boxes.",
            flush=True,
        )

    poses = PoseBuffer(body_id, matrix)
    estimates = EstimateBuffer()
    print(f"Listening for rigid body {body_id} on {MULTICAST_GROUP}:{DATA_PORT}.")
    print(f"  Client {client_ip}    Server {server}")
    motive = MotiveListener(client_ip, poses.on_rigid_body)
    started = {"flight": None}
    try:
        motive.start()
        wait_for_motive(motive, poses, body_id, server)
        lock_wall_heading(poses, matrix)

        cflib.crtp.init_drivers()
        print(f"Connecting to {uri} ...", flush=True)
        land = threading.Event()
        landed = threading.Event()
        stop_tx = threading.Event()
        yaw_locked = threading.Event()
        sender = None
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            try:
                cf = scf.cf
                ensure_kalman(cf)
                try:
                    cf.param.set_value("locSrv.extQuatStdDev", "0.05")
                except Exception as exc:
                    print(f"Could not set locSrv.extQuatStdDev ({exc}).", flush=True)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not pose_fresh(poses):
                    time.sleep(0.05)
                if not pose_fresh(poses):
                    raise SystemExit(f"No recent packets for rigid body {body_id}.")
                sender = threading.Thread(
                    target=send_extpos_loop,
                    args=(cf, poses, estimates, stop_tx, yaw_locked),
                    name="extpos",
                    daemon=True,
                )
                sender.start()
                time.sleep(0.3)
                _motive, _cf_pos, _age, yaw = poses.snapshot()
                reset_kalman(cf, None if yaw is None else math.radians(yaw))
                start_state_log(cf, estimates)
                if hover:
                    print(
                        "OptiTrack position is going to the drone. "
                        "The window opens next, then a slow climb.",
                        flush=True,
                    )
                else:
                    print("OptiTrack position is going to the drone. Pass --hover to take off.", flush=True)
                show_and_maybe_hover(
                    viewer,
                    cf,
                    poses,
                    estimates,
                    state,
                    frame_lock,
                    land,
                    landed,
                    yaw_locked,
                    scale,
                    save,
                    hover,
                    started,
                    detections,
                )
            finally:
                land.set()
                if started["flight"] is not None and not landed.wait(timeout=20):
                    print("Landing did not finish. Disconnecting.", flush=True)
                stop_tx.set()
                if sender is not None:
                    sender.join(timeout=1.0)
    finally:
        motive.shutdown()
        stop_stream.set()
        deck.close()
        receiver.join(timeout=1.0)
        if detector is not None:
            detector.join(timeout=1.0)
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the AI-deck stream, draw YOLO nano boxes with --detection, and hover with --hover."
    )
    parser.add_argument("server", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This computer's IP on the OptiTrack network")
    parser.add_argument("--body-id", type=int, default=BODY_ID, help="Motive rigid body id")
    parser.add_argument("--address", required=True, help="Crazyradio address, for example 0xE7E7E7E7E7")
    parser.add_argument(
        "--hover",
        action="store_true",
        help=f"Climb {HOVER_CLIMB_M:.1f} m over {CLIMB_S:.0f} s, hold {HOVER_HOLD_S:.0f} s, then land",
    )
    parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
    parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
    parser.add_argument("--scale", type=float, default=2.0, help="Display scale")
    parser.add_argument("--save", action="store_true", help="Save the frames shown in the window")
    parser.add_argument(
        "--detection",
        action="store_true",
        help="Run YOLO nano on the camera stream and draw its boxes in the window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        fly(
            args.address,
            args.hover,
            args.server,
            client=args.client,
            body_id=args.body_id,
            deck_ip=args.n,
            deck_port=args.p,
            scale=args.scale,
            save=args.save,
            detection=args.detection,
        )
    except KeyboardInterrupt:
        print(flush=True)


if __name__ == "__main__":
    main()
