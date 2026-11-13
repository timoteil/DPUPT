#!/usr/bin/env python3
"""Build a replay video from a --save-imgs folder.

The left window plays the camera frames. The right window draws the
OptiTrack position in 3D up to the same timestamp as the frame on screen.
Playback follows the recorded arrival times.

    python 8_pose_video.py stream_out/imgs/1794487743345
    python 8_pose_video.py stream_out/imgs/1794487743345 --fps 20 --speed 1
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

WINDOW_TITLE_H = 28
MARGIN = 16
GAP = 12
STATUS_H = 42
CAMERA_SCALE = 2
PANEL_BG = (245, 245, 245)
# OpenCV colors are BGR. This is a dark slate title bar.
TITLE_BG = (90, 62, 36)
TITLE_FG = (255, 255, 255)
STATUS_FG = (20, 20, 20)
PATH_COLOR = "#1f4e79"
NOW_COLOR = "#c0392b"
START_COLOR = "#1e8449"
SHADOW_COLOR = "#8d8d8d"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a video with the camera frames and the OptiTrack position in 3D."
    )
    parser.add_argument(
        "session",
        type=Path,
        help="Folder from --save-imgs, or the positions.csv inside it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Video path. Default: <session>/pose_video.mp4",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Output frames per second")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed. 1 follows the recorded arrival times",
    )
    return parser.parse_args()


def session_folder(path):
    path = path.expanduser().resolve()
    if path.is_file():
        path = path.parent
    if not (path / "positions.csv").is_file():
        raise SystemExit(f"No positions.csv in {path}")
    return path


def image_file(folder, stamp_ms):
    for suffix in (".jpg", ".jpeg", ".png"):
        path = folder / f"{stamp_ms}{suffix}"
        if path.is_file():
            return path
    return None


def optional_float(text):
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


def load_samples(folder):
    samples = []
    missing = 0
    with (folder / "positions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            stamp_ms = int(row["timestamp_ms"])
            path = image_file(folder, stamp_ms)
            if path is None:
                missing += 1
                continue
            position = None
            x = optional_float(row.get("x"))
            y = optional_float(row.get("y"))
            z = optional_float(row.get("z"))
            if x is not None and y is not None and z is not None:
                position = (x, y, z)
            samples.append((stamp_ms, path, position, optional_float(row.get("yaw_deg"))))
    if missing:
        print(f"Skipped {missing} row(s) with no image file.", flush=True)
    if not samples:
        raise SystemExit(f"No images matched positions.csv in {folder}")
    samples.sort(key=lambda sample: sample[0])
    return samples


def axis_limits(values):
    """Limits with equal padding. A short axis still gets enough room to see."""
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 0.5)
    mid = (lo + hi) / 2.0
    half = span / 2.0 + span * 0.12
    return mid - half, mid + half


def set_line_3d(line, xs, ys, zs):
    if hasattr(line, "set_data_3d"):
        line.set_data_3d(xs, ys, zs)
        return
    line.set_data(xs, ys)
    line.set_3d_properties(zs)


class PositionView:
    """One 3D axes. render() draws the path from the start through pose_index."""

    def __init__(self, samples):
        positions = [sample[2] for sample in samples if sample[2] is not None]
        if not positions:
            raise SystemExit("positions.csv has no x, y, z values.")
        self.xs = [position[0] for position in positions]
        self.ys = [position[1] for position in positions]
        self.zs = [position[2] for position in positions]
        self.yaws = [yaw for _stamp, _path, position, yaw in samples if position is not None]
        self.pose_index = []
        seen = -1
        for _stamp, _path, position, _yaw in samples:
            if position is not None:
                seen += 1
            self.pose_index.append(seen)

        x_lo, x_hi = axis_limits(self.xs)
        y_lo, y_hi = axis_limits(self.ys)
        z_lo, z_hi = axis_limits(self.zs)
        self.floor_z = z_lo
        span = max(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo)
        self.arrow_m = max(0.3, 0.22 * span)

        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
        self.fig = plt.figure(figsize=(6.4, 4.88), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.patch.set_facecolor("#f5f5f5")
        self.ax.set_facecolor("#f5f5f5")
        self.ax.set_xlim(x_lo, x_hi)
        self.ax.set_ylim(y_lo, y_hi)
        self.ax.set_zlim(z_lo, z_hi)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.view_init(elev=24, azim=-58)
        if hasattr(self.ax, "set_proj_type"):
            self.ax.set_proj_type("ortho")
        if hasattr(self.ax, "set_box_aspect"):
            self.ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
        self.ax.tick_params(labelsize=8, pad=2)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_major_formatter(FormatStrFormatter("%.1f"))
            axis.set_major_locator(MaxNLocator(nbins=5))
        self.ax.xaxis.labelpad = 6
        self.ax.yaxis.labelpad = 8
        self.ax.zaxis.labelpad = 8
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.98)

        self.shadow, = self.ax.plot([], [], [], color=SHADOW_COLOR, lw=1.0, label="ground track")
        self.trail, = self.ax.plot([], [], [], color=PATH_COLOR, lw=2.0, label="path")
        self.drop, = self.ax.plot([], [], [], color=SHADOW_COLOR, lw=1.0, label="_nolegend_")
        self.heading, = self.ax.plot([], [], [], color=NOW_COLOR, lw=1.6, label="heading")
        self.ax.plot(
            [self.xs[0]],
            [self.ys[0]],
            [self.zs[0]],
            linestyle="None",
            marker="o",
            color=START_COLOR,
            markersize=5,
            label="start",
        )
        self.point, = self.ax.plot(
            [],
            [],
            [],
            linestyle="None",
            marker="o",
            color=NOW_COLOR,
            markersize=7,
            label="now",
        )
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self._cache = {}

    def render(self, pose_index):
        if pose_index in self._cache:
            return self._cache[pose_index]
        if pose_index < 0:
            set_line_3d(self.trail, [], [], [])
            set_line_3d(self.shadow, [], [], [])
            set_line_3d(self.drop, [], [], [])
            set_line_3d(self.heading, [], [], [])
            set_line_3d(self.point, [], [], [])
        else:
            end = pose_index + 1
            xs = self.xs[:end]
            ys = self.ys[:end]
            zs = self.zs[:end]
            floor = [self.floor_z] * len(xs)
            set_line_3d(self.trail, xs, ys, zs)
            set_line_3d(self.shadow, xs, ys, floor)
            x, y, z = xs[-1], ys[-1], zs[-1]
            set_line_3d(self.point, [x], [y], [z])
            set_line_3d(self.drop, [x, x], [y, y], [z, self.floor_z])
            yaw = self.yaws[pose_index]
            if yaw is None:
                set_line_3d(self.heading, [], [], [])
            else:
                rad = math.radians(yaw)
                set_line_3d(
                    self.heading,
                    [x, x + self.arrow_m * math.cos(rad)],
                    [y, y + self.arrow_m * math.sin(rad)],
                    [z, z],
                )
        self.fig.canvas.draw()
        rgba = np.array(self.fig.canvas.buffer_rgba())
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        self._cache[pose_index] = image
        return image

    def close(self):
        plt.close(self.fig)


def playback_runs(samples, fps, speed):
    """Repeat each image so the video lasts as long as the recording, scaled by speed."""
    stamps = [sample[0] for sample in samples]
    span_ms = stamps[-1] - stamps[0]
    duration_s = 0.0 if speed <= 0 else (span_ms / 1000.0) / speed
    count = 1 if duration_s <= 0 else max(2, int(round(duration_s * fps)))
    runs = []
    for step in range(count):
        alpha = 0.0 if count == 1 else step / (count - 1)
        stamp = stamps[0] + alpha * span_ms
        index = bisect.bisect_right(stamps, stamp) - 1
        if index < 0:
            index = 0
        if runs and runs[-1][0] == index:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    return count, duration_s, runs


def draw_centered(image, origin, size, text, color):
    x, y = origin
    width, height = size
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), _baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = x + (width - text_w) // 2
    ty = y + (height + text_h) // 2
    cv2.putText(image, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)


def status_text(sample, t0_ms):
    elapsed = (sample[0] - t0_ms) / 1000.0
    position = sample[2]
    if position is None:
        return f"t = {elapsed:6.2f} s     no OptiTrack position for this image"
    x, y, z = position
    yaw = sample[3]
    yaw_text = "yaw = n/a" if yaw is None else f"yaw = {yaw:7.1f} deg"
    return (
        f"t = {elapsed:6.2f} s     "
        f"x = {x:7.3f} m     y = {y:7.3f} m     z = {z:7.3f} m     {yaw_text}"
    )


def compose(camera, plot, status):
    camera_h, camera_w = camera.shape[:2]
    plot_h, plot_w = plot.shape[:2]
    panel_h = max(camera_h, plot_h)
    if camera_h != panel_h:
        camera = cv2.resize(camera, (int(round(camera_w * panel_h / camera_h)), panel_h))
        camera_h, camera_w = camera.shape[:2]
    if plot_h != panel_h:
        plot = cv2.resize(plot, (int(round(plot_w * panel_h / plot_h)), panel_h))
        plot_h, plot_w = plot.shape[:2]

    width = MARGIN + camera_w + GAP + plot_w + MARGIN
    height = MARGIN + WINDOW_TITLE_H + panel_h + STATUS_H + MARGIN
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)

    camera_x = MARGIN
    plot_x = MARGIN + camera_w + GAP
    title_y = MARGIN
    image_y = MARGIN + WINDOW_TITLE_H
    cv2.rectangle(
        canvas,
        (camera_x, title_y),
        (camera_x + camera_w, title_y + WINDOW_TITLE_H),
        TITLE_BG,
        -1,
    )
    cv2.rectangle(
        canvas,
        (plot_x, title_y),
        (plot_x + plot_w, title_y + WINDOW_TITLE_H),
        TITLE_BG,
        -1,
    )
    draw_centered(canvas, (camera_x, title_y), (camera_w, WINDOW_TITLE_H), "Camera", TITLE_FG)
    draw_centered(canvas, (plot_x, title_y), (plot_w, WINDOW_TITLE_H), "OptiTrack", TITLE_FG)
    canvas[image_y : image_y + camera_h, camera_x : camera_x + camera_w] = camera
    canvas[image_y : image_y + plot_h, plot_x : plot_x + plot_w] = plot

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        status,
        (MARGIN, image_y + panel_h + 28),
        font,
        0.55,
        STATUS_FG,
        1,
        cv2.LINE_AA,
    )
    return canvas


def even_frame(frame):
    height, width = frame.shape[:2]
    pad_w = width % 2
    pad_h = height % 2
    if pad_w == 0 and pad_h == 0:
        return frame
    even = np.full((height + pad_h, width + pad_w, 3), PANEL_BG, dtype=np.uint8)
    even[:height, :width] = frame
    return even


def open_writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    # avc1 plays in QuickTime. mp4v is the next choice. MJPG always writes.
    for codec, suffix in (("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi")):
        target = path.with_suffix(suffix) if path.suffix.lower() not in {".mp4", ".avi"} else path
        if codec == "MJPG":
            target = path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(target), fourcc, fps, size)
        if writer.isOpened():
            return writer, target
        writer.release()
    raise SystemExit(f"Could not open a video writer for {path}")


def scale_camera(image):
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (width * CAMERA_SCALE, height * CAMERA_SCALE),
        interpolation=cv2.INTER_LINEAR,
    )


def build_video(folder, output, fps, speed):
    if fps <= 0:
        raise SystemExit("--fps must be positive")
    if speed <= 0:
        raise SystemExit("--speed must be positive")
    samples = load_samples(folder)
    count, duration_s, runs = playback_runs(samples, fps, speed)
    view = PositionView(samples)
    output = output if output is not None else folder / "pose_video.mp4"
    writer = None
    written = 0
    try:
        for index, repeats in runs:
            stamp_ms, path, _position, _yaw = samples[index]
            camera = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if camera is None:
                raise SystemExit(f"Could not read {path}")
            plot = view.render(view.pose_index[index])
            frame = even_frame(compose(scale_camera(camera), plot, status_text(samples[index], samples[0][0])))
            if writer is None:
                writer, output = open_writer(output, fps, (frame.shape[1], frame.shape[0]))
                print(f"Writing {output}  {frame.shape[1]}x{frame.shape[0]}  {fps:.0f} fps", flush=True)
            for _repeat in range(repeats):
                writer.write(frame)
                written += 1
            if written == repeats or written == count or index == runs[-1][0]:
                print(f"  frame {written}/{count}  image {stamp_ms}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        view.close()
    print(
        f"Saved {written} frame(s), {duration_s:.1f} s of playback, from {len(samples)} image(s).",
        flush=True,
    )
    return output


def main():
    args = parse_args()
    folder = session_folder(args.session)
    build_video(folder, args.output, args.fps, args.speed)


if __name__ == "__main__":
    main()
