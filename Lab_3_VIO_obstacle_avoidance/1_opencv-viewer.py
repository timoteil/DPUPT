#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2021 Bitcraze AB
#
#  AI-deck demo
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License along with
#  this program; if not, write to the Free Software Foundation, Inc., 51
#  Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#  Demo for showing streamed images from the AI-deck example.
#
#  By default this demo connects to the IP of the AI-deck example when in
#  Access point mode.
#
#  A reader thread pulls CPX packets as fast as they arrive and keeps only the
#  newest frame. The main thread decodes that frame and draws it immediately,
#  so the window stays on the live image instead of lagging behind the socket.

import argparse
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

WINDOW = "AI-deck"
HEADER_STRUCT = struct.Struct("<BHHBBI")
PACKET_STRUCT = struct.Struct("<HBB")
MAX_FRAME_BYTES = 2 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Show the AI-deck image stream in a live window")
    parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
    parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
    parser.add_argument("--save", action="store_true", help="Save the frames shown in the window")
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Display scale. The 324x244 camera image is 648x488 at 2",
    )
    parser.add_argument("--flip", action="store_true", help="Rotate the camera image 180 degrees")
    return parser.parse_args()


def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("AI-deck closed the connection")
        data.extend(chunk)
    return data


def read_frame(sock):
    """Read one CPX image. Return width, height, format, and the image bytes."""
    length, _routing, _function = PACKET_STRUCT.unpack(rx_bytes(sock, 4))
    payload_len = length - 2
    if payload_len < HEADER_STRUCT.size or payload_len > MAX_FRAME_BYTES:
        raise ConnectionError(f"Bad image header length {length}")
    header = rx_bytes(sock, payload_len)
    magic, width, height, _depth, fmt, size = HEADER_STRUCT.unpack_from(header)
    if magic != 0xBC:
        raise ConnectionError(f"Unexpected AI-deck magic byte: 0x{magic:02X}")
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise ConnectionError(f"Bad image size {size}")

    image = bytearray()
    while len(image) < size:
        length, _dst, _src = PACKET_STRUCT.unpack(rx_bytes(sock, 4))
        chunk_len = length - 2
        if chunk_len < 0 or chunk_len > MAX_FRAME_BYTES:
            raise ConnectionError(f"Bad chunk length {length}")
        image.extend(rx_bytes(sock, chunk_len))
    return width, height, fmt, bytes(image[:size])


def decode_frame(width, height, fmt, payload, flip=False):
    """Return (color BGR image, raw Bayer array or None). flip rotates the image 180 degrees."""
    if fmt == 0:
        expected = width * height
        if width <= 0 or height <= 0 or len(payload) < expected:
            raise ValueError(f"Bayer frame is {len(payload)} bytes, expected {expected}")
        bayer = np.frombuffer(payload[:expected], dtype=np.uint8).reshape(height, width)
        color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
        if flip:
            return cv2.flip(color, -1), cv2.flip(bayer, -1)
        return color, bayer

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode JPEG")
    if flip:
        decoded = cv2.flip(decoded, -1)
    return decoded, None


def draw_overlay(image, text):
    view = image.copy()
    origin = (8, 22)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(view, text, origin, font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, text, origin, font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def window_closed():
    # 0 means the user closed the window. -1 means the property is unavailable.
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0


def present(image, scale, sized):
    """Draw one frame. Return False when the user asks to quit."""
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(WINDOW, image)
    if not sized:
        height, width = image.shape[:2]
        cv2.resizeWindow(WINDOW, width, height)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return False
    if window_closed():
        return False
    return True


def pump_window(check_close):
    """Keep the window responsive while the next frame is still on the wire."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return False
    # Before the first image, the window exists but may not report itself visible yet.
    if check_close and window_closed():
        return False
    return True


def save_frame(count, color, bayer):
    if bayer is not None:
        cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer)
        cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color)
    else:
        cv2.imwrite(f"stream_out/jpeg/img_{count:06d}.jpg", color)


def receive_loop(sock, state, lock, stop):
    try:
        while not stop.is_set():
            frame = read_frame(sock)
            with lock:
                state["frame"] = frame
                state["seq"] += 1
                state["received"] += 1
    except Exception as exc:
        if not stop.is_set():
            with lock:
                state["error"] = exc
    finally:
        stop.set()


def main():
    args = parse_args()
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
    if args.save:
        os.makedirs("stream_out/raw", exist_ok=True)
        os.makedirs("stream_out/debayer", exist_ok=True)
        os.makedirs("stream_out/jpeg", exist_ok=True)

    print(f"Connecting to socket on {args.n}:{args.p}...", flush=True)
    sock = socket.create_connection((args.n, args.p))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Socket connected", flush=True)
    print("Live view. Press q or Esc, or close the window, to stop.", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    state = {"frame": None, "seq": 0, "received": 0, "error": None}
    lock = threading.Lock()
    stop = threading.Event()
    thread = threading.Thread(
        target=receive_loop,
        args=(sock, state, lock, stop),
        name="aideck-rx",
        daemon=True,
    )
    thread.start()

    shown = 0
    last_seq = 0
    sized = False
    hz = 0.0
    last_frame_time = None
    report_time = time.perf_counter()
    try:
        while True:
            with lock:
                seq = state["seq"]
                frame = state["frame"] if seq != last_seq else None
                error = state["error"]
                received = state["received"]
            if frame is None:
                if error is not None:
                    raise error
                if stop.is_set() or not pump_window(shown > 0):
                    break
                continue

            width, height, fmt, payload = frame
            last_seq = seq
            try:
                color, bayer = decode_frame(width, height, fmt, payload, args.flip)
            except ValueError as exc:
                print(exc, flush=True)
                continue

            now = time.perf_counter()
            if last_frame_time is not None:
                dt = now - last_frame_time
                if dt > 0:
                    instant = 1.0 / dt
                    hz = instant if hz == 0.0 else (0.9 * hz + 0.1 * instant)
            last_frame_time = now
            shown += 1

            if now - report_time >= 1.0:
                kind = "bayer" if fmt == 0 else "jpeg"
                print(
                    f"{hz:.1f} Hz  shown={shown}  received={received}  {width}x{height} {kind}",
                    flush=True,
                )
                report_time = now

            kind = "bayer" if fmt == 0 else "jpeg"
            label = f"{hz:.1f} Hz  {color.shape[1]}x{color.shape[0]} {kind}"
            if not present(draw_overlay(color, label), args.scale, sized):
                break
            sized = True
            if args.save:
                save_frame(shown, color, bayer)
    except KeyboardInterrupt:
        print(flush=True)
    except (ConnectionError, OSError) as exc:
        print(f"Stream error: {exc}", flush=True)
    finally:
        stop.set()
        sock.close()
        thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print(f"Showed {shown} frame(s).", flush=True)


if __name__ == "__main__":
    main()
