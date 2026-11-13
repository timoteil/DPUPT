#!/usr/bin/env python3
"""Monocular Visual Odometry with switchable parts, to find out what each part buys you (ablation study).

This is 5_visual_odometry_orb.py with five switches. The defaults reproduce script 5 exactly:

  --detector  orb | akaze | sift     which keypoints and descriptors to use
  --matcher   ratio | crosscheck     how the descriptors of two images are paired
  --tracking  keyframe | previous    compare with the last KEYFRAME, or with the previous FRAME
  --no-clahe                         do not boost the image contrast before detecting keypoints
  --no-mask                          do not hide the drone itself (propellers, frame) from the detector

The algorithm, for every new frame and the reference frame:
  1. detect keypoints and compute their descriptors in both images,
  2. match the descriptors,
  3. undistort the matched pixels and estimate the essential matrix E with RANSAC,
  4. decompose E into the camera rotation R and the translation DIRECTION t (|t| = 1),
  5. give t a length (the OptiTrack displacement between the two frames, see below) and chain the steps:
         position    <- position + R_world_body . (scale * t_body)
         orientation <- orientation . R_body

Scale: a single camera cannot measure distances, only directions, so the LENGTH of every step is borrowed
from OptiTrack. The DIRECTION of every step and all rotations come from the camera only.

Validation: the VO starts once from the OptiTrack pose at the first frame in flight and is never corrected.
The VO path is also shown after the single best rotation about z (around the start) that maps it onto OptiTrack.

Outputs, in <session>/vo_<configuration>/ (or --output):
    vo_trajectory.csv   one row per step: VO position, OptiTrack position, error, matches, inliers
    vo_trajectory.png   top view, x/y/z over time, and the position error
    vo_matches.mp4      the two frames side by side with the inlier matches drawn on top
and one line per run is appended to <session>/vo_ablation_summary.csv, so the runs can be compared in a table.

Examples (one change at a time, compare with the baseline):
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video --no-clahe
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video --no-mask
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video --matcher crosscheck
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video --detector akaze
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video --detector sift
    python 6_visual_odometry_ablations.py stream_out/imgs/1794489705263 --no-video --tracking previous
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Camera axes (x right, y down, z forward) -> drone body axes (x forward, y left, z up).
R_BODY_CAM_LEVEL = np.array([[0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0]])
FLIGHT_MIN_Z = 0.3  # [m] OptiTrack height above which the drone counts as flying


# --------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description="Monocular visual odometry with switchable parts (ablation study).")
    p.add_argument("session", type=Path, help="Folder from --save-imgs (with positions.csv)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output folder. Default: <session>/vo_<configuration>")

    ablation = p.add_argument_group("ablation switches (defaults = script 5)")
    ablation.add_argument("--detector", choices=["orb", "akaze", "sift"], default="orb",
                          help="Keypoint detector and descriptor. orb/akaze: binary (Hamming), sift: float (L2)")
    ablation.add_argument("--matcher", choices=["ratio", "crosscheck"], default="ratio",
                          help="ratio: 2 nearest neighbours + Lowe's ratio test. "
                               "crosscheck: keep only mutual best matches, no ratio test")
    ablation.add_argument("--tracking", choices=["keyframe", "previous"], default="keyframe",
                          help="keyframe: compare with the last keyframe, which only moves on after enough parallax. "
                               "previous: compare every frame with the one before it")
    ablation.add_argument("--no-clahe", action="store_true", help="Skip the contrast enhancement (CLAHE)")
    ablation.add_argument("--no-mask", action="store_true",
                          help="Ignore ranging/self_mask.png: keypoints may land on the drone's own propellers")

    tuning = p.add_argument_group("tuning")
    tuning.add_argument("--min-parallax", type=float, default=4.0,
                        help="[keyframe] Parallax [px] needed before a frame becomes the next keyframe")
    tuning.add_argument("--max-gap", type=int, default=8,
                        help="[keyframe] Maximum number of frames between two keyframes")
    tuning.add_argument("--features", type=int, default=1000, help="Maximum number of keypoints per image")
    tuning.add_argument("--ratio", type=float, default=0.75, help="[ratio matcher] Lowe's ratio test threshold")
    tuning.add_argument("--ransac-px", type=float, default=1.0, help="RANSAC inlier threshold [px]")
    tuning.add_argument("--min-inliers", type=int, default=20, help="Fewer inliers than this = skip the step")

    misc = p.add_argument_group("output")
    misc.add_argument("--tag", default="", help="Extra text added to the output folder name and the summary")
    misc.add_argument("--summary", type=Path, default=None,
                      help="CSV where one result line per run is appended. Default: <session>/vo_ablation_summary.csv")
    misc.add_argument("--max-frames", type=int, default=0, help="Use only the first N frames (0 = all, for quick tests)")
    misc.add_argument("--no-video", action="store_true", help="Do not write vo_matches.mp4 (much faster)")
    misc.add_argument("--fps", type=float, default=8.0, help="Video frames per second")
    return p.parse_args()


def describe(args):
    """(short name for folders, long label for plots) of the chosen configuration."""
    name = f"{args.detector}_{args.matcher}_{args.tracking}"
    label = f"{args.detector.upper()}, {args.matcher} matcher, {args.tracking} tracking"
    if args.no_clahe:
        name += "_noclahe"
        label += ", no CLAHE"
    if args.no_mask:
        name += "_nomask"
        label += ", no self-mask"
    if args.tag:
        name += f"_{args.tag}"
        label += f" [{args.tag}]"
    return name, label


# --------------------------------------------------------------------------- small helpers
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


# --------------------------------------------------------------------------- data loading
def load_camera(session, image_shape):
    """Intrinsics K, distortion and camera tilt from ranging/camera_calibration.json (if present)."""
    calib = {"fx": 180.0, "k1": 0.0, "tilt_deg": 0.0, "latency_ms": 0.0}
    path = session / "ranging" / "camera_calibration.json"
    if path.is_file():
        calib.update(json.loads(path.read_text()))
    h, w = image_shape
    fx = calib["fx"]
    K = np.array([[fx, 0.0, (w - 1) / 2],
                  [0.0, fx, (h - 1) / 2],
                  [0.0, 0.0, 1.0]])
    dist = np.array([calib["k1"], 0.0, 0.0, 0.0, 0.0])
    # The camera is tilted about its x axis (negative tilt_deg = pointing down).
    R_body_cam = R_BODY_CAM_LEVEL @ rot_x(-math.radians(calib["tilt_deg"]))
    return K, dist, R_body_cam, calib["latency_ms"] / 1000.0


def load_session(session):
    """Returns the frames [(time_s, image_path)] and the OptiTrack samples as arrays t, xyz, yaw."""
    frames, pose_t, pose_xyz, pose_yaw = [], [], [], []
    last = None
    with (session / "positions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            image = session / f"{row['timestamp_ms']}.jpg"
            if not image.is_file():
                continue
            frames.append((int(row["timestamp_ms"]) / 1000.0, image))
            if not row.get("x") or not row.get("pose_timestamp_ms"):
                continue
            values = (row["x"], row["y"], row["z"], row["yaw_deg"])
            if values == last:  # the same pose repeated = OptiTrack lost the drone
                continue
            last = values
            pose_t.append(int(row["pose_timestamp_ms"]) / 1000.0)
            pose_xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
            pose_yaw.append(math.radians(float(row["yaw_deg"])))
    frames.sort()
    return frames, np.array(pose_t), np.array(pose_xyz), np.unwrap(np.array(pose_yaw))


def optitrack_at(t, pose_t, pose_xyz, pose_yaw):
    """Linearly interpolated OptiTrack position and yaw at time t, or None if not tracked."""
    if len(pose_t) < 2 or t < pose_t[0] or t > pose_t[-1]:
        return None
    k = int(np.clip(np.searchsorted(pose_t, t) - 1, 0, len(pose_t) - 2))
    if pose_t[k + 1] - pose_t[k] > 0.5:  # a gap in the tracking
        return None
    a = (t - pose_t[k]) / (pose_t[k + 1] - pose_t[k])
    return (1 - a) * pose_xyz[k] + a * pose_xyz[k + 1], (1 - a) * pose_yaw[k] + a * pose_yaw[k + 1]


# --------------------------------------------------------------------------- features and matching
def make_extractor(name, n_features):
    """Returns (extract, norm). extract(image, mask) -> (keypoints, descriptors); norm = distance to compare them."""
    if name == "orb":
        det = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
        norm = cv2.NORM_HAMMING  # binary descriptors: count the different bits

        def extract(img, mask):
            kp, des = det.detectAndCompute(img, mask)
            return list(kp), des
    elif name == "akaze":
        create = getattr(cv2, "AKAZE_create", None) or getattr(getattr(cv2, "AKAZE", None), "create", None)
        if create is None:  # OpenCV 5 moved AKAZE out of the main package
            raise SystemExit(f"This OpenCV ({cv2.__version__}) has no AKAZE. Use OpenCV 4: "
                             f"pip install \"opencv-python<5\"   (or choose --detector orb / sift)")
        det = create()
        norm = cv2.NORM_HAMMING  # AKAZE's default descriptor (MLDB) is binary too

        def extract(img, mask):
            # AKAZE has no "max features" setting: keep the strongest n_features ourselves.
            kp = sorted(det.detect(img, mask), key=lambda k: -k.response)[:n_features]
            kp, des = det.compute(img, kp)
            return list(kp), des
    elif name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise SystemExit("This OpenCV has no SIFT. Update with: pip install -U opencv-python")
        det = cv2.SIFT_create(nfeatures=n_features)
        norm = cv2.NORM_L2  # float descriptors: Euclidean distance

        def extract(img, mask):
            kp, des = det.detectAndCompute(img, mask)
            return list(kp), des
    else:
        raise ValueError(name)
    return extract, norm


def make_matcher(kind, norm):
    # crossCheck=True keeps a pair only when each descriptor is the other one's best match.
    return cv2.BFMatcher(norm, crossCheck=(kind == "crosscheck"))


def match(matcher, kind, kp1, des1, kp2, des2, ratio):
    """Matches between two sets of descriptors, and their pixel positions as two N x 2 arrays."""
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    if kind == "ratio":
        # For every descriptor of image 1, its two nearest neighbours in image 2.
        # Keep the match only if the best one is clearly better than the second best (Lowe's ratio test).
        good = [m for m, n in (pair for pair in matcher.knnMatch(des1, des2, k=2) if len(pair) == 2)
                if m.distance < ratio * n.distance]
    else:
        # Plain cross-check: a mutual best match is kept however ambiguous it is. RANSAC must remove the rest.
        good = list(matcher.match(des1, des2))
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2, good


# --------------------------------------------------------------------------- visual odometry
def relative_pose(pts1, pts2, K, dist, ransac_px):
    """Camera motion from image 1 to image 2: rotation R_cam and unit translation t_cam (in camera 1 axes).

    Returns (R_cam, t_cam, inlier_mask, parallax_px) or None when the geometry cannot be estimated.
    parallax_px is how far the matches move once the rotation is removed. Only that part of the image
    motion tells us anything about the translation: with little parallax, t is mostly noise.
    """
    if len(pts1) < 8:  # the 5-point algorithm with RANSAC needs a handful of points to be stable
        return None
    # Undistort and remove K: points become "normalised" coordinates (x/z, y/z) of a perfect pinhole camera.
    n1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    n2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    # With normalised coordinates the camera matrix is the identity, and the threshold is in normalised units.
    E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                   threshold=ransac_px / K[0, 0])
    if E is None or E.shape != (3, 3):
        return None
    inliers = mask.ravel() > 0
    # recoverPose picks, out of the 4 solutions of E, the one that puts the points in front of both cameras.
    # (Its own output mask also drops very distant points, which is most of them for small motions,
    # so the RANSAC inliers above are the better quality measure.)
    _, R, t, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask)

    # Where would the points of image 1 land in image 2 if the camera had only rotated?
    rotated = np.column_stack([n1, np.ones(len(n1))]) @ R.T
    rotated = rotated[:, :2] / rotated[:, 2:3]
    parallax_px = float(np.median(np.linalg.norm(rotated - n2, axis=1)[inliers])) * K[0, 0]

    # recoverPose describes the POINTS: X2 = R X1 + t. The CAMERA moves the opposite way:
    R_cam = R.T
    t_cam = (-R.T @ t).ravel()
    return R_cam, t_cam, inliers, parallax_px


# --------------------------------------------------------------------------- validation helpers
def align_about_start(vo_xy, gt_xy):
    """Best rotation about z (least squares) of the VO xy path around its start point onto OptiTrack."""
    f = vo_xy - vo_xy[0]
    g = gt_xy - gt_xy[0]
    theta = math.atan2(np.sum(f[:, 0] * g[:, 1] - f[:, 1] * g[:, 0]), np.sum(f * g))
    c, s = math.cos(theta), math.sin(theta)
    return f @ np.array([[c, -s], [s, c]]).T + gt_xy[0], math.degrees(theta)


def draw_matches(img1, img2, kp1, kp2, good, inliers, text):
    inlier_matches = [m for m, ok in zip(good, inliers) if ok]
    canvas = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, matchColor=(0, 255, 0),
                             singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
    canvas = cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    bar = np.full((22 * len(text) + 8, canvas.shape[1], 3), 245, np.uint8)
    for i, line in enumerate(text):
        cv2.putText(bar, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def open_video(path, frame, fps):
    h, w = frame.shape[:2]
    for codec in ("avc1", "mp4v"):  # avc1 (H.264) plays in browsers and IDEs, mp4v is the fallback
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            return writer
    return None


def plot_results(rows, theta_deg, session_name, label, path):
    t = np.array([r["t_s"] for r in rows])
    t -= t[0]
    vo = np.array([[r["vo_x"], r["vo_y"], r["vo_z"]] for r in rows])
    al = np.array([[r["aligned_x"], r["aligned_y"], r["vo_z"]] for r in rows])
    gt = np.array([[r["ot_x"], r["ot_y"], r["ot_z"]] for r in rows])
    err = np.linalg.norm(vo - gt, axis=1)
    err_al = np.linalg.norm(al - gt, axis=1)
    err_still = np.linalg.norm(gt - gt[0], axis=1)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(gt[:, 0], gt[:, 1], color="k", lw=2, alpha=0.7, label="OptiTrack (ground truth)")
    ax.plot(vo[:, 0], vo[:, 1], color="tab:red", lw=1, ls=":", label="VO, as computed")
    ax.plot(al[:, 0], al[:, 1], color="tab:red", lw=1.6, label=f"VO, rotated by {theta_deg:+.0f} deg")
    ax.plot(*gt[0, :2], "o", color="tab:green", ms=10, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    for k, name in enumerate("xyz"):
        ax = fig.add_subplot(gs[k // 2, 1 + k % 2]) if k < 2 else fig.add_subplot(gs[1, 1])
        ax.plot(t, gt[:, k], color="k", lw=2, alpha=0.6, label="OptiTrack")
        ax.plot(t, (al if k < 2 else vo)[:, k], color="tab:red", lw=1.2, label="VO" + (" (rotated)" if k < 2 else ""))
        ax.set_title(f"{name} position")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{name} [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, err, color="tab:red", lw=1, ls=":", label="VO, as computed")
    ax.plot(t, err_al, color="tab:red", lw=1.6, label="VO, rotated")
    ax.plot(t, err_still, color="gray", lw=1.2, label="staying at the start")
    ax.set_title("3D position error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    path_len = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    fig.suptitle(f"Visual odometry vs OptiTrack - {session_name}\n{label}\n"
                 f"Final error {err_al[-1]:.2f} m after {path_len:.1f} m flown "
                 f"(median {np.median(err_al):.2f} m, staying put: median {np.median(err_still):.2f} m)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def append_summary(path, record):
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if new:
            w.writeheader()
        w.writerow(record)


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    name, label = describe(args)
    session = args.session.expanduser().resolve()
    out_dir = (args.output or session / f"vo_{name}").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (args.summary or session / "vo_ablation_summary.csv").expanduser().resolve()
    print(f"Configuration: {label}", flush=True)

    frames, pose_t, pose_xyz, pose_yaw = load_session(session)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    if len(frames) < 2 or len(pose_t) < 2:
        raise SystemExit(f"Need images and OptiTrack poses in {session}")
    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_GRAYSCALE)
    K, dist, R_body_cam, latency = load_camera(session, first.shape)
    gt = lambda t: optitrack_at(t - latency, pose_t, pose_xyz, pose_yaw)

    # ABLATION --no-mask: pixels of the drone itself (propellers, frame) move with the camera, so they must
    # not be used as keypoints. With --no-mask they are used anyway.
    mask = None
    mask_path = session / "ranging" / "self_mask.png"
    if args.no_mask:
        print("Self-mask ignored: keypoints on the drone itself are allowed.", flush=True)
    elif mask_path.is_file():
        mask = np.where(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0, 0, 255).astype(np.uint8)

    # ABLATION --detector and --matcher.
    extract, norm = make_extractor(args.detector, args.features)
    matcher = make_matcher(args.matcher, norm)
    # ABLATION --no-clahe: contrast enhancement gives more keypoints in dark, flat images.
    clahe = None if args.no_clahe else cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    # Start at the first frame where the drone flies and is tracked.
    start = next((i for i, (t, _) in enumerate(frames) if gt(t) is not None and gt(t)[0][2] > FLIGHT_MIN_Z), None)
    if start is None:
        raise SystemExit("The drone never flies in this session.")
    p0, yaw0 = gt(frames[start][0])
    position = p0.copy()          # VO position in the OptiTrack (world) frame [m]
    R_world_body = rot_z(yaw0)    # VO orientation, initialised from the OptiTrack yaw
    print(f"{len(frames)} frames, starting at frame {start} (t = {frames[start][0] - frames[0][0]:.1f} s) "
          f"from OptiTrack position {np.round(p0, 2)} and yaw {math.degrees(yaw0):.0f} deg", flush=True)

    def load(k):
        """Image k (contrast enhanced unless --no-clahe) with its keypoints and descriptors."""
        img = cv2.imread(str(frames[k][1]), cv2.IMREAD_GRAYSCALE)
        if clahe is not None:
            img = clahe.apply(img)
        kp, des = extract(img, mask)
        return img, kp, des

    # ABLATION --tracking.
    # keyframe: every new frame is compared with the last KEYFRAME. The keyframe moves on only once the camera
    #   has translated enough to see parallax (or after --max-gap frames). Between consecutive frames the drone
    #   moves only a few cm, and the translation direction of such a tiny step is mostly noise.
    # previous: the keyframe is always the previous frame, so every step is tiny.
    use_keyframes = args.tracking == "keyframe"
    key = start
    key_img, key_kp, key_des = load(key)
    rows, writer, skipped = [], None, 0
    started = time.perf_counter()
    for j in range(start + 1, len(frames)):
        img, kp, des = load(j)

        # Steps 1-2: keypoints (in load) and matches with the reference frame.
        pts1, pts2, good = match(matcher, args.matcher, key_kp, key_des, kp, des, args.ratio)
        # Steps 3-4: relative camera motion (rotation + translation direction).
        pose = relative_pose(pts1, pts2, K, dist, args.ransac_px)
        n_inliers = int(pose[2].sum()) if pose is not None else 0
        parallax = pose[3] if pose is not None else 0.0

        lost = pose is None or n_inliers < args.min_inliers
        if use_keyframes and not lost and parallax < args.min_parallax and j - key < args.max_gap:
            continue  # not enough parallax yet: keep the keyframe and try the next frame

        # Step 5: scale of this step = how far OptiTrack says the drone moved between the two frames.
        g1, g2 = gt(frames[key][0]), gt(frames[j][0])
        scale = float(np.linalg.norm(g2[0] - g1[0])) if g1 is not None and g2 is not None else 0.0

        if lost:
            status = "skipped (too few inliers)"
            skipped += 1
        else:
            status = "ok" if parallax >= args.min_parallax else "low parallax"
            R_cam, t_cam = pose[0], pose[1]
            # Express the camera motion in drone body axes, then in world axes, and chain it.
            R_body = R_body_cam @ R_cam @ R_body_cam.T
            t_body = R_body_cam @ t_cam
            position = position + R_world_body @ (scale * t_body)
            R_world_body = R_world_body @ R_body

        ot = g2[0] if g2 is not None else np.full(3, np.nan)
        rows.append({
            "keyframe": key, "frame": j, "t_s": frames[j][0], "status": status, "matches": len(good),
            "inliers": n_inliers, "parallax_px": parallax, "step_m": scale,
            "vo_x": position[0], "vo_y": position[1], "vo_z": position[2],
            "vo_yaw_deg": math.degrees(yaw_of(R_world_body)),
            "ot_x": ot[0], "ot_y": ot[1], "ot_z": ot[2],
            "ot_yaw_deg": math.degrees(g2[1]) if g2 is not None else np.nan,
        })

        if not args.no_video:
            inliers = pose[2] if pose is not None else np.zeros(len(good), bool)
            text = [f"{label}",
                    f"keyframe {key} -> frame {j}  keypoints {len(key_kp)}/{len(kp)}  matches {len(good)}  "
                    f"inliers {n_inliers}  parallax {parallax:.1f}px  [{status}]",
                    f"VO position  x={position[0]:+.2f}  y={position[1]:+.2f}  z={position[2]:+.2f} m",
                    f"OptiTrack    x={ot[0]:+.2f}  y={ot[1]:+.2f}  z={ot[2]:+.2f} m"]
            panel = draw_matches(key_img, img, key_kp, kp, good, inliers, text)
            writer = writer or open_video(out_dir / "vo_matches.mp4", panel, args.fps)
            if writer is not None:
                writer.write(panel)

        if len(rows) % 25 == 1:
            print(f"keyframe {key:4d} -> {j:4d}: {len(good):4d} matches, {n_inliers:4d} inliers, "
                  f"parallax {parallax:4.1f}px, VO {np.round(position, 2)}  OptiTrack {np.round(ot, 2)}", flush=True)

        # The current frame becomes the new reference frame (in "previous" mode this happens every frame).
        key, key_img, key_kp, key_des = j, img, kp, des

    if writer is not None:
        writer.release()
    run_s = time.perf_counter() - started

    # Validation: only the frames where OptiTrack knows where the drone is.
    rows = [r for r in rows if np.isfinite(r["ot_x"])]
    if not rows:
        raise SystemExit("No step could be compared with OptiTrack.")
    vo_xy = np.array([[r["vo_x"], r["vo_y"]] for r in rows])
    gt_xy = np.array([[r["ot_x"], r["ot_y"]] for r in rows])
    aligned_xy, theta_deg = align_about_start(vo_xy, gt_xy)
    for r, (ax_, ay_) in zip(rows, aligned_xy):
        r["aligned_x"], r["aligned_y"] = ax_, ay_
        r["err_m"] = math.dist((r["vo_x"], r["vo_y"], r["vo_z"]), (r["ot_x"], r["ot_y"], r["ot_z"]))
        r["err_aligned_m"] = math.dist((ax_, ay_, r["vo_z"]), (r["ot_x"], r["ot_y"], r["ot_z"]))

    with (out_dir / "vo_trajectory.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_results(rows, theta_deg, session.name, label, out_dir / "vo_trajectory.png")

    err = np.array([r["err_m"] for r in rows])
    err_al = np.array([r["err_aligned_m"] for r in rows])
    yaw_err = np.degrees(np.angle(np.exp(1j * np.radians(
        [r["vo_yaw_deg"] - r["ot_yaw_deg"] for r in rows]))))
    path_len = float(np.sum(np.linalg.norm(np.diff(np.array([[r["ot_x"], r["ot_y"], r["ot_z"]] for r in rows]),
                                                   axis=0), axis=1)))
    median_inliers = float(np.median([r["inliers"] for r in rows]))
    median_matches = float(np.median([r["matches"] for r in rows]))
    print(f"{len(rows)} steps, {skipped} skipped. Median matches {median_matches:.0f}, inliers {median_inliers:.0f}. "
          f"{run_s:.1f} s ({1000 * run_s / max(len(rows), 1):.0f} ms per step)")
    print(f"Position error as computed:          median {np.median(err):.2f} m, final {err[-1]:.2f} m")
    print(f"Position error rotated by {theta_deg:+.0f} deg:   median {np.median(err_al):.2f} m, final {err_al[-1]:.2f} m")
    print(f"Yaw (VO - OptiTrack): start {yaw_err[0]:+.0f} deg, final {yaw_err[-1]:+.0f} deg")
    print(f"Wrote {out_dir}/vo_trajectory.csv, vo_trajectory.png" + ("" if args.no_video else ", vo_matches.mp4"))

    append_summary(summary_path, {
        "configuration": name, "detector": args.detector, "matcher": args.matcher, "tracking": args.tracking,
        "clahe": not args.no_clahe, "self_mask": not args.no_mask, "features": args.features,
        "steps": len(rows), "skipped": skipped,
        "median_matches": round(median_matches), "median_inliers": round(median_inliers),
        "median_err_m": round(float(np.median(err)), 3), "final_err_m": round(float(err[-1]), 3),
        "median_err_rotated_m": round(float(np.median(err_al)), 3),
        "final_err_rotated_m": round(float(err_al[-1]), 3),
        "final_err_percent_of_path": round(100 * float(err_al[-1]) / path_len, 1) if path_len > 0 else "",
        "rotation_deg": round(theta_deg, 1), "final_yaw_err_deg": round(float(yaw_err[-1]), 1),
        "run_s": round(run_s, 1),
    })
    print(f"Appended a line to {summary_path}")


if __name__ == "__main__":
    main()
