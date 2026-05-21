"""Standalone MediaPipe Holistic keypoint extractor.

Input:  a video file, an image, or a directory of either.
Output: an .npz (and optional .json) file per input containing pose / hands / face
        landmarks plus per-landmark confidence. Schema matches the rest of the
        UzSL pipeline (see scripts/extract_keypoints.py and src/pretrain/landmarks.py).

Required dependencies (pip):
    mediapipe>=0.10        # Holistic model — pose + hands + face mesh
    opencv-python>=4.5     # video/image I/O + overlay drawing (cv2)
    numpy>=1.23            # array storage

    Python 3.9–3.11 recommended (mediapipe wheels are limited on 3.12+).
    The repo's venv already has all of these; outside it:
        pip install mediapipe opencv-python numpy

Standard library only (no extra installs): argparse, json, sys, dataclasses,
pathlib, typing.

Usage examples:
    python keypoint_extraction.py --input path/to/video.mp4
    python keypoint_extraction.py --input path/to/image.jpg --output out/img.npz
    python keypoint_extraction.py --input some_dir/ --output-dir out/ --overlay
    python keypoint_extraction.py --input vid.mp4 --no-face --model-complexity 2

Tune the CONFIG block below or pass CLI flags; everything is intended to be
hackable.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG — tweak these to taste. CLI flags override.
# ----------------------------------------------------------------------------

# Face subset used by the UzSL pipeline (nose, mouth, eyes, jaw, cheeks).
# Mirrors src/pretrain/landmarks.py:NMM_INDICES so downstream code keeps working.
NMM_INDICES: list[int] = [
    70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 80, 81, 82, 13, 312, 311, 310,
    159, 160, 161, 144, 145, 153,
    386, 387, 388, 373, 374, 380,
    6, 1, 2, 98, 327, 152, 172, 397,
    50, 101, 36, 280, 330, 266,
]

N_POSE = 33
N_HAND = 21
N_FACE_FULL = 478
N_FACE_NMM = len(NMM_INDICES)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """One result per input file. Arrays have leading time dim T (T=1 for images)."""

    pose: np.ndarray          # (T, 33, 4)  — x, y, z, visibility
    left_hand: np.ndarray     # (T, 21, 3)
    right_hand: np.ndarray    # (T, 21, 3)
    face_full: np.ndarray     # (T, 478, 3)
    face_nmm: np.ndarray      # (T, 73, 3)
    conf_pose: np.ndarray     # (T, 33)
    conf_left_hand: np.ndarray   # (T, 21)
    conf_right_hand: np.ndarray  # (T, 21)
    conf_face: np.ndarray     # (T, 478)
    detected: np.ndarray      # (T, 4) bool — [pose, left_hand, right_hand, face]
    fps: float
    width: int
    height: int

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "pose": self.pose,
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            "face_full": self.face_full,
            "face_nmm": self.face_nmm,
            "conf_pose": self.conf_pose,
            "conf_left_hand": self.conf_left_hand,
            "conf_right_hand": self.conf_right_hand,
            "conf_face": self.conf_face,
            "detected": self.detected,
            "meta": np.array(
                [self.fps, self.width, self.height], dtype=np.float32
            ),
        }


# ----------------------------------------------------------------------------
# Allocation helpers
# ----------------------------------------------------------------------------

def _alloc(num_frames: int, with_face: bool) -> dict[str, np.ndarray]:
    return {
        "pose": np.zeros((num_frames, N_POSE, 4), dtype=np.float32),
        "left_hand": np.zeros((num_frames, N_HAND, 3), dtype=np.float32),
        "right_hand": np.zeros((num_frames, N_HAND, 3), dtype=np.float32),
        "face_full": np.zeros(
            (num_frames, N_FACE_FULL if with_face else 0, 3), dtype=np.float32
        ),
        "face_nmm": np.zeros(
            (num_frames, N_FACE_NMM if with_face else 0, 3), dtype=np.float32
        ),
        "conf_pose": np.zeros((num_frames, N_POSE), dtype=np.float32),
        "conf_left_hand": np.zeros((num_frames, N_HAND), dtype=np.float32),
        "conf_right_hand": np.zeros((num_frames, N_HAND), dtype=np.float32),
        "conf_face": np.zeros(
            (num_frames, N_FACE_FULL if with_face else 0), dtype=np.float32
        ),
        "detected": np.zeros((num_frames, 4), dtype=bool),
    }


def _landmark_conf(lm, fallback: bool) -> float:
    """MediaPipe hand/face landmarks lack a 'visibility' field; fall back to
    a binary present/absent signal so downstream confidence math still works."""
    vis = getattr(lm, "visibility", None)
    if vis is not None and vis > 0:
        return float(vis)
    return 1.0 if fallback else 0.0


# ----------------------------------------------------------------------------
# Per-frame fill
# ----------------------------------------------------------------------------

def _fill_frame(buf: dict[str, np.ndarray], t: int, results, with_face: bool) -> None:
    if results.pose_landmarks is not None:
        buf["detected"][t, 0] = True
        for i, lm in enumerate(results.pose_landmarks.landmark[:N_POSE]):
            buf["pose"][t, i] = (lm.x, lm.y, lm.z, lm.visibility)
            buf["conf_pose"][t, i] = lm.visibility

    if results.left_hand_landmarks is not None:
        buf["detected"][t, 1] = True
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:N_HAND]):
            buf["left_hand"][t, i] = (lm.x, lm.y, lm.z)
            buf["conf_left_hand"][t, i] = _landmark_conf(
                lm, fallback=abs(lm.x) + abs(lm.y) + abs(lm.z) > 1e-8
            )

    if results.right_hand_landmarks is not None:
        buf["detected"][t, 2] = True
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:N_HAND]):
            buf["right_hand"][t, i] = (lm.x, lm.y, lm.z)
            buf["conf_right_hand"][t, i] = _landmark_conf(
                lm, fallback=abs(lm.x) + abs(lm.y) + abs(lm.z) > 1e-8
            )

    if with_face and results.face_landmarks is not None:
        buf["detected"][t, 3] = True
        for i, lm in enumerate(results.face_landmarks.landmark[:N_FACE_FULL]):
            buf["face_full"][t, i] = (lm.x, lm.y, lm.z)
            buf["conf_face"][t, i] = _landmark_conf(
                lm, fallback=abs(lm.x) + abs(lm.y) + abs(lm.z) > 1e-8
            )
        buf["face_nmm"][t] = buf["face_full"][t, NMM_INDICES, :]


# ----------------------------------------------------------------------------
# Extractors
# ----------------------------------------------------------------------------

def _build_holistic(args: argparse.Namespace, *, static: bool):
    return mp.solutions.holistic.Holistic(
        static_image_mode=static,
        model_complexity=args.model_complexity,
        smooth_landmarks=not static,
        refine_face_landmarks=args.refine_face,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


def extract_from_video(path: Path, args: argparse.Namespace) -> ExtractionResult:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.max_frames > 0:
        total = min(total, args.max_frames) if total else args.max_frames
    if total <= 0:
        total = 0  # unknown; we'll grow lists instead

    with_face = not args.no_face

    if total > 0:
        buf = _alloc(total, with_face)
    else:
        buf = None  # will allocate after we know the frame count

    frames_dyn: list[dict] = []
    overlay_writer = None
    overlay_path = None
    if args.overlay:
        overlay_path = path.with_suffix(".overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(
            str(overlay_path), fourcc, fps or 25.0, (width, height)
        )

    t = 0
    with _build_holistic(args, static=False) as holistic:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and t >= args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if buf is not None:
                _fill_frame(buf, t, results, with_face)
            else:
                tmp = _alloc(1, with_face)
                _fill_frame(tmp, 0, results, with_face)
                frames_dyn.append(tmp)

            if overlay_writer is not None:
                overlay_writer.write(_draw_overlay(frame_bgr, results, with_face))

            t += 1

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
        print(f"  overlay -> {overlay_path}")

    if buf is None:
        buf = _concat_dyn(frames_dyn, with_face) if frames_dyn else _alloc(0, with_face)
    else:
        # Trim if we ran short of expected frame count.
        if t < buf["pose"].shape[0]:
            for k in buf:
                buf[k] = buf[k][:t]

    return ExtractionResult(
        pose=buf["pose"],
        left_hand=buf["left_hand"],
        right_hand=buf["right_hand"],
        face_full=buf["face_full"],
        face_nmm=buf["face_nmm"],
        conf_pose=buf["conf_pose"],
        conf_left_hand=buf["conf_left_hand"],
        conf_right_hand=buf["conf_right_hand"],
        conf_face=buf["conf_face"],
        detected=buf["detected"],
        fps=fps,
        width=width,
        height=height,
    )


def extract_from_image(path: Path, args: argparse.Namespace) -> ExtractionResult:
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    height, width = frame_bgr.shape[:2]

    with_face = not args.no_face
    buf = _alloc(1, with_face)

    with _build_holistic(args, static=True) as holistic:
        results = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        _fill_frame(buf, 0, results, with_face)

        if args.overlay:
            overlay_path = path.with_suffix(".overlay.png")
            cv2.imwrite(str(overlay_path), _draw_overlay(frame_bgr, results, with_face))
            print(f"  overlay -> {overlay_path}")

    return ExtractionResult(
        pose=buf["pose"],
        left_hand=buf["left_hand"],
        right_hand=buf["right_hand"],
        face_full=buf["face_full"],
        face_nmm=buf["face_nmm"],
        conf_pose=buf["conf_pose"],
        conf_left_hand=buf["conf_left_hand"],
        conf_right_hand=buf["conf_right_hand"],
        conf_face=buf["conf_face"],
        detected=buf["detected"],
        fps=0.0,
        width=width,
        height=height,
    )


def _concat_dyn(frames: list[dict], with_face: bool) -> dict[str, np.ndarray]:
    out = _alloc(len(frames), with_face)
    for t, f in enumerate(frames):
        for k in out:
            out[k][t] = f[k][0]
    return out


# ----------------------------------------------------------------------------
# Optional overlay drawing
# ----------------------------------------------------------------------------

def _draw_overlay(frame_bgr: np.ndarray, results, with_face: bool) -> np.ndarray:
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    img = frame_bgr.copy()
    if results.pose_landmarks is not None:
        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
    if results.left_hand_landmarks is not None:
        mp_draw.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks is not None:
        mp_draw.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if with_face and results.face_landmarks is not None:
        mp_draw.draw_landmarks(
            img,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
    return img


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def save_result(result: ExtractionResult, out_path: Path, also_json: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **result.to_npz_dict())
    if also_json:
        json_path = out_path.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(
                {
                    "shape": {
                        "pose": list(result.pose.shape),
                        "left_hand": list(result.left_hand.shape),
                        "right_hand": list(result.right_hand.shape),
                        "face_full": list(result.face_full.shape),
                        "face_nmm": list(result.face_nmm.shape),
                    },
                    "fps": result.fps,
                    "width": result.width,
                    "height": result.height,
                    "detected_per_stream": result.detected.mean(axis=0).tolist(),
                },
                f,
                indent=2,
            )


def iter_inputs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS or p.suffix.lower() in IMAGE_EXTS:
                yield p
        return
    raise FileNotFoundError(input_path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Video, image, or directory.")
    p.add_argument("--output", type=Path, default=None, help="Output .npz path (single-file mode).")
    p.add_argument("--output-dir", type=Path, default=Path("keypoints_out"), help="Output dir for batch mode.")
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    p.add_argument("--refine-face", action="store_true", help="Use refined face mesh (slower).")
    p.add_argument("--no-face", action="store_true", help="Skip the 478-point face mesh.")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames.")
    p.add_argument("--overlay", action="store_true", help="Also save an annotated overlay video/image.")
    p.add_argument("--json", action="store_true", help="Also write a .json sidecar with shapes & metadata.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    inputs = list(iter_inputs(args.input))
    if not inputs:
        print(f"No video/image inputs found under {args.input}", file=sys.stderr)
        return 1

    single_file = len(inputs) == 1 and args.output is not None

    print(f"MediaPipe Holistic {mp.__version__} — {len(inputs)} input(s)")
    for path in inputs:
        suffix = path.suffix.lower()
        try:
            if suffix in VIDEO_EXTS:
                result = extract_from_video(path, args)
            elif suffix in IMAGE_EXTS:
                result = extract_from_image(path, args)
            else:
                print(f"  skip (unknown ext): {path}")
                continue
        except Exception as e:
            print(f"  FAILED {path}: {e}", file=sys.stderr)
            continue

        if single_file:
            out_path = args.output
        else:
            out_path = args.output_dir / (path.stem + ".npz")
        save_result(result, out_path, also_json=args.json)

        det = result.detected.mean(axis=0)
        print(
            f"  {path.name}  T={result.pose.shape[0]}  "
            f"pose={det[0]:.2f} lh={det[1]:.2f} rh={det[2]:.2f} face={det[3]:.2f}  "
            f"-> {out_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
