#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import mediapipe as mp
import numpy as np

NMM_INDICES = [
    70,
    63,
    105,
    66,
    107,
    46,
    53,
    52,
    65,
    55,
    336,
    296,
    334,
    293,
    300,
    276,
    283,
    282,
    295,
    285,
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    159,
    160,
    161,
    144,
    145,
    153,
    386,
    387,
    388,
    373,
    374,
    380,
    6,
    1,
    2,
    98,
    327,
    152,
    172,
    397,
    50,
    101,
    36,
    280,
    330,
    266,
]
NMM_INDEX_MAP = {full_idx: i for i, full_idx in enumerate(NMM_INDICES)}

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
POSE_DOT_IDXS = [11, 12, 13, 14, 15, 16]
HAND_CONNECTIONS = sorted([(min(a, b), max(a, b)) for (a, b) in mp.solutions.hands.HAND_CONNECTIONS])
FACE_POLYLINES_FULL = [
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # left eyebrow
    [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],  # right eyebrow
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],  # mouth outer
    [78, 80, 81, 82, 13, 312, 311, 310, 78],  # mouth inner
    [159, 160, 161, 144, 145, 153, 159],  # left eye
    [386, 387, 388, 373, 374, 380, 386],  # right eye
]
FACE_POLYLINES = [[NMM_INDEX_MAP[i] for i in line if i in NMM_INDEX_MAP] for line in FACE_POLYLINES_FULL]
EXTRA_FACE_FULL_POLYLINES = [
    [6, 1, 2, 98, 327],  # nose: bridge -> tip -> under nose -> nostrils
    [172, 152, 397],  # chin: left -> center -> right
    [50, 101, 36],  # left cheek
    [280, 330, 266],  # right cheek
]
EXTRA_FACE_FULL_EDGES = sorted(
    {
        tuple(sorted((int(poly[i]), int(poly[i + 1]))))
        for poly in EXTRA_FACE_FULL_POLYLINES
        for i in range(len(poly) - 1)
    }
)
EXTRA_FACE_NMM_EDGES = [
    (NMM_INDEX_MAP[a], NMM_INDEX_MAP[b])
    for (a, b) in EXTRA_FACE_FULL_EDGES
    if a in NMM_INDEX_MAP and b in NMM_INDEX_MAP
]

FACE_CONNECTION_SETS = [
    mp.solutions.face_mesh_connections.FACEMESH_LIPS,
    mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYE,
    mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYE,
    mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYEBROW,
    mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYEBROW,
]
if hasattr(mp.solutions.face_mesh_connections, "FACEMESH_LEFT_IRIS"):
    FACE_CONNECTION_SETS.append(mp.solutions.face_mesh_connections.FACEMESH_LEFT_IRIS)
if hasattr(mp.solutions.face_mesh_connections, "FACEMESH_RIGHT_IRIS"):
    FACE_CONNECTION_SETS.append(mp.solutions.face_mesh_connections.FACEMESH_RIGHT_IRIS)
FACE_FULL_CONNECTIONS = sorted(
    {tuple(sorted((int(a), int(b)))) for conn_set in FACE_CONNECTION_SETS for (a, b) in conn_set}
)
FACE_FULL_DRAW_POINTS = sorted(
    {i for e in FACE_FULL_CONNECTIONS for i in e} | {i for poly in EXTRA_FACE_FULL_POLYLINES for i in poly}
)

POSE_LINE = (255, 255, 255)
POSE_DOT = (160, 160, 160)
LEFT_COLOR = (255, 0, 0)
RIGHT_COLOR = (0, 255, 0)
FACE_COLOR = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize keypoints from extracted HDF5/chunks.")
    p.add_argument("--video", required=True, help="Video ID, e.g. kun-uz-ep001")
    p.add_argument("--mode", choices=["full", "chunk"], required=True)
    p.add_argument("--chunk", type=int, help="Chunk number (required for --mode chunk)")
    p.add_argument("--show-original", action="store_true")
    p.add_argument("--keypoints-dir", default="data/keypoints/news")
    p.add_argument("--chunks-dir", default="data/chunks/news")
    p.add_argument("--index-path", default="data/chunks/news/index.jsonl")
    p.add_argument("--cleaned-videos-dir", default="data/processed/news_cleaned/videos")
    p.add_argument("--output-dir", default="data/debug")
    return p.parse_args()


def color_scale(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return (int(color[0] * scale), int(color[1] * scale), int(color[2] * scale))


def point_visible(xyz: np.ndarray, conf: float | None = None) -> bool:
    if conf is not None and conf <= 0:
        return False
    return bool(np.any(np.abs(xyz) > 1e-8))


def to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return (int(np.clip(x, 0.0, 1.0) * (w - 1)), int(np.clip(y, 0.0, 1.0) * (h - 1)))


def draw_pose(canvas: np.ndarray, pose: np.ndarray, conf_pose: np.ndarray, dim: float) -> None:
    line_color = color_scale(POSE_LINE, dim)
    dot_color = color_scale(POSE_DOT, dim)
    h, w = canvas.shape[:2]

    for a, b in POSE_CONNECTIONS:
        if not (point_visible(pose[a, :3]) and point_visible(pose[b, :3])):
            continue
        pa = to_px(float(pose[a, 0]), float(pose[a, 1]), w, h)
        pb = to_px(float(pose[b, 0]), float(pose[b, 1]), w, h)
        cv2.line(canvas, pa, pb, line_color, 2, cv2.LINE_AA)

    for i in POSE_DOT_IDXS:
        if not point_visible(pose[i, :3]):
            continue
        p = to_px(float(pose[i, 0]), float(pose[i, 1]), w, h)
        cv2.circle(canvas, p, 3, dot_color, -1, cv2.LINE_AA)


def draw_hand(
    canvas: np.ndarray,
    hand: np.ndarray,
    conf_hand: np.ndarray,
    color: tuple[int, int, int],
    dim: float,
) -> None:
    c = color_scale(color, dim)
    h, w = canvas.shape[:2]
    for a, b in HAND_CONNECTIONS:
        if not (point_visible(hand[a], conf_hand[a]) and point_visible(hand[b], conf_hand[b])):
            continue
        pa = to_px(float(hand[a, 0]), float(hand[a, 1]), w, h)
        pb = to_px(float(hand[b, 0]), float(hand[b, 1]), w, h)
        cv2.line(canvas, pa, pb, c, 1, cv2.LINE_AA)
    for i in range(21):
        if not point_visible(hand[i], conf_hand[i]):
            continue
        p = to_px(float(hand[i, 0]), float(hand[i, 1]), w, h)
        cv2.circle(canvas, p, 2, c, -1, cv2.LINE_AA)


def draw_face_nmm(canvas: np.ndarray, face_nmm: np.ndarray, color: tuple[int, int, int], dim: float) -> None:
    c = color_scale(color, dim)
    h, w = canvas.shape[:2]
    for poly in FACE_POLYLINES:
        pts: list[tuple[int, int]] = []
        for idx in poly:
            xyz = face_nmm[idx]
            if not point_visible(xyz):
                continue
            pts.append(to_px(float(xyz[0]), float(xyz[1]), w, h))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, c, 1, cv2.LINE_AA)
    for a, b in EXTRA_FACE_NMM_EDGES:
        if not (point_visible(face_nmm[a]) and point_visible(face_nmm[b])):
            continue
        pa = to_px(float(face_nmm[a, 0]), float(face_nmm[a, 1]), w, h)
        pb = to_px(float(face_nmm[b, 0]), float(face_nmm[b, 1]), w, h)
        cv2.line(canvas, pa, pb, c, 1, cv2.LINE_AA)
    for i in range(face_nmm.shape[0]):
        if not point_visible(face_nmm[i]):
            continue
        p = to_px(float(face_nmm[i, 0]), float(face_nmm[i, 1]), w, h)
        cv2.circle(canvas, p, 1, c, -1, cv2.LINE_AA)


def draw_face_full(
    canvas: np.ndarray,
    face_full: np.ndarray,
    conf_face: np.ndarray | None,
    color: tuple[int, int, int],
    dim: float,
) -> None:
    c = color_scale(color, dim)
    h, w = canvas.shape[:2]

    for a, b in FACE_FULL_CONNECTIONS:
        conf_ok = True
        if conf_face is not None:
            conf_ok = bool(conf_face[a] > 0 and conf_face[b] > 0)
        if not conf_ok:
            continue
        if not (point_visible(face_full[a]) and point_visible(face_full[b])):
            continue
        pa = to_px(float(face_full[a, 0]), float(face_full[a, 1]), w, h)
        pb = to_px(float(face_full[b, 0]), float(face_full[b, 1]), w, h)
        cv2.line(canvas, pa, pb, c, 1, cv2.LINE_AA)

    for a, b in EXTRA_FACE_FULL_EDGES:
        conf_ok = True
        if conf_face is not None:
            conf_ok = bool(conf_face[a] > 0 and conf_face[b] > 0)
        if not conf_ok:
            continue
        if not (point_visible(face_full[a]) and point_visible(face_full[b])):
            continue
        pa = to_px(float(face_full[a, 0]), float(face_full[a, 1]), w, h)
        pb = to_px(float(face_full[b, 0]), float(face_full[b, 1]), w, h)
        cv2.line(canvas, pa, pb, c, 1, cv2.LINE_AA)

    for i in FACE_FULL_DRAW_POINTS:
        conf_ok = True
        if conf_face is not None:
            conf_ok = bool(conf_face[i] > 0)
        if not conf_ok or not point_visible(face_full[i]):
            continue
        p = to_px(float(face_full[i, 0]), float(face_full[i, 1]), w, h)
        cv2.circle(canvas, p, 1, c, -1, cv2.LINE_AA)


def draw_keypoints_frame(
    pose: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    face_nmm: np.ndarray,
    face_full: np.ndarray | None,
    conf_pose: np.ndarray,
    conf_lh: np.ndarray,
    conf_rh: np.ndarray,
    conf_face: np.ndarray | None,
    frame_quality: float,
    frame_num_text: str,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    dim = 0.3 if frame_quality < 0.5 else 1.0
    draw_pose(canvas, pose, conf_pose, dim)
    draw_hand(canvas, left_hand, conf_lh, LEFT_COLOR, dim)
    draw_hand(canvas, right_hand, conf_rh, RIGHT_COLOR, dim)
    if face_full is not None and face_full.shape[0] == 478 and np.any(np.abs(face_full) > 1e-8):
        draw_face_full(canvas, face_full, conf_face, FACE_COLOR, dim)
    else:
        draw_face_nmm(canvas, face_nmm, FACE_COLOR, dim)
    cv2.putText(
        canvas,
        f"{frame_num_text} | quality={frame_quality:.3f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_scale((255, 255, 255), dim),
        2,
        cv2.LINE_AA,
    )
    return canvas


def load_chunk_start(index_path: Path, chunk_id: str) -> int:
    if not index_path.exists():
        return 0
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("chunk_id") == chunk_id:
                return int(row.get("start_frame", 0))
    return 0


def full_mode(args: argparse.Namespace, project_root: Path) -> None:
    video_id = args.video
    h5_path = project_root / args.keypoints_dir / f"{video_id}.h5"
    if not h5_path.exists():
        raise SystemExit(f"Missing HDF5: {h5_path}")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.show_original:
        out_path = output_dir / f"{video_id}_full_sidebyside.mp4"
    else:
        out_path = output_dir / f"{video_id}_full_keypoints.mp4"

    with h5py.File(h5_path, "r") as h5:
        total = int(h5.attrs["total_frames"])
        fps = float(h5.attrs.get("fps", 25.0))
        w = int(h5.attrs.get("frame_w", 0))
        h = int(h5.attrs.get("frame_h", 0))

        cap = None
        if args.show_original or w <= 0 or h <= 0:
            clean_video = project_root / args.cleaned_videos_dir / f"{video_id}.cropped.cleaned.mp4"
            cap = cv2.VideoCapture(str(clean_video))
            if not cap.isOpened():
                raise SystemExit(f"Cannot open cleaned video for visualization: {clean_video}")
            if w <= 0 or h <= 0:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w = w * 2 if args.show_original else w
        out_h = h
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )

        for i in range(total):
            key = draw_keypoints_frame(
                pose=h5["pose"][i],
                left_hand=h5["left_hand"][i],
                right_hand=h5["right_hand"][i],
                face_nmm=h5["face_nmm"][i],
                face_full=h5["face_full"][i] if "face_full" in h5 else None,
                conf_pose=h5["confidence/pose"][i],
                conf_lh=h5["confidence/left_hand"][i],
                conf_rh=h5["confidence/right_hand"][i],
                conf_face=h5["confidence/face"][i] if "confidence/face" in h5 else None,
                frame_quality=float(h5["frame_quality"][i]),
                frame_num_text=f"frame {i}",
                width=w,
                height=h,
            )

            if args.show_original:
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                combo = np.hstack([frame, key])
                cv2.line(combo, (w, 0), (w, h - 1), (255, 255, 255), 1)
                text = f"frame {i}"
                tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(
                    combo,
                    text,
                    ((combo.shape[1] - tw) // 2, combo.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(combo)
            else:
                writer.write(key)

        writer.release()
        if cap is not None:
            cap.release()

    print(f"Saved: {out_path.as_posix()}")


def chunk_mode(args: argparse.Namespace, project_root: Path) -> None:
    if args.chunk is None:
        raise SystemExit("--chunk is required when --mode chunk")
    video_id = args.video

    chunk_id = f"{video_id}_chunk_{args.chunk:05d}"
    chunk_path = project_root / args.chunks_dir / f"{chunk_id}.npz"
    if not chunk_path.exists():
        raise SystemExit(f"Missing chunk file: {chunk_path}")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.show_original:
        out_path = output_dir / f"{video_id}_chunk_{args.chunk:03d}_sidebyside.mp4"
    else:
        out_path = output_dir / f"{video_id}_chunk_{args.chunk:03d}_keypoints.mp4"

    h5_path = project_root / args.keypoints_dir / f"{video_id}.h5"
    fps = 25.0
    w = h = 0
    total_h5_frames = 0
    if h5_path.exists():
        with h5py.File(h5_path, "r") as h5:
            fps = float(h5.attrs.get("fps", 25.0))
            w = int(h5.attrs.get("frame_w", 0))
            h = int(h5.attrs.get("frame_h", 0))
            total_h5_frames = int(h5.attrs.get("total_frames", 0))

    data = np.load(chunk_path)
    chunk_len = data["pose"].shape[0]
    start_frame = load_chunk_start(project_root / args.index_path, chunk_id)
    if w <= 0 or h <= 0:
        clean_video = project_root / args.cleaned_videos_dir / f"{video_id}.cropped.cleaned.mp4"
        cap_size = cv2.VideoCapture(str(clean_video))
        if not cap_size.isOpened():
            raise SystemExit(f"Cannot open cleaned video: {clean_video}")
        w = int(cap_size.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_size.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_size.release()

    out_w = w * 2 if args.show_original else w
    out_h = h
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    cap = None
    if args.show_original:
        clean_video = project_root / args.cleaned_videos_dir / f"{video_id}.cropped.cleaned.mp4"
        cap = cv2.VideoCapture(str(clean_video))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open cleaned video for side-by-side: {clean_video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    face_full_slice = None
    conf_face_slice = None
    if h5_path.exists() and start_frame + chunk_len <= total_h5_frames:
        with h5py.File(h5_path, "r") as h5:
            if "face_full" in h5:
                face_full_slice = h5["face_full"][start_frame : start_frame + chunk_len]
            if "confidence/face" in h5:
                conf_face_slice = h5["confidence/face"][start_frame : start_frame + chunk_len]

    for i in range(chunk_len):
        frame_no = start_frame + i
        key = draw_keypoints_frame(
            pose=data["pose"][i],
            left_hand=data["left_hand"][i],
            right_hand=data["right_hand"][i],
            face_nmm=data["face_nmm"][i],
            face_full=face_full_slice[i] if face_full_slice is not None else None,
            conf_pose=data["confidence_pose"][i],
            conf_lh=data["confidence_lh"][i],
            conf_rh=data["confidence_rh"][i],
            conf_face=conf_face_slice[i] if conf_face_slice is not None else None,
            frame_quality=float(data["frame_quality"][i]),
            frame_num_text=f"frame {frame_no}",
            width=w,
            height=h,
        )

        if args.show_original:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            combo = np.hstack([frame, key])
            cv2.line(combo, (w, 0), (w, h - 1), (255, 255, 255), 1)
            text = f"frame {frame_no}"
            tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(
                combo,
                text,
                ((combo.shape[1] - tw) // 2, combo.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(combo)
        else:
            writer.write(key)

    writer.release()
    if cap is not None:
        cap.release()
    data.close()
    print(f"Saved: {out_path.as_posix()}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    if args.mode == "full":
        full_mode(args, project_root)
    else:
        chunk_mode(args, project_root)


if __name__ == "__main__":
    main()
