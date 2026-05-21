"""
Keypoint visualizer — skeleton and side-by-side modes.

Given a rep directory (e.g. .../sign/videos/rep-0), reads the corresponding
keypoints.npz from .../sign/keypoints/rep-0/keypoints.npz and renders an
output video saved to debug/ at the project root.

Output filename pattern:
    debug/{topic}__{signer}__{sign}__{rep}__{mode}.mp4

Modes:
  skeleton    — 1280×720 black canvas; pose (white lines / gray dots),
                LH (blue), RH (green), face (cyan — 478-pt mesh when available,
                74-pt NMM fallback otherwise).
  sidebyside  — original video (left) + skeleton canvas (right) concatenated.

Usage:
    python keypoints/visualize_keypoints.py --rep .../Alifbo/signer01/A/videos/rep-0
    python keypoints/visualize_keypoints.py --rep ... --mode sidebyside
    python keypoints/visualize_keypoints.py --rep ... --mode skeleton --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT    = Path(__file__).resolve().parent   # keypoints/
_PROJECT = _ROOT.parent                      # project root
sys.path.insert(0, str(_ROOT))               # keypoint_extraction.py
sys.path.insert(0, str(_PROJECT / "app"))    # mod01_config.py

from keypoint_extraction import NMM_INDICES  # noqa: E402
from mod01_config import DATA_ROOT as _DATA_ROOT_STR, POSE_KEEP_CONNECTIONS  # noqa: E402
import mediapipe as mp  # noqa: E402

_DEFAULT_ROOT = (_PROJECT / _DATA_ROOT_STR).resolve()
_DEBUG_DIR    = _PROJECT / "debug"

# ── Skeleton colours (BGR) — matches reference machine ───────────────────────
COL_POSE_LINE = (255, 255, 255)   # white  — lines
COL_POSE_DOT  = (160, 160, 160)   # gray   — dots (reference uses separate colors)
COL_LH        = (255,   0,   0)   # blue  (BGR)
COL_RH        = (  0, 255,   0)   # green (BGR)
COL_FACE      = (  0, 255, 255)   # yellow (BGR)

POSE_DOT_IDXS = [11, 12, 13, 14, 15, 16]  # only these get dots

CANVAS_W, CANVAS_H = 1280, 720

# ── MediaPipe connections ─────────────────────────────────────────────────────
_holistic = mp.solutions.holistic
HAND_CONNECTIONS = list(_holistic.HAND_CONNECTIONS)

# ── face_full (478-point mesh) connection sets ────────────────────────────────
_fmc = mp.solutions.face_mesh_connections
_FACE_FULL_CONN_SETS = [
    _fmc.FACEMESH_LIPS,
    _fmc.FACEMESH_LEFT_EYE,      _fmc.FACEMESH_RIGHT_EYE,
    _fmc.FACEMESH_LEFT_EYEBROW,  _fmc.FACEMESH_RIGHT_EYEBROW,
]
for _attr in ("FACEMESH_LEFT_IRIS", "FACEMESH_RIGHT_IRIS"):
    if hasattr(_fmc, _attr):
        _FACE_FULL_CONN_SETS.append(getattr(_fmc, _attr))

FACE_FULL_CONNECTIONS = sorted(
    {tuple(sorted((int(a), int(b)))) for cs in _FACE_FULL_CONN_SETS for a, b in cs}
)

_EXTRA_FULL_POLYS = [
    [6, 1, 2, 98, 327],   # nose bridge → tip → under-nose → nostrils
    [172, 152, 397],       # chin left → center → right
    [50, 101, 36],         # left cheek
    [280, 330, 266],       # right cheek
]
_EXTRA_FULL_EDGES = sorted(
    {tuple(sorted((p[i], p[i + 1]))) for p in _EXTRA_FULL_POLYS for i in range(len(p) - 1)}
)
FACE_FULL_DRAW_POINTS = sorted(
    {i for e in FACE_FULL_CONNECTIONS for i in e}
  | {i for p in _EXTRA_FULL_POLYS for i in p}
)

# ── face_nmm (74-point fallback) polylines — correct drawing order ────────────
_NMM_MAP = {v: i for i, v in enumerate(NMM_INDICES)}

_FACE_NMM_POLYS_FULL = [
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],                                    # L eyebrow
    [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],                            # R eyebrow
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,                                 # mouth outer
     291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
    [78, 80, 81, 82, 13, 312, 311, 310, 78],                                       # mouth inner
    [159, 160, 161, 144, 145, 153, 159],                                           # L eye
    [386, 387, 388, 373, 374, 380, 386],                                           # R eye
]
FACE_NMM_POLYLINES = [
    [_NMM_MAP[i] for i in poly if i in _NMM_MAP]
    for poly in _FACE_NMM_POLYS_FULL
]
FACE_NMM_EXTRA_EDGES = [
    (a, b)
    for a, b in [(_NMM_MAP.get(ea), _NMM_MAP.get(eb)) for ea, eb in _EXTRA_FULL_EDGES]
    if a is not None and b is not None
]


# ── Path helpers ──────────────────────────────────────────────────────────────

def rep_to_kp_path(rep_dir: Path) -> Path:
    """videos/rep-N  →  keypoints/rep-N/keypoints.npz"""
    sign_dir = rep_dir.parent.parent    # skip videos/
    return sign_dir / "keypoints" / rep_dir.name / "keypoints.npz"


def parse_rep_parts(rep_dir: Path, data_root: Path) -> tuple[str, str, str, str]:
    """Return (topic, signer, sign, rep_name) from a rep directory path."""
    try:
        rel = rep_dir.resolve().relative_to(data_root.resolve())
        parts = rel.parts   # topic / signer / sign / videos / rep-N
        return parts[0], parts[1], parts[2], parts[4]
    except (ValueError, IndexError):
        # fallback: use last 3 path components
        p = rep_dir.resolve().parts
        return "unknown", "unknown", p[-3] if len(p) >= 3 else "unknown", rep_dir.name


# ── Drawing ───────────────────────────────────────────────────────────────────

def _to_px(xy: np.ndarray, w: int, h: int) -> tuple[int, int]:
    return (int(np.clip(float(xy[0]), 0.0, 1.0) * (w - 1)),
            int(np.clip(float(xy[1]), 0.0, 1.0) * (h - 1)))


def draw_skeleton(kp: dict[str, np.ndarray], t: int,
                  canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H) -> np.ndarray:
    """Draw keypoints for frame t onto a black canvas. Returns BGR image."""
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    W, H = canvas_w, canvas_h

    pose = kp["pose"][t]          # (33, 4)
    lh   = kp["left_hand"][t]     # (21, 3)
    rh   = kp["right_hand"][t]    # (21, 3)
    face = kp["face_nmm"][t]      # (74, 3)

    def _point_visible(xy: np.ndarray) -> bool:
        return bool(np.any(np.abs(xy) > 1e-8))

    def draw_lines(pts, conns, colour, thickness=2):
        for a, b in conns:
            if a >= len(pts) or b >= len(pts):
                continue
            if not (_point_visible(pts[a]) and _point_visible(pts[b])):
                continue
            pa = _to_px(pts[a], W, H)
            pb = _to_px(pts[b], W, H)
            cv2.line(canvas, pa, pb, colour, thickness, cv2.LINE_AA)

    def draw_dots(pts, idxs, colour, radius=2):
        for i in idxs:
            if i >= len(pts) or not _point_visible(pts[i]):
                continue
            p = _to_px(pts[i], W, H)
            cv2.circle(canvas, p, radius, colour, -1, cv2.LINE_AA)

    # pose — white lines, gray dots only at the 6 upper-body joints
    draw_lines(pose[:, :2], POSE_KEEP_CONNECTIONS, COL_POSE_LINE, thickness=2)
    draw_dots(pose[:, :2], POSE_DOT_IDXS, COL_POSE_DOT, radius=3)

    # hands — full 21-point skeleton + dots at every landmark
    draw_lines(lh[:, :2], HAND_CONNECTIONS, COL_LH, thickness=1)
    draw_dots(lh[:, :2], range(21), COL_LH, radius=2)
    draw_lines(rh[:, :2], HAND_CONNECTIONS, COL_RH, thickness=1)
    draw_dots(rh[:, :2], range(21), COL_RH, radius=2)

    # face — use 478-point dense mesh when available, fallback to 74-point NMM
    face_full_arr = kp.get("face_full")
    if (face_full_arr is not None
            and face_full_arr.ndim == 3
            and face_full_arr.shape[1] == 478
            and np.any(np.abs(face_full_arr[t]) > 1e-8)):
        draw_face_full(canvas, face_full_arr[t], COL_FACE)
    else:
        draw_face_nmm(canvas, face, COL_FACE)

    return canvas


def _pv(xy: np.ndarray) -> bool:
    return bool(np.any(np.abs(xy) > 1e-8))


def draw_face_full(canvas: np.ndarray, face_full: np.ndarray, colour: tuple) -> None:
    W, H = canvas.shape[1], canvas.shape[0]
    for a, b in FACE_FULL_CONNECTIONS:
        if not (_pv(face_full[a]) and _pv(face_full[b])):
            continue
        cv2.line(canvas, _to_px(face_full[a], W, H), _to_px(face_full[b], W, H),
                 colour, 1, cv2.LINE_AA)
    for a, b in _EXTRA_FULL_EDGES:
        if not (_pv(face_full[a]) and _pv(face_full[b])):
            continue
        cv2.line(canvas, _to_px(face_full[a], W, H), _to_px(face_full[b], W, H),
                 colour, 1, cv2.LINE_AA)
    for i in FACE_FULL_DRAW_POINTS:
        if _pv(face_full[i]):
            cv2.circle(canvas, _to_px(face_full[i], W, H), 1, colour, -1, cv2.LINE_AA)


def draw_face_nmm(canvas: np.ndarray, face_nmm: np.ndarray, colour: tuple) -> None:
    W, H = canvas.shape[1], canvas.shape[0]
    for poly in FACE_NMM_POLYLINES:
        pts = [_to_px(face_nmm[i], W, H) for i in poly if _pv(face_nmm[i])]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, colour, 1, cv2.LINE_AA)
    for a, b in FACE_NMM_EXTRA_EDGES:
        if not (_pv(face_nmm[a]) and _pv(face_nmm[b])):
            continue
        cv2.line(canvas, _to_px(face_nmm[a], W, H), _to_px(face_nmm[b], W, H),
                 colour, 1, cv2.LINE_AA)
    for i in range(len(face_nmm)):
        if _pv(face_nmm[i]):
            cv2.circle(canvas, _to_px(face_nmm[i], W, H), 1, colour, -1, cv2.LINE_AA)


# ── Writer helpers ────────────────────────────────────────────────────────────

def make_writer(out_path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    return writer


# ── Render modes ──────────────────────────────────────────────────────────────

def render_skeleton(kp: dict[str, np.ndarray], out_path: Path, fps: float) -> None:
    T = kp["pose"].shape[0]
    writer = make_writer(out_path, fps, CANVAS_W, CANVAS_H)
    for t in range(T):
        writer.write(draw_skeleton(kp, t))
    writer.release()


def render_sidebyside(kp: dict[str, np.ndarray], video_path: Path,
                      out_path: Path, fps: float) -> None:
    T = kp["pose"].shape[0]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = orig_w + CANVAS_W
    out_h = max(orig_h, CANVAS_H)

    writer = make_writer(out_path, fps, out_w, out_h)

    for t in range(T):
        ok, frame_bgr = cap.read()
        if not ok:
            frame_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # resize original to canvas height if needed
        if orig_h != CANVAS_H:
            scale = CANVAS_H / orig_h
            frame_bgr = cv2.resize(frame_bgr, (int(orig_w * scale), CANVAS_H))
        left_w = frame_bgr.shape[1]

        skel = draw_skeleton(kp, t, CANVAS_W, CANVAS_H)

        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:frame_bgr.shape[0], :left_w] = frame_bgr
        combined[:CANVAS_H, left_w:left_w + CANVAS_W] = skel

        writer.write(combined)

    cap.release()
    writer.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep",  type=Path, required=True, metavar="REP_DIR",
                    help="Path to a rep directory (e.g. .../videos/rep-0)")
    ap.add_argument("--mode", choices=["skeleton", "sidebyside"], default="skeleton",
                    help="Visualization mode (default: skeleton)")
    ap.add_argument("--data-root", type=Path, default=_DEFAULT_ROOT, metavar="PATH")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="Override output FPS (0 = use value from keypoints meta)")
    args = ap.parse_args()

    rep_dir  = args.rep.resolve()
    kp_path  = rep_to_kp_path(rep_dir)
    vid_path = rep_dir / "video.mp4"

    if not kp_path.exists():
        print(f"ERROR: keypoints.npz not found: {kp_path}", file=sys.stderr)
        print("  Run keypoints/extract_keypoints.py first.", file=sys.stderr)
        return 1

    if args.mode == "sidebyside" and not vid_path.exists():
        print(f"ERROR: video not found for sidebyside: {vid_path}", file=sys.stderr)
        return 1

    # Load keypoints
    raw = np.load(str(kp_path))
    kp  = {k: raw[k] for k in raw.files}

    fps = args.fps if args.fps > 0 else float(kp["meta"][0]) if "meta" in kp else 30.0
    if fps <= 0:
        fps = 30.0

    T = kp["pose"].shape[0]
    print(f"Loaded keypoints: T={T} frames, fps={fps:.1f}")

    # Build output filename
    topic, signer, sign, rep_name = parse_rep_parts(rep_dir, args.data_root)
    out_name = f"{topic}__{signer}__{sign}__{rep_name}__{args.mode}.mp4"
    out_path = _DEBUG_DIR / out_name

    print(f"Mode:   {args.mode}")
    print(f"Output: {out_path}")

    if args.mode == "skeleton":
        render_skeleton(kp, out_path, fps)
    else:
        render_sidebyside(kp, vid_path, out_path, fps)

    print(f"Done — {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
