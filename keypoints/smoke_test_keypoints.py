"""
Smoke test for the keypoint extraction pipeline.

Finds the first video.mp4 under DATA_ROOT, runs extract_from_video(),
checks all array shapes and value ranges, writes keypoints.npz, reloads it,
and verifies the reloaded arrays match.

Usage:
    python keypoints/smoke_test_keypoints.py
    python keypoints/smoke_test_keypoints.py --data-root /path/to/data
    python keypoints/smoke_test_keypoints.py --video /explicit/path/video.mp4

Exit code: 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT    = Path(__file__).resolve().parent   # keypoints/
_PROJECT = _ROOT.parent                      # project root
sys.path.insert(0, str(_ROOT))               # keypoint_extraction.py
sys.path.insert(0, str(_PROJECT / "app"))    # mod01_config.py

from keypoint_extraction import extract_from_video, N_POSE, N_HAND, N_FACE_NMM  # noqa: E402
from mod01_config import DATA_ROOT as _DATA_ROOT_STR, POSE_REMOVE_IDX           # noqa: E402

_POSE_KEEP_IDX = [i for i in range(N_POSE) if i not in set(POSE_REMOVE_IDX)]

_DEFAULT_ROOT = (_PROJECT / _DATA_ROOT_STR).resolve()

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


# ── Path helpers (mirrors keypoints/extract_keypoints.py) ────────────────────

def default_kp_args() -> argparse.Namespace:
    return argparse.Namespace(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face=False,
        no_face=False,
        max_frames=0,
        overlay=False,
    )


def video_to_kp_path(video_path: Path) -> Path:
    rep_dir  = video_path.parent
    sign_dir = rep_dir.parent.parent
    return sign_dir / "keypoints" / rep_dir.name / "keypoints.npz"


def find_first_video(data_root: Path) -> Path | None:
    for v in sorted(data_root.rglob("video.mp4")):
        parts = v.relative_to(data_root).parts
        if len(parts) >= 6:
            return v
    return None


# ── Checks ────────────────────────────────────────────────────────────────────

def run_checks(video_path: Path) -> int:
    fails = 0

    # 1 — video found
    _pass(f"Found video: {video_path}")

    # 2 — extraction runs without exception
    result = None
    try:
        result = extract_from_video(video_path, default_kp_args())
        _pass("extract_from_video() completed without exception")
    except Exception:
        _fail("extract_from_video() raised an exception:")
        traceback.print_exc()
        fails += 1
        return fails   # further checks all need result

    T = result.pose.shape[0]

    # 3 — pose shape
    expected_pose = (T, N_POSE, 4)
    if result.pose.shape == expected_pose and T > 0:
        _pass(f"pose.shape == {result.pose.shape}")
    else:
        _fail(f"pose.shape expected (T>0, {N_POSE}, 4), got {result.pose.shape}")
        fails += 1

    # 4 — left_hand shape
    expected_hand = (T, N_HAND, 3)
    if result.left_hand.shape == expected_hand:
        _pass(f"left_hand.shape == {result.left_hand.shape}")
    else:
        _fail(f"left_hand.shape expected {expected_hand}, got {result.left_hand.shape}")
        fails += 1

    # 5 — right_hand shape
    if result.right_hand.shape == expected_hand:
        _pass(f"right_hand.shape == {result.right_hand.shape}")
    else:
        _fail(f"right_hand.shape expected {expected_hand}, got {result.right_hand.shape}")
        fails += 1

    # 6 — face_nmm shape
    expected_face = (T, N_FACE_NMM, 3)
    if result.face_nmm.shape == expected_face:
        _pass(f"face_nmm.shape == {result.face_nmm.shape}")
    else:
        _fail(f"face_nmm.shape expected {expected_face}, got {result.face_nmm.shape}")
        fails += 1

    # 7 — x/y coords of kept pose landmarks in [-0.15, 1.15]
    # Only check the 6 arm landmarks (11-16); removed indices are zeroed and skipped.
    xy = result.pose[:, _POSE_KEEP_IDX, :2]
    lo, hi = float(xy.min()), float(xy.max())
    if -0.15 <= lo and hi <= 1.15:
        _pass(f"kept pose x/y coords in [-0.15, 1.15]  (min={lo:.4f}, max={hi:.4f})")
    else:
        _fail(f"kept pose x/y coords out of expected range: min={lo:.4f}, max={hi:.4f}")
        fails += 1

    # 8 — npz written to canonical path and is readable
    out_path = video_to_kp_path(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(str(out_path), **result.to_npz_dict())
        if out_path.exists():
            _pass(f"keypoints.npz written: {out_path}")
        else:
            _fail(f"np.savez_compressed ran but file not found at {out_path}")
            fails += 1
    except Exception as exc:
        _fail(f"Failed to write keypoints.npz: {exc}")
        fails += 1
        return fails

    # 9 — reloaded arrays match original shapes
    try:
        loaded = np.load(str(out_path))
        shape_checks = [
            ("pose",       result.pose.shape),
            ("left_hand",  result.left_hand.shape),
            ("right_hand", result.right_hand.shape),
            ("face_nmm",   result.face_nmm.shape),
            ("detected",   result.detected.shape),
        ]
        all_ok = True
        for key, expected in shape_checks:
            if loaded[key].shape != expected:
                _fail(f"Reloaded '{key}' shape {loaded[key].shape} != original {expected}")
                fails += 1
                all_ok = False
        if all_ok:
            _pass("Reloaded arrays match original shapes")
    except Exception as exc:
        _fail(f"Failed to reload keypoints.npz: {exc}")
        fails += 1

    return fails


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=_DEFAULT_ROOT, metavar="PATH")
    ap.add_argument("--video",     type=Path, default=None, metavar="PATH",
                    help="Use a specific video instead of auto-discovering the first one")
    args = ap.parse_args()

    print(f"\n{BOLD}Keypoint extraction smoke test{RESET}")
    print(f"Data root: {args.data_root}\n")

    # Check 1: find a video
    if args.video:
        if not args.video.exists():
            print(f"  {RED}FAIL{RESET}  Video not found: {args.video}", file=sys.stderr)
            return 1
        video_path = args.video
    else:
        video_path = find_first_video(args.data_root)
        if video_path is None:
            print(f"  {RED}FAIL{RESET}  No video.mp4 found under {args.data_root}", file=sys.stderr)
            return 1

    fails = run_checks(video_path)

    total = 9
    passed = total - fails
    print()
    if fails == 0:
        print(f"{BOLD}{GREEN}All {total} checks passed.{RESET}")
    else:
        print(f"{BOLD}{RED}{fails}/{total} checks failed.{RESET}")

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
