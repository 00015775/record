"""
Batch keypoint extractor for the UzSL recording dataset.

Walks DATA_ROOT, finds every recorded video.mp4, and extracts 149 MediaPipe
keypoints (33 pose + 21 left-hand + 21 right-hand + 74 face NMM) using the
existing keypoint_extraction.py library.

Output path mirrors the videos/ directory structure:
    DATA_ROOT/topic/signer/sign/videos/rep-N/video.mp4
→   DATA_ROOT/topic/signer/sign/keypoints/rep-N/keypoints.npz

Reps whose keypoints.npz already exist are skipped unless --force is given.

Usage:
    python extract_keypoints.py
    python extract_keypoints.py --signer signer01 --topic Alifbo
    python extract_keypoints.py --force
    python extract_keypoints.py --model-complexity 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate project root so imports work regardless of CWD
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from keypoint_extraction import extract_from_video  # noqa: E402
from mod01_config import DATA_ROOT as _DATA_ROOT_STR, POSE_REMOVE_IDX  # noqa: E402

_DEFAULT_ROOT = (_ROOT / _DATA_ROOT_STR).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_kp_args(model_complexity: int = 1) -> argparse.Namespace:
    """Minimal args namespace accepted by keypoint_extraction.extract_from_video."""
    return argparse.Namespace(
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face=False,
        no_face=False,
        max_frames=0,
        overlay=False,
    )


def video_to_kp_path(video_path: Path) -> Path:
    """Derive the canonical keypoints output path from a video path.

    .../sign/videos/rep-N/video.mp4  →  .../sign/keypoints/rep-N/keypoints.npz
    """
    rep_dir  = video_path.parent        # .../sign/videos/rep-N
    sign_dir = rep_dir.parent.parent    # .../sign  (skip videos/)
    return sign_dir / "keypoints" / rep_dir.name / "keypoints.npz"


def iter_videos(data_root: Path,
                signer: str | None = None,
                topic:  str | None = None):
    """Yield all video.mp4 paths under data_root, optionally filtered."""
    for video in sorted(data_root.rglob("video.mp4")):
        # Expected structure: topic / signer / sign / videos / rep-N / video.mp4
        parts = video.relative_to(data_root).parts
        if len(parts) < 6:
            continue
        if topic  and parts[0] != topic:
            continue
        if signer and parts[1] != signer:
            continue
        yield video


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-root", type=Path, default=_DEFAULT_ROOT,
                    metavar="PATH", help="Root data directory (default: %(default)s)")
    ap.add_argument("--topic",  default=None, metavar="TOPIC",
                    help="Process only this topic folder")
    ap.add_argument("--signer", default=None, metavar="SIGNER",
                    help="Process only this signer (e.g. signer01)")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if keypoints.npz already exists")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                    metavar="{0,1,2}", help="MediaPipe model complexity (default: 1)")
    args = ap.parse_args()

    videos = list(iter_videos(args.data_root, args.signer, args.topic))
    if not videos:
        print(f"No video.mp4 files found under {args.data_root}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video(s) under {args.data_root}")
    if args.topic  or args.signer:
        filters = " | ".join(filter(None, [args.topic, args.signer]))
        print(f"Filter: {filters}")
    print()

    kp_args = default_kp_args(args.model_complexity)
    done = skipped = failed = 0

    for video_path in videos:
        parts    = video_path.relative_to(args.data_root).parts
        label    = f"{parts[0]} | {parts[1]} | {parts[2]} | {parts[4]}"
        out_path = video_to_kp_path(video_path)

        if out_path.exists() and not args.force:
            print(f"  SKIP   {label}")
            skipped += 1
            continue

        try:
            result = extract_from_video(video_path, kp_args)
            result.pose[:, POSE_REMOVE_IDX, :]   = 0.0
            result.conf_pose[:, POSE_REMOVE_IDX] = 0.0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_path), **result.to_npz_dict())

            det = result.detected.mean(axis=0)
            T   = result.pose.shape[0]
            print(
                f"  OK     {label}  "
                f"T={T:<4d}  "
                f"pose={det[0]:.2f}  lh={det[1]:.2f}  "
                f"rh={det[2]:.2f}  face={det[3]:.2f}"
            )
            done += 1
        except Exception as exc:
            print(f"  FAIL   {label}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone {done}  |  Skipped {skipped}  |  Failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
