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
    python keypoints/extract_keypoints.py
    python keypoints/extract_keypoints.py --signer signer01 --topic Alifbo
    python keypoints/extract_keypoints.py --force
    python keypoints/extract_keypoints.py --model-complexity 2
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np

# ---------------------------------------------------------------------------
# Locate project root so imports work regardless of CWD
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parent   # keypoints/
_PROJECT = _ROOT.parent                      # project root
sys.path.insert(0, str(_ROOT))               # keypoint_extraction.py
sys.path.insert(0, str(_PROJECT / "app"))    # mod01_config.py

from keypoint_extraction import extract_from_video  # noqa: E402
from mod01_config import DATA_ROOT as _DATA_ROOT_STR, POSE_REMOVE_IDX  # noqa: E402

_DEFAULT_ROOT = (_PROJECT / _DATA_ROOT_STR).resolve()


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
    ap.add_argument("--exec", choices=["local", "remote"], default="local",
                    help="Run locally or execute remote extraction")
    ap.add_argument("--remote-host", default="akmalov@100.77.75.72",
                    help="SSH host for remote execution")
    ap.add_argument("--remote-project", default="~/record",
                    help="Remote project root containing keypoints/")
    ap.add_argument("--remote-python", default=".venv/bin/python",
                    help="Python executable path on remote")
    ap.add_argument("--remote-data-root", default=None,
                    help="Data root path on remote (default: same as --data-root)")
    ap.add_argument("--topic",  default=None, metavar="TOPIC",
                    help="Process only this topic folder")
    ap.add_argument("--signer", default=None, metavar="SIGNER",
                    help="Process only this signer (e.g. signer01)")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if keypoints.npz already exists")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                    metavar="{0,1,2}", help="MediaPipe model complexity (default: 1)")
    args = ap.parse_args()

    if args.exec == "remote":
        remote_data_root = args.remote_data_root or str(args.data_root)
        cmd = [
            str(args.remote_project) + "/" + str(Path("keypoints") / "extract_keypoints.py"),
            "--data-root", remote_data_root,
        ]
        if args.topic:
            cmd += ["--topic", args.topic]
        if args.signer:
            cmd += ["--signer", args.signer]
        if args.force:
            cmd += ["--force"]
        if args.model_complexity != 1:
            cmd += ["--model-complexity", str(args.model_complexity)]
        quoted = " ".join(shlex.quote(c) for c in cmd)
        remote_py = shlex.quote(str(args.remote_python))
        remote_proj = shlex.quote(str(args.remote_project))
        ssh_host = shlex.quote(str(args.remote_host))
        ssh_cmd = f"ssh {ssh_host} \"cd {remote_proj} && {remote_py} {quoted}\""
        print("Remote extraction command:")
        print(ssh_cmd)
        subprocess.run(ssh_cmd, shell=True, check=True)

        sync_roots: list[str] = []
        if args.topic and args.signer:
            sync_roots = [f"{remote_data_root}/{args.topic}/{args.signer}"]
        elif args.topic:
            sync_roots = [f"{remote_data_root}/{args.topic}"]
        elif args.signer:
            find_cmd = (
                "find "
                + shlex.quote(remote_data_root)
                + " -maxdepth 2 -mindepth 2 -type d -name "
                + shlex.quote(args.signer)
            )
            list_cmd = f"ssh {ssh_host} {shlex.quote(find_cmd)}"
            result = subprocess.run(list_cmd, shell=True, check=True, capture_output=True, text=True)
            sync_roots = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        else:
            sync_roots = [remote_data_root]

        if not sync_roots:
            print("No matching remote paths found to sync.")
            return 0

        for remote_root in sync_roots:
            remote_root_posix = PurePosixPath(remote_root)
            remote_data_posix = PurePosixPath(remote_data_root)
            try:
                rel_root = remote_root_posix.relative_to(remote_data_posix)
            except ValueError:
                rel_root = PurePosixPath()
            local_root = args.data_root / rel_root.as_posix()
            rsync_cmd = [
                "rsync", "-az", "--prune-empty-dirs",
                "--include", "*/",
                "--include", "keypoints/***",
                "--exclude", "*",
                f"{args.remote_host}:{remote_root}/",
                str(local_root) + "/",
            ]
            print("Syncing keypoints:")
            print(" ".join(shlex.quote(part) for part in rsync_cmd))
            subprocess.run(rsync_cmd, check=True)
        return 0


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
