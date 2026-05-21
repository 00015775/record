#!/usr/bin/env python3
"""
# default: process pending only, in episode order
./venv/bin/python scripts/visual_news_crop.py

# start from a specific episode (still respects pending-only unless --include-done)
./venv/bin/python scripts/visual_news_crop.py kun-uz-ep037.mp4

# include already-done episodes too
./venv/bin/python scripts/visual_news_crop.py --include-done

# recrop exactly one episode and stop immediately
./venv/bin/python scripts/visual_news_crop.py --redo kun-uz-ep037.mp4

# lower preview CPU usage
UZSL_PREVIEW_THREADS=4 ./venv/bin/python scripts/visual_news_crop.py

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    import cv2
except ImportError:
    print("OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw/news"
OUT_DIR = PROJECT_ROOT / "data/processed/news_cropped"
MANIFEST_PATH = PROJECT_ROOT / "data/sources/news_manifest.json"
PROGRESS_CSV = PROJECT_ROOT / "data/sources/news_crop_progress.csv"
PREVIEW_CACHE_DIR = PROJECT_ROOT / "data/processed/news_preview_cache"
WINDOW = "UzSL visual crop"
EP_RE = re.compile(r"kun-uz-ep(\d+)\.mp4$")


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int


state = {
    "roi": None,
    "dragging": False,
    "drag_mode": None,  # "move" | "resize" | "new"
    "drag_start": None,
    "drag_offset": (0, 0),
    "resize_edges": {"left": False, "right": False, "top": False, "bottom": False},
    "frame_size": (0, 0),
}

CSV_FIELDS = [
    "episode",
    "file_name",
    "status",
    "raw_local_path",
    "cropped_local_path",
    "crop_keep_w",
    "crop_keep_h",
    "crop_x",
    "crop_y",
    "crop_w_px",
    "crop_h_px",
    "crop_x_px",
    "crop_y_px",
    "frame_w_px",
    "frame_h_px",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual per-video crop for news signer")
    p.add_argument("start_from", nargs="?", help="Optional start file, e.g. kun-uz-ep037.mp4")
    p.add_argument(
        "--redo",
        help="Recrop exactly one episode (e.g. kun-uz-ep037.mp4), then stop.",
    )
    p.add_argument(
        "--include-done",
        action="store_true",
        help="Include already completed episodes in queue (default skips done).",
    )
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def make_even(v: int) -> int:
    return v - (v % 2)


def normalize_roi(roi: ROI, width: int, height: int) -> ROI:
    roi.x = clamp(make_even(roi.x), 0, max(0, width - 2))
    roi.y = clamp(make_even(roi.y), 0, max(0, height - 2))
    roi.w = clamp(make_even(roi.w), 2, max(2, width - roi.x))
    roi.h = clamp(make_even(roi.h), 2, max(2, height - roi.y))
    if roi.x + roi.w > width:
        roi.x = max(0, make_even(width - roi.w))
    if roi.y + roi.h > height:
        roi.y = max(0, make_even(height - roi.h))
    return roi


def to_rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def episode_num(path: Path) -> int:
    m = EP_RE.match(path.name)
    if not m:
        return 10**9
    return int(m.group(1))


def get_all_files() -> list[Path]:
    files = [p for p in RAW_DIR.glob("kun-uz-ep*.mp4") if p.is_file()]
    return sorted(files, key=episode_num)


def read_progress() -> dict[str, dict[str, str]]:
    if not PROGRESS_CSV.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with PROGRESS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("file_name", "")
            if name:
                out[name] = row
    return out


def bootstrap_progress_from_manifest(files_sorted: list[Path]) -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_local = {to_rel(p): p for p in files_sorted}
    out: dict[str, dict[str, str]] = {}
    for entry in data.get("entries", {}).values():
        raw_local = entry.get("local_path")
        ratio = entry.get("crop_ratio")
        if not raw_local or not isinstance(ratio, dict):
            continue
        p = by_local.get(raw_local)
        if not p:
            continue
        # Pixel dimensions are unknown from manifest, keep ratio fields only.
        out[p.name] = {
            "episode": str(episode_num(p)),
            "file_name": p.name,
            "status": "done",
            "raw_local_path": raw_local,
            "cropped_local_path": entry.get("cropped_local_path", ""),
            "crop_keep_w": str(ratio.get("keep_w", "")),
            "crop_keep_h": str(ratio.get("keep_h", "")),
            "crop_x": str(ratio.get("x", "")),
            "crop_y": str(ratio.get("y", "")),
            "crop_w_px": "",
            "crop_h_px": "",
            "crop_x_px": "",
            "crop_y_px": "",
            "frame_w_px": "",
            "frame_h_px": "",
            "updated_at": entry.get("cropped_at", data.get("updated_at", "")),
        }
    return out


def write_progress(progress: dict[str, dict[str, str]], files_sorted: list[Path]) -> None:
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    file_lookup = {p.name: p for p in files_sorted}
    for name, row in progress.items():
        if name in file_lookup:
            rows.append(row)
    rows.sort(key=lambda r: int(r.get("episode", "999999")))
    with PROGRESS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def update_progress(progress: dict[str, dict[str, str]], in_path: Path, out_path: Path, roi: ROI, frame_w: int, frame_h: int) -> None:
    ep = episode_num(in_path)
    progress[in_path.name] = {
        "episode": str(ep),
        "file_name": in_path.name,
        "status": "done",
        "raw_local_path": to_rel(in_path),
        "cropped_local_path": to_rel(out_path),
        "crop_keep_w": f"{roi.w / frame_w:.6f}",
        "crop_keep_h": f"{roi.h / frame_h:.6f}",
        "crop_x": f"{roi.x / frame_w:.6f}",
        "crop_y": f"{roi.y / frame_h:.6f}",
        "crop_w_px": str(roi.w),
        "crop_h_px": str(roi.h),
        "crop_x_px": str(roi.x),
        "crop_y_px": str(roi.y),
        "frame_w_px": str(frame_w),
        "frame_h_px": str(frame_h),
        "updated_at": now_iso(),
    }


def find_roi_from_last_done(progress: dict[str, dict[str, str]], files_sorted: list[Path], start_ep: int) -> Optional[tuple[ROI, int, int]]:
    prev_rows = []
    for p in files_sorted:
        ep = episode_num(p)
        if ep >= start_ep:
            break
        row = progress.get(p.name)
        if row and row.get("status") == "done":
            prev_rows.append(row)
    if not prev_rows:
        return None
    row = prev_rows[-1]
    try:
        return (
            ROI(
                x=int(row["crop_x_px"]),
                y=int(row["crop_y_px"]),
                w=int(row["crop_w_px"]),
                h=int(row["crop_h_px"]),
            ),
            int(row["frame_w_px"]),
            int(row["frame_h_px"]),
        )
    except (KeyError, ValueError):
        return None


def scale_roi(roi: ROI, from_w: int, from_h: int, to_w: int, to_h: int) -> ROI:
    return normalize_roi(
        ROI(
            x=round(roi.x * to_w / from_w),
            y=round(roi.y * to_h / from_h),
            w=round(roi.w * to_w / from_w),
            h=round(roi.h * to_h / from_h),
        ),
        to_w,
        to_h,
    )


def default_roi(width: int, height: int) -> ROI:
    return normalize_roi(
        ROI(
            x=int(width * 0.76),
            y=int(height * 0.50),
            w=int(width * 0.24),
            h=int(height * 0.50),
        ),
        width,
        height,
    )


def hit_edges(roi: ROI, x: int, y: int, margin: int = 12) -> dict[str, bool]:
    left = abs(x - roi.x) <= margin and roi.y - margin <= y <= roi.y + roi.h + margin
    right = abs(x - (roi.x + roi.w)) <= margin and roi.y - margin <= y <= roi.y + roi.h + margin
    top = abs(y - roi.y) <= margin and roi.x - margin <= x <= roi.x + roi.w + margin
    bottom = abs(y - (roi.y + roi.h)) <= margin and roi.x - margin <= x <= roi.x + roi.w + margin
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def on_mouse(event, x, y, _flags, _userdata):
    roi: Optional[ROI] = state["roi"]
    if roi is None:
        return
    fw, fh = state["frame_size"]
    inside = roi.x <= x <= roi.x + roi.w and roi.y <= y <= roi.y + roi.h
    edges = hit_edges(roi, x, y)
    on_edge = any(edges.values())

    if event == cv2.EVENT_LBUTTONDOWN:
        if on_edge:
            state["dragging"] = True
            state["drag_mode"] = "resize"
            state["resize_edges"] = edges
        elif inside:
            state["dragging"] = True
            state["drag_mode"] = "move"
            state["drag_offset"] = (x - roi.x, y - roi.y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["dragging"] = True
        state["drag_mode"] = "new"
        state["drag_start"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
        mode = state["drag_mode"]
        if mode == "move":
            ox, oy = state["drag_offset"]
            moved = ROI(x=x - ox, y=y - oy, w=roi.w, h=roi.h)
            state["roi"] = normalize_roi(moved, fw, fh)
        elif mode == "resize":
            left = roi.x
            right = roi.x + roi.w
            top = roi.y
            bottom = roi.y + roi.h
            e = state["resize_edges"]
            if e["left"]:
                left = clamp(make_even(x), 0, right - 2)
            if e["right"]:
                right = clamp(make_even(x), left + 2, fw)
            if e["top"]:
                top = clamp(make_even(y), 0, bottom - 2)
            if e["bottom"]:
                bottom = clamp(make_even(y), top + 2, fh)
            state["roi"] = normalize_roi(ROI(left, top, right - left, bottom - top), fw, fh)
        elif mode == "new":
            sx, sy = state["drag_start"]
            x0, x1 = sorted((sx, x))
            y0, y1 = sorted((sy, y))
            state["roi"] = normalize_roi(ROI(x0, y0, max(2, x1 - x0), max(2, y1 - y0)), fw, fh)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        state["dragging"] = False
        state["drag_mode"] = None


def update_manifest(raw_rel: str, out_rel: str, ratio: dict[str, float]) -> None:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    found = None
    for entry in data.get("entries", {}).values():
        if entry.get("local_path") == raw_rel:
            found = entry
            break
    if found is None:
        raise RuntimeError(f"Manifest entry not found for {raw_rel}")
    ts = now_iso()
    found["crop_ratio"] = ratio
    found["cropped_local_path"] = out_rel
    found["cropped_at"] = ts
    data["updated_at"] = ts
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_ffmpeg_crop(in_path: Path, out_path: Path, roi: ROI) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-vf",
        f"crop={roi.w}:{roi.h}:{roi.x}:{roi.y}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    shutil.move(tmp, out_path)


def get_video_codec(in_path: Path) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(in_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip().lower()


def build_preview_proxy(in_path: Path) -> Path:
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = PREVIEW_CACHE_DIR / f"{in_path.stem}.preview.mp4"
    if tmp_path.exists() and tmp_path.stat().st_mtime >= in_path.stat().st_mtime:
        return tmp_path
    threads = os.environ.get("UZSL_PREVIEW_THREADS", "6")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-map",
        "0:v:0",
        "-an",
        "-threads",
        threads,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "38",
        "-vf",
        "fps=12",
        str(tmp_path),
    ]
    print(f"Building preview cache for {in_path.name} (one-time)...")
    subprocess.run(cmd, check=True)
    return tmp_path


def open_preview_capture(in_path: Path) -> tuple[cv2.VideoCapture, Optional[Path]]:
    codec = get_video_codec(in_path)
    if codec == "av1":
        print(f"{in_path.name}: AV1 detected, using proxy preview decode")
        proxy = build_preview_proxy(in_path)
        cap = cv2.VideoCapture(str(proxy), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to preview {in_path.name} via proxy")
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Unable to decode proxy preview for {in_path.name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, proxy

    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(str(in_path), backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap, None
        cap.release()

    proxy = build_preview_proxy(in_path)
    cap = cv2.VideoCapture(str(proxy), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to preview {in_path.name} (direct and proxy failed)")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Unable to decode preview frames for {in_path.name}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, proxy


def build_queue(
    files_sorted: list[Path],
    progress: dict[str, dict[str, str]],
    start_from: Optional[str],
    include_done: bool,
    redo: Optional[str],
) -> tuple[list[Path], bool]:
    if redo:
        target = next((p for p in files_sorted if p.name == redo), None)
        if not target:
            raise ValueError(f"--redo target not found: {redo}")
        return [target], True

    queue = files_sorted[:]
    if start_from:
        names = [p.name for p in queue]
        if start_from not in names:
            raise ValueError(f"Start file not found: {start_from}")
        queue = queue[names.index(start_from) :]

    if not include_done:
        queue = [p for p in queue if progress.get(p.name, {}).get("status") != "done"]
    return queue, False


def main() -> int:
    args = parse_args()
    if not RAW_DIR.exists():
        print(f"Raw dir not found: {RAW_DIR}")
        return 1
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1

    files_sorted = get_all_files()
    if not files_sorted:
        print("No news videos found.")
        return 1

    progress = read_progress()
    if not progress and not PROGRESS_CSV.exists():
        progress = bootstrap_progress_from_manifest(files_sorted)
        if progress:
            write_progress(progress, files_sorted)
    try:
        queue, single_mode = build_queue(files_sorted, progress, args.start_from, args.include_done, args.redo)
    except ValueError as e:
        print(e)
        return 1

    if not queue:
        print("No pending videos to crop (all done).")
        return 0

    print(f"Queue size: {len(queue)}")
    print(f"Progress file: {to_rel(PROGRESS_CSV)}")
    if single_mode:
        print("Single recrop mode: will stop after this episode.")

    start_ep = episode_num(queue[0])
    prev_seed = find_roi_from_last_done(progress, files_sorted, start_ep)
    prev_roi: Optional[ROI] = prev_seed[0] if prev_seed else None
    prev_fw: Optional[int] = prev_seed[1] if prev_seed else None
    prev_fh: Optional[int] = prev_seed[2] if prev_seed else None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    for idx, in_path in enumerate(queue, start=1):
        out_path = OUT_DIR / f"{in_path.stem}.cropped.mp4"
        try:
            cap, proxy_path = open_preview_capture(in_path)
        except Exception as e:
            print(f"Preview init failed for {in_path.name}: {e}")
            cv2.destroyAllWindows()
            return 1

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if prev_roi is not None and prev_fw and prev_fh:
            if prev_fw == fw and prev_fh == fh:
                roi = normalize_roi(ROI(prev_roi.x, prev_roi.y, prev_roi.w, prev_roi.h), fw, fh)
            else:
                roi = scale_roi(prev_roi, prev_fw, prev_fh, fw, fh)
        else:
            row = progress.get(in_path.name)
            if row and row.get("status") == "done":
                try:
                    roi = normalize_roi(
                        ROI(
                            x=int(row["crop_x_px"]),
                            y=int(row["crop_y_px"]),
                            w=int(row["crop_w_px"]),
                            h=int(row["crop_h_px"]),
                        ),
                        fw,
                        fh,
                    )
                except (KeyError, ValueError):
                    roi = default_roi(fw, fh)
            else:
                roi = default_roi(fw, fh)

        state["roi"] = roi
        state["frame_size"] = (fw, fh)
        playing = True
        frame_pos = 0
        last_frame = None
        action = "skip"

        print(f"\n[{idx}/{len(queue)}] {in_path.name} ({fw}x{fh})")
        print("Mouse: left-drag INSIDE/EDGE move-resize, right-drag draw new box")
        print("Keys: space play/pause, ,/. seek -/+2s, r reset, Enter save+next, x skip, q quit")

        while True:
            if playing:
                ok, frame = cap.read()
                if ok:
                    last_frame = frame.copy()
                    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                else:
                    playing = False
                    frame = last_frame
            else:
                frame = last_frame
                if frame is None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    last_frame = frame.copy()
                    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            roi = state["roi"]
            if frame is None or roi is None:
                continue
            roi = normalize_roi(roi, fw, fh)
            state["roi"] = roi

            vis = frame.copy()
            cv2.rectangle(vis, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2)
            ratio = {
                "keep_w": round(roi.w / fw, 6),
                "keep_h": round(roi.h / fh, 6),
                "x": round(roi.x / fw, 6),
                "y": round(roi.y / fh, 6),
            }
            t1 = f"{in_path.name} frame {frame_pos}/{total} w={ratio['keep_w']} h={ratio['keep_h']} x={ratio['x']} y={ratio['y']}"
            t2 = "L-drag inside/edge move-resize | R-drag new box | Enter save | X skip | Q quit"
            cv2.putText(vis, t1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(vis, t2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(20) & 0xFF
            if key == ord(" "):
                playing = not playing
            elif key == ord(","):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_pos - int(fps * 2)))
                playing = False
                last_frame = None
            elif key == ord("."):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, total - 1), frame_pos + int(fps * 2)))
                playing = False
                last_frame = None
            elif key == ord("r"):
                if prev_roi is not None and prev_fw and prev_fh:
                    if prev_fw == fw and prev_fh == fh:
                        state["roi"] = normalize_roi(ROI(prev_roi.x, prev_roi.y, prev_roi.w, prev_roi.h), fw, fh)
                    else:
                        state["roi"] = scale_roi(prev_roi, prev_fw, prev_fh, fw, fh)
                else:
                    state["roi"] = default_roi(fw, fh)
            elif key in (13, 10):
                action = "save"
                break
            elif key == ord("x"):
                action = "skip"
                break
            elif key == ord("q"):
                action = "quit"
                break

        cap.release()

        if action == "quit":
            cv2.destroyAllWindows()
            print("Stopped by user.")
            return 0
        if action == "skip":
            print(f"Skipped {in_path.name}")
            if single_mode:
                cv2.destroyAllWindows()
                return 0
            continue

        roi = state["roi"]
        if roi is None:
            print(f"No ROI set for {in_path.name}; skipped.")
            continue

        try:
            run_ffmpeg_crop(in_path, out_path, roi)
            ratio = {
                "keep_w": round(roi.w / fw, 6),
                "keep_h": round(roi.h / fh, 6),
                "x": round(roi.x / fw, 6),
                "y": round(roi.y / fh, 6),
            }
            update_manifest(to_rel(in_path), to_rel(out_path), ratio)
            update_progress(progress, in_path, out_path, roi, fw, fh)
            write_progress(progress, files_sorted)
            prev_roi = ROI(roi.x, roi.y, roi.w, roi.h)
            prev_fw, prev_fh = fw, fh
            print(f"Saved {out_path.name} and progress updated.")
        except Exception as e:
            print(f"Failed on {in_path.name}: {e}")
            return 1

        if single_mode:
            cv2.destroyAllWindows()
            print("Single recrop done; stopping.")
            return 0

    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
