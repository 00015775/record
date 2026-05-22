# UzSL Recording & Keypoint Pipeline

A tool for collecting Uzbek Sign Language (UzSL) videos and extracting MediaPipe keypoints for downstream model training. Each recording session produces a structured video archive alongside `.npz` keypoint files containing 149 landmarks per frame — 33 pose, 21 left-hand, 21 right-hand, and 74 face NMM points (plus the raw 478-point face mesh).

---

## Requirements

Python 3.9–3.11 recommended (mediapipe wheels are not available for 3.12+ on all platforms).

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `mediapipe==0.10.14` | Holistic model — pose, hands, face mesh |
| `opencv-python>=4.5` | Video I/O and camera capture |
| `numpy>=1.23` | Array storage and `.npz` serialisation |
| `flask>=2.0` | Web UI for browser-based recording |

---

## Project structure

```
record/
├── app/               # Recording application (web UI + terminal UI)
│   ├── mod01_config.py    # Global settings (camera, paths, FPS)
│   ├── mod02_storage.py   # File-system helpers (paths, sign/topic lists)
│   ├── mod03_recorder.py  # Core OpenCV recording loop
│   ├── mod04_ui.py        # Terminal menu helpers
│   ├── mod05_main.py      # Entry point — terminal UI
│   ├── mod06_webapp.py    # Entry point — Flask web UI
│   └── translations.py    # UI strings (en / uz / ru)
├── keypoints/         # Keypoint extraction pipeline
│   ├── keypoint_extraction.py   # Library — MediaPipe Holistic wrapper
│   ├── extract_keypoints.py     # Batch extractor for the full dataset
│   ├── smoke_test_keypoints.py  # 9-check sanity test
│   └── visualize_keypoints.py  # Render skeleton or side-by-side video
├── scripts/           # One-off utilities
│   ├── _seed_signs.py   # Populate signs.json from the words CSV
│   └── recorder.py      # Minimal standalone camera recorder
├── templates/         # Flask HTML templates
├── debug/             # Visualizer output videos
└── Data_Numpy_Arrays_RSL_UzSL/   # Dataset root (created on first run)
```

---

## Data layout

All recordings and keypoints live under `Data_Numpy_Arrays_RSL_UzSL/` (configurable in `app/mod01_config.py`):

```
Data_Numpy_Arrays_RSL_UzSL/
└── <topic>/
    └── <signer>/
        └── <sign>/
            ├── videos/
            │   ├── rep-0/
            │   │   └── video.mp4
            │   ├── rep-1/
            │   │   └── video.mp4
            │   └── ...
            └── keypoints/
                ├── rep-0/
                │   └── keypoints.npz
                ├── rep-1/
                │   └── keypoints.npz
                └── ...
```

Signer IDs follow the pattern `signer01`, `signer02`, etc.

---

## One-time setup: seed sign lists

Before recording, populate the `signs.json` file for each topic by reading the words CSV:

```bash
python scripts/_seed_signs.py
```

This reads `imo-ishora-so'zlar - so'zlar.csv` and writes a `signs.json` into each topic directory. Only needed once, or when the CSV is updated.

---

## Recording videos

### A. Web UI (recommended)

```bash
python app/mod06_webapp.py
```

A browser tab opens automatically at `http://127.0.0.1:5000`. The workflow:

1. Enter a signer ID (e.g. `signer01`) and press **Enter**.
2. Select a topic from the grid.
3. Select a sign from the list — already-recorded signs are highlighted.
4. Press **Start Recording**. A countdown plays, then recording begins.
5. Press **Stop** when the sign is complete.
6. Choose **Save** (keep the rep) or **Discard** (delete and try again).

Additional controls:
- **Camera selector** — switch between connected cameras without restarting.
- **Countdown selector** — choose 3, 4, or 5 second countdown.
- **Language switcher** — UI available in English, Uzbek, and Russian.

### B. Terminal UI

```bash
python app/mod05_main.py
```

Keyboard-driven workflow in an OpenCV window:

1. Enter a signer ID at the prompt.
2. Select a topic by number.
3. Select a sign by number.
4. Press `s` in the camera window to start recording.
5. Press `s` again to stop.
6. Press `s` to record another rep or `d` to return to the sign list.
7. Press `q` at any time to quit.

### C. Standalone recorder

```bash
python scripts/recorder.py
```

Minimal recorder with no sign/topic structure. Measures actual camera FPS on startup, then saves timestamped `.mp4` files to `data/` at the project root.

- `r` — start / stop recording
- `q` — quit

---

## Extracting keypoints

After recording, run the batch extractor to produce `.npz` files for every rep. Existing `.npz` files are skipped unless `--force` is given.

```bash
# Process everything under DATA_ROOT
python keypoints/extract_keypoints.py

# Filter to a specific topic
python keypoints/extract_keypoints.py --topic Alifbo

# Filter to a specific signer
python keypoints/extract_keypoints.py --signer signer01

# Filter to both
python keypoints/extract_keypoints.py --topic Alifbo --signer signer01

# Re-extract even if keypoints.npz already exists
python keypoints/extract_keypoints.py --force

# Higher accuracy model (slower, more VRAM)
python keypoints/extract_keypoints.py --model-complexity 2
```

### Output — keypoints.npz schema

Each `keypoints.npz` contains the following arrays:

| Key | Shape | Description |
|---|---|---|
| `pose` | `(T, 33, 4)` | x, y, z, visibility per landmark. Indices 0–10 and 17–32 are zeroed (face, hips, legs). |
| `left_hand` | `(T, 21, 3)` | x, y, z for the 21 left-hand landmarks. All zeros when hand not detected. |
| `right_hand` | `(T, 21, 3)` | Same for right hand. |
| `face_full` | `(T, 478, 3)` | Full 478-point MediaPipe face mesh. |
| `face_nmm` | `(T, 74, 3)` | 74-point NMM subset (mouth, eyes, eyebrows, nose, chin, cheeks). |
| `conf_pose` | `(T, 33)` | Per-landmark visibility confidence. |
| `conf_left_hand` | `(T, 21)` | Per-landmark presence confidence. |
| `conf_right_hand` | `(T, 21)` | Per-landmark presence confidence. |
| `conf_face` | `(T, 478)` | Per-landmark presence confidence. |
| `detected` | `(T, 4)` | Boolean flags — `[pose, left_hand, right_hand, face]` detected per frame. |
| `meta` | `(3,)` | `[fps, width, height]` as float32. |

**Note on pose landmarks:** MediaPipe extrapolates face and leg landmarks even when the subject is off-frame, producing out-of-range coordinates. Indices 0–10 (face), 17–22 (torso extras), and 23–32 (hips, legs, feet) are zeroed before saving. Only indices 11–16 (left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist) carry meaningful data.

---

## Visualizing keypoints

Generate a debug video from any rep that has a `keypoints.npz`.

```bash
# Skeleton on a black 1280x720 canvas
python keypoints/visualize_keypoints.py --rep path/to/sign/videos/rep-0

# Side-by-side: original video (left) + skeleton (right)
python keypoints/visualize_keypoints.py --rep path/to/sign/videos/rep-0 --mode sidebyside

# Override the output frame rate
python keypoints/visualize_keypoints.py --rep path/to/sign/videos/rep-0 --fps 25
```

Output is saved to `debug/` at the project root:

```
debug/<topic>__<signer>__<sign>__<rep>__<mode>.mp4
```

### Colour legend

| Colour | Stream |
|---|---|
| White lines, gray dots | Pose — shoulders, elbows, wrists |
| Blue | Left hand (21 landmarks) |
| Green | Right hand (21 landmarks) |
| Cyan | Face — 478-point dense mesh when available, 74-point NMM fallback otherwise |

---

## Smoke test

Verifies the full extraction pipeline on a single video without needing to run the entire dataset.

```bash
# Auto-discover the first video under DATA_ROOT
python keypoints/smoke_test_keypoints.py

# Run on a specific video
python keypoints/smoke_test_keypoints.py --video path/to/video.mp4
```

The test runs 9 checks:

1. Video file found
2. `extract_from_video()` completes without exception
3. `pose.shape == (T, 33, 4)` with T > 0
4. `left_hand.shape == (T, 21, 3)`
5. `right_hand.shape == (T, 21, 3)`
6. `face_nmm.shape == (T, 74, 3)`
7. Kept pose landmarks (indices 11–16) have x/y coordinates in `[-0.15, 1.15]`
8. `keypoints.npz` is written to the canonical path and the file exists
9. Reloaded arrays match the original shapes

Exit code 0 = all pass, 1 = any fail.

---

## Configuration

All global settings are in `app/mod01_config.py`:

| Setting | Default | Description |
|---|---|---|
| `DATA_ROOT` | `./Data_Numpy_Arrays_RSL_UzSL` | Root directory for all recordings and keypoints |
| `VIDEO_DEVICE` | `0` | OpenCV camera index |
| `FRAME_WIDTH` | `1280` | Recording resolution width |
| `FRAME_HEIGHT` | `720` | Recording resolution height |
| `FPS` | `30` | Target recording frame rate |
| `COUNTDOWN_SECONDS` | `2` | Countdown before recording starts (terminal UI) |
