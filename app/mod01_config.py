import os
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent

# 1. GLOBAL SETTINGS
DATA_ROOT = str(_PROJECT_ROOT / "Data_Numpy_Arrays_RSL_UzSL")
# DATA_ROOT = "/Volumes/Untitled/Data_Numpy_Arrays_RSL_UzSL"
VIDEO_DEVICE = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
FPS = 30
COUNTDOWN_SECONDS = 2


# 2. TOPICS (27 UzSL topic categories)
TOPICS: list[str] = [
    "Tibbiyot",
    "Sport va dam olish",
    "Inson. Qarindoshlik. Oila",
    "Shahar. Transport",
    "Mahalla",
    "Bozor va bozorlik",
    "Kiyim. Poyabzal. Ranglar",
    "Vaqt. Taqvim",
    "Kasb. Biznes",
    "Idish-tovoq. Oziq-ovqat",
    "Sonlar",
    "Emotsiyalar. Tuyg'ular. Holatlar",
    "Oshxona",
    "Hayvonlar dunyosi",
    "Maktab",
    "Intellektual faoliyat",
    "Davlat. Dunyo mamlakatlari",
    "Huquq. Qonun",
    "Sayohat",
    "O'simliklar",
    "Tanishish",
    "Tabiat. Fasllar",
    "Uy",
    "Uy-ro'zg'or buyumlari",
    "Alifbo",
    "Tartib sonlar",
    "Hujjatlar mavzusi",
]


# 3. POSE LANDMARK FILTERING
# Keep only upper-body arm landmarks (shoulders, elbows, wrists).
# MediaPipe predicts face and leg landmarks even when they are off-frame,
# producing extrapolated out-of-range values. These indices are zeroed out
# when saving keypoints so downstream code only sees meaningful data.
POSE_REMOVE_IDX: list[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,        # face landmarks
    17, 18, 19, 20, 21, 22,                    # torso / hip extras
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,   # hips, legs, feet
]

# Kept indices: 11 (L shoulder), 12 (R shoulder), 13 (L elbow),
#               14 (R elbow), 15 (L wrist), 16 (R wrist)
POSE_KEEP_CONNECTIONS: frozenset = frozenset([
    (11, 12),  # left shoulder — right shoulder
    (11, 13),  # left shoulder — left elbow
    (12, 14),  # right shoulder — right elbow
    (13, 15),  # left elbow — left wrist
    (14, 16),  # right elbow — right wrist
])
