import os

# 1. GLOBAL SETTINGS
DATA_ROOT = "./Data_Numpy_Arrays_RSL_UzSL"
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
