"""One-off script: read the trilingual CSV and write signs.json + topic_translations.json.

Run:    python scripts/_seed_signs.py

Reads imo-ishora-so_zlar_with_en.csv (columns: topic_uz, topic_ru, word_uz, word_ru,
topic_en, word_en) and writes:
  - DATA_ROOT/<topic>/signs.json  — list of {"uz","ru","en"} dicts
  - DATA_ROOT/topic_translations.json  — {"topic_uz": {"ru":..., "en":...}, ...}
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

DATA_ROOT = _PROJECT_ROOT / "Data_Numpy_Arrays_RSL_UzSL"
CSV_PATH  = _PROJECT_ROOT / "imo-ishora-so_zlar_with_en.csv"

topic_translations: dict[str, dict] = {}
words_by_topic: dict[str, list] = defaultdict(list)

with CSV_PATH.open(encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        topic_uz = row["topic_uz"].strip()
        topic_ru = row["topic_ru"].strip()
        topic_en = row["topic_en"].strip()
        word_uz  = row["word_uz"].strip()
        word_ru  = row["word_ru"].strip()
        word_en  = row["word_en"].strip()
        if topic_uz and word_uz:
            if topic_uz not in topic_translations:
                topic_translations[topic_uz] = {"ru": topic_ru, "en": topic_en}
            words_by_topic[topic_uz].append({"uz": word_uz, "ru": word_ru, "en": word_en})

for topic_uz, words in sorted(words_by_topic.items()):
    topic_dir = DATA_ROOT / topic_uz
    topic_dir.mkdir(parents=True, exist_ok=True)
    signs_path = topic_dir / "signs.json"
    signs_path.write_text(json.dumps(words, ensure_ascii=False, indent=2))
    print(f"  {topic_uz:45s}  {len(words):3d} words  →  {signs_path}")

tt_path = DATA_ROOT / "topic_translations.json"
tt_path.write_text(json.dumps(topic_translations, ensure_ascii=False, indent=2))
print(f"\nTopic translations → {tt_path}")
print(f"\nTotal: {sum(len(w) for w in words_by_topic.values())} words across {len(words_by_topic)} topics")
