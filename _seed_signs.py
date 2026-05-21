"""One-off script: read the words CSV and write signs.json for every topic."""
import csv
import json
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("./Data_Numpy_Arrays_RSL_UzSL")
CSV_PATH  = Path("imo-ishora-so'zlar - so'zlar.csv")

words_by_topic: dict[str, list[str]] = defaultdict(list)

with CSV_PATH.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        topic = row["topic_uz"].strip()
        word  = row["word_uz"].strip()
        if topic and word:
            words_by_topic[topic].append(word)

for topic, words in sorted(words_by_topic.items()):
    topic_dir = DATA_ROOT / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    signs_path = topic_dir / "signs.json"
    signs_path.write_text(json.dumps(words, ensure_ascii=False, indent=2))
    print(f"  {topic:45s}  {len(words):3d} words  →  {signs_path}")

print(f"\nTotal: {sum(len(w) for w in words_by_topic.values())} words across {len(words_by_topic)} topics")
