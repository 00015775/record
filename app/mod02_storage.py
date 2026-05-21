import json
from pathlib import Path
from typing import Optional
from mod01_config import DATA_ROOT, TOPICS

Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)


# ── Paths ─────────────────────────────────────────────────────────────────────

def path_topic(topic: str) -> Path:
    return Path(DATA_ROOT) / topic

def path_signer(topic: str, signer_id: str) -> Path:
    return path_topic(topic) / signer_id

def path_sign(topic: str, signer_id: str, sign: str) -> Path:
    return path_signer(topic, signer_id) / sign

def path_videos(topic: str, signer_id: str, sign: str) -> Path:
    return path_sign(topic, signer_id, sign) / "videos"

def ensure_folders(topic: str, signer_id: str, sign: str):
    path_videos(topic, signer_id, sign).mkdir(parents=True, exist_ok=True)


# ── Topic list ────────────────────────────────────────────────────────────────

def _topics_path() -> Path:
    return Path(DATA_ROOT) / "topics.json"

def load_topic_list() -> list[str]:
    p = _topics_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return TOPICS[:]

def save_topic_list(topics: list[str]):
    _topics_path().write_text(json.dumps(topics, ensure_ascii=False, indent=2))

def add_topic(new_topic: str):
    topics = load_topic_list()
    if new_topic not in topics:
        topics.append(new_topic)
        save_topic_list(topics)


# ── Sign list (per topic) ─────────────────────────────────────────────────────

def _signs_path(topic: str) -> Path:
    return path_topic(topic) / "signs.json"

def load_sign_list(topic: str) -> list[str]:
    p = _signs_path(topic)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []

def save_sign_list(topic: str, signs: list[str]):
    p = _signs_path(topic)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(signs, ensure_ascii=False, indent=2))

def add_sign(topic: str, new_sign: str):
    signs = load_sign_list(topic)
    if new_sign not in signs:
        signs.append(new_sign)
        save_sign_list(topic, signs)


# ── Progress ──────────────────────────────────────────────────────────────────

def count_repetitions(topic: str, signer_id: str, sign: str) -> int:
    vid_dir = path_videos(topic, signer_id, sign)
    if not vid_dir.exists():
        return 0
    return sum(1 for d in vid_dir.iterdir() if d.is_dir() and d.name.startswith("rep-"))

def count_all_repetitions(topic: str, signer_id: str) -> dict[str, int]:
    """Single directory scan → {sign: rep_count} for all signs with recordings."""
    signer_dir = path_signer(topic, signer_id)
    if not signer_dir.exists():
        return {}
    counts: dict[str, int] = {}
    for sign_dir in signer_dir.iterdir():
        if not sign_dir.is_dir() or sign_dir.name.startswith("."):
            continue
        vids = sign_dir / "videos"
        if vids.exists():
            n = sum(1 for d in vids.iterdir() if d.is_dir() and d.name.startswith("rep-"))
            if n > 0:
                counts[sign_dir.name] = n
    return counts

def recorded_signs(topic: str, signer_id: str) -> set[str]:
    return set(count_all_repetitions(topic, signer_id).keys())


# ── Signer list ───────────────────────────────────────────────────────────────

def list_signers() -> list[str]:
    signers: set[str] = set()
    root = Path(DATA_ROOT)
    if not root.exists():
        return []
    for item in root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            for d in item.iterdir():
                if d.is_dir() and d.name.startswith("signer"):
                    signers.add(d.name)
    return sorted(signers)
