"""Generate an HTML report of recorded signs and repetitions.

Usage:
  /Users/macbookair/Projects/record/.venv/bin/python scripts/report_recordings.py \
    --out reports/recordings_report.html

Optional:
  --signer signer01   # filter to a single signer
  --data-root /path/to/Data_Numpy_Arrays_RSL_UzSL
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from typing import Any

from app.mod01_config import DATA_ROOT as DEFAULT_DATA_ROOT, TOPICS as DEFAULT_TOPICS


def _load_topics(data_root: Path) -> list[str]:
    topics_path = data_root / "topics.json"
    if topics_path.exists():
        try:
            return json.loads(topics_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_TOPICS[:]


def _load_signs(data_root: Path, topic: str) -> list[dict[str, str]]:
    signs_path = data_root / topic / "signs.json"
    if not signs_path.exists():
        return []
    try:
        data = json.loads(signs_path.read_text(encoding="utf-8"))
        if data and isinstance(data[0], str):
            return [{"uz": s, "ru": s, "en": s} for s in data]
        return data
    except Exception:
        return []


def _list_signers(topic_dir: Path) -> list[str]:
    if not topic_dir.exists():
        return []
    signers = [d.name for d in topic_dir.iterdir() if d.is_dir() and d.name.startswith("signer")]
    return sorted(signers)


def _count_reps(vid_dir: Path) -> int:
    if not vid_dir.exists():
        return 0
    return sum(
        1
        for d in vid_dir.iterdir()
        if d.is_dir() and d.name.startswith("rep-") and (d / "video.mp4").exists()
    )


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/recordings_report.html"))
    ap.add_argument("--signer", default=None, help="Filter to a single signer (e.g. signer01)")
    ap.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    args = ap.parse_args()

    data_root = args.data_root
    topics = _load_topics(data_root)

    rows: list[dict[str, Any]] = []
    topic_summary: dict[str, dict[str, int]] = {}
    signer_signs: dict[str, set[tuple[str, str]]] = {}
    signer_topics: dict[str, set[str]] = {}
    all_signs: set[tuple[str, str]] = set()

    for topic in topics:
        topic_dir = data_root / topic
        signers = [args.signer] if args.signer else _list_signers(topic_dir)
        signs = _load_signs(data_root, topic)
        sign_index = {s["uz"]: s for s in signs}

        recorded_signs = set()
        total_reps = 0

        for signer in signers:
            if not signer:
                continue
            signer_dir = topic_dir / signer
            if not signer_dir.exists():
                continue

            for sign_name in sorted(sign_index.keys()):
                vid_dir = signer_dir / sign_name / "videos"
                reps = _count_reps(vid_dir)
                if reps <= 0:
                    continue

                recorded_signs.add(sign_name)
                total_reps += reps
                all_signs.add((topic, sign_name))
                signer_signs.setdefault(signer, set()).add((topic, sign_name))
                signer_topics.setdefault(signer, set()).add(topic)
                entry = sign_index.get(sign_name, {"uz": sign_name, "ru": "", "en": ""})
                rows.append({
                    "topic": topic,
                    "signer": signer,
                    "uz": entry.get("uz", sign_name),
                    "ru": entry.get("ru", ""),
                    "en": entry.get("en", ""),
                    "reps": reps,
                })

        total_signs = len(sign_index)
        topic_summary[topic] = {
            "recorded": len(recorded_signs),
            "total": total_signs,
            "reps": total_reps,
        }

    total_recorded = sum(v["recorded"] for v in topic_summary.values())
    total_signs = sum(v["total"] for v in topic_summary.values())
    total_reps = sum(v["reps"] for v in topic_summary.values())
    total_topics = len(topic_summary)
    recorded_topics = sum(1 for v in topic_summary.values() if v["recorded"] > 0)
    total_unique_signs = len(all_signs)
    signer_summary = []
    for signer, signs in sorted(signer_signs.items()):
        topics_covered = len(signer_topics.get(signer, set()))
        count = len(signs)
        pct = (count / total_unique_signs * 100.0) if total_unique_signs else 0.0
        signer_summary.append({
            "signer": signer,
            "signs": count,
            "topics": topics_covered,
            "pct": pct,
        })
    overlap_signs: list[tuple[str, str]] = []
    if signer_signs:
        sets = list(signer_signs.values())
        common = set.intersection(*sets) if len(sets) > 1 else set()
        overlap_signs = sorted(common)

    rows.sort(key=lambda r: (r["topic"], r["uz"], r["signer"]))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [
        '<!doctype html>',
        '<html>',
        '<head>',
        '  <meta charset="utf-8">',
        '  <title>UzSL Recording Report</title>',
        '  <style>',
        '    body { font-family: Arial, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }',
        '    h1 { font-size: 22px; margin-bottom: 6px; }',
        '    .summary { margin-bottom: 18px; color: #94a3b8; }',
        '    table { width: 100%; border-collapse: collapse; margin-top: 10px; }',
        '    th, td { padding: 8px 10px; border-bottom: 1px solid #1f2430; }',
        '    th { text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #94a3b8; }',
        '    tr:hover { background: #141925; }',
        '    .topic { color: #a5b4fc; font-weight: 600; }',
        '  </style>',
        '</head>',
        '<body>',
        '  <h1>UzSL Recording Report</h1>',
        f'  <div class="summary">Recorded topics: {recorded_topics} / {total_topics} · Recorded signs: {total_recorded} / {total_signs} · Total reps: {total_reps}</div>',
        '  <h2 style="margin-top:16px; font-size:16px;">Signer Summary</h2>',
        '  <table>',
        '    <thead>',
        '      <tr>',
        '        <th>Signer</th>',
        '        <th>Recorded Signs</th>',
        '        <th>% of Total Signs</th>',
        '        <th>Topics Covered</th>',
        '      </tr>',
        '    </thead>',
        '    <tbody>',
    ]

    for s in signer_summary:
        html_parts.append(
            '      <tr>'
            f'<td>{_html_escape(s["signer"])}</td>'
            f'<td>{s["signs"]}</td>'
            f'<td>{s["pct"]:.1f}%</td>'
            f'<td>{s["topics"]}</td>'
            '</tr>'
        )

    html_parts += [
        '    </tbody>',
        '  </table>',
        '  <h2 style="margin-top:16px; font-size:16px;">Topic Summary</h2>',
        '  <table>',
        '    <thead>',
        '      <tr>',
        '        <th>Topic</th>',
        '        <th>Recorded Signs</th>',
        '        <th>Total Signs</th>',
        '        <th>Total Reps</th>',
        '      </tr>',
        '    </thead>',
        '    <tbody>',
    ]

    for topic in sorted(topic_summary.keys()):
        stats = topic_summary[topic]
        html_parts.append(
            '      <tr>'
            f'<td class="topic">{_html_escape(topic)}</td>'
            f'<td>{stats["recorded"]}</td>'
            f'<td>{stats["total"]}</td>'
            f'<td>{stats["reps"]}</td>'
            '</tr>'
        )

    html_parts += [
        '    </tbody>',
        '  </table>',
        '  <h2 style="margin-top:18px; font-size:16px;">Overlap (All Signers)</h2>',
        f'  <div class="summary">Common signs recorded by all signers: {len(overlap_signs)}</div>',
        '  <table>',
        '    <thead>',
        '      <tr>',
        '        <th>Topic</th>',
        '        <th>Sign</th>',
        '      </tr>',
        '    </thead>',
        '    <tbody>',
    ]

    for topic, sign_name in overlap_signs:
        html_parts.append(
            '      <tr>'
            f'<td class="topic">{_html_escape(topic)}</td>'
            f'<td>{_html_escape(sign_name)}</td>'
            '</tr>'
        )
    if not overlap_signs:
        html_parts.append('      <tr><td colspan="2">No overlapping signs.</td></tr>')

    html_parts += [
        '    </tbody>',
        '  </table>',
        '  <h2 style="margin-top:18px; font-size:16px;">Recorded Signs</h2>',
        '  <table>',
        '    <thead>',
        '      <tr>',
        '        <th>Topic</th>',
        '        <th>Signer</th>',
        '        <th>Uz</th>',
        '        <th>Ru</th>',
        '        <th>En</th>',
        '        <th>Reps</th>',
        '      </tr>',
        '    </thead>',
        '    <tbody>',
    ]

    for row in rows:
        html_parts.append(
            '      <tr>'
            f'<td class="topic">{_html_escape(row["topic"])}</td>'
            f'<td>{_html_escape(row["signer"])}</td>'
            f'<td>{_html_escape(row["uz"])}</td>'
            f'<td>{_html_escape(row["ru"])}</td>'
            f'<td>{_html_escape(row["en"])}</td>'
            f'<td>{row["reps"]}</td>'
            '</tr>'
        )

    html_parts += [
        '    </tbody>',
        '  </table>',
        '</body>',
        '</html>',
    ]

    out.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
