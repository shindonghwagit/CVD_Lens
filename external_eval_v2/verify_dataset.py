"""Integrity checks for the frozen external-evaluation manifest."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_csv(ROOT / "manifest.csv")
    old_path = ROOT.parent / "external_eval" / "manifest.csv"
    old = read_csv(old_path) if old_path.exists() else []
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    old_titles = {row["commons_title"] for row in old}
    old_hashes = {row["sha256"] for row in old}

    for row in rows:
        path = ROOT / "images" / row["filename"]
        if row["id"] in seen_ids:
            errors.append(f"duplicate id: {row['id']}")
        seen_ids.add(row["id"])
        if row["sha256"] in seen_hashes:
            errors.append(f"duplicate sha256: {row['id']}")
        seen_hashes.add(row["sha256"])
        if row["commons_title"] in old_titles or row["sha256"] in old_hashes:
            errors.append(f"overlap with original set: {row['id']}")
        if not path.is_file():
            errors.append(f"missing file: {row['filename']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            errors.append(f"hash mismatch: {row['filename']}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"invalid image: {row['filename']}: {exc}")

    counts = Counter(row["category"] for row in rows)
    print(f"images={len(rows)} categories={len(counts)} min_per_category={min(counts.values())} max_per_category={max(counts.values())}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("PASS: files, image decoding, hashes, uniqueness, and v1 independence")


if __name__ == "__main__":
    main()
