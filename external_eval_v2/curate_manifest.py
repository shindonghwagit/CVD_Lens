"""Freeze the first manual visual review without deleting raw candidates."""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Clear non-photographs, query mismatches, and near-duplicate scenes identified
# from the labelled category contact sheets. The raw 300 remain on disk.
EXCLUDED = {
    "road_transit_03",
    "home_living_02", "home_living_04", "home_living_05", "home_living_06",
    "home_living_07", "home_living_08", "home_living_09", "home_living_10",
    "home_living_11", "home_living_14", "home_living_15",
    "kitchen_dining_04", "kitchen_dining_13",
    "food_drink_04", "food_drink_05", "food_drink_06", "food_drink_08",
    "office_school_11", "office_school_12", "office_school_13",
    "office_school_14", "office_school_15",
    "screens_controls_05", "screens_controls_06", "screens_controls_12",
    "screens_controls_14", "screens_controls_15",
    "tools_workshop_05", "tools_workshop_06", "tools_workshop_10",
    "tools_workshop_11", "tools_workshop_12", "tools_workshop_13",
    "clothing_fashion_05", "clothing_fashion_06", "clothing_fashion_10",
    "clothing_fashion_11",
    "people_social_12",
    "sports_play_12", "sports_play_13", "sports_play_14",
    "garden_plants_01", "garden_plants_02", "garden_plants_12",
    "landscape_water_02", "landscape_water_06", "landscape_water_13",
    "landscape_water_15",
    "animals_pets_01", "animals_pets_02", "animals_pets_03", "animals_pets_04",
    "animals_pets_05", "animals_pets_06", "animals_pets_07", "animals_pets_08",
    "animals_pets_09", "animals_pets_11", "animals_pets_15",
    "night_lowlight_04", "night_lowlight_06", "night_lowlight_09",
    "weather_seasons_03", "weather_seasons_10", "weather_seasons_12",
    "weather_seasons_13", "weather_seasons_14", "weather_seasons_15",
    "small_color_items_02", "small_color_items_03", "small_color_items_08",
    "small_color_items_10", "small_color_items_13", "small_color_items_14",
    "small_color_items_15",
    "grocery_retail_02", "grocery_retail_03", "grocery_retail_11",
    "grocery_retail_12",
}


def main() -> None:
    manifest = ROOT / "manifest.csv"
    candidates = ROOT / "manifest_candidates.csv"
    if not candidates.exists():
        shutil.copy2(manifest, candidates)

    with candidates.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = [row for row in rows if row["id"] not in EXCLUDED]
    found = {row["id"] for row in rows}
    unknown = EXCLUDED - found
    if unknown:
        raise RuntimeError(f"Unknown exclusion IDs: {sorted(unknown)}")

    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)

    counts = Counter(row["category"] for row in kept)
    review = ROOT / "REVIEW.md"
    lines = [
        "# Dataset visual review", "",
        f"- Raw candidates: {len(rows)}",
        f"- Accepted: {len(kept)}",
        f"- Excluded: {len(rows) - len(kept)}",
        "- Review method: labelled 5 x 3 contact sheet for each category",
        "- Exclusion reasons: non-photograph, clear query/category mismatch, or near-duplicate scene",
        "- Raw candidates and downloaded files are retained for auditability.", "",
        "## Accepted count by category", "",
        "| Category | Accepted |", "|---|---:|",
    ]
    lines.extend(f"| {category} | {counts[category]} |" for category in counts)
    lines += ["", "## Excluded IDs", "", *[f"- `{item}`" for item in sorted(EXCLUDED)]]
    review.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Accepted {len(kept)}/{len(rows)}; excluded {len(rows) - len(kept)}")


if __name__ == "__main__":
    main()
