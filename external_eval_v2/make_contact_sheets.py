"""Create labelled contact sheets for manual dataset review."""
from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
CELL_W, CELL_H = 260, 205
THUMB_H = 165


def main() -> None:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out_dir = ROOT / "contact_sheets"
    out_dir.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=14)

    categories = list(dict.fromkeys(row["category"] for row in rows))
    for category in categories:
        selected = [row for row in rows if row["category"] == category]
        sheet = Image.new("RGB", (CELL_W * 5, CELL_H * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected):
            image = Image.open(ROOT / "images" / row["filename"]).convert("RGB")
            thumb = ImageOps.contain(image, (CELL_W - 8, THUMB_H - 8))
            x = (index % 5) * CELL_W
            y = (index // 5) * CELL_H
            sheet.paste(thumb, (x + (CELL_W - thumb.width) // 2, y + 4))
            draw.text((x + 5, y + THUMB_H + 2), row["id"], fill="black", font=font)
            draw.text((x + 5, y + THUMB_H + 20), row["commons_title"][:34], fill="#444444", font=font)
        sheet.save(out_dir / f"{category}.jpg", quality=90)
    print(f"Wrote {len(categories)} contact sheets to {out_dir}")


if __name__ == "__main__":
    main()
