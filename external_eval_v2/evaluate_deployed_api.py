"""Stratified HTTP E2E parity check against the deployed /infer endpoint."""
from __future__ import annotations

import csv
import argparse
import io
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "cvd-lens" / "inference"
sys.path.insert(0, str(INFERENCE))
import main as local  # noqa: E402

DATASET = Path(__file__).resolve().parent
OUT = DATASET / "results"
API = "https://cvd-lens.onrender.com"


def jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def post_infer(data: bytes, cvd_type: str, timeout: int = 120) -> tuple[bytes, str, int]:
    boundary = f"----cvdlens-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"cvd_type\"\r\n\r\n{cvd_type}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{API}/infer", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get_content_type(), response.status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="test every frozen manifest image")
    parser.add_argument("--resume", action="store_true", help="resume the --all checkpoint")
    args = parser.parse_args()
    with (DATASET / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.all:
        selected = rows
    else:
        selected = []
        seen: set[str] = set()
        for row in rows:
            if row["category"] not in seen:
                selected.append(row)
                seen.add(row["category"])

    checkpoint = OUT / "deployed_api_e2e_all.checkpoint.json"
    records = []
    if args.all and args.resume and checkpoint.exists():
        records = json.loads(checkpoint.read_text(encoding="utf-8"))
    completed = {(row["id"], row["cvd_type"]) for row in records}
    for index, row in enumerate(selected, 1):
        with Image.open(DATASET / "images" / row["filename"]) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        upload = jpeg_bytes(source, 92)
        upload_image = Image.open(io.BytesIO(upload)).convert("RGB")
        upload_array = np.asarray(upload_image, np.float32) / 255.0
        # /infer caps the returned image to a 2048-pixel longest side before
        # correction; mirror that server behavior for parity comparison.
        upload_array = local._cap_long_side(upload_array)
        for cvd_type in ("p", "d"):
            if (row["id"], cvd_type) in completed:
                continue
            started = time.perf_counter()
            response, content_type, status = post_infer(upload, cvd_type)
            elapsed = time.perf_counter() - started
            deployed_image = Image.open(io.BytesIO(response)).convert("RGB")
            deployed = np.asarray(deployed_image, np.float32) / 255.0

            expected_float = local._correct_image(upload_array, cvd_type, 1.0)
            expected_u8 = local._to_u8(expected_float)
            expected_jpeg = jpeg_bytes(Image.fromarray(expected_u8), local.config.RESPONSE_JPEG_QUALITY)
            expected = np.asarray(Image.open(io.BytesIO(expected_jpeg)).convert("RGB"), np.float32) / 255.0
            delta = np.abs(deployed - expected)
            records.append({
                "id": row["id"], "category": row["category"], "cvd_type": cvd_type,
                "status": status, "content_type": content_type,
                "width": deployed_image.width, "height": deployed_image.height,
                "expected_width": expected.shape[1], "expected_height": expected.shape[0],
                "parity_mae": float(delta.mean()), "parity_p99": float(np.quantile(delta, 0.99)),
                "elapsed_seconds": elapsed,
            })
            if args.all:
                OUT.mkdir(exist_ok=True)
                checkpoint.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[{index}/{len(selected)}] {row['category']}", flush=True)

    checks = {
        "http_200": all(row["status"] == 200 for row in records),
        "jpeg_response": all(row["content_type"] == "image/jpeg" for row in records),
        "dimensions_match": all(row["width"] == row["expected_width"] and row["height"] == row["expected_height"] for row in records),
        "pixel_parity_mae_le_2_255": all(row["parity_mae"] <= 2 / 255 for row in records),
        "pixel_parity_p99_le_4_255": all(row["parity_p99"] <= 4 / 255 for row in records),
    }
    result = {
        "api": API, "images": len(selected), "requests": len(records), "types": ["p", "d"],
        "browser_upload_jpeg_quality": 92,
        "expected_response_jpeg_quality": local.config.RESPONSE_JPEG_QUALITY,
        "checks": checks,
        "parity_mae_mean": float(np.mean([row["parity_mae"] for row in records])),
        "parity_mae_max": float(np.max([row["parity_mae"] for row in records])),
        "parity_p99_max": float(np.max([row["parity_p99"] for row in records])),
        "latency_seconds_mean": float(np.mean([row["elapsed_seconds"] for row in records])),
        "latency_seconds_max": float(np.max([row["elapsed_seconds"] for row in records])),
        "passed": all(checks.values()), "rows": records,
    }
    OUT.mkdir(exist_ok=True)
    suffix = "_all" if args.all else ""
    (OUT / f"deployed_api_e2e{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Deployed API E2E parity", "",
        f"- Endpoint: `{API}/infer`", f"- Images: {len(selected)} ({'full frozen manifest' if args.all else 'one per category'})",
        f"- HTTP requests: {len(records)} (P and D)",
        f"- Verdict: **{'PASS' if result['passed'] else 'FAIL'}**", "",
        "| Check | Result |", "|---|---|",
        *[f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in checks.items()], "",
        f"- Mean pixel parity MAE: {result['parity_mae_mean']:.6f}",
        f"- Worst pixel parity MAE: {result['parity_mae_max']:.6f}",
        f"- Worst pixel parity p99: {result['parity_p99_max']:.6f}",
        f"- Mean latency: {result['latency_seconds_mean']:.3f} s",
        f"- Maximum latency: {result['latency_seconds_max']:.3f} s",
    ]
    (OUT / f"DEPLOYED_API_E2E{suffix.upper()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.all and checkpoint.exists():
        checkpoint.unlink()
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
