"""
batch_eval.py — Batch accuracy evaluation for the Monixor 2.0 pipeline.

Usage:
    1. Make sure pipeline_api.py is running (python pipeline_api.py)
    2. python batch_eval.py

Scans DATASET_DIR for images, matches each to the ground-truth CSV by
filename stem, runs the pipeline, and writes a per-image results CSV plus
an accuracy summary.
"""

import csv
import io
import os
import time
import urllib.request

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
FLASK_URL   = "http://localhost:5000/run_pipeline"

DATASET_DIR = r"C:\Users\Tela\Documents\Thesis\Dataset"
GT_CSV_NAME = "Monixor2.0_Ground Truth - Sheet1.csv"

# Fallback: load GT from Google Sheets if local CSV not found
GT_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1FFgE1AiqdIHChd5Tf83lhl43m8DiAsEoOs8pykgtXXA"
    "/export?format=csv"
)

OUTPUT_CSV   = "batch_results.csv"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}
LIMIT        = None   # e.g. 10 to run only the first N images; None = all

# Tolerance for numeric vitals (|predicted - ground_truth| <= TOL → correct)
TOLERANCE    = 1

# API key → GT column mapping
VITALS = [
    ("HR",   "HR"),
    ("SpO2", "Sp02"),
    ("PR",   "PR"),
    ("RR",   "Resp"),
    ("NIBP", "NIBP"),
    ("MAP",  "MAP"),
    ("Temp", "Temp"),
]
# ──────────────────────────────────────────────────────────────────────────────


def load_ground_truth():
    """Return a dict keyed by image name stem (lowercase) → row dict."""
    local_path = os.path.join(DATASET_DIR, GT_CSV_NAME)
    if os.path.exists(local_path):
        print(f"Loading ground truth from local file: {local_path}")
        with open(local_path, encoding="utf-8") as f:
            raw = f.read()
    else:
        print("Local GT CSV not found — loading from Google Sheets…")
        raw = urllib.request.urlopen(GT_URL).read().decode("utf-8")

    rows = list(csv.DictReader(io.StringIO(raw)))
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
    print(f"  {len(rows)} ground-truth rows loaded.\n")

    # Build lookup: image stem (lowercase) → row
    gt_lookup = {}
    for row in rows:
        stem = row.get("Image", "").strip().lower()
        if stem:
            gt_lookup[stem] = row
    return gt_lookup


def scan_images(folder):
    """Return sorted list of image file paths found in folder (flat scan)."""
    images = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            images.append(os.path.join(folder, fname))
    return images


def is_correct(pred_str, gt_str, vital_key):
    """Exact match for NIBP; ±TOLERANCE for all numeric vitals."""
    pred_str = str(pred_str).strip()
    gt_str   = str(gt_str).strip()

    if not pred_str or not gt_str:
        return False

    if vital_key == "NIBP":
        return pred_str == gt_str

    try:
        diff = abs(float(pred_str) - float(gt_str))
        return diff <= TOLERANCE
    except ValueError:
        return pred_str == gt_str


def run_batch():
    gt_lookup = load_ground_truth()
    images    = scan_images(DATASET_DIR)

    if not images:
        print(f"No images found in {DATASET_DIR}")
        return

    if LIMIT:
        images = images[:LIMIT]
        print(f"Running on first {LIMIT} images (LIMIT={LIMIT}).\n")
    else:
        print(f"Found {len(images)} image(s) in folder.\n")

    results   = []
    skipped   = 0
    no_gt     = 0
    processed = 0

    fieldnames = (
        ["Session", "Angle", "Image", "Status", "Time_s"]
        + [f"{api_key}_gt"   for api_key, _ in VITALS]
        + [f"{api_key}_pred" for api_key, _ in VITALS]
        + [f"{api_key}_ok"   for api_key, _ in VITALS]
    )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, img_path in enumerate(images, 1):
            fname      = os.path.basename(img_path)
            stem       = os.path.splitext(fname)[0]
            gt_row     = gt_lookup.get(stem.lower())

            print(f"[{idx:>3}/{len(images)}] {fname} … ", end="", flush=True)

            if gt_row is None:
                print("SKIP — no ground truth in CSV")
                no_gt += 1
                continue

            angle   = gt_row.get("Angle", "").strip()
            session = gt_row.get("Session", "").strip()

            t0 = time.time()
            try:
                with open(img_path, "rb") as img_f:
                    resp = requests.post(
                        FLASK_URL,
                        files={"monitor_photo": img_f},
                        timeout=180,
                    )
                elapsed = round(time.time() - t0, 1)

                if resp.status_code != 200:
                    print(f"ERR HTTP {resp.status_code} ({elapsed}s)")
                    skipped += 1
                    continue

                predicted = resp.json()

            except requests.exceptions.Timeout:
                print("ERR — timeout")
                skipped += 1
                continue
            except Exception as exc:
                print(f"ERR — {exc}")
                skipped += 1
                continue

            record = {
                "Session": session,
                "Angle":   angle,
                "Image":   stem,
                "Status":  "OK",
                "Time_s":  elapsed,
            }

            correct_count = 0
            for api_key, gt_key in VITALS:
                gt_val   = gt_row.get(gt_key, "").strip()
                pred_val = str(predicted.get(api_key, "") or "").strip()
                ok       = is_correct(pred_val, gt_val, api_key)
                if ok:
                    correct_count += 1
                record[f"{api_key}_gt"]   = gt_val
                record[f"{api_key}_pred"] = pred_val
                record[f"{api_key}_ok"]   = "1" if ok else "0"

            results.append(record)
            writer.writerow(record)
            out_f.flush()
            processed += 1

            vital_str = "  ".join(
                f"{api_key}={'✓' if record[f'{api_key}_ok']=='1' else '✗'}"
                f"({record[f'{api_key}_pred']}|gt:{record[f'{api_key}_gt']})"
                for api_key, _ in VITALS
            )
            print(f"OK ({elapsed}s)  {correct_count}/{len(VITALS)}  —  {vital_str}")

    print(f"\n{'='*70}")
    print(f"Processed: {processed}  |  No GT: {no_gt}  |  Errors/Skipped: {skipped}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"{'='*70}\n")

    if not results:
        print("No results to summarise.")
        return

    # ── Overall accuracy summary ───────────────────────────────────────────
    print("ACCURACY SUMMARY")
    print(f"{'Vital':<14}  {'Correct':>7}  {'%':>7}")
    print("-" * 34)

    for api_key, _ in VITALS:
        correct = sum(1 for r in results if r[f"{api_key}_ok"] == "1")
        pct     = correct / len(results) * 100
        label   = api_key if api_key == "NIBP" else f"{api_key} (±{TOLERANCE})"
        print(f"{label:<14}  {correct:>3}/{len(results):<3}  {pct:>6.1f}%")

    all_correct = sum(
        1 for r in results
        if all(r[f"{api_key}_ok"] == "1" for api_key, _ in VITALS)
    )
    print("-" * 34)
    print(f"{'All correct':<14}  {all_correct:>3}/{len(results):<3}  {all_correct/len(results)*100:>6.1f}%")

    # ── Per-angle breakdown ────────────────────────────────────────────────
    angles = sorted({r["Angle"] for r in results if r["Angle"]})
    if len(angles) > 1:
        header = " | ".join(api_key for api_key, _ in VITALS)
        print(f"\nPER-ANGLE BREAKDOWN  ({header})")
        print("-" * 70)
        for angle in angles:
            sub   = [r for r in results if r["Angle"] == angle]
            parts = []
            for api_key, _ in VITALS:
                c = sum(1 for r in sub if r[f"{api_key}_ok"] == "1")
                parts.append(f"{c}/{len(sub)}")
            print(f"  {angle:<8}: {' | '.join(parts)}")


if __name__ == "__main__":
    run_batch()
