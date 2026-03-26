"""
pipeline_api.py — Flask bridge between Monixor 2.0 UI and the vision pipeline.

Install dependencies:
    pip install flask flask-cors opencv-python easyocr numpy

Run:
    python pipeline_api.py

The server listens on http://localhost:5000
Endpoint: POST /run_pipeline
"""

import os
import re
import math
import uuid
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr

# ── App setup ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML file (file:// or any origin)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# EasyOCR reader — initialised once at startup (heavy; takes ~30 s first run)
logger.info("Initialising EasyOCR reader (this may take a moment)…")
reader = easyocr.Reader(["en"], gpu=False)
logger.info("EasyOCR reader ready.")


# ── Pipeline helpers (ported from Pipeline.ipynb, Colab calls stripped) ───────

VITAL_LABELS = {
    "HR"  : ["ecg", "hr", "heart"],
    "SpO2": ["spo2", "spoz", "spo", "oxygen"],
    "PR"  : ["pr"],
    "Resp": ["resp", "rosp", "rsp"],
    "NIBP": ["nibp"],
    "Temp": ["temp", "tmp"],
}

IGNORE_PHRASES = [
    "too low", "too high", "alarm", "alert", "sys", "sya",
    "source", "list", "configuration", "monitor", "patient",
    "manual", "standby", "review",
]


def _get_center(bbox):
    x = sum(pt[0] for pt in bbox) / 4
    y = sum(pt[1] for pt in bbox) / 4
    return (x, y)


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _clean_text(text):
    text = re.sub(r"^[^a-zA-Z0-9(]+", "", text.strip())
    text = re.sub(r"^[a-zA-Z](\d)", r"\1", text)
    return text


def _should_ignore(text):
    tl = text.lower()
    for phrase in IGNORE_PHRASES:
        if phrase in tl:
            return True
    return len(text.strip()) > 10


def _is_value(text):
    cleaned = text.strip().replace(" ", "")
    if not any(c.isdigit() for c in cleaned):
        return False
    noise = re.sub(r"[\d\/\.\(\)\-]", "", cleaned)
    return len(noise) <= 1


def _identify_label(text):
    if _should_ignore(text):
        return None
    tl = text.lower().strip()
    if len(tl) < 2:
        return None
    for label_name, keywords in VITAL_LABELS.items():
        for kw in keywords:
            if kw in tl or tl in kw:
                return label_name
    return None


def _nibp_systolic(paired):
    nibp = paired.get("NIBP", {}).get("value", "")
    return nibp.split("/")[0] if "/" in nibp else ""


def _spo2_center(paired, values):
    spo2_val = paired.get("SpO2", {}).get("value", "")
    if not spo2_val:
        return None
    for (vtext, vcenter, vconf, vbbox) in values:
        if vtext == spo2_val:
            return vcenter
    return None


# ── Main pipeline function ─────────────────────────────────────────────────────

def run_pipeline(monitor_path: str, reference_path: str) -> dict:
    """
    Full pipeline: geometry correction → de-glare → gamma/CLAHE → dual-pass OCR
    → label-value pairing.  Returns a dict with keys: HR, SpO2, NIBP, MAP, RR, Temp.
    """

    # ── Stage 1: Geometry Correction (SIFT + FLANN + RANSAC) ──────────────────
    img_tilted    = cv2.imread(monitor_path)
    img_reference = cv2.imread(reference_path)

    if img_tilted is None or img_reference is None:
        raise ValueError("Could not read one or both image files.")

    target_h, target_w = img_reference.shape[:2]
    if img_tilted.shape[:2] != (target_h, target_w):
        img_tilted = cv2.resize(img_tilted, (target_w, target_h))

    gray_t = cv2.cvtColor(img_tilted,    cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)

    clahe_init = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_t = clahe_init.apply(gray_t)
    gray_r = clahe_init.apply(gray_r)

    sift = cv2.SIFT_create(nfeatures=1500)
    kp_t, desc_t = sift.detectAndCompute(gray_t, None)
    kp_r, desc_r = sift.detectAndCompute(gray_r, None)

    img_straightened = img_tilted.copy()   # fallback if matching fails

    if desc_t is not None and desc_r is not None and len(kp_t) >= 4 and len(kp_r) >= 4:
        FLANN_INDEX_KDTREE = 1
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=50),
        )
        raw = matcher.knnMatch(desc_t, desc_r, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        good = sorted(good, key=lambda x: x.distance)[:100]

        if len(good) >= 4:
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                img_straightened = cv2.warpPerspective(img_tilted, H, (target_w, target_h))
                logger.info("Geometry correction applied (SIFT+FLANN+RANSAC).")
            else:
                logger.warning("Homography returned None — skipping geometry correction.")
        else:
            logger.warning(f"Only {len(good)} good matches — skipping geometry correction.")
    else:
        logger.warning("Insufficient descriptors — skipping geometry correction.")

    # ── Stage 2: De-Glare (Telea inpainting) ──────────────────────────────────
    gray_c     = cv2.cvtColor(img_straightened, cv2.COLOR_BGR2GRAY)
    glare_mask = (gray_c > 240).astype(np.uint8) * 255
    glare_pct  = np.count_nonzero(glare_mask) / glare_mask.size * 100

    if glare_pct >= 5.0:
        kernel       = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(glare_mask, kernel, iterations=1)
        img_deglared = cv2.inpaint(img_straightened, mask_dilated, 2, cv2.INPAINT_TELEA)
        logger.info(f"De-glare applied ({glare_pct:.1f}% glare).")
    else:
        img_deglared = img_straightened.copy()
        logger.info(f"De-glare skipped ({glare_pct:.1f}% glare — below threshold).")

    # ── Stage 3: Gamma Correction + CLAHE ─────────────────────────────────────
    gray_dg  = cv2.cvtColor(img_deglared, cv2.COLOR_BGR2GRAY)
    mean_b   = np.mean(gray_dg)

    gamma    = 1.5 if mean_b < 100 else (0.7 if mean_b > 150 else 1.0)
    lut      = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    img_gam  = cv2.LUT(img_deglared, lut)

    gray_gam = cv2.cvtColor(img_gam, cv2.COLOR_BGR2GRAY)
    clahe_f  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enh = clahe_f.apply(gray_gam)
    img_final = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)
    logger.info(f"Gamma={gamma:.1f} applied; mean brightness {mean_b:.0f}→{np.mean(gray_enh):.0f}.")

    # ── OCR: Dual-pass EasyOCR ────────────────────────────────────────────────
    # Pass 1 (labels): less-processed image preserves label text
    # Pass 2 (values): fully-enhanced image improves digit recognition
    det_labels = reader.readtext(img_deglared, detail=1)
    det_values = reader.readtext(img_final,    detail=1)
    detections = det_labels + det_values
    logger.info(f"OCR: {len(det_labels)} label-pass + {len(det_values)} value-pass detections.")

    img_h, img_w = img_final.shape[:2]

    # ── Label-value pairing (Cell 19 logic) ───────────────────────────────────
    labels_found = []
    values_found = []

    for (bbox, text, conf) in detections:
        center  = _get_center(bbox)
        cleaned = _clean_text(text)
        lbl     = _identify_label(cleaned)
        if lbl and conf > 0.10:
            labels_found.append((lbl, center, conf, cleaned, bbox))
        elif _is_value(cleaned) and conf > 0.25:
            values_found.append((cleaned, center, conf, bbox))

    # Deduplicate labels (keep highest-confidence per vital name)
    seen_lbl = {}
    for item in labels_found:
        n = item[0]
        if n not in seen_lbl or item[2] > seen_lbl[n][2]:
            seen_lbl[n] = item
    labels_deduped = list(seen_lbl.values())

    # Deduplicate values (same text + similar position → keep best)
    seen_val = {}
    for (vtext, vcenter, vconf, vbbox) in values_found:
        key = (vtext, round(vcenter[0] / 20) * 20, round(vcenter[1] / 20) * 20)
        if key not in seen_val or vconf > seen_val[key][2]:
            seen_val[key] = (vtext, vcenter, vconf, vbbox)
    values_found = list(seen_val.values())

    logger.info(f"Deduped: {len(labels_deduped)} labels, {len(values_found)} values.")

    def label_priority(item):
        return {"NIBP": 0, "HR": 1, "SpO2": 2, "Temp": 3, "Resp": 4, "PR": 5}.get(item[0], 99)

    labels_deduped.sort(key=label_priority)

    paired  = {}
    map_val = ""

    for (lname, lcenter, lconf, raw_text, lbbox) in labels_deduped:
        lx, ly = lcenter

        if lname == "NIBP":
            cands = [(t, c, f) for (t, c, f, b) in values_found if c[1] > ly and "/" in t]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                best = cands[0]
                paired["NIBP"] = {"value": best[0], "value_conf": round(best[2], 2)}
                nibp_y = best[1][1]
                mc = [(t, c, f) for (t, c, f, b) in values_found
                      if t.startswith("(") and t.endswith(")") and c[1] > nibp_y]
                if mc:
                    mc.sort(key=lambda v: _dist(best[1], v[1]))
                    map_val = mc[0][0]
            continue

        if lname == "HR":
            cands = [(t, c, f) for (t, c, f, b) in values_found
                     if c[1] > ly and c[0] > lx + 50
                     and "." not in t and "/" not in t and not t.startswith("(")
                     and f >= 0.5 and t.isdigit() and 40 <= int(t) <= 200]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["HR"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "Temp":
            cands = [(t, c, f) for (t, c, f, b) in values_found if "." in t and c[1] > ly]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["Temp"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "SpO2":
            cands = [(t, c, f) for (t, c, f, b) in values_found
                     if c[1] > ly and "." not in t and "/" not in t
                     and not t.startswith("(") and t.isdigit() and int(t) >= 90]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["SpO2"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "PR":
            nibp_sys   = _nibp_systolic(paired)
            spo2_ctr   = _spo2_center(paired, values_found)
            if spo2_ctr:
                cands = [(t, c, f) for (t, c, f, b) in values_found
                         if c[0] > spo2_ctr[0] and abs(c[1] - spo2_ctr[1]) < 150
                         and "." not in t and "/" not in t and not t.startswith("(")
                         and t.isdigit() and 20 < int(t) <= 200 and t != nibp_sys]
            else:
                cands = []
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["PR"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        # Default: Resp
        nibp_sys = _nibp_systolic(paired)
        cands = [(t, c, f) for (t, c, f, b) in values_found
                 if c[1] > ly and "." not in t and "/" not in t
                 and not t.startswith("(") and t.isdigit() and int(t) > 10
                 and t != nibp_sys]
        if cands:
            cands.sort(key=lambda v: _dist(lcenter, v[1]))
            paired[lname] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}

    # ── Fallbacks (Cell 19 fallback block) ────────────────────────────────────
    if "Temp" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found if "." in t and "/" not in t]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["Temp"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "SpO2" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if t.isdigit() and int(t) >= 90 and c[0] > img_w * 0.55]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["SpO2"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "PR" not in paired and "SpO2" in paired:
        nibp_sys  = _nibp_systolic(paired)
        spo2_ctr  = _spo2_center(paired, values_found)
        if spo2_ctr:
            fc = [(t, c, f) for (t, c, f, b) in values_found
                  if c[0] > spo2_ctr[0] and abs(c[1] - spo2_ctr[1]) < 150
                  and "." not in t and "/" not in t and not t.startswith("(")
                  and t != paired.get("SpO2", {}).get("value", "")
                  and t.isdigit() and 20 < int(t) <= 200 and t != nibp_sys]
            if fc:
                fc.sort(key=lambda v: _dist(spo2_ctr, v[1]))
                paired["PR"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "HR" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if t.isdigit() and 40 <= int(t) <= 200
              and c[1] < img_h * 0.35 and c[0] > img_w * 0.60
              and f >= 0.5
              and t not in [paired.get("SpO2", {}).get("value", ""),
                            paired.get("PR",   {}).get("value", "")]]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["HR"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "Resp" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if t.isdigit() and 8 <= int(t) <= 40
              and img_h * 0.35 < c[1] < img_h * 0.75 and c[0] > img_w * 0.55
              and t not in [paired.get("HR",   {}).get("value", ""),
                            paired.get("SpO2", {}).get("value", ""),
                            paired.get("PR",   {}).get("value", "")]]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["Resp"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if not map_val:
        mc = [(t, c, f) for (t, c, f, b) in values_found if t.startswith("(") and t.endswith(")")]
        if mc:
            mc.sort(key=lambda v: v[2], reverse=True)
            map_val = mc[0][0]

    # ── Build response (map pipeline names → UI field names) ──────────────────
    result = {
        "HR"  : paired.get("HR",   {}).get("value", ""),
        "SpO2": paired.get("SpO2", {}).get("value", ""),
        "NIBP": paired.get("NIBP", {}).get("value", ""),
        "MAP" : map_val.strip("()").strip() if map_val else "",
        "RR"  : paired.get("Resp", {}).get("value", ""),
        "Temp": paired.get("Temp", {}).get("value", ""),
    }
    logger.info(f"Extracted vitals: {result}")
    return result


# ── Flask endpoints ────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Quick health check — useful to confirm the server is up."""
    return jsonify({"status": "ok"})


@app.route("/run_pipeline", methods=["POST"])
def pipeline_endpoint():
    """
    POST /run_pipeline
    Form-data fields:
        monitor_photo  — the uploaded monitor photo (image file)
        reference_image — the reference (base) image for this monitor model (image file)
    Returns JSON: { HR, SpO2, NIBP, MAP, RR, Temp }
    """
    if "monitor_photo" not in request.files or "reference_image" not in request.files:
        return jsonify({"error": "Both 'monitor_photo' and 'reference_image' files are required."}), 400

    uid            = uuid.uuid4().hex[:8]
    monitor_path   = os.path.join(TEMP_DIR, f"monitor_{uid}.jpg")
    reference_path = os.path.join(TEMP_DIR, f"reference_{uid}.jpg")

    request.files["monitor_photo"].save(monitor_path)
    request.files["reference_image"].save(reference_path)
    logger.info(f"[{uid}] Images saved — starting pipeline…")

    try:
        result = run_pipeline(monitor_path, reference_path)
        return jsonify(result)
    except Exception as exc:
        logger.error(f"[{uid}] Pipeline error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        for path in (monitor_path, reference_path):
            try:
                os.remove(path)
            except OSError:
                pass
        logger.info(f"[{uid}] Temp files cleaned up.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
