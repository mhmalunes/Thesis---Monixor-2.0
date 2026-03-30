"""
pipeline_api.py — Flask bridge between Monixor 2.0 UI and the vision pipeline.
Branch: dev/different-method

Key differences from main:
  - No reference image required (reference-free, generalizable to any monitor)
  - Deskew via contour/edge-based monitor screen detection instead of SIFT+FLANN
  - Single-pass OCR instead of dual-pass (faster)
  - Auto GPU detection for EasyOCR

Install dependencies:
    pip install flask flask-cors opencv-python easyocr numpy torch

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

try:
    import torch
    _GPU = torch.cuda.is_available()
except ImportError:
    _GPU = False

# ── App setup ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"Initialising EasyOCR reader (gpu={_GPU})…")
reader = easyocr.Reader(["en"], gpu=_GPU)
logger.info("EasyOCR reader ready.")


# ── Pipeline helpers ───────────────────────────────────────────────────────────

VITAL_LABELS = {
    "HR"  : ["ecg", "hr", "heart"],
    "SpO2": ["spo2", "spoz", "spo", "sp02", "sp0", "sp0z", "oxygen"],
    "PR"  : ["pr"],
    "Resp": ["resp", "rosp", "rsp", "resp.", "rr"],
    "NIBP": ["nibp", "n1bp", "nibp:"],
    "Temp": ["temp", "tmp", "t1", "temp.", "tc"],
}

IGNORE_PHRASES = [
    "too low", "too high", "alarm", "alert", "sys", "sya",
    "source", "list", "configuration", "monitor", "patient",
    "manual", "standby", "review",
    # alarm banner variants (e.g. "** NIBP-Mean Too Low")
    "nibp-mean", "nibp mean", "mean too", "- mean",
]

# Short keywords that must match exactly (not as substrings) to avoid false positives
# e.g. "pr" should not match "pressure", "spo" should not match random noise
_EXACT_MATCH_KEYWORDS = {"hr", "pr", "rr", "spo", "sp0", "rsp", "tmp", "tc", "t1"}


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
    return len(text.strip()) > 15


def _is_value(text):
    cleaned = text.strip().replace(" ", "")
    if not any(c.isdigit() for c in cleaned):
        return False
    noise = re.sub(r"[\d\/\.\(\)\-]", "", cleaned)
    return len(noise) <= 1


def _identify_label(text):
    if _should_ignore(text):
        return None
    tl = text.lower().strip().rstrip(".:,")
    if len(tl) < 2:
        return None
    for label_name, keywords in VITAL_LABELS.items():
        for kw in keywords:
            if kw in _EXACT_MATCH_KEYWORDS:
                # Short/ambiguous keywords: exact match only to avoid false positives
                if tl == kw:
                    return label_name
            else:
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


# ── Waveform masking ──────────────────────────────────────────────────────────

def _mask_waveforms(img: np.ndarray) -> np.ndarray:
    """
    Blacks out the oscillating ECG/Pleth/Resp waveform lines in the LEFT 60%
    of the image using HSV saturation.  The right 40% (where all vital values
    and labels live) is left untouched.

    Benefit: removes hundreds of false text-detection candidates → OCR runs
    significantly faster AND picks up fewer spurious numbers.
    """
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Highly saturated, non-dark pixels → coloured waveform lines
    wave_mask = ((hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 80)).astype(np.uint8) * 255

    # Only apply in the left 60 % — vital values on the right stay intact.
    # Zero AFTER dilation so dilated edges don't bleed into the right side.
    kernel    = np.ones((9, 9), np.uint8)
    wave_mask = cv2.dilate(wave_mask, kernel, iterations=2)
    wave_mask[:, int(w * 0.60):] = 0

    result              = img.copy()
    result[wave_mask > 0] = 0
    logger.info(f"Waveform mask: {np.count_nonzero(wave_mask) // 255} px blacked out.")
    return result


# ── Stage 1 (new): Reference-free deskew via screen contour detection ─────────

def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Attempts to find the monitor screen as the largest rectangular contour and
    perform a perspective warp to a canonical upright view.

    Works on any monitor model — no reference image needed.
    Falls back to the original image if no clear rectangle is found.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur + edge detect
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    edges   = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screen_corners = None
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.10):   # must cover at least 10% of image area
            break
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            screen_corners = approx.reshape(4, 2).astype(np.float32)
            break

    if screen_corners is None:
        logger.info("Deskew: no clear rectangle found — using original image.")
        return img

    # Order corners: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = screen_corners.sum(axis=1)
    diff = np.diff(screen_corners, axis=1)
    rect[0] = screen_corners[np.argmin(s)]    # top-left
    rect[2] = screen_corners[np.argmax(s)]    # bottom-right
    rect[1] = screen_corners[np.argmin(diff)] # top-right
    rect[3] = screen_corners[np.argmax(diff)] # bottom-left

    # Output size: use the bounding box of the detected screen
    width  = int(max(_dist(rect[0], rect[1]), _dist(rect[2], rect[3])))
    height = int(max(_dist(rect[0], rect[3]), _dist(rect[1], rect[2])))

    if width < 100 or height < 100:
        logger.info("Deskew: detected rect too small — using original image.")
        return img

    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    logger.info(f"Deskew: screen detected, warped to {width}×{height}.")
    return warped


# ── Main pipeline function ─────────────────────────────────────────────────────

def run_pipeline(monitor_path: str) -> dict:
    """
    Reference-free pipeline:
      1. Contour-based deskew
      2. De-glare (Telea inpainting)
      3. Gamma correction + CLAHE
      4. Single-pass EasyOCR
      5. Label-value pairing
    Returns a dict with keys: HR, SpO2, NIBP, MAP, RR, Temp.
    """

    img = cv2.imread(monitor_path)
    if img is None:
        raise ValueError("Could not read image file.")

    # ── Stage 1: Deskew ───────────────────────────────────────────────────────
    img_straightened = _deskew(img)

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
    gray_dg = cv2.cvtColor(img_deglared, cv2.COLOR_BGR2GRAY)
    mean_b  = np.mean(gray_dg)

    gamma   = 1.5 if mean_b < 100 else (0.7 if mean_b > 150 else 1.0)
    lut     = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    img_gam = cv2.LUT(img_deglared, lut)

    gray_gam  = cv2.cvtColor(img_gam, cv2.COLOR_BGR2GRAY)
    clahe_f   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enh  = clahe_f.apply(gray_gam)
    img_final = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)
    logger.info(f"Gamma={gamma:.1f}; mean brightness {mean_b:.0f}→{np.mean(gray_enh):.0f}.")

    # ── Stage 4: Waveform masking + Single-pass EasyOCR ──────────────────────
    # Mask is derived from the DEGLARED COLOR image — img_final is grayscale
    # so HSV saturation is zero there and masking would do nothing on it.
    # Using the color image preserves cyan/yellow label text (SpO2, Resp, etc.)
    # while still removing the bright colored waveform lines.
    img_masked = _mask_waveforms(img_deglared)

    if img_masked.shape[1] > 900:
        scale   = 900 / img_masked.shape[1]
        ocr_img = cv2.resize(img_masked, (900, int(img_masked.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        logger.info(f"Resized for OCR: {img_masked.shape[1]}→900px wide.")
    else:
        ocr_img = img_masked

    detections = reader.readtext(ocr_img, detail=1)
    logger.info(f"OCR: {len(detections)} detections (single-pass, waveform-masked).")

    img_h, img_w = ocr_img.shape[:2]

    # ── Stage 5: Label-value pairing ──────────────────────────────────────────
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

    # Deduplicate labels
    seen_lbl = {}
    for item in labels_found:
        n = item[0]
        if n not in seen_lbl or item[2] > seen_lbl[n][2]:
            seen_lbl[n] = item
    labels_deduped = list(seen_lbl.values())

    # Deduplicate values
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

        # Small upward tolerance: if OCR places a label center slightly below
        # its value (multi-line read), we still find the value.
        BELOW = ly - 30

        if lname == "NIBP":
            # Primary: look for SYS/DIA as single "xx/xx" token
            cands = [(t, c, f) for (t, c, f, b) in values_found if c[1] > BELOW and "/" in t]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                best = cands[0]
                paired["NIBP"] = {"value": best[0], "value_conf": round(best[2], 2)}
                nibp_y = best[1][1]
                # MAP: accept "(83)" or bare "83" that is close below the NIBP value
                mc = [(t, c, f) for (t, c, f, b) in values_found
                      if c[1] > nibp_y and _dist(best[1], c) < 200
                      and (t.startswith("(") or t.isdigit())
                      and t not in (best[0], paired.get("HR", {}).get("value", ""))]
                if mc:
                    mc.sort(key=lambda v: _dist(best[1], v[1]))
                    raw_map = mc[0][0]
                    map_val = raw_map.strip("() ")
            else:
                # Fallback: two separate integers near the NIBP label → "sys/dia"
                nearby = [(t, c, f) for (t, c, f, b) in values_found
                          if c[1] > BELOW and t.isdigit()
                          and 40 <= int(t) <= 250
                          and _dist(lcenter, c) < img_w * 0.35]
                nearby.sort(key=lambda v: v[1][0])   # left → right
                if len(nearby) >= 2:
                    sys_v, dia_v = nearby[0], nearby[1]
                    if int(sys_v[0]) > int(dia_v[0]):   # sanity: sys > dia
                        paired["NIBP"] = {
                            "value": f"{sys_v[0]}/{dia_v[0]}",
                            "value_conf": round(min(sys_v[2], dia_v[2]), 2),
                        }
            continue

        if lname == "HR":
            cands = [(t, c, f) for (t, c, f, b) in values_found
                     if c[1] > BELOW and c[0] > lx + 30
                     and "." not in t and "/" not in t and not t.startswith("(")
                     and f >= 0.25 and t.isdigit() and 20 <= int(t) <= 300]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["HR"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "Temp":
            cands = [(t, c, f) for (t, c, f, b) in values_found if "." in t and c[1] > BELOW]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["Temp"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "SpO2":
            cands = [(t, c, f) for (t, c, f, b) in values_found
                     if c[1] > BELOW and "." not in t and "/" not in t
                     and not t.startswith("(") and t.isdigit() and 70 <= int(t) <= 100]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["SpO2"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "PR":
            nibp_sys  = _nibp_systolic(paired)
            spo2_ctr  = _spo2_center(paired, values_found)
            hr_val    = paired.get("HR",   {}).get("value", "")
            spo2_val  = paired.get("SpO2", {}).get("value", "")
            if spo2_ctr:
                cands = [(t, c, f) for (t, c, f, b) in values_found
                         if c[0] > spo2_ctr[0] - 20 and abs(c[1] - spo2_ctr[1]) < 200
                         and "." not in t and "/" not in t and not t.startswith("(")
                         and t.isdigit() and 20 < int(t) <= 250
                         and t not in (nibp_sys, spo2_val, hr_val)]
            else:
                cands = []
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["PR"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        # Default: Resp
        nibp_sys = _nibp_systolic(paired)
        cands = [(t, c, f) for (t, c, f, b) in values_found
                 if c[1] > BELOW and "." not in t and "/" not in t
                 and not t.startswith("(") and t.isdigit() and 4 <= int(t) <= 60
                 and t != nibp_sys]
        if cands:
            cands.sort(key=lambda v: _dist(lcenter, v[1]))
            paired[lname] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}

    # ── Fallbacks ──────────────────────────────────────────────────────────────
    if "Temp" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found if "." in t and "/" not in t]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["Temp"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    # HR fallback runs BEFORE SpO2 — both share the 70-100 range so HR must
    # claim its top-right value first, otherwise SpO2 fallback steals it.
    if "HR" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if t.isdigit() and 20 <= int(t) <= 300
              and c[1] < img_h * 0.40 and c[0] > img_w * 0.55
              and f >= 0.25
              and t not in [paired.get("SpO2", {}).get("value", ""),
                            paired.get("PR",   {}).get("value", "")]]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["HR"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "SpO2" not in paired:
        # Use vertical position to separate from HR — do NOT exclude by value text
        # because HR and SpO2 can legitimately be the same number (e.g. both 98).
        # HR lives in the top ~30%, SpO2 in the middle ~35-55%.
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if t.isdigit() and 70 <= int(t) <= 100
              and c[0] > img_w * 0.50 and c[1] > img_h * 0.30]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["SpO2"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "PR" not in paired and "SpO2" in paired:
        nibp_sys = _nibp_systolic(paired)
        spo2_ctr = _spo2_center(paired, values_found)
        hr_val   = paired.get("HR",   {}).get("value", "")
        spo2_val = paired.get("SpO2", {}).get("value", "")
        if spo2_ctr:
            fc = [(t, c, f) for (t, c, f, b) in values_found
                  if c[0] > spo2_ctr[0] and abs(c[1] - spo2_ctr[1]) < 200
                  and "." not in t and "/" not in t and not t.startswith("(")
                  and t.isdigit() and 20 < int(t) <= 250
                  and t not in (nibp_sys, spo2_val, hr_val)]
            if fc:
                fc.sort(key=lambda v: _dist(spo2_ctr, v[1]))
                paired["PR"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    if "Resp" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if t.isdigit() and 4 <= int(t) <= 60
              and img_h * 0.30 < c[1] < img_h * 0.80 and c[0] > img_w * 0.50
              and t not in [paired.get("HR",   {}).get("value", ""),
                            paired.get("SpO2", {}).get("value", ""),
                            paired.get("PR",   {}).get("value", ""),
                            _nibp_systolic(paired)]]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["Resp"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    # ── NIBP standalone fallback (runs when NIBP label was never detected) ────
    if "NIBP" not in paired:
        # Look for a "sys/dia" token in the right half with valid BP range
        fc = [(t, c, f) for (t, c, f, b) in values_found
              if "/" in t and c[0] > img_w * 0.45]
        valid_nibp = []
        for (t, c, f) in fc:
            parts = t.split("/")
            if len(parts) == 2:
                try:
                    s, d = int(parts[0]), int(parts[1])
                    if 50 <= s <= 220 and 30 <= d <= 130 and s > d:
                        valid_nibp.append((t, c, f))
                except ValueError:
                    pass
        if valid_nibp:
            # Prefer the rightmost (main reading, not the small history list)
            valid_nibp.sort(key=lambda v: v[1][0], reverse=True)
            best = valid_nibp[0]
            paired["NIBP"] = {"value": best[0], "value_conf": round(best[2], 2)}
            # Try to find MAP near it
            nibp_pos = best[1]
            mc = [(t, c, f) for (t, c, f, b) in values_found
                  if _dist(nibp_pos, c) < img_w * 0.20
                  and (t.startswith("(") or (t.isdigit() and 40 <= int(t) <= 130))
                  and t != best[0]]
            if mc:
                mc.sort(key=lambda v: _dist(nibp_pos, v[1]))
                map_val = mc[0][0]

    if not map_val:
        mc = [(t, c, f) for (t, c, f, b) in values_found if t.startswith("(") and t.endswith(")")]
        if mc:
            mc.sort(key=lambda v: v[2], reverse=True)
            map_val = mc[0][0]

    # MAP bare-integer fallback: if still no MAP but NIBP found, look for a
    # plausible MAP integer (mean ≈ dia + 1/3 pulse pressure) near the NIBP value
    if not map_val and "NIBP" in paired:
        nibp_parts = paired["NIBP"]["value"].split("/")
        if len(nibp_parts) == 2:
            try:
                s, d   = int(nibp_parts[0]), int(nibp_parts[1])
                lo, hi = d - 5, s - 5        # MAP must be between dia and sys
                nibp_ctr = next(
                    (c for (t, c, f, b) in values_found if t == paired["NIBP"]["value"]),
                    None,
                )
                if nibp_ctr:
                    mc = [(t, c, f) for (t, c, f, b) in values_found
                          if t.isdigit() and lo <= int(t) <= hi
                          and _dist(nibp_ctr, c) < img_w * 0.25]
                    if mc:
                        mc.sort(key=lambda v: _dist(nibp_ctr, v[1]))
                        map_val = mc[0][0]
            except ValueError:
                pass

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
    return jsonify({"status": "ok"})


@app.route("/debug_ocr", methods=["POST"])
def debug_ocr():
    """
    POST /debug_ocr  — temporary debug endpoint
    Same input as /run_pipeline but returns raw OCR detections so we can
    see exactly what text was read and at what positions.
    Remove this endpoint once debugging is done.
    """
    if "monitor_photo" not in request.files:
        return jsonify({"error": "'monitor_photo' file is required."}), 400

    uid          = uuid.uuid4().hex[:8]
    monitor_path = os.path.join(TEMP_DIR, f"debug_{uid}.jpg")
    request.files["monitor_photo"].save(monitor_path)

    try:
        img          = cv2.imread(monitor_path)
        img_straight = _deskew(img)

        gray_c     = cv2.cvtColor(img_straight, cv2.COLOR_BGR2GRAY)
        glare_mask = (gray_c > 240).astype(np.uint8) * 255
        if np.count_nonzero(glare_mask) / glare_mask.size * 100 >= 5.0:
            kernel = np.ones((3, 3), np.uint8)
            img_dg = cv2.inpaint(img_straight, cv2.dilate(glare_mask, kernel), 2, cv2.INPAINT_TELEA)
        else:
            img_dg = img_straight.copy()

        img_masked = _mask_waveforms(img_dg)
        if img_masked.shape[1] > 900:
            scale      = 900 / img_masked.shape[1]
            ocr_img    = cv2.resize(img_masked, (900, int(img_masked.shape[0] * scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            ocr_img = img_masked

        raw = reader.readtext(ocr_img, detail=1)
        detections = []
        for (bbox, text, conf) in raw:
            cx = sum(pt[0] for pt in bbox) / 4
            cy = sum(pt[1] for pt in bbox) / 4
            cleaned = _clean_text(text)
            detections.append({
                "text"   : text,
                "cleaned": cleaned,
                "conf"   : round(conf, 3),
                "center" : [round(cx, 1), round(cy, 1)],
                "is_label": _identify_label(cleaned),
                "is_value": _is_value(cleaned),
            })

        return jsonify({
            "ocr_image_size": [ocr_img.shape[1], ocr_img.shape[0]],
            "total_detections": len(detections),
            "detections": sorted(detections, key=lambda d: d["center"][1]),
        })
    except Exception as exc:
        logger.error(f"Debug OCR error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(monitor_path)
        except OSError:
            pass


@app.route("/run_pipeline", methods=["POST"])
def pipeline_endpoint():
    """
    POST /run_pipeline
    Form-data fields:
        monitor_photo  — the uploaded monitor photo (image file)
        (reference_image no longer required)
    Returns JSON: { HR, SpO2, NIBP, MAP, RR, Temp }
    """
    if "monitor_photo" not in request.files:
        return jsonify({"error": "'monitor_photo' file is required."}), 400

    uid          = uuid.uuid4().hex[:8]
    monitor_path = os.path.join(TEMP_DIR, f"monitor_{uid}.jpg")

    request.files["monitor_photo"].save(monitor_path)
    logger.info(f"[{uid}] Image saved — starting pipeline…")

    try:
        result = run_pipeline(monitor_path)
        return jsonify(result)
    except Exception as exc:
        logger.error(f"[{uid}] Pipeline error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(monitor_path)
        except OSError:
            pass
        logger.info(f"[{uid}] Temp file cleaned up.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
