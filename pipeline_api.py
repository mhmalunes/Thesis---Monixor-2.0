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
from difflib import SequenceMatcher

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
    "HR"  : ["ecg", "hr", "heart", "eco", "ecd", "ecq"],
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
    # bottom button bar labels that contain vital-sign keywords
    "measure", "setup", "zero ibp", "zero bp",
]

# Short keywords that must match exactly (not as substrings) to avoid false positives
# e.g. "pr" should not match "pressure", "spo" should not match random noise
_EXACT_MATCH_KEYWORDS = {"hr", "pr", "rr", "spo", "sp0", "rsp", "tmp", "tc", "t1", "eco", "ecd", "ecq"}


def _get_center(bbox):
    x = sum(pt[0] for pt in bbox) / 4
    y = sum(pt[1] for pt in bbox) / 4
    return (x, y)


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _bbox_area(bbox):
    """Return the pixel area of an EasyOCR bounding box (list of 4 [x,y] corners)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _clean_text(text):
    text = re.sub(r"^[^a-zA-Z0-9(]+", "", text.strip())
    text = re.sub(r"^[a-zA-Z](\d)", r"\1", text)
    # If a pure numeric/fraction token is followed by whitespace + noise
    # (e.g. OCR merges "100" and "77" into "100 kn"), strip the trailing noise.
    m = re.match(r"^([\d./()]+)\s+\S", text)
    if m:
        text = m.group(1)
    return text


def _should_ignore(text):
    tl = text.lower()
    for phrase in IGNORE_PHRASES:
        if phrase in tl:
            return True
    # Multi-word tokens containing vital keywords are annotation text, not primary labels.
    # e.g. "Source SpO2", "Souce Spo2" (OCR misread) — not the actual SpO2 label.
    if " " in tl.strip():
        for keywords in VITAL_LABELS.values():
            if any(kw in tl for kw in keywords if len(kw) >= 3):
                return True
    return len(text.strip()) > 15


def _is_value(text):
    cleaned = text.strip().replace(" ", "")
    if not any(c.isdigit() for c in cleaned):
        return False
    noise = re.sub(r"[\d\/\.\(\)\-]", "", cleaned)
    return len(noise) <= 1


def _fuzzy_score(a: str, b: str) -> float:
    """Similarity ratio between two strings (0–1)."""
    return SequenceMatcher(None, a, b).ratio()


def _identify_label(text):
    """
    Match OCR text to a vital-sign label using both exact/substring rules
    and fuzzy similarity.  Fuzzy matching catches OCR misreads like
    'Sp0z'→SpO2, 'N1BP'→NIBP, 'Reap'→Resp without hardcoding positions.
    """
    if _should_ignore(text):
        return None
    tl = text.lower().strip().rstrip(".:,")
    if len(tl) < 2:
        return None

    # ── Pass 1: exact / substring rules ──────────────────────────────────────
    for label_name, keywords in VITAL_LABELS.items():
        for kw in keywords:
            if kw in _EXACT_MATCH_KEYWORDS:
                if tl == kw:
                    return label_name
            else:
                if kw in tl or tl in kw:
                    return label_name

    # ── Pass 2: fuzzy match (handles OCR character-level errors) ─────────────
    # Only try fuzzy on short tokens (≤8 chars) to avoid false positives on
    # longer strings that happen to be similar to a keyword.
    if len(tl) <= 8:
        best_label, best_score = None, 0.0
        for label_name, keywords in VITAL_LABELS.items():
            for kw in keywords:
                score = _fuzzy_score(tl, kw)
                if score > best_score:
                    best_score = score
                    best_label = label_name
        if best_score >= 0.75:
            logger.debug(f"Fuzzy label match: '{tl}' → {best_label} (score={best_score:.2f})")
            return best_label

    return None


def _nibp_systolic(paired):
    nibp = paired.get("NIBP", {}).get("value", "")
    return nibp.split("/")[0] if "/" in nibp else ""


def _spo2_center(paired, values):
    # Prefer the authoritative center stored when SpO2 was paired.
    vc = paired.get("SpO2", {}).get("value_center")
    if vc:
        return vc
    spo2_val = paired.get("SpO2", {}).get("value", "")
    if not spo2_val:
        return None
    for (vtext, vcenter, vconf, vbbox) in values:
        if vtext == spo2_val:
            return vcenter
    return None


# ── Waveform masking ──────────────────────────────────────────────────────────

def _compute_wave_mask(color_img: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask (uint8, same H×W) where waveform pixels = 255.
    Derived from the color image's HSV saturation; only covers the left 60%.
    """
    h, w = color_img.shape[:2]
    hsv  = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 80)).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask[:, int(w * 0.60):] = 0
    return mask


def _mask_waveforms(img: np.ndarray) -> np.ndarray:
    """
    Blacks out the oscillating ECG/Pleth/Resp waveform lines in the LEFT 60%
    of the image using HSV saturation.  The right 40% (where all vital values
    and labels live) is left untouched.
    """
    mask   = _compute_wave_mask(img)
    result = img.copy()
    result[mask > 0] = 0
    logger.info(f"Waveform mask: {np.count_nonzero(mask) // 255} px blacked out.")
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

    # ── Stage 4: Waveform masking + Dual-pass EasyOCR ────────────────────────
    # Waveform mask (from color image) is applied to both passes so ECG/Pleth
    # lines don't generate false OCR detections.
    # Pass 1 (color image)  → preserves colored label text (SpO2, Resp, etc.)
    # Pass 2 (CLAHE image)  → boosts contrast for small values like (83), PR 77
    wm             = _compute_wave_mask(img_deglared)
    ocr_color      = img_deglared.copy();  ocr_color[wm > 0] = 0
    ocr_clahe      = img_final.copy();     ocr_clahe[wm > 0] = 0

    # No resize — small text like "(83)" and PR value become too small at 900px.
    # Run OCR at full deskewed resolution (same as main branch behaviour).

    det_labels = reader.readtext(ocr_color, detail=1)
    det_values = reader.readtext(ocr_clahe, detail=1)
    detections = det_labels + det_values
    logger.info(f"OCR: {len(det_labels)} (color) + {len(det_values)} (CLAHE) detections.")

    img_h, img_w = ocr_color.shape[:2]

    # ── Stage 5: Label-value pairing ──────────────────────────────────────────
    labels_found = []
    values_found = []
    paren_values = []   # parenthesized readings (e.g. "(83)") at lower conf threshold

    # Find the y-position of "BeneView T8" or equivalent external hardware text
    # so we can exclude any detections below it (they are outside the monitor screen).
    external_y = img_h  # default: no external hardware detected
    for (bbox, text, conf) in detections:
        if "beneview" in text.lower() or "bene view" in text.lower():
            external_y = min(external_y, _get_center(bbox)[1])
            logger.info(f"External hardware text detected at y={external_y:.0f} — values below excluded.")

    for (bbox, text, conf) in detections:
        center  = _get_center(bbox)
        if center[1] >= external_y:
            continue   # below external hardware boundary
        cleaned = _clean_text(text)
        lbl     = _identify_label(cleaned)
        if lbl and conf > 0.10:
            labels_found.append((lbl, center, conf, cleaned, bbox))
        elif _is_value(cleaned) and conf > 0.25:
            values_found.append((cleaned, center, conf, bbox))
        # Collect parenthesized values at lower confidence — monitors show MAP as "(83)"
        # which can be low-confidence due to small font.
        if (cleaned.startswith("(") and cleaned.endswith(")")
                and len(cleaned) >= 3 and conf > 0.10):
            paren_values.append((cleaned, center, conf, bbox))

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
                paired["NIBP"] = {"value": best[0], "value_conf": round(best[2], 2),
                                  "value_center": best[1]}
                nibp_y = best[1][1]
                # MAP: prefer parenthesized "(83)" from paren_values (lower conf threshold)
                # then fall back to bare digit — monitors always show MAP in parens.
                pm = [(t, c, f) for (t, c, f, b) in paren_values
                      if c[1] > nibp_y and _dist(best[1], c) < 200]
                if pm:
                    pm.sort(key=lambda v: _dist(best[1], v[1]))
                    map_val = pm[0][0].strip("() ")
                else:
                    _nibp_sys_str = best[0].split("/")[0]
                    mc = [(t, c, f) for (t, c, f, b) in values_found
                          if c[1] > nibp_y and _dist(best[1], c) < 120
                          and t.isdigit() and 40 <= int(t) <= 130
                          and t not in (best[0], paired.get("HR", {}).get("value", ""),
                                        paired.get("SpO2", {}).get("value", ""))
                          and int(t) < int(_nibp_sys_str)]
                    if mc:
                        mc.sort(key=lambda v: _dist(best[1], v[1]))
                        map_val = mc[0][0]
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
            cands = [(t, c, f, b) for (t, c, f, b) in values_found
                     if c[1] > ly + img_h * 0.04 and c[0] > lx + 30
                     and "." not in t and "/" not in t and not t.startswith("(")
                     and f >= 0.25 and t.isdigit() and 20 <= int(t) <= 300]
            if cands:
                # Drop scale markers (small font) by keeping only candidates whose
                # bbox area is at least 25% of the largest detected bbox.
                # Then y-ascending: ECG is always the topmost channel.
                max_area = max(_bbox_area(v[3]) for v in cands)
                large    = [v for v in cands if _bbox_area(v[3]) >= max_area * 0.25]
                best     = sorted(large or cands, key=lambda v: v[1][1])[0]
                paired["HR"] = {"value": best[0], "value_conf": round(best[2], 2),
                                "value_center": best[1]}
            continue

        if lname == "Temp":
            cands = [(t, c, f) for (t, c, f, b) in values_found if "." in t and c[1] > BELOW]
            if cands:
                cands.sort(key=lambda v: _dist(lcenter, v[1]))
                paired["Temp"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2)}
            continue

        if lname == "SpO2":
            _hr_vc = paired.get("HR", {}).get("value_center")
            cands = [(t, c, f, b) for (t, c, f, b) in values_found
                     if c[1] > BELOW and "." not in t and "/" not in t
                     and not t.startswith("(") and t.isdigit() and 70 <= int(t) <= 100
                     and (_hr_vc is None or _dist(c, _hr_vc) > 60)]
            if cands:
                # Largest bounding box = main SpO2 value; alarm limits are small text
                cands.sort(key=lambda v: -_bbox_area(v[3]))
                paired["SpO2"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2),
                                  "value_center": cands[0][1]}
            continue

        if lname == "PR":
            nibp_sys  = _nibp_systolic(paired)
            spo2_ctr  = _spo2_center(paired, values_found)
            spo2_c    = paired.get("SpO2", {}).get("value_center")
            hr_c      = paired.get("HR",   {}).get("value_center")
            if spo2_ctr:
                cands = [(t, c, f) for (t, c, f, b) in values_found
                         if c[0] > spo2_ctr[0] - 20 and abs(c[1] - spo2_ctr[1]) < 100
                         and "." not in t and "/" not in t and not t.startswith("(")
                         and t.isdigit() and 20 < int(t) <= 250
                         and t != nibp_sys
                         and (spo2_c is None or _dist(c, spo2_c) > 30)
                         and (hr_c   is None or _dist(c, hr_c)   > 30)]
            else:
                cands = []
            if cands:
                # Sort by distance to SpO2 center — PR is always adjacent to SpO2.
                # Using lcenter is unreliable because "PR" also appears as a column
                # header in the NIBP history table (often higher OCR confidence).
                cands.sort(key=lambda v: _dist(spo2_ctr, v[1]))
                paired["PR"] = {"value": cands[0][0], "value_conf": round(cands[0][2], 2),
                                "value_center": cands[0][1]}
            continue

        # Default: Resp
        # Minimum y-offset skips same-row scale markers (30, 8); bbox-area picks
        # the large display value over any small residual digit.
        nibp_sys = _nibp_systolic(paired)
        cands = [(t, c, f, b) for (t, c, f, b) in values_found
                 if c[1] > ly + img_h * 0.03 and "." not in t and "/" not in t
                 and not t.startswith("(") and t.isdigit() and 4 <= int(t) <= 60
                 and t != nibp_sys]
        if cands:
            cands.sort(key=lambda v: -_bbox_area(v[3]))
            best_t, best_c, best_f, best_b = cands[0]
            paired[lname] = {"value": best_t, "value_conf": round(best_f, 2)}
            # Targeted re-OCR: upscale only the detected value's own bbox at 3×
            # to correct single-digit misreads (e.g. "19" → "29").
            # Cropping the label area instead would capture waveform scale markers
            # ("30", "8") and produce wrong corrections.
            bx1 = max(0,     int(min(p[0] for p in best_b)) - 4)
            bx2 = min(img_w, int(max(p[0] for p in best_b)) + 4)
            by1 = max(0,     int(min(p[1] for p in best_b)) - 4)
            by2 = min(img_h, int(max(p[1] for p in best_b)) + 4)
            resp_crop = img_deglared[by1:by2, bx1:bx2]
            if resp_crop.size > 0:
                resp_up = cv2.resize(resp_crop,
                                     (resp_crop.shape[1] * 3, resp_crop.shape[0] * 3),
                                     interpolation=cv2.INTER_CUBIC)
                for (_, rr_text, rr_conf) in sorted(
                        reader.readtext(resp_up, detail=1), key=lambda x: -x[2]):
                    rr_cl = _clean_text(rr_text)
                    if (rr_cl.isdigit() and 4 <= int(rr_cl) <= 60
                            and rr_conf > 0.25 and rr_cl != nibp_sys):
                        if rr_cl != best_t:
                            logger.info(
                                f"Resp re-OCR corrected: {best_t} → {rr_cl} "
                                f"(conf={rr_conf:.2f})"
                            )
                            paired[lname]["value"]      = rr_cl
                            paired[lname]["value_conf"] = round(rr_conf, 2)
                        break

    # ── Fallbacks ──────────────────────────────────────────────────────────────
    if "Temp" not in paired:
        fc = [(t, c, f) for (t, c, f, b) in values_found if "." in t and "/" not in t]
        if fc:
            fc.sort(key=lambda v: v[2], reverse=True)
            paired["Temp"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2)}

    # HR fallback runs BEFORE SpO2 — both share the 70-100 range so HR must
    # claim its top-right value first, otherwise SpO2 fallback steals it.
    if "HR" not in paired:
        _spo2_vc = paired.get("SpO2", {}).get("value_center")
        _pr_vc   = paired.get("PR",   {}).get("value_center")
        fc = [(t, c, f, b) for (t, c, f, b) in values_found
              if t.isdigit() and 20 <= int(t) <= 300
              and c[1] < img_h * 0.40 and c[0] > img_w * 0.55
              and f >= 0.25
              # Exclude by position (not value string) — HR and PR can share the same number
              and (_spo2_vc is None or _dist(c, _spo2_vc) > 60)
              and (_pr_vc   is None or _dist(c, _pr_vc)   > 60)]
        if fc:
            # Drop scale markers (small font); among remaining large-font values,
            # topmost y = ECG channel (always the top channel on the monitor).
            max_area = max(_bbox_area(v[3]) for v in fc)
            large    = [v for v in fc if _bbox_area(v[3]) >= max_area * 0.25]
            best     = sorted(large or fc, key=lambda v: v[1][1])[0]
            paired["HR"] = {"value": best[0], "value_conf": round(best[2], 2),
                            "value_center": best[1]}

    if "SpO2" not in paired:
        # SpO2 is always left of PR and in the upper portion of the screen.
        # Exclude candidates near the NIBP value to avoid alarm-limit numbers
        # (e.g. "100" on the NIBP bar) being mistaken for SpO2.
        nibp_vc = paired.get("NIBP", {}).get("value_center")
        hr_vc   = paired.get("HR",   {}).get("value_center")
        fc = [(t, c, f, b) for (t, c, f, b) in values_found
              if t.isdigit() and 70 <= int(t) <= 100
              and img_w * 0.50 < c[0] < img_w * 0.80
              and img_h * 0.15 < c[1] < img_h * 0.65   # SpO2 is in upper 65% of screen
              and (nibp_vc is None or _dist(c, nibp_vc) > img_w * 0.15)
              and (hr_vc   is None or _dist(c, hr_vc)   > 60)]
        if fc:
            fc.sort(key=lambda v: -_bbox_area(v[3]))  # largest bbox = main SpO2 value
            paired["SpO2"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2),
                              "value_center": fc[0][1]}

    if "PR" not in paired and "SpO2" in paired:
        nibp_sys = _nibp_systolic(paired)
        spo2_ctr = _spo2_center(paired, values_found)
        spo2_c   = paired.get("SpO2", {}).get("value_center")
        hr_c     = paired.get("HR",   {}).get("value_center")
        if spo2_ctr:
            fc = [(t, c, f) for (t, c, f, b) in values_found
                  if c[0] > spo2_ctr[0] - 30 and abs(c[1] - spo2_ctr[1]) < 100
                  and "." not in t and "/" not in t and not t.startswith("(")
                  and t.isdigit() and 20 < int(t) <= 250
                  and t != nibp_sys
                  and (spo2_c is None or _dist(c, spo2_c) > 30)
                  and (hr_c   is None or _dist(c, hr_c)   > 30)]
            if fc:
                fc.sort(key=lambda v: _dist(spo2_ctr, v[1]))
                paired["PR"] = {"value": fc[0][0], "value_conf": round(fc[0][2], 2),
                                "value_center": fc[0][1]}

    # ── PR targeted re-OCR (SpO2 and PR often merged into one OCR token) ─────
    # When OCR merges "100" and "77" into "100 kn", PR is unrecoverable from
    # values_found.  Crop the area just right of the SpO2 value, upscale 3×,
    # and re-run OCR to find PR at higher effective resolution.
    if "PR" not in paired and "SpO2" in paired:
        spo2_vc = paired["SpO2"].get("value_center")
        if spo2_vc:
            sx, sy   = int(spo2_vc[0]), int(spo2_vc[1])
            x1 = min(img_w, sx + 10)
            x2 = min(img_w, sx + int(img_w * 0.12))
            y1 = max(0,     sy - int(img_h * 0.12))
            y2 = min(img_h, sy + int(img_h * 0.10))
            crop = img_deglared[y1:y2, x1:x2]
            if crop.size > 0:
                crop_up = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                                     interpolation=cv2.INTER_CUBIC)
                # Exclude NIBP systolic only — HR can equal PR so don't exclude HR value
                _excl = {_nibp_systolic(paired)}
                for (_, pr_text, pr_conf) in sorted(
                        reader.readtext(crop_up, detail=1), key=lambda x: -x[2]):
                    pr_cl = _clean_text(pr_text)
                    if (pr_cl.isdigit() and 20 < int(pr_cl) <= 250
                            and pr_conf > 0.10 and pr_cl not in _excl):
                        paired["PR"] = {"value": pr_cl, "value_conf": round(pr_conf, 2)}
                        logger.info(f"PR from targeted re-OCR: {pr_cl} (conf={pr_conf:.2f})")
                        break

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
            paired["NIBP"] = {"value": best[0], "value_conf": round(best[2], 2),
                              "value_center": best[1]}
            # Try to find MAP near it — prefer parenthesized value
            nibp_pos = best[1]
            pm = [(t, c, f) for (t, c, f, b) in paren_values
                  if _dist(nibp_pos, c) < img_w * 0.20]
            if pm:
                pm.sort(key=lambda v: _dist(nibp_pos, v[1]))
                map_val = pm[0][0]
            else:
                mc = [(t, c, f) for (t, c, f, b) in values_found
                      if _dist(nibp_pos, c) < img_w * 0.15
                      and t.isdigit() and 40 <= int(t) <= 130
                      and t != best[0]]
                if mc:
                    mc.sort(key=lambda v: _dist(nibp_pos, v[1]))
                    map_val = mc[0][0]

    if not map_val and paren_values:
        mc = sorted(paren_values, key=lambda v: v[2], reverse=True)
        map_val = mc[0][0]

    # MAP bare-integer fallback: if still no MAP but NIBP found, look for a
    # plausible MAP integer (mean ≈ dia + 1/3 pulse pressure) near the NIBP value
    if not map_val and "NIBP" in paired:
        nibp_parts = paired["NIBP"]["value"].split("/")
        if len(nibp_parts) == 2:
            try:
                s, d      = int(nibp_parts[0]), int(nibp_parts[1])
                map_est   = round((s + 2 * d) / 3)   # standard MAP formula
                lo, hi    = map_est - 10, map_est + 10
                nibp_ctr = (paired["NIBP"].get("value_center") or
                            next((c for (t, c, f, b) in values_found
                                  if t == paired["NIBP"]["value"]), None))
                if nibp_ctr:
                    mc = [(t, c, f) for (t, c, f, b) in values_found
                          if t.isdigit() and lo <= int(t) <= hi
                          and _dist(nibp_ctr, c) < img_w * 0.25
                          and t != paired.get("SpO2", {}).get("value", "")]
                    if mc:
                        mc.sort(key=lambda v: _dist(nibp_ctr, v[1]))
                        map_val = mc[0][0]
            except ValueError:
                pass

    # ── NIBP-MAP cross-validation: correct common OCR digit errors ──────────────
    # If MAP was detected (from parentheses, highly reliable) but the NIBP formula
    # MAP ≠ detected MAP, try swapping single-digit OCR confusions in diastolic.
    if "NIBP" in paired and map_val:
        parts = paired["NIBP"]["value"].split("/")
        try:
            s, d  = int(parts[0]), int(parts[1])
            map_v = int(map_val.strip("() "))
            if abs(round((s + 2 * d) / 3) - map_v) > 5:
                for old, new in [("8", "7"), ("9", "4"), ("6", "0"), ("8", "3")]:
                    d_str = str(d)
                    if old in d_str:
                        d_try = int(d_str.replace(old, new, 1))
                        if 30 <= d_try <= 130 and d_try < s:
                            if abs(round((s + 2 * d_try) / 3) - map_v) <= 5:
                                logger.info(
                                    f"NIBP diastolic corrected {d}→{d_try} "
                                    f"(MAP cross-check: {map_v})"
                                )
                                paired["NIBP"]["value"] = f"{s}/{d_try}"
                                break
        except (ValueError, IndexError, ZeroDivisionError):
            pass

    # ── Formula MAP last resort ────────────────────────────────────────────────
    # If OCR never found the MAP value (no parens detected, no nearby digit),
    # derive it from the NIBP formula so the field is never left blank.
    if not map_val and "NIBP" in paired:
        parts = paired["NIBP"]["value"].split("/")
        if len(parts) == 2:
            try:
                s, d    = int(parts[0]), int(parts[1])
                map_val = str(round((s + 2 * d) / 3))
                logger.info(f"MAP derived from formula: ({s}+2×{d})/3 = {map_val}")
            except ValueError:
                pass

    result = {
        "HR"  : paired.get("HR",   {}).get("value", ""),
        "SpO2": paired.get("SpO2", {}).get("value", ""),
        "PR"  : paired.get("PR",   {}).get("value", ""),
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

        # Same CLAHE preprocessing as run_pipeline
        gray_dg  = cv2.cvtColor(img_dg, cv2.COLOR_BGR2GRAY)
        mean_b   = np.mean(gray_dg)
        gamma    = 1.5 if mean_b < 100 else (0.7 if mean_b > 150 else 1.0)
        lut      = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        gray_gam = cv2.LUT(cv2.cvtColor(img_dg, cv2.COLOR_BGR2GRAY), lut)
        clahe_f  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enh = clahe_f.apply(gray_gam)
        img_enh  = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)

        wm         = _compute_wave_mask(img_dg)
        ocr_color  = img_dg.copy();   ocr_color[wm > 0] = 0
        ocr_clahe  = img_enh.copy();  ocr_clahe[wm > 0] = 0

        raw = reader.readtext(ocr_color, detail=1) + reader.readtext(ocr_clahe, detail=1)
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
            "ocr_image_size": [ocr_color.shape[1], ocr_color.shape[0]],
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
