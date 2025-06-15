#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py  –  extract *all* text (vector + raster), then mask it.

 ▸ JSON written as raw UTF-8  → bypasses Windows code-page 1252
 ▸ CLAHE + 400 dpi raster     → sharp tiny text
 ▸ EasyOCR (detector only) + optional PaddleOCR detector
 ▸ Tesseract for recognition
 ▸ dilate / close / erode     → thorough blanking incl. arrow-tips
"""

from __future__ import annotations
from pathlib import Path
import io, json, os, re, sys, time

import cv2
import numpy as np
import typer
import easyocr                        # detector only
import pytesseract                    # recogniser
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ───────────────── 1. Poppler (pdf2image) ────────────────────────
POPPLER_PATH = (r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop"
                r"\poppler-24.08.0\Library\bin")
if os.name == "nt":
    print("Using poppler from:", POPPLER_PATH)

# ───────────────── 2. UTF-8 console (optional) ───────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace")

# ───────────────── 3. Hyper-parameters (can be CLI-overridden) ───
DEFAULTS = dict(
    dpi        = 400,   # 300–500: higher ⇒ crisper glyphs
    pad_vec    = 10,    # grow PDF-text boxes
    pad_ras    = 14,    # grow raster-text boxes
    clahe_clip = 3.0,   # CLAHE contrast
)

DILATE_KRN = np.ones((3, 3), np.uint8)
CLOSE_KRN  = np.ones((5, 5), np.uint8)
ERODE_KRN  = np.ones((3, 3), np.uint8)

# ───────────────── 4. OCR models ─────────────────────────────────
# ◆ EasyOCR detector-only (recognizer disabled)
eocr = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False,
    detector=True,
    recognizer=False
)

# optional PaddleOCR detector fallback
try:
    from paddleocr import PaddleOCR
    HAVE_PADDLE = True
except ImportError:
    HAVE_PADDLE = False
pocr = None               # built lazily if --paddle supplied


# ═════════════════════ helpers ═══════════════════════════════════
def pdf_page_count(pdf: Path) -> int:
    return pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)["Pages"]

def parse_pages(spec: str, n_pages: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n_pages))
    if re.fullmatch(r"\d+(,\d+)+", spec):
        return [int(x) for x in spec.split(",")]
    m = re.fullmatch(r"(\d+)\.\.(\d+)", spec)
    return list(range(int(m[1]), int(m[2]) + 1)) if m else [int(spec)]

def vector_boxes(pdf: Path, page: int) -> list[dict]:
    out = []
    for idx, layout in enumerate(extract_pages(str(pdf))):
        if idx != page:
            continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1, y1, x2, y2 = map(int, obj.bbox)
                txt = obj.get_text().strip()
                if txt:
                    out.append({"bbox": [x1, y1, x2, y2], "text": txt})
    return out

def raster_page(pdf: Path, page: int, dpi: int) -> np.ndarray:
    img = convert_from_path(
        str(pdf), dpi=dpi,
        first_page=page + 1, last_page=page + 1,
        fmt="png", poppler_path=POPPLER_PATH
    )[0].convert("L")
    clahe = cv2.createCLAHE(
        clipLimit=DEFAULTS["clahe_clip"], tileGridSize=(8, 8)
    )
    return clahe.apply(np.array(img))

def detector_boxes(gray: np.ndarray, pad_ras: int,
                   use_paddle: bool) -> list[list[int]]:
    h, w = gray.shape
    out: list[list[int]] = []

    # 1️⃣ EasyOCR detector
    for pts, _ in eocr.detect(gray, width_ths=0.5)[0]:
        xs = [int(p[0]) for p in pts];  ys = [int(p[1]) for p in pts]
        x1 = max(0, min(xs) - pad_ras);  y1 = max(0, min(ys) - pad_ras)
        x2 = min(w, max(xs) + pad_ras);  y2 = min(h, max(ys) + pad_ras)
        out.append([x1, y1, x2, y2])

    # 2️⃣ optional PaddleOCR detector fallback
    if use_paddle and HAVE_PADDLE:
        global pocr
        if pocr is None:                         # build once
            pocr = PaddleOCR(use_angle_cls=False, show_log=False)
        for line in pocr.ocr(gray, cls=False, rec=False):
            for pts in line:
                xs = [int(p[0]) for p in pts];  ys = [int(p[1]) for p in pts]
                x1 = max(0, min(xs) - pad_ras);  y1 = max(0, min(ys) - pad_ras)
                x2 = min(w, max(xs) + pad_ras);  y2 = min(h, max(ys) + pad_ras)
                out.append([x1, y1, x2, y2])
    return out

def recognise(gray: np.ndarray, boxes: list[list[int]]) -> list[str]:
    out = []
    for x1, y1, x2, y2 in boxes:
        roi = gray[y1:y2, x1:x2]
        out.append(
            pytesseract.image_to_string(roi, config="--psm 6").strip()
        )
    return out

def build_mask(shape, boxes, pad_vec: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    h, w = shape
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            m,
            (max(0, x1 - pad_vec), max(0, y1 - pad_vec)),
            (min(w, x2 + pad_vec), min(h, y2 + pad_vec)),
            255, -1
        )
    m = cv2.dilate(m, DILATE_KRN, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, CLOSE_KRN)
    m = cv2.erode(m, ERODE_KRN, 1)      # arrow tips, straggling pixels
    return m

# ═══════════════════ per-page work ═══════════════════════════════
def process(pdf: Path, idx: int, root: Path,
            dpi: int, pad_vec: int, pad_ras: int, use_paddle: bool):
    t0 = time.time()
    gray = raster_page(pdf, idx, dpi)
    v    = vector_boxes(pdf, idx)
    r_bb = detector_boxes(gray, pad_ras, use_paddle)
    r_tx = recognise(gray, r_bb)

    r = [{"bbox": bb, "text": tx} for bb, tx in zip(r_bb, r_tx)]
    notes = (
        [{"id": f"V{i:04d}", **b} for i, b in enumerate(v)] +
        [{"id": f"R{i:04d}", **b} for i, b in enumerate(r)]
    )

    pg = root / f"page{idx:03d}"
    pg.mkdir(parents=True, exist_ok=True)
    pg.joinpath("sidecar_text.json").write_bytes(
        json.dumps(notes, indent=2, ensure_ascii=False).encode("utf-8")
    )

    geom = gray.copy()
    geom[build_mask(gray.shape,
                    [b["bbox"] for b in (*v, *r)],
                    pad_vec) > 0] = 255

    Image.fromarray(gray).save(pg / "page_raw.png")
    Image.fromarray(geom).save(pg / "geom_only.png")

    print(f"page {idx:03d}: {len(notes):4d} texts masked "
          f"[{time.time() - t0:.1f}s]")

# ─────────────────── Typer CLI ────────────────────────────────────
cli = typer.Typer()

@cli.command()
def main(pdf: str  = typer.Option(..., help="PDF file"),
         pages: str = typer.Option("all", help="'all', '0..3', '0,2,7'"),
         out: str   = typer.Option("build/coordination", help="output dir"),
         dpi: int   = typer.Option(DEFAULTS["dpi"], help="raster DPI ≥ 300"),
         pad_vec: int=typer.Option(DEFAULTS["pad_vec"], help="vector pad px"),
         pad_ras: int=typer.Option(DEFAULTS["pad_ras"], help="raster pad px"),
         paddle: bool = typer.Option(False, help="add PaddleOCR detector")):
    p   = Path(pdf).expanduser().resolve()
    ids = parse_pages(pages, pdf_page_count(p))
    print(f"Processing pages {ids} of {p.name}")
    root = Path(out).expanduser()
    for idx in ids:
        process(p, idx, root, dpi, pad_vec, pad_ras, paddle)

# ─────────────────── main guard ───────────────────────────────────
if __name__ == "__main__":
    cli()



