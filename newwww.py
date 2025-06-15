#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py – extract *all* text (vector + raster) and blank it
on every requested page.

Typical use
-----------
python pdf_mask_all.py --pdf "C:\\docs\\coordination.pdf" --pages all --out build\\coord
python pdf_mask_all.py --pdf coordination.pdf            --pages 0..3 --out build/coord
"""

from __future__ import annotations
from pathlib import Path
import io, json, os, re, sys, time

import cv2
import numpy as np
import typer
import easyocr
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


# ─────────────────── 1.  Poppler (needed only on Windows) ──────────────────
POPPLER_PATH = (
    r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop"
    r"\poppler-24.08.0\Library\bin"
)
if os.name == "nt":   # Windows
    print("Using poppler from:", POPPLER_PATH)


# ─────────────────── 2.  UTF-8 console (nice but optional) ─────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                              encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,
                              encoding="utf-8", errors="replace")


# ─────────────────── 3.  Parameters & OCR engine  ──────────────────────────
DPI          = 300        # raster resolution
PAD_VEC      = 8          # grow vector-text boxes
PAD_RAS      = 12         # grow raster-text boxes
CLOSE_KERNEL = np.ones((5, 5), np.uint8)

#  ▸ stronger detection model → better tiny-text recall
reader = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False,
    recog_network="english_g2"           # <= *only* supported kwarg set
)


# ═════════════════════ helpers ═════════════════════════════════════════════
def pdf_page_count(pdf: Path) -> int:
    return pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)["Pages"]


def parse_pages(spec: str, total: int) -> list[int]:
    """
    'all' | '7' | '0..3' | '0,2,8'  →  list[int]
    """
    if spec.lower() == "all":
        return list(range(total))
    if re.fullmatch(r"\d+(,\d+)+", spec):
        return [int(x) for x in spec.split(",")]
    rng = re.fullmatch(r"(\d+)\.\.(\d+)", spec)
    if rng:
        a, b = map(int, rng.groups())
        return list(range(a, b + 1))
    return [int(spec)]


def vector_boxes(pdf: Path, page: int) -> list[dict]:
    """text that is *already* vectors inside the PDF"""
    out: list[dict] = []
    for p_no, layout in enumerate(extract_pages(str(pdf))):
        if p_no != page:
            continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1, y1, x2, y2 = map(int, obj.bbox)
                txt = obj.get_text().strip()
                if txt:
                    out.append({"bbox": [x1, y1, x2, y2], "text": txt})
    return out


def raster_page(pdf: Path, page: int) -> np.ndarray:
    """300-dpi grayscale numpy array of one page"""
    img = convert_from_path(
        str(pdf),
        dpi=DPI,
        first_page=page + 1,
        last_page=page + 1,
        fmt="png",
        poppler_path=POPPLER_PATH,
    )[0].convert("L")
    return np.array(img)


def raster_boxes(gray: np.ndarray) -> list[dict]:
    """OCR every raster glyph (EasyOCR)"""
    h, w = gray.shape
    out: list[dict] = []
    for pts, txt, _ in reader.readtext(gray, detail=1, paragraph=False):
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1 = max(0, min(xs) - PAD_RAS)
        y1 = max(0, min(ys) - PAD_RAS)
        x2 = min(w, max(xs) + PAD_RAS)
        y2 = min(h, max(ys) + PAD_RAS)
        out.append({"bbox": [x1, y1, x2, y2], "text": txt.strip()})
    return out


def mask_from_boxes(shape: tuple[int, int], boxes: list[list[int]]) -> np.ndarray:
    """return a binary mask (uint8) for all given boxes"""
    mask = np.zeros(shape, np.uint8)
    h, w = shape
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            mask,
            (max(0, x1 - PAD_VEC), max(0, y1 - PAD_VEC)),
            (min(w, x2 + PAD_VEC), min(h, y2 + PAD_VEC)),
            255,
            -1,
        )
    # close tiny holes inside glyphs so arrow shafts get filled too
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)


# ═════════════════════ per-page work ══════════════════════════════
def process_page(pdf: Path, idx: int, out_root: Path) -> None:
    start = time.time()

    gray  = raster_page(pdf, idx)
    vect  = vector_boxes(pdf, idx)
    rast  = raster_boxes(gray)
    notes = (
        [{"id": f"V{i:04d}", **b} for i, b in enumerate(vect)] +
        [{"id": f"R{i:04d}", **b} for i, b in enumerate(rast)]
    )

    page_dir = out_root / f"page{idx:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    # --- side-car JSON (raw UTF-8, so Windows code page is never used) ----
    page_dir.joinpath("sidecar_text.json").write_bytes(
        json.dumps(notes, indent=2, ensure_ascii=False).encode("utf-8")
    )

    # build one big mask and (optionally) erode once more to catch arrowheads
    full_mask = mask_from_boxes(gray.shape, [b["bbox"] for b in (*vect, *rast)])
    arrow_fix = cv2.erode(full_mask, np.ones((3, 3), np.uint8), iterations=1)
    geom = gray.copy()
    geom[np.logical_or(full_mask > 0, arrow_fix > 0)] = 255  # blank them all

    Image.fromarray(gray).save(page_dir / "page_raw.png")
    Image.fromarray(geom).save(page_dir / "geom_only.png")

    print(f"page {idx:03d}: {len(notes):4d} texts masked "
          f"[{time.time() - start:.1f}s]")


# ───────────────────── Typer CLI ───────────────────────────────────────────
cli = typer.Typer()

@cli.command()
def main(
    pdf:   str = typer.Option(..., help="PDF file"),
    pages: str = typer.Option("all", help="'all', '0..3', '0,2,7', …"),
    out:   str = typer.Option("build/coordination", help="output dir"),
):
    pdf_path = Path(pdf).expanduser().resolve()
    ids      = parse_pages(pages, pdf_page_count(pdf_path := pdf_path))
    print(f"Processing pages {ids} of {pdf_path.name}")

    out_root = Path(out).expanduser()
    for idx in ids:
        process_page(pdf_path, idx, out_root)


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli()




