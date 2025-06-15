#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py  – extract EVERY text string, then mask them on the
requested pages.  Side-car JSON is written in *binary* UTF-8 so Windows
code-page 1252 never bites us.

examples
--------
python pdf_mask_all.py --pdf "C:\docs\coordination.pdf" --pages all   --out build\coord
python pdf_mask_all.py --pdf "C:\docs\coordination.pdf" --pages 0..3  --out build\coord
"""

from __future__ import annotations
from pathlib import Path
import json, time, re, io, os, sys

import cv2, numpy as np, typer, easyocr
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ───────────────────── 1. Poppler (pdf2image) ─────────────────────
POPPLER_PATH = (
    r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop"
    r"\poppler-24.08.0\Library\bin"
)
print("Using poppler from:", POPPLER_PATH)

# ───────────────────── 2. UTF-8 console (nice) ────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ───────────────────── 3. Parameters & OCR ────────────────────────
DPI           = 300
PAD_VEC       = 8           # ← larger than before
PAD_RAS       = 12
CLOSE_KERNEL  = np.ones((5, 5), np.uint8)

# stronger EasyOCR network (better tiny-text recall)
reader = easyocr.Reader(
    ["en"], gpu=False, verbose=False,
    recog_network="english_g2",
    detect_batch_size=1, recog_batch_size=1
)

# ═════════════════════ helpers ════════════════════════════════════
def pdf_pages(pdf: Path) -> int:
    return pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)["Pages"]

def page_list(spec: str, n: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n))
    if re.fullmatch(r"\d+(,\d+)+", spec):
        return [int(x) for x in spec.split(",")]
    rng = re.fullmatch(r"(\d+)\.\.(\d+)", spec)
    return list(range(int(rng[1]), int(rng[2])+1)) if rng else [int(spec)]

def vector_boxes(pdf: Path, idx: int) -> list[dict]:
    out = []
    for p, layout in enumerate(extract_pages(str(pdf))):
        if p != idx:                # only the page we want
            continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1, y1, x2, y2 = map(int, obj.bbox)
                txt = obj.get_text().strip()
                if txt:
                    out.append({"bbox": [x1, y1, x2, y2], "text": txt})
    return out

def raster_page(pdf: Path, idx: int) -> np.ndarray:
    img = convert_from_path(
        str(pdf), dpi=DPI,
        first_page=idx + 1, last_page=idx + 1,
        fmt="png", poppler_path=POPPLER_PATH
    )[0].convert("L")
    return np.array(img)

def raster_boxes(gray: np.ndarray) -> list[dict]:
    h, w = gray.shape; out = []
    for pts, txt, _ in reader.readtext(gray, detail=1, paragraph=False):
        xs = [int(p[0]) for p in pts];  ys = [int(p[1]) for p in pts]
        x1 = max(0, min(xs) - PAD_RAS);  y1 = max(0, min(ys) - PAD_RAS)
        x2 = min(w, max(xs) + PAD_RAS);  y2 = min(h, max(ys) + PAD_RAS)
        out.append({"bbox": [x1, y1, x2, y2], "text": txt.strip()})
    return out

def mask_from_boxes(shape, boxes):
    m = np.zeros(shape, np.uint8);  h, w = shape
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            m,
            (max(0, x1 - PAD_VEC), max(0, y1 - PAD_VEC)),
            (min(w, x2 + PAD_VEC), min(h, y2 + PAD_VEC)),
            255, -1
        )
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, CLOSE_KERNEL)

# ════════════════════ per-page work ═══════════════════════════════
def process(pdf: Path, idx: int, root: Path):
    t0   = time.time()
    gray = raster_page(pdf, idx)
    v    = vector_boxes(pdf, idx)
    r    = raster_boxes(gray)

    notes = (
        [{"id": f"V{i:04d}", **x} for i, x in enumerate(v)] +
        [{"id": f"R{i:04d}", **x} for i, x in enumerate(r)]
    )

    pg = root / f"page{idx:03d}"
    pg.mkdir(parents=True, exist_ok=True)

    # — write raw UTF-8 bytes (never hits cp-1252) —
    pg.joinpath("sidecar_text.json").write_bytes(
        json.dumps(notes, indent=2, ensure_ascii=False).encode("utf-8")
    )

    full_mask = mask_from_boxes(gray.shape, [b["bbox"] for b in (*v, *r)])

    # optional arrow-head removal —— uncomment to KEEP heads instead
    arrow_mask = cv2.erode(full_mask, np.ones((3, 3), np.uint8), 2)
    geom = gray.copy()
    geom[np.logical_or(full_mask > 0, arrow_mask > 0)] = 255
    # geom[np.logical_and(full_mask > 0, arrow_mask == 0)] = 255  # keep heads

    Image.fromarray(gray).save(pg / "page_raw.png")
    Image.fromarray(geom).save(pg / "geom_only.png")
    print(f"page {idx:03d}: {len(notes):4d} texts masked "
          f"[{time.time() - t0:.1f}s]")

# ───────────────────── Typer CLI ──────────────────────────────────
cli = typer.Typer()

@cli.command()
def main(pdf: str  = typer.Option(..., help="PDF file"),
         pages: str = typer.Option("all", help="'all', '0..3', '0,2,7'"),
         out: str   = typer.Option("build/coordination", help="output dir")):
    p = Path(pdf).expanduser().resolve()
    ids = page_list(pages, pdf_pages(p))
    print(f"Processing pages {ids} of {p.name}")
    root = Path(out).expanduser()
    for i in ids:
        process(p, i, root)

if __name__ == "__main__":
    cli()




