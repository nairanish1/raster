#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py  –  extract *all* text (vector + raster) and mask it.

 ▸ JSON written in raw UTF-8  → never hits Windows code-page 1252
 ▸ enlarged boxes + dilation  → more complete masking
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


# ─────────────────── 1.  Poppler for Windows ─────────────────────
POPPLER_PATH = (
    r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop"
    r"\poppler-24.08.0\Library\bin"
)
if os.name == "nt":
    print("Using poppler from:", POPPLER_PATH)

# ─────────────────── 2.  UTF-8 console (optional) ────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────── 3.  Parameters & OCR ────────────────────────
DPI         = 300
PAD_VEC     = 8          # ← was 5 : vector-text padding
PAD_RAS     = 12         # ← was 8 : raster-text padding
DILATE_KRN  = np.ones((3, 3), np.uint8)   # 1-px dilation
CLOSE_KRN   = np.ones((5, 5), np.uint8)   # close tiny holes

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False,
    recog_network="english_g2"   # ▸ stronger net = better recall
)

# ═══════════════════ helpers ═════════════════════════════════════
def pdf_page_count(pdf: Path) -> int:
    return pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)["Pages"]

def parse_pages(spec: str, n_pages: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n_pages))
    if re.fullmatch(r"\d+(,\d+)+", spec):
        return [int(x) for x in spec.split(",")]
    m = re.fullmatch(r"(\d+)\.\.(\d+)", spec)
    return list(range(int(m[1]), int(m[2])+1)) if m else [int(spec)]

def vector_boxes(pdf: Path, p: int) -> list[dict]:
    out=[]
    for idx, layout in enumerate(extract_pages(str(pdf))):
        if idx != p:
            continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1,y1,x2,y2 = map(int, obj.bbox)
                txt = obj.get_text().strip()
                if txt:
                    out.append({"bbox":[x1,y1,x2,y2], "text":txt})
    return out

def raster_page(pdf: Path, p: int) -> np.ndarray:
    img = convert_from_path(
        str(pdf), dpi=DPI,
        first_page=p+1, last_page=p+1,
        fmt="png", poppler_path=POPPLER_PATH
    )[0].convert("L")
    return np.array(img)

def raster_boxes(gray: np.ndarray) -> list[dict]:
    h,w = gray.shape; out=[]
    for pts, txt, _ in reader.readtext(gray, detail=1, paragraph=False):
        xs=[int(p[0]) for p in pts]; ys=[int(p[1]) for p in pts]
        x1=max(0, min(xs)-PAD_RAS);  y1=max(0, min(ys)-PAD_RAS)
        x2=min(w, max(xs)+PAD_RAS);  y2=min(h, max(ys)+PAD_RAS)
        out.append({"bbox":[x1,y1,x2,y2], "text":txt.strip()})
    return out

def build_mask(shape, boxes) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    h, w = shape
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(
            m,
            (max(0, x1-PAD_VEC), max(0, y1-PAD_VEC)),
            (min(w, x2+PAD_VEC), min(h, y2+PAD_VEC)),
            255, -1
        )
    # one-pixel dilation joins neighbouring words & erases arrow heads
    m = cv2.dilate(m, DILATE_KRN, iterations=1)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, CLOSE_KRN)

# ═══════════════════ per-page work ═══════════════════════════════
def process(pdf: Path, idx: int, root: Path):
    t0 = time.time()
    gray = raster_page(pdf, idx)
    v    = vector_boxes(pdf, idx)
    r    = raster_boxes(gray)

    notes = (
        [{"id": f"V{i:04d}", **b} for i, b in enumerate(v)] +
        [{"id": f"R{i:04d}", **b} for i, b in enumerate(r)]
    )

    pg = root / f"page{idx:03d}"
    pg.mkdir(parents=True, exist_ok=True)
    pg.joinpath("sidecar_text.json").write_bytes(
        json.dumps(notes, indent=2, ensure_ascii=False).encode("utf-8")
    )

    mask = build_mask(gray.shape, [b["bbox"] for b in (*v, *r)])

    # optional – erode once more to be extra safe with arrow tips
    mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)

    geom = gray.copy()
    geom[mask > 0] = 255

    Image.fromarray(gray).save(pg / "page_raw.png")
    Image.fromarray(geom).save(pg / "geom_only.png")

    print(f"page {idx:03d}: {len(notes):4d} texts masked "
          f"[{time.time()-t0:.1f}s]")

# ─────────────────── Typer CLI ────────────────────────────────────
cli = typer.Typer()

@cli.command()
def main(pdf: str  = typer.Option(..., help="PDF file"),
         pages: str = typer.Option("all", help="'all', '0..3', '0,5,7' …"),
         out: str   = typer.Option("build/coordination", help="output dir")):
    p   = Path(pdf).expanduser().resolve()
    ids = parse_pages(pages, pdf_page_count(p))
    print(f"Processing pages {ids} of {p.name}")
    root = Path(out).expanduser()
    for idx in ids:
        process(p, idx, root)

# ─────────────────── main guard ───────────────────────────────────
if __name__ == "__main__":
    cli()





