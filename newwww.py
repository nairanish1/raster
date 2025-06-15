#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py – extract EVERY text string, then blank them on the
requested pages.  Writes sidecar_text.json with UTF-8 encoding so no
'cp1252' / 'charmap' errors can occur.

examples
--------
python pdf_mask_all.py --pdf "C:\\docs\\coordination.pdf" --pages all   --out build\\coord
python pdf_mask_all.py --pdf "C:\\docs\\coordination.pdf" --pages 0..3 --out build\\coord
"""

from __future__ import annotations
from pathlib import Path
import json, time, re, io, os, sys

import cv2, numpy as np, typer, easyocr
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ───────────────── 1. poppler (pdf2image backend) ─────────────────
POPPLER_PATH = (
    r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop"
    r"\poppler-24.08.0\Library\bin"
)
print("Using poppler from:", POPPLER_PATH)

# ───────────────── 2. force UTF-8 for console output ──────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

# ───────────────── 3. configuration ───────────────────────────────
DPI      = 300
PAD_VEC  = 5
PAD_RAS  = 8
LANGS    = ["en"]

reader = easyocr.Reader(LANGS, gpu=False, verbose=False)

# ═════════════════════ helpers ════════════════════════════════════
def pdf_page_count(pdf: Path) -> int:
    return pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)["Pages"]

def parse_pages(spec: str, total: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(total))
    if re.fullmatch(r"\d+(,\d+)+", spec):          # 0,2,7
        return [int(x) for x in spec.split(",")]
    m = re.fullmatch(r"(\d+)\.\.(\d+)", spec):     # 0..3
    if m:
        a, b = int(m[1]), int(m[2])
        return list(range(a, b + 1))
    return [int(spec)]

def vector_boxes(pdf: Path, idx: int) -> list[dict]:
    out : list[dict] = []
    for p_no, layout in enumerate(extract_pages(str(pdf))):
        if p_no != idx:
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

def raster_boxes_and_text(gray: np.ndarray) -> list[dict]:
    h, w = gray.shape
    out  = []
    for pts, txt, _ in reader.readtext(gray, detail=1, paragraph=False):
        xs = [int(p[0]) for p in pts]; ys = [int(p[1]) for p in pts]
        x1 = max(0, min(xs) - PAD_RAS);  y1 = max(0, min(ys) - PAD_RAS)
        x2 = min(w, max(xs) + PAD_RAS);  y2 = min(h, max(ys) + PAD_RAS)
        out.append({"bbox": [x1, y1, x2, y2], "text": txt.strip()})
    return out

def build_mask(shape, v_boxes, r_boxes) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in v_boxes + r_boxes:
        cv2.rectangle(mask,
                      (max(0, x1 - PAD_VEC), max(0, y1 - PAD_VEC)),
                      (min(shape[1], x2 + PAD_VEC), min(shape[0], y2 + PAD_VEC)),
                      255, -1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# ═════════════════════════ processing ═════════════════════════════
def process_page(pdf: Path, idx: int, root: Path) -> None:
    t0   = time.time()
    gray = raster_page(pdf, idx)
    vect = vector_boxes(pdf, idx)
    rast = raster_boxes_and_text(gray)

    notes = (
        [{"id": f"V{i:04d}", **v} for i, v in enumerate(vect)] +
        [{"id": f"R{j:04d}", **r} for j, r in enumerate(rast)]
    )

    page_dir = root / f"page{idx:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    # ←―― save UTF-8 directly (no cp-1252 surprises)
    with (page_dir / "sidecar_text.json").open("w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)

    geom = gray.copy()
    geom[build_mask(gray.shape,
                    [v["bbox"] for v in vect],
                    [r["bbox"] for r in rast]) > 0] = 255

    Image.fromarray(gray).save(page_dir / "page_raw.png")
    Image.fromarray(geom).save(page_dir / "geom_only.png")
    print(f"page {idx:03d}: {len(notes):4d} texts masked "
          f"[{time.time() - t0:.1f}s]")

# ───────────────────────── Typer CLI ──────────────────────────────
cli = typer.Typer()

@cli.command(no_args_is_help=True)
def main(
    pdf:   str = typer.Option(..., help="Path to PDF file"),
    pages: str = typer.Option("all", help="'all', '7', '0..3', '0,2,8'"),
    out:   str = typer.Option("build/coordination", help="Output root dir")
):
    pdf_path = Path(pdf).expanduser().resolve()
    ids      = parse_pages(pages, pdf_page_count(pdf_path))
    print(f"Processing pages {ids} of {pdf_path.name}")

    out_root = Path(out).expanduser()
    for idx in ids:
        process_page(pdf_path, idx, out_root)

# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli()


