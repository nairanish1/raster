#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py – extract every string first, then mask all pages.

Usage examples
  python pdf_mask_all.py --pdf C:\docs\file.pdf --pages all --out build\outdir
  python pdf_mask_all.py --pdf file.pdf --pages 0..3 --out outdir
"""

# ───────────────── 1  force the entire process to speak UTF-8 ──────────────
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                              encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,
                              encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
# ───────────────────────────────────────────────────────────────────────────

from pathlib import Path
import json, time, re, typer, cv2, numpy as np, easyocr
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ---------- POPPLER --------------------------------------------------------
# ← CHANGE THIS to the *bin* folder that contains pdfinfo.exe & pdftoppm.exe
POPPLER_PATH = r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop\poppler-24.08.0\Library\bin"
print("Using poppler from:", POPPLER_PATH)
# --------------------------------------------------------------------------

# ---------- CONFIG ---------------------------------------------------------
DPI      = 300            # raster resolution
PAD_VEC  = 5              # expand vector-text boxes (px)
PAD_RAS  = 8              # expand raster-text boxes (px)
LANGS    = ["en"]         # OCR language(s) for EasyOCR
# --------------------------------------------------------------------------

reader = easyocr.Reader(LANGS, gpu=False, verbose=False)

# ═════════════════════════════════ HELPERS ════════════════════════════════
def pdf_page_count(pdf: Path) -> int:
    info = pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)
    return info["Pages"]

def parse_pages(spec: str, n_pages: int):
    """'all' | '5' | '0..3' | '0,2,7'  →  [list[int]]"""
    if spec.lower() == "all":
        return list(range(n_pages))
    if re.fullmatch(r"\d+(,\d+)+", spec):          # 0,2,7
        return [int(x) for x in spec.split(",")]
    m = re.fullmatch(r"(\d+)\.\.(\d+)", spec)      # 0..3
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return list(range(a, b + 1))
    return [int(spec)]                            # single page

def vector_boxes(pdf: Path, page_idx: int):
    """PDF vector glyphs → [{'bbox':[x1,y1,x2,y2], 'text':...}, ...]"""
    out = []
    for p_no, layout in enumerate(extract_pages(str(pdf))):
        if p_no != page_idx:
            continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1, y1, x2, y2 = map(int, obj.bbox)
                txt = obj.get_text().strip()
                if txt:
                    out.append({"bbox": [x1, y1, x2, y2], "text": txt})
    return out

def raster_page(pdf: Path, page_idx: int) -> np.ndarray:
    img = convert_from_path(str(pdf),
                            dpi=DPI,
                            first_page=page_idx + 1,
                            last_page=page_idx + 1,
                            fmt="png",
                            poppler_path=POPPLER_PATH)[0].convert("L")
    return np.array(img)

def raster_boxes_and_text(gray: np.ndarray):
    h, w = gray.shape
    res  = reader.readtext(gray, detail=1, paragraph=False)
    out  = []
    for pts, txt, conf in res:
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1 = max(0, min(xs) - PAD_RAS)
        y1 = max(0, min(ys) - PAD_RAS)
        x2 = min(w, max(xs) + PAD_RAS)
        y2 = min(h, max(ys) + PAD_RAS)
        out.append({"bbox": [x1, y1, x2, y2], "text": txt.strip()})
    return out

def build_mask(shape, v_boxes, r_boxes):
    mask = np.zeros(shape, np.uint8)
    h, w = shape
    for x1, y1, x2, y2 in v_boxes:
        cv2.rectangle(mask,
                      (max(0, x1 - PAD_VEC), max(0, y1 - PAD_VEC)),
                      (min(w, x2 + PAD_VEC), min(h, y2 + PAD_VEC)),
                      255, -1)
    for x1, y1, x2, y2 in r_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
# ══════════════════════════════════════════════════════════════════════════

def process_page(pdf: Path, idx: int, root: Path):
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

    # ---- write UTF-8 JSON (binary) ---------------------------------------
    page_dir.joinpath("sidecar_text.json").write_bytes(
        json.dumps(notes, indent=2, ensure_ascii=False).encode("utf-8")
    )
    # ----------------------------------------------------------------------

    mask = build_mask(gray.shape,
                      [v["bbox"] for v in vect],
                      [r["bbox"] for r in rast])
    geom = gray.copy(); geom[mask > 0] = 255

    Image.fromarray(gray).save(page_dir / "page_raw.png")
    Image.fromarray(geom).save(page_dir / "geom_only.png")

    print(f"page {idx:03d}: {len(notes):4d} texts masked "
          f"[{time.time() - t0:.1f}s]")

# ───────────────────────── CLI ────────────────────────────────────────────
cli = typer.Typer()

@cli.command(no_args_is_help=True)
def main(pdf: str = typer.Option(..., help="Path to PDF file"),
         pages: str = typer.Option("all",
                                   help="'all', '5', '0..3', '0,2,7'"),
         out: str = typer.Option("build/coordination",
                                 help="Output root dir")):
    pdf_path = Path(pdf).expanduser().resolve()
    total    = pdf_page_count(pdf_path)
    ids      = parse_pages(pages, total)
    print(f"Processing pages {ids} of {pdf_path.name}")
    out_root = Path(out).expanduser()
    for idx in ids:
        process_page(pdf_path, idx, out_root)

if __name__ == "__main__":
    cli()
