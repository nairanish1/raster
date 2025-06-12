#!/usr/bin/env python3
"""
data.py  –  extract every string first, then mask all pages.

* pages arg:  "all"  |  "N"  |  "start..end"  |  "0,3,7"
"""

from pathlib import Path
import json, time, cv2, numpy as np, re, typer, easyocr
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ───── POPPLER PATH ──────────────────────────────────────────────────────
# Change this to wherever you unzipped poppler-24.08.0-0\Library\bin
POPPLER_PATH = r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop\poppler-24.08.0\Library\bin"
# ────────────────────────────────────────────────────────────────────────

# ───── CONFIG ───────────────────────────────────────────────────────────
DPI      = 300
PAD_VEC  = 5
PAD_RAS  = 8
LANGS    = ['en']
# ────────────────────────────────────────────────────────────────────────

reader = easyocr.Reader(LANGS, gpu=False, verbose=False)


# ──────────────── helpers ──────────────────────────────────────────────
def pdf_page_count(pdf: Path) -> int:
    info = pdfinfo_from_path(
        str(pdf),
        poppler_path=POPPLER_PATH
    )
    return info['Pages']


def parse_pages(spec: str, n_pages: int):
    if spec.lower() == "all":
        return list(range(n_pages))
    if re.fullmatch(r"\d+(,\d+)+", spec):          # comma list
        return [int(x) for x in spec.split(',')]
    m = re.fullmatch(r"(\d+)\.\.(\d+)", spec)      # range a..b
        # e.g. "2..5"
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return list(range(a, b + 1))
    return [int(spec)]                             # single number


def vector_boxes(pdf: Path, p: int):
    out = []
    for p_no, layout in enumerate(extract_pages(str(pdf))):
        if p_no != p:
            continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1, y1, x2, y2 = map(int, obj.bbox)
                txt = obj.get_text().strip()
                if txt:
                    out.append({"bbox": [x1, y1, x2, y2], "text": txt})
    return out


def raster_page(pdf: Path, p: int):
    img = convert_from_path(
        str(pdf),
        dpi=DPI,
        first_page=p + 1, last_page=p + 1,
        fmt="png",
        poppler_path=POPPLER_PATH
    )[0].convert("L")
    return np.array(img)


def raster_boxes_and_text(gray: np.ndarray):
    h, w = gray.shape
    res = reader.readtext(gray, detail=1, paragraph=False)
    out = []
    for (pts, txt, conf) in res:
        xs = [int(pt[0]) for pt in pts]
        ys = [int(pt[1]) for pt in pts]
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
        cv2.rectangle(
            mask,
            (max(0, x1 - PAD_VEC), max(0, y1 - PAD_VEC)),
            (min(w, x2 + PAD_VEC), min(h, y2 + PAD_VEC)),
            255, -1
        )
    for x1, y1, x2, y2 in r_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# ────────────────────────────────────────────────────────────────────────


def process_page(pdf: Path, p: int, root: Path):
    t0 = time.time()
    gray = raster_page(pdf, p)
    vect = vector_boxes(pdf, p)
    rast = raster_boxes_and_text(gray)

    notes = (
        [{"id": f"V{i:04d}", **v} for i, v in enumerate(vect)] +
        [{"id": f"R{j:04d}", **r} for j, r in enumerate(rast)]
    )

    page_dir = root / f"page{p:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        notes,
        (page_dir / "sidecar_text.json").open("w"),
        indent=2,
        ensure_ascii=False
    )

    mask = build_mask(
        gray.shape,
        [v["bbox"] for v in vect],
        [r["bbox"] for r in rast]
    )
    geom = gray.copy()
    geom[mask > 0] = 255

    Image.fromarray(gray).save(page_dir / "page_raw.png")
    Image.fromarray(geom).save(page_dir / "geom_only.png")

    print(f"page {p:03d}: {len(notes):3d} texts masked  "
          f"[{time.time() - t0:.1f}s]")


# ─────────── CLI ───────────────────────────────────────────────────────
cli = typer.Typer()

@cli.command()
def run(
    pdf:   str = typer.Option(..., help="PDF file path"),
    pages: str = typer.Option("all", help="'all', '0', '2..5', '0,3,7'"),
    out:   str = typer.Option("build/coordination", help="output root")
):
    pdf_path = Path(pdf).expanduser().resolve()
    total    = pdf_page_count(pdf_path)
    page_ids = parse_pages(pages, total)
    print(f"Processing pages {page_ids} of {pdf_path.name}")
    out_root = Path(out).expanduser()
    for p in page_ids:
        process_page(pdf_path, p, out_root)

if __name__ == "__main__":
    cli()
