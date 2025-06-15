#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_mask_all.py  –  extract every text string and then mask them on the
requested pages.  The side-car JSON is written in *binary* UTF-8, which
bypasses Windows code-page 1252 and eliminates UnicodeEncodeError.
"""

from __future__ import annotations
from pathlib import Path
import json, time, re, io, os, sys

import cv2, numpy as np, typer, easyocr
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ────────────────────────── 1. Poppler  ──────────────────────────
POPPLER_PATH = (
    r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop"
    r"\poppler-24.08.0\Library\bin"
)
print("Using poppler from:", POPPLER_PATH)

# ─────────────── 2. UTF-8 console (optional, but nice) ───────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────── 3. Parameters ───────────────────────────────────
DPI, PAD_VEC, PAD_RAS = 300, 5, 8
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

# ═══════════════ helpers ═════════════════════════════════════════
def pdf_pages(pdf: Path) -> int:
    return pdfinfo_from_path(str(pdf), poppler_path=POPPLER_PATH)["Pages"]

def page_list(spec: str, n: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n))
    if re.fullmatch(r"\d+(,\d+)+", spec):
        return [int(x) for x in spec.split(",")]
    rng = re.fullmatch(r"(\d+)\.\.(\d+)", spec)
    return list(range(int(rng[1]), int(rng[2])+1)) if rng else [int(spec)]

def vector_boxes(pdf: Path, i: int) -> list[dict]:
    out=[]
    for p, layout in enumerate(extract_pages(str(pdf))):
        if p!=i: continue
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                x1,y1,x2,y2 = map(int,obj.bbox)
                txt=obj.get_text().strip()
                if txt: out.append({"bbox":[x1,y1,x2,y2],"text":txt})
    return out

def raster_page(pdf: Path, i: int) -> np.ndarray:
    img = convert_from_path(str(pdf), dpi=DPI,
                            first_page=i+1, last_page=i+1,
                            fmt="png", poppler_path=POPPLER_PATH)[0].convert("L")
    return np.array(img)

def raster_boxes(gray: np.ndarray) -> list[dict]:
    h,w = gray.shape; out=[]
    for pts, txt, _ in reader.readtext(gray, detail=1, paragraph=False):
        xs=[int(p[0]) for p in pts]; ys=[int(p[1]) for p in pts]
        x1=max(0,min(xs)-PAD_RAS); y1=max(0,min(ys)-PAD_RAS)
        x2=min(w,max(xs)+PAD_RAS); y2=min(h,max(ys)+PAD_RAS)
        out.append({"bbox":[x1,y1,x2,y2],"text":txt.strip()})
    return out

def mask_from_boxes(shape, boxes):
    m=np.zeros(shape,np.uint8); h,w=shape
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(m,(max(0,x1-PAD_VEC),max(0,y1-PAD_VEC)),
                         (min(w,x2+PAD_VEC),min(h,y2+PAD_VEC)),255,-1)
    return cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

# ═══════════════ per-page work ═══════════════════════════════════
def process(pdf:Path, i:int, root:Path):
    t=time.time()
    gray=raster_page(pdf,i)
    v = vector_boxes(pdf,i)
    r = raster_boxes(gray)
    notes=[*({"id":f"V{n:04d}",**x} for n,x in enumerate(v)),
           *({"id":f"R{n:04d}",**x} for n,x in enumerate(r))]

    pg=root/f"page{i:03d}"; pg.mkdir(parents=True,exist_ok=True)

    # —— HERE: write raw UTF-8 bytes ——
    pg.joinpath("sidecar_text.json").write_bytes(
        json.dumps(notes, indent=2, ensure_ascii=False).encode("utf-8")
    )

    geom=gray.copy()
    geom[ mask_from_boxes(gray.shape,[b["bbox"] for b in (*v,*r)])>0 ] = 255
    Image.fromarray(gray).save(pg/"page_raw.png")
    Image.fromarray(geom).save(pg/"geom_only.png")

    print(f"page {i:03d}: {len(notes):4d} texts masked [{time.time()-t:.1f}s]")

# ─────────────── Typer CLI ───────────────────────────────────────
cli = typer.Typer()

@cli.command()
def main(pdf:str=typer.Option(...),
         pages:str=typer.Option("all"),
         out:str=typer.Option("build/coordination")):
    p=Path(pdf).expanduser().resolve()
    ids=page_list(pages, pdf_pages(p))
    print(f"Processing pages {ids} of {p.name}")
    root=Path(out).expanduser()
    for i in ids:
        process(p,i,root)

if __name__ == "__main__":
    cli()




