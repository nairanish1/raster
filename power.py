def pdf_page_count(pdf: Path) -> int:
    info = pdfinfo_from_path(
        str(pdf),
        poppler_path=POPPLER_PATH        
    )
    return info['Pages']



def raster_page(pdf: Path, p: int):
    img = convert_from_path(
        str(pdf),
        dpi=DPI,
        first_page=p + 1,
        last_page=p + 1,
        fmt="png",
        poppler_path=POPPLER_PATH        
    )[0].convert("L")
    return np.array(img)
