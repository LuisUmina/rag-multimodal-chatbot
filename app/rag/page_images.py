from __future__ import annotations

from pathlib import Path

import fitz


def extract_embedded_images(pdf_path: str, images_output_dir: str = "data/processed/images",) -> dict[str, list]:
    """Extract embedded PDF images and return metadata for captioning."""
    
    output_dir = Path(images_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1

            for image_index, image_ref in enumerate(page.get_images(full=True), start=1):
                xref = image_ref[0]
                pix = fitz.Pixmap(document, xref)

                filename = f"page_{page_number:03d}_img_{image_index}.png"
                image_path = output_dir / filename
                _save_pixmap_as_png(pixmap=pix, output_path=image_path)

                items.append(
                    {
                        "page": page_number,
                        "image_index": image_index,
                        "filename": filename,
                        "path": str(image_path),
                    }
                )

    return {"images": items}


def render_page_images(pdf_path: str, pages_output_dir: str = "data/processed/page_images") -> dict[str, list]:
    
    """Render full PDF pages as PNG images and return page metadata."""
    output_dir = Path(pages_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1

            # Render at 2x to improve readability for OCR and vision.
            matrix = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            filename = f"page_{page_number:03d}.png"
            page_path = output_dir / filename
            pix.save(str(page_path))

            items.append(
                {
                    "page": page_number,
                    "filename": filename,
                    "path": str(page_path),
                }
            )

    return {"pages": items}


def _save_pixmap_as_png(pixmap: fitz.Pixmap, output_path: Path) -> None:
    """Save a pixmap as PNG, converting to RGB when required."""
    try:
        if pixmap.n - pixmap.alpha < 4:
            pixmap.save(str(output_path))
            return

        rgb_pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        try:
            rgb_pixmap.save(str(output_path))
        finally:
            rgb_pixmap = None
    finally:
        pixmap = None