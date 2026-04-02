from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from app.config import settings

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
except ImportError:
    pytesseract = None
    Image = None
    ImageFilter = None
    ImageOps = None


def extract_ocr_from_page_images(page_images_dir: str = "data/processed/page_images", page_ocr_output_file: str = "data/processed/page_ocr.json",) -> dict[str, list]:
    """Extract literal text from rendered full-page images."""
    
    image_dir = Path(page_images_dir)
    items: list[dict] = []

    if not image_dir.exists():
        return {"pages": items}

    _configure_tesseract_binary()

    for image_path in sorted(image_dir.glob("page_*.png")):
        page_number = int(image_path.stem.split("_")[-1])

        try:
            ocr_text = _extract_text_with_local_ocr(str(image_path))
        except Exception as exc:
            print(f"OCR error on page {page_number}: {exc}")
            ocr_text = ""

        items.append(
            {
                "page": page_number,
                "filename": image_path.name,
                "path": str(image_path),
                "ocr_text": ocr_text,
            }
        )

    output_path = Path(page_ocr_output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"pages": items}


def _configure_tesseract_binary() -> None:
    if not pytesseract:
        return

    configured_cmd = settings.tesseract_cmd.strip()
    if configured_cmd:
        pytesseract.pytesseract.tesseract_cmd = configured_cmd


def _extract_text_with_local_ocr(image_path: str) -> str:
    if not pytesseract or not Image:
        raise RuntimeError("OCR dependencies are missing. Install pytesseract and Pillow.")

    # Validate where Tesseract binary should be loaded from.
    if settings.tesseract_cmd:
        if not Path(settings.tesseract_cmd).exists():
            raise RuntimeError(f"TESSERACT_CMD does not exist: {settings.tesseract_cmd}")
    elif not shutil.which("tesseract"):
        raise RuntimeError(
            "Tesseract executable was not found in PATH. "
            "Install Tesseract OCR or set TESSERACT_CMD in .env."
        )

    assert Image is not None  # Narrow type for static checkers.

    with Image.open(image_path) as image:
        prepared = _preprocess_for_ocr(image)
        raw_text = pytesseract.image_to_string(
            prepared,
            lang=settings.ocr_lang,
            config="--psm 6",
        )

    return _normalize_ocr_text(raw_text)


def _preprocess_for_ocr(image: Any) -> Any:
    """Apply a lightweight preprocessing chain to improve OCR readability."""
    grayscale = ImageOps.grayscale(image)
    autocontrast = ImageOps.autocontrast(grayscale)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)
    return sharpened


def _normalize_ocr_text(text: str) -> str:
    """Normalize OCR output by collapsing extra whitespace and blank lines."""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()
