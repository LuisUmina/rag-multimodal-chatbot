from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from app.config import settings

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
except ImportError:
    pytesseract = None
    Image = None
    ImageFilter = None
    ImageOps = None


def extract_ocr_from_page_images(
    page_images_dir: str = "data/processed/page_images",
    page_ocr_output_file: str = "data/processed/page_ocr.json",
) -> dict[str, list]:
    """
    Extrae texto literal visible desde imágenes de página completa.
    """
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
            print(f"Error OCR en página {page_number}: {exc}")
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
        raise RuntimeError("Dependencias OCR no instaladas. Instala pytesseract y Pillow.")

    tesseract_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if not settings.tesseract_cmd and not shutil.which("tesseract"):
        raise RuntimeError(
            "No se encontró el ejecutable tesseract en PATH. "
            "Instala Tesseract OCR o define TESSERACT_CMD en .env."
        )
    if settings.tesseract_cmd and not Path(settings.tesseract_cmd).exists():
        raise RuntimeError(f"TESSERACT_CMD no existe: {settings.tesseract_cmd}")

    with Image.open(image_path) as image:
        prepared = _preprocess_for_ocr(image)
        raw_text = pytesseract.image_to_string(
            prepared,
            lang=settings.ocr_lang,
            config="--psm 6",
        )

    return _normalize_ocr_text(raw_text)


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    autocontrast = ImageOps.autocontrast(grayscale)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)
    return sharpened


def _normalize_ocr_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()
