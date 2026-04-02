from __future__ import annotations
from pathlib import Path
from time import perf_counter

import fitz

from app.config import settings
from app.rag.image_caption import extract_and_caption_images, render_and_caption_pages
from app.rag.ocr import extract_ocr_from_page_images
from app.rag.text_extraction import extract_page_text


def ingest_pdf(pdf_path: str) -> dict[str, int | str]:
	"""Run multimodal PDF ingestion and return summary metrics."""

	start = perf_counter()
	source = Path(pdf_path)

	if not source.exists():
		raise FileNotFoundError(f"The PDF was not found in the path: {source}")

	# 1) Extract classic page text.
	pages_processed, pages_with_text = _run_page_text_step(source=source)

	# 2) Extract embedded images and generate image captions.
	images_captioned = _run_image_caption_step(source=source)

	# 3) Generate page-level semantic vision output (also renders page images).
	pages_with_vision = _run_page_vision_step(source=source)

	# 4) Run local OCR on full-page images.
	pages_with_ocr = _run_page_ocr_step(source=source)

	chunks_created = pages_with_text + images_captioned + pages_with_vision + pages_with_ocr
	duration_ms = int((perf_counter() - start) * 1000)

	return {
		"pdf_path": str(source),
		"pages_processed": pages_processed,
		"images_extracted": images_captioned,
		"chunks_created": chunks_created,
		"processing_time_ms": duration_ms,
	}


def _run_page_text_step(source: Path) -> tuple[int, int]:
	"""Run page-text extraction."""
	try:
		return extract_page_text(source=source, save_extracted_text=settings.save_extracted_text)
	
	except Exception as exc:
		print(f"Error during page text extraction: {exc}")
		raise


def _run_image_caption_step(source: Path) -> int:
	"""Extract embedded images and return how many captions were produced."""
	
	if not settings.save_extracted_images:
		return 0

	try:
		multimodal_data = extract_and_caption_images(
			pdf_path=str(source),
			images_output_dir="data/processed/images",
			captions_output_file="data/processed/image_captions.json",
		)
	
		return len(multimodal_data.get("captions", []))
	
	except Exception as exc:
		print(f"Error during image caption extraction: {exc}")
		return 0


def _run_page_vision_step(source: Path) -> int:
	"""Run full-page vision analysis and return pages with non-empty vision text."""
	if not settings.save_page_vision:
		return 0

	try:
		page_vision_data = render_and_caption_pages(
			pdf_path=str(source),
			pages_output_dir="data/processed/page_images",
			page_vision_output_file="data/processed/page_vision.json",
		)
		return sum(1 for item in page_vision_data.get("pages", []) if item.get("vision_text", "").strip())
	except Exception as exc:
		print(f"Error during page-level vision extraction: {exc}")
		return 0


def _run_page_ocr_step(source: Path) -> int:
	"""Run local OCR for page images and return pages with non-empty OCR text."""
	if not settings.save_page_ocr:
		return 0

	# Ensure page images exist when OCR is enabled without page-vision step.
	if not settings.save_page_vision:
		_render_page_images(source, pages_output_dir="data/processed/page_images")

	try:
		ocr_data = extract_ocr_from_page_images(
			page_images_dir="data/processed/page_images",
			page_ocr_output_file="data/processed/page_ocr.json",
		)
		return sum(1 for item in ocr_data.get("pages", []) if item.get("ocr_text", "").strip())
	except Exception as exc:
		print(f"Error during page OCR extraction: {exc}")
		return 0


def _render_page_images(source: Path, pages_output_dir: str) -> None:
	"""Render full-page PNG images used by OCR and vision steps."""
	output_dir = Path(pages_output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	with fitz.open(source) as document:
		for page_index in range(len(document)):
			page = document[page_index]
			page_number = page_index + 1

			matrix = fitz.Matrix(2, 2)
			pix = page.get_pixmap(matrix=matrix, alpha=False)
			page_filename = f"page_{page_number:03d}.png"
			pix.save(str(output_dir / page_filename))
