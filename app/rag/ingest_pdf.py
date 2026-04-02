from __future__ import annotations
from pathlib import Path
from time import perf_counter

from app.config import settings
from app.rag.image_caption import caption_extracted_images, caption_page_images
from app.rag.ocr import extract_ocr_from_page_images
from app.rag.page_images import extract_embedded_images, render_page_images
from app.rag.text_extraction import extract_page_text


def ingest_pdf(pdf_path: str) -> dict[str, int | str]:
	"""Run multimodal PDF ingestion and return summary metrics."""

	start = perf_counter()
	source = Path(pdf_path)

	if not source.exists():
		raise FileNotFoundError(f"The PDF was not found in the path: {source}")

	# 1) Extract classic page text.
	pages_processed, pages_with_text = _run_page_text_step(source=source)

	# 2) Extract embedded images and then caption them.
	extracted_images = _run_embedded_image_extraction_step(source=source)
	images_captioned = _run_embedded_image_caption_step(extracted_images=extracted_images)

	# 3) Render full-page images once for downstream OCR and page vision.
	_run_page_image_render_step(source=source)

	# 4) Generate page-level semantic vision output.
	pages_with_vision = _run_page_vision_step()

	# 5) Run local OCR on rendered full-page images.
	pages_with_ocr = _run_page_ocr_step()

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


def _run_embedded_image_extraction_step(source: Path) -> list[dict]:
	"""Extract embedded images and return item metadata for captioning."""

	if not settings.save_extracted_images:
		return []

	try:
		extraction = extract_embedded_images(
			pdf_path=str(source),
			images_output_dir="data/processed/images",
		)
		return extraction.get("images", [])
	
	except Exception as exc:
		print(f"Error during embedded image extraction: {exc}")
		return []


def _run_embedded_image_caption_step(extracted_images: list[dict]) -> int:
	"""Caption extracted embedded images and return caption count."""

	if not settings.save_extracted_images or not extracted_images:
		return 0

	try:
		caption_data = caption_extracted_images(
			extracted_images=extracted_images,
			captions_output_file="data/processed/image_captions.json",
		)
		return len(caption_data.get("captions", []))

	except Exception as exc:
		print(f"Error during image caption generation: {exc}")
		return 0
	

def _run_page_image_render_step(source: Path) -> None:
	"""Render page images once to be shared by vision and OCR."""

	if not settings.save_page_vision and not settings.save_page_ocr:
		return

	try:
		render_page_images(pdf_path=str(source), pages_output_dir="data/processed/page_images",)
		
	except Exception as exc:
		print(f"Error during page image rendering: {exc}")
		raise


def _run_page_vision_step() -> int:
	"""Run full-page vision analysis and return pages with non-empty vision text."""
	if not settings.save_page_vision:
		return 0

	try:
		page_vision_data = caption_page_images(
			pages_input_dir="data/processed/page_images",
			page_vision_output_file="data/processed/page_vision.json",
		)
		return sum(1 for item in page_vision_data.get("pages", []) if item.get("vision_text", "").strip())
	except Exception as exc:
		print(f"Error during page-level vision extraction: {exc}")
		return 0


def _run_page_ocr_step() -> int:
	"""Run local OCR for page images and return pages with non-empty OCR text."""
	if not settings.save_page_ocr:
		return 0

	try:
		ocr_data = extract_ocr_from_page_images(
			page_images_dir="data/processed/page_images",
			page_ocr_output_file="data/processed/page_ocr.json",
		)
		return sum(1 for item in ocr_data.get("pages", []) if item.get("ocr_text", "").strip())
	except Exception as exc:
		print(f"Error during page OCR extraction: {exc}")
		return 0
