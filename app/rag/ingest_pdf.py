from __future__ import annotations
from pathlib import Path
from time import perf_counter

import fitz

from app.config import settings
from app.rag.image_caption import extract_and_caption_images, render_and_caption_pages
from app.rag.ocr import extract_ocr_from_page_images


def ingest_pdf(pdf_path: str, save_extracted_text: bool = True) -> dict[str, int | str]:
	"""
	Ingesta completa del PDF: texto, imágenes y captions multimodales.
	"""

	start = perf_counter()
	source = Path(pdf_path)

	if not source.exists():
		raise FileNotFoundError(f"No se encontró el PDF en la ruta: {source}")

	text_output_dir = Path("data/processed/text")
	if save_extracted_text:
		text_output_dir.mkdir(parents=True, exist_ok=True)

	pages_processed = 0
	images_extracted = 0
	chunks_created = 0

	# Paso 1: Extraer texto por página
	with fitz.open(source) as document:
		for page_index in range(len(document)):
			page = document[page_index]
			page_number = page_index + 1

			page_text = page.get_text("text").strip()
			image_count = len(page.get_images(full=True))

			pages_processed += 1
			images_extracted += image_count
			if page_text:
				chunks_created += 1

			if save_extracted_text:
				target = text_output_dir / f"page_{page_number:03d}.txt"
				target.write_text(page_text, encoding="utf-8")

	# Paso 2: Extraer imágenes internas y generar captions
	if settings.save_extracted_images:
		try:
			multimodal_data = extract_and_caption_images(pdf_path=str(source), images_output_dir="data/processed/images", captions_output_file="data/processed/image_captions.json",)
			
			# Contar chunks adicionales por imágenes con captions
			images_with_captions = len(multimodal_data["captions"])
			chunks_created += images_with_captions
		
		except Exception as exc:
			print(f"Error durante extracción multimodal: {exc}")

	# Paso 3: Renderizar páginas completas y extraer contenido visual
	if settings.save_page_vision:
		try:
			page_vision_data = render_and_caption_pages(
				pdf_path=str(source),
				pages_output_dir="data/processed/page_images",
				page_vision_output_file="data/processed/page_vision.json",
			)

			pages_with_vision = sum(1 for item in page_vision_data["pages"] if item.get("vision_text", "").strip())
			chunks_created += pages_with_vision
		except Exception as exc:
			print(f"Error durante extracción visual por página: {exc}")

	# Paso 4: OCR literal de páginas completas
	if settings.save_page_ocr:
		try:
			ocr_data = extract_ocr_from_page_images(
				page_images_dir="data/processed/page_images",
				page_ocr_output_file="data/processed/page_ocr.json",
			)

			pages_with_ocr = sum(1 for item in ocr_data["pages"] if item.get("ocr_text", "").strip())
			chunks_created += pages_with_ocr
		except Exception as exc:
			print(f"Error durante OCR por página: {exc}")

	duration_ms = int((perf_counter() - start) * 1000)

	return {
		"pdf_path": str(source),
		"pages_processed": pages_processed,
		"images_extracted": images_extracted,
		"chunks_created": chunks_created,
		"processing_time_ms": duration_ms,
	}
