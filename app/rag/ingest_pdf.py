from __future__ import annotations
from pathlib import Path
from time import perf_counter
import fitz

from app.rag.image_caption import extract_and_caption_images


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

	# Paso 2: Extraer imágenes y generar captions
	try:
		multimodal_data = extract_and_caption_images(
			pdf_path=str(source),
			images_output_dir="data/processed/images",
			captions_output_file="data/processed/image_captions.json",
		)
		# Contar chunks adicionales por imágenes con captions
		images_with_captions = len(multimodal_data["captions"])
		chunks_created += images_with_captions
	except Exception as exc:
		print(f"Error durante extracción multimodal: {exc}")
		# Continuar sin captions si falla (ej: sin API key)

	duration_ms = int((perf_counter() - start) * 1000)

	return {
		"pdf_path": str(source),
		"pages_processed": pages_processed,
		"images_extracted": images_extracted,
		"chunks_created": chunks_created,
		"processing_time_ms": duration_ms,
	}
