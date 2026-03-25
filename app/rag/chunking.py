from __future__ import annotations
import json
from pathlib import Path


def _is_relevant_caption(caption: str) -> bool:
	text = caption.lower()

	noisy_patterns = [
		"retrato",
		"hombre sonriendo",
		"fondo negro",
		"línea horizontal simple",
		"elemento decorativo",
	]
	return not any(pattern in text for pattern in noisy_patterns)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
	cleaned = " ".join(text.split())
	if not cleaned:
		return []

	if len(cleaned) <= chunk_size:
		return [cleaned]

	chunks: list[str] = []
	step = max(1, chunk_size - chunk_overlap)
	for start in range(0, len(cleaned), step):
		chunk = cleaned[start : start + chunk_size]
		if chunk:
			chunks.append(chunk)
		if start + chunk_size >= len(cleaned):
			break
	return chunks


def build_multimodal_chunks(captions_file: str = "data/processed/image_captions.json", text_dir: str = "data/processed/text", page_vision_file: str = "data/processed/page_vision.json", chunk_size: int = 700, chunk_overlap: int = 150,) -> list[dict]:
	chunks: list[dict] = []
	chunk_id = 1

	captions_path = Path(captions_file)
	if captions_path.exists():
		caption_items = json.loads(captions_path.read_text(encoding="utf-8"))
		for item in caption_items:
			caption = item.get("caption", "")
			if not caption or not _is_relevant_caption(caption):
				continue

			for part in _chunk_text(caption, chunk_size, chunk_overlap):
				chunks.append(
					{
						"id": f"chunk_{chunk_id:05d}",
						"text": part,
						"metadata": {
							"source_type": "image_caption",
							"page": item.get("page"),
							"image_index": item.get("image_index"),
							"filename": item.get("filename"),
						},
					}
				)
				chunk_id += 1

	text_path = Path(text_dir)
	if text_path.exists():
		for file_path in sorted(text_path.glob("page_*.txt")):
			content = file_path.read_text(encoding="utf-8").strip()
			if not content:
				continue

			page = int(file_path.stem.split("_")[-1])
			for part in _chunk_text(content, chunk_size, chunk_overlap):
				chunks.append(
					{
						"id": f"chunk_{chunk_id:05d}",
						"text": part,
						"metadata": {
							"source_type": "page_text",
							"page": page,
							"filename": file_path.name,
						},
					}
				)
				chunk_id += 1

	page_vision_path = Path(page_vision_file)
	if page_vision_path.exists():
		page_items = json.loads(page_vision_path.read_text(encoding="utf-8"))
		for item in page_items:
			vision_text = item.get("vision_text", "").strip()
			if not vision_text:
				continue

			for part in _chunk_text(vision_text, chunk_size, chunk_overlap):
				chunks.append(
					{
						"id": f"chunk_{chunk_id:05d}",
						"text": part,
						"metadata": {
							"source_type": "page_vision",
							"page": item.get("page"),
							"filename": item.get("filename"),
						},
					}
				)
				chunk_id += 1

	page_ocr_path = Path("data/processed/page_ocr.json")
	if page_ocr_path.exists():
		page_items = json.loads(page_ocr_path.read_text(encoding="utf-8"))
		for item in page_items:
			ocr_text = item.get("ocr_text", "").strip()
			if not ocr_text:
				continue

			for part in _chunk_text(ocr_text, chunk_size, chunk_overlap):
				chunks.append(
					{
						"id": f"chunk_{chunk_id:05d}",
						"text": part,
						"metadata": {
							"source_type": "page_ocr",
							"page": item.get("page"),
							"filename": item.get("filename"),
						},
					}
				)
				chunk_id += 1

	return chunks
