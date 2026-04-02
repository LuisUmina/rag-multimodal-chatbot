from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.rag.chunking import build_multimodal_chunks
from app.rag.embeddings import embed_texts


DEFAULT_CAPTIONS_FILE = "data/processed/image_captions.json"
DEFAULT_TEXT_DIR = "data/processed/text"
DEFAULT_PAGE_VISION_FILE = "data/processed/page_vision.json"
DEFAULT_PAGE_OCR_FILE = "data/processed/page_ocr.json"
DEFAULT_INDEX_PATH = "data/index/faiss.index"
DEFAULT_METADATA_PATH = "data/index/metadata.json"


def build_faiss_index(
	chunks: list[dict],
	index_path: str = DEFAULT_INDEX_PATH,
	metadata_path: str = DEFAULT_METADATA_PATH,
	model: str | None = None,
) -> dict[str, str | int]:
	if not chunks:
		raise ValueError("No chunks available for indexing.")

	embedding_model = model or settings.openai_embed_model
	texts = [item["text"] for item in chunks]
	vectors = embed_texts(texts, model=embedding_model)
	if not vectors:
		raise ValueError("Embeddings could not be generated.")

	vectors_np = np.array(vectors, dtype="float32")
	dimension = vectors_np.shape[1]

	index = faiss.IndexFlatL2(dimension)
	index.add(vectors_np)

	index_file = Path(index_path)
	index_file.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(index_file))

	metadata_file = Path(metadata_path)
	metadata_file.parent.mkdir(parents=True, exist_ok=True)
	metadata_file.write_text(
		json.dumps(chunks, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	return {
		"chunks_indexed": len(chunks),
		"index_path": str(index_file),
		"metadata_path": str(metadata_file),
		"embedding_model": embedding_model,
	}


def index_processed_content() -> dict[str, str | int]:
	chunks = build_multimodal_chunks(
		captions_file=DEFAULT_CAPTIONS_FILE,
		text_dir=DEFAULT_TEXT_DIR,
		page_vision_file=DEFAULT_PAGE_VISION_FILE,
		page_ocr_file=DEFAULT_PAGE_OCR_FILE,
		chunk_size=settings.chunk_size,
		chunk_overlap=settings.chunk_overlap,
	)

	return build_faiss_index(
		chunks=chunks,
		index_path=DEFAULT_INDEX_PATH,
		metadata_path=DEFAULT_METADATA_PATH,
		model=settings.openai_embed_model,
	)
