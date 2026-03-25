from __future__ import annotations
import json
from pathlib import Path
import faiss
import numpy as np
from app.config import settings
from app.rag.chunking import build_multimodal_chunks
from app.rag.embeddings import embed_texts


def build_faiss_index(chunks: list[dict], index_path: str = "data/index/faiss.index", metadata_path: str = "data/index/metadata.json",) -> dict[str, str | int]:
	
	if not chunks:
		raise ValueError("No hay chunks para indexar.")

	texts = [item["text"] for item in chunks]
	vectors = embed_texts(texts)
	if not vectors:
		raise ValueError("No se pudieron generar embeddings.")

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
		"embedding_model": settings.openai_embed_model,
	}


def index_processed_content() -> dict[str, str | int]:
	chunks = build_multimodal_chunks(
		captions_file="data/processed/image_captions.json",
		text_dir="data/processed/text",
		chunk_size=settings.chunk_size,
		chunk_overlap=settings.chunk_overlap,
	)

	return build_faiss_index(
		chunks=chunks,
		index_path="data/index/faiss.index",
		metadata_path="data/index/metadata.json",
	)
