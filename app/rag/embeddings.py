from __future__ import annotations

from app.config import settings
from app.services.openai_client import get_openai_client


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
	"""Generate embeddings for a list of texts using the configured OpenAI model."""
	if not texts:
		return []

	client = get_openai_client()
	embedding_model = model or settings.openai_embed_model

	response = client.embeddings.create(
		model=embedding_model,
		input=texts,
	)

	return [item.embedding for item in response.data]
