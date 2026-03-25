from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    pdf_path: str = Field(default="rag-challenge.pdf", description="Ruta del PDF a procesar",)
    save_extracted_text: bool = Field(default=True,description="Guardar texto extraído por página en data/processed/text",)


class IngestResponse(BaseModel):
    pdf_path: str
    pages_processed: int
    images_extracted: int
    chunks_created: int
    processing_time_ms: int


class IndexRequest(BaseModel):
    force_rebuild: bool = Field(default=True,description="Reconstruir índice desde data/processed",)


class IndexResponse(BaseModel):
    chunks_indexed: int
    index_path: str
    metadata_path: str
    embedding_model: str
