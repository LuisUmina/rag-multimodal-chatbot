from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación."""

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    openai_vision_model: str = "gpt-4o-mini"

    # Vector Store
    vector_store: str = "faiss"

    # PDF Processing
    pdf_path: str = "rag-challenge.pdf"
    save_extracted_text: bool = True
    save_extracted_images: bool = True
    save_page_vision: bool = True
    save_page_ocr: bool = True
    ocr_lang: str = "eng"
    tesseract_cmd: str = ""

    # Chunking
    chunk_size: int = 700
    chunk_overlap: int = 150
    top_k: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
