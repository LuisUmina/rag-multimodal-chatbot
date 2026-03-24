from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.rag.ingest_pdf import ingest_pdf

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "RAG Multimodal Chatbot API"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        result = ingest_pdf(
            pdf_path=request.pdf_path,
            save_extracted_text=request.save_extracted_text,
        )
        return IngestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la ingesta: {exc}") from exc
