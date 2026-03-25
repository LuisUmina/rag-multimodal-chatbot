from fastapi import APIRouter, HTTPException
from app.api.schemas import IndexRequest, IndexResponse, IngestRequest, IngestResponse
from app.rag.ingest_pdf import ingest_pdf
from app.rag.vector_store import index_processed_content

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


@router.post("/index", response_model=IndexResponse)
def index_content(request: IndexRequest) -> IndexResponse:
    try:

        if not request.force_rebuild:
            raise HTTPException(status_code=400, detail="Actualmente solo se soporta force_rebuild=true")

        result = index_processed_content()
        return IndexResponse(**result)
    
    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante el indexado: {exc}") from exc
