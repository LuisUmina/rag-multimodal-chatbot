from fastapi import FastAPI
from app.api.routes import router


def create_app() -> FastAPI:
	app = FastAPI(
		title="RAG Multimodal Chatbot API",
		description="Backend en FastAPI para chatbot RAG multimodal.",
		version="0.1.0",
	)
	app.include_router(router)
	return app


app = create_app()
