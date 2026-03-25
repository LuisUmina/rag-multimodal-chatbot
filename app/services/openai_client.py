from openai import OpenAI
from app.config import settings


def get_openai_client() -> OpenAI:
    """Create the shared OpenAI client from application settings."""
    
    return OpenAI(api_key=settings.openai_api_key)
