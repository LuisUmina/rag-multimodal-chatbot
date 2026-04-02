from __future__ import annotations

import base64
import json
from pathlib import Path

from app.config import settings
from app.rag.page_images import extract_embedded_images
from app.services.openai_client import get_openai_client

client = get_openai_client()

IMAGE_CAPTION_PROMPT = (
    "Briefly describe this image. "
    "If it is a diagram, include main components and relationships. "
    "Maximum 120 words."
)

PAGE_VISION_PROMPT = (
    "Analyze this complete document page. "
    "Return: (1) page summary, "
    "(2) key visible text, (3) important components/entities, "
    "(4) relationships or flow if diagram present. Maximum 220 words."
)


def caption_extracted_images(extracted_images: list[dict], captions_output_file: str) -> dict[str, list]:
    """Generate captions for already extracted embedded image items."""
    captions_data: list[dict] = []

    for item in extracted_images:
        page_number = item.get("page")
        image_index = item.get("image_index")
        filename = item.get("filename")
        image_path = item.get("path", "")

        try:
            caption = _run_vision_prompt(
                image_path=str(image_path),
                prompt_text=IMAGE_CAPTION_PROMPT,
                max_tokens=250,
                empty_fallback="Image without description",
            )
            captions_data.append(
                {
                    "page": page_number,
                    "image_index": image_index,
                    "filename": filename,
                    "caption": caption,
                }
            )
        except Exception as exc:
            print(f"Error processing image on page {page_number}: {exc}")
            continue

    captions_path = Path(captions_output_file)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text(
        json.dumps(captions_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {"captions": captions_data}


def caption_page_images(pages_input_dir: str = "data/processed/page_images", page_vision_output_file: str = "data/processed/page_vision.json",) -> dict[str, list]:
    """Generate page-level vision text from pre-rendered page images."""
    
    input_dir = Path(pages_input_dir)
    page_items: list[dict] = []

    if not input_dir.exists():
        return {"pages": page_items}

    for page_path in sorted(input_dir.glob("page_*.png")):
        page_number = int(page_path.stem.split("_")[-1])

        try:
            vision_text = _run_vision_prompt(
                image_path=str(page_path),
                prompt_text=PAGE_VISION_PROMPT,
                max_tokens=420,
                empty_fallback="",
            )
        except Exception as exc:
            print(f"Error processing full page {page_number}: {exc}")
            vision_text = ""

        page_items.append(
            {
                "page": page_number,
                "filename": page_path.name,
                "path": str(page_path),
                "vision_text": vision_text,
            }
        )

    output_path = Path(page_vision_output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(page_items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {"pages": page_items}


def _run_vision_prompt(image_path: str, prompt_text: str, max_tokens: int, empty_fallback: str) -> str:
    """Run a vision prompt with model fallback and return text output."""
    with open(image_path, "rb") as image_file:
        image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

    candidate_models = [settings.openai_vision_model, "gpt-4o-mini", "gpt-4o"]
    models = list(dict.fromkeys(model for model in candidate_models if model))
    last_error: Exception | None = None

    for model_name in models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"},
                            },
                            {
                                "type": "text",
                                "text": prompt_text,
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or empty_fallback
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "deprecated" in error_text or "model_not_found" in error_text:
                continue
            raise

    raise RuntimeError(f"Could not complete vision prompt with available models: {last_error}")

