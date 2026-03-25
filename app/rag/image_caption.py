from __future__ import annotations
import base64
import json
from pathlib import Path
import fitz
from app.services.openai_client import get_openai_client
from app.config import settings

client = get_openai_client()

def extract_and_caption_images(pdf_path: str, images_output_dir: str = "data/processed/images", captions_output_file: str = "data/processed/image_captions.json",) -> dict[str, list]:
    """
    Extrae imágenes del PDF y genera captions con OpenAI Vision.
    """
    
    images_dir = Path(images_output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    captions_data = []
    extracted_images = []

    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1
            image_list = page.get_images(full=True)

            for img_index, img_reference in enumerate(image_list):
                try:
                    # ID of image
                    xref = img_reference[0]
                    # Select image by ID
                    pix = fitz.Pixmap(document, xref)

                    image_filename = f"page_{page_number:03d}_img_{img_index + 1}.png"
                    image_path = images_dir / image_filename

                    # Is this image RGB (or simpler)?
                    if pix.n - pix.alpha < 4:
                        pix.save(str(image_path))
                    else:
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix_rgb.save(str(image_path))
                        pix_rgb = None
                    pix = None

                    extracted_images.append(
                        {
                            "page": page_number,
                            "image_index": img_index + 1,
                            "filename": image_filename,
                            "path": str(image_path),
                        }
                    )

                    caption = _generate_caption_with_vision(str(image_path))
                    captions_data.append(
                        {
                            "page": page_number,
                            "image_index": img_index + 1,
                            "filename": image_filename,
                            "caption": caption,
                        }
                    )

                except Exception as exc:
                    print(f"Error procesando imagen en página {page_number}: {exc}")
                    continue

    captions_path = Path(captions_output_file)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)

    return {
        "extracted_images": extracted_images,
        "captions": captions_data,
    }


def render_and_caption_pages(
    pdf_path: str,
    pages_output_dir: str = "data/processed/page_images",
    page_vision_output_file: str = "data/processed/page_vision.json",
) -> dict[str, list]:
    """
    Renderiza cada página completa como imagen y genera descripción/lectura visual.
    """
    output_dir = Path(pages_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_items: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1

            # Render básico (2x) para mejor lectura visual
            matrix = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            page_filename = f"page_{page_number:03d}.png"
            page_path = output_dir / page_filename
            pix.save(str(page_path))

            try:
                vision_text = _generate_page_vision_text(str(page_path))
            except Exception as exc:
                print(f"Error procesando página completa {page_number}: {exc}")
                vision_text = ""

            page_items.append(
                {
                    "page": page_number,
                    "filename": page_filename,
                    "path": str(page_path),
                    "vision_text": vision_text,
                }
            )

    output_path = Path(page_vision_output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(page_items, file, indent=2, ensure_ascii=False)

    return {"pages": page_items}


def _generate_caption_with_vision(image_path: str) -> str:
    """Genera descripción usando un modelo vision vigente con fallback automático."""

    with open(image_path, "rb") as img_file:
        image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    candidate_models = [settings.openai_vision_model, "gpt-4o-mini", "gpt-4o"]
    unique_models: list[str] = []

    for model_name in candidate_models:
        if model_name and model_name not in unique_models:
            unique_models.append(model_name)

    last_error: Exception | None = None

    for model_name in unique_models:
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
                                "text": (
                                    "Briefly describe this image. "
                                    "If it is a diagram, include main components and relationships. "
                                    "Maximum 120 words."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=250,
            )
            return response.choices[0].message.content or "Imagen sin descripción"
        
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "deprecated" in error_text or "model_not_found" in error_text:
                continue
            raise

    raise RuntimeError(f"No fue posible generar caption con modelos vision disponibles: {last_error}")


def _generate_page_vision_text(image_path: str) -> str:
    """Extrae información densa de una página completa renderizada."""
    with open(image_path, "rb") as img_file:
        image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    candidate_models = [settings.openai_vision_model, "gpt-4o-mini", "gpt-4o"]
    unique_models: list[str] = []

    for model_name in candidate_models:
        if model_name and model_name not in unique_models:
            unique_models.append(model_name)

    last_error: Exception | None = None

    for model_name in unique_models:
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
                                "text": (
                                    "Analyze this complete document page. "
                                    "Return: (1) page summary, "
                                    "(2) key visible text, (3) important components/entities, "
                                    "(4) relationships or flow if diagram present. Maximum 220 words."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=420,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "deprecated" in error_text or "model_not_found" in error_text:
                continue
            raise

    raise RuntimeError(f"No fue posible generar extracción visual por página: {last_error}")
