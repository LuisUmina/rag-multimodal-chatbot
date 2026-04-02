from __future__ import annotations

from pathlib import Path

import fitz


def extract_page_text(source: Path, save_extracted_text: bool) -> tuple[int, int]:
    """Extract per-page selectable text and optionally persist page txt files."""

    text_output_dir = Path("data/processed/text")
    if save_extracted_text:
        text_output_dir.mkdir(parents=True, exist_ok=True)

    pages_processed = 0
    pages_with_text = 0

    with fitz.open(source) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1
            page_text = page.get_text("text").strip()

            pages_processed += 1
            if page_text:
                pages_with_text += 1

            # Keep one text artifact per page for traceability.
            if save_extracted_text:
                target = text_output_dir / f"page_{page_number:03d}.txt"
                target.write_text(page_text, encoding="utf-8")

    return pages_processed, pages_with_text