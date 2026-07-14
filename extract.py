"""Resume and job-description text extraction (PDF / DOCX / TXT)."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

from config import SUPPORTED_EXTENSIONS


def extract_text_from_pdf(path: Path) -> str:
    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()


def extract_text_from_docx(path: Path) -> str:
    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".txt":
        return extract_text_from_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def list_resume_files(resume_path: Path) -> list[Path]:
    """Accept either a single resume file or a folder of resumes."""
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume path not found: {resume_path}")

    if resume_path.is_file():
        if resume_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported resume type: {resume_path.suffix}. Use PDF, DOCX, or TXT."
            )
        return [resume_path]

    if not resume_path.is_dir():
        raise ValueError(f"Resume path must be a file or folder: {resume_path}")

    files = [
        p
        for p in sorted(resume_path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No PDF/DOCX/TXT resumes found in {resume_path}")
    return files


def load_job_description(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {path}")
    text = extract_text(path) if path.suffix.lower() in SUPPORTED_EXTENSIONS else path.read_text(
        encoding="utf-8", errors="ignore"
    )
    text = text.strip()
    if not text:
        raise ValueError(f"Job description is empty: {path}")
    return text
