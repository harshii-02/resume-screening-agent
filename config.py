"""Configuration for the Resume Screening Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent

RESUME_DIR = Path(os.getenv("RESUME_DIR", ROOT_DIR / "sample_resumes"))
JD_PATH = Path(os.getenv("JD_PATH", ROOT_DIR / "JD" / "job_description.txt"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "outputs"))

# LLM: OpenAI and/or Groq (OpenAI-compatible). Prefer Groq if only GROQ_API_KEY is set.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()  # openai | groq | auto

# Local embedding model (runs offline after first download)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Final score weights (must sum to 1.0)
WEIGHT_SIMILARITY = float(os.getenv("WEIGHT_SIMILARITY", "0.50"))
WEIGHT_SKILLS = float(os.getenv("WEIGHT_SKILLS", "0.25"))
WEIGHT_EXPERIENCE = float(os.getenv("WEIGHT_EXPERIENCE", "0.15"))
WEIGHT_EDUCATION = float(os.getenv("WEIGHT_EDUCATION", "0.10"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def resolve_llm_provider() -> tuple[str | None, str]:
    """Return (provider, model). provider is openai|groq|None."""
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return "openai", OPENAI_MODEL
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return "groq", GROQ_MODEL
    if GROQ_API_KEY and (LLM_PROVIDER in {"", "auto"} or not OPENAI_API_KEY):
        return "groq", GROQ_MODEL
    if OPENAI_API_KEY:
        return "openai", OPENAI_MODEL
    return None, ""
