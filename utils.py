"""Shared helpers for the Resume Screening Agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())


def skill_aliases() -> dict[str, str]:
    """Map common aliases to a canonical skill name."""
    return {
        "js": "javascript",
        "ts": "typescript",
        "node": "node.js",
        "nodejs": "node.js",
        "node.js": "node.js",
        "react.js": "react",
        "reactjs": "react",
        "vue.js": "vue",
        "postgres": "postgresql",
        "psql": "postgresql",
        "k8s": "kubernetes",
        "tf": "tensorflow",
        "pytorch": "pytorch",
        "scikit-learn": "scikit-learn",
        "sklearn": "scikit-learn",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "ci/cd": "ci/cd",
        "github actions": "ci/cd",
        "fastapi": "fastapi",
        "flask": "flask",
        "django": "django",
        "aws": "aws",
        "gcp": "gcp",
        "azure": "azure",
        "docker": "docker",
        "git": "git",
        "sql": "sql",
        "nosql": "nosql",
        "mongodb": "mongodb",
        "redis": "redis",
        "kafka": "kafka",
        "python": "python",
        "java": "java",
        "c++": "c++",
        "c#": "c#",
        "go": "go",
        "golang": "go",
        "rust": "rust",
        "linux": "linux",
        "rest": "rest api",
        "rest api": "rest api",
        "graphql": "graphql",
        "pandas": "pandas",
        "numpy": "numpy",
        "spark": "apache spark",
        "apache spark": "apache spark",
        "airflow": "airflow",
        "terraform": "terraform",
        "ansible": "ansible",
        "jenkins": "jenkins",
        "power bi": "power bi",
        "tableau": "tableau",
        "excel": "excel",
        "llm": "llm",
        "openai": "openai",
        "langchain": "langchain",
        "hugging face": "hugging face",
        "transformers": "transformers",
    }


def canonicalize_skills(skills: list[str]) -> list[str]:
    aliases = skill_aliases()
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in skills:
        key = normalize_skill(raw)
        if not key:
            continue
        canonical = aliases.get(key, key)
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def extract_years_of_experience(text: str) -> float:
    """Best-effort years-of-experience estimate from free text."""
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w/#+\-. ]{0,40}?\s)?(?:experience|exp)\b",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:in|as)\s+",
        r"with\s+(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b",
    ]
    years: list[float] = []
    lowered = text.lower()
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            value = float(match.group(1))
            if 0 < value <= 40:
                years.append(value)
    return max(years) if years else 0.0


def guess_candidate_name(filename: str, text: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    if stem and not stem.lower().startswith("resume"):
        return stem.title()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:5]:
        if "@" in line or "http" in line.lower():
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", line):
            return line.title()
    return stem.title() or "Unknown Candidate"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
