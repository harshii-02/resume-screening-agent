"""Embedding similarity and weighted ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    EMBEDDING_MODEL,
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_SIMILARITY,
    WEIGHT_SKILLS,
)
from utils import canonicalize_skills


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def skill_match_score(candidate_skills: list[str], required_skills: list[str]) -> tuple[float, list[str], list[str]]:
    cand = set(canonicalize_skills(candidate_skills))
    req = canonicalize_skills(required_skills)
    if not req:
        return 100.0, [], []
    matched = [s for s in req if s in cand]
    missing = [s for s in req if s not in cand]
    score = 100.0 * (len(matched) / len(req))
    return score, matched, missing


def experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        # Prefer some experience but don't heavily punish juniors
        if candidate_years <= 0:
            return 55.0
        if candidate_years >= 3:
            return 100.0
        return 55.0 + (candidate_years / 3.0) * 45.0
    ratio = candidate_years / required_years
    return float(min(100.0, max(0.0, ratio * 100.0)))


def education_score(education_items: list[str], keywords: list[str]) -> float:
    if not keywords:
        return 70.0 if education_items else 50.0
    blob = " ".join(education_items).lower()
    hits = sum(1 for kw in keywords if kw.lower() in blob)
    if hits == 0:
        return 40.0
    return float(min(100.0, 60.0 + (hits / max(len(keywords), 1)) * 40.0))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def overall_score(
    similarity_pct: float,
    skills_pct: float,
    experience_pct: float,
    education_pct: float,
) -> float:
    total = (
        WEIGHT_SIMILARITY * similarity_pct
        + WEIGHT_SKILLS * skills_pct
        + WEIGHT_EXPERIENCE * experience_pct
        + WEIGHT_EDUCATION * education_pct
    )
    return round(float(total), 2)


def build_profile_text(candidate: dict[str, Any], resume_text: str) -> str:
    """Compact, JD-comparable text for embedding (skills + structured facts)."""
    parts = [
        f"Candidate: {candidate.get('name', '')}",
        f"Skills: {', '.join(candidate.get('skills', []))}",
        f"Experience years: {candidate.get('experience_years', 0)}",
        f"Education: {'; '.join(candidate.get('education', []))}",
        f"Projects: {'; '.join(candidate.get('projects', []))}",
        f"Experience: {'; '.join(candidate.get('experience', []))}",
        resume_text[:2500],
    ]
    return "\n".join(parts)


def build_jd_embedding_text(jd: dict[str, Any], jd_text: str) -> str:
    parts = [
        f"Role: {jd.get('title', '')}",
        f"Required skills: {', '.join(jd.get('required_skills', []))}",
        f"Preferred skills: {', '.join(jd.get('preferred_skills', []))}",
        f"Minimum experience years: {jd.get('min_experience_years', 0)}",
        f"Education: {', '.join(jd.get('education_keywords', []))}",
        jd_text[:2000],
    ]
    return "\n".join(parts)


def score_candidate(
    resume_text: str,
    jd_text: str,
    candidate: dict[str, Any],
    jd: dict[str, Any],
    engine: EmbeddingEngine,
) -> dict[str, Any]:
    left = build_profile_text(candidate, resume_text)
    right = build_jd_embedding_text(jd, jd_text)
    resume_vec, jd_vec = engine.encode([left, right])
    sim = max(0.0, cosine_sim(resume_vec, jd_vec))
    similarity_pct = round(sim * 100.0, 2)

    skills_pct, matched, missing = skill_match_score(
        candidate.get("skills", []),
        jd.get("required_skills", []),
    )
    exp_pct = experience_score(
        float(candidate.get("experience_years") or 0),
        float(jd.get("min_experience_years") or 0),
    )
    edu_pct = education_score(
        [str(x) for x in candidate.get("education", [])],
        [str(x) for x in jd.get("education_keywords", [])],
    )
    total = overall_score(similarity_pct, skills_pct, exp_pct, edu_pct)

    return {
        "similarity": similarity_pct,
        "skills_score": round(skills_pct, 2),
        "experience_score": round(exp_pct, 2),
        "education_score": round(edu_pct, 2),
        "score": total,
        "matched_skills": matched,
        "missing_skills": missing,
    }
