"""Core screening agent orchestration (CLI + UI share this)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_SIMILARITY,
    WEIGHT_SKILLS,
)
from extract import extract_text, list_resume_files, load_job_description
from llm import (
    explain_rank_difference,
    generate_recruiter_summary,
    llm_status,
    parse_job_description,
    parse_resume,
)
from ranking import EmbeddingEngine, score_candidate
from utils import ensure_dir, write_csv, write_json


def screen_resumes(
    resume_path: Path,
    jd_path: Path,
    output_dir: Path | None = None,
    limit: int = 0,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Agent loop:
      Input (JD + resumes) → extract text → LLM/heuristic structure
      → embed + score → reason → optional save CSV/JSON.
    """
    jd_text = load_job_description(jd_path)
    jd = parse_job_description(jd_text)
    files = list_resume_files(resume_path)
    if limit > 0:
        files = files[:limit]

    engine = EmbeddingEngine()
    results: list[dict[str, Any]] = []

    for idx, path in enumerate(files, start=1):
        if progress_callback:
            progress_callback(idx, len(files), path.name)
        text = extract_text(path)
        if not text.strip():
            continue

        candidate = parse_resume(path.name, text)
        scores = score_candidate(text, jd_text, candidate, jd, engine)
        narrative = generate_recruiter_summary(
            candidate,
            jd,
            scores["matched_skills"],
            scores["missing_skills"],
            scores["score"],
        )
        results.append(
            {
                "candidate": candidate["name"],
                "file": path.name,
                "score": scores["score"],
                "similarity": scores["similarity"],
                "skills_score": scores["skills_score"],
                "experience_score": scores["experience_score"],
                "education_score": scores["education_score"],
                "experience_years": candidate.get("experience_years", 0),
                "matched_skills": scores["matched_skills"],
                "missing_skills": scores["missing_skills"],
                "skills": candidate.get("skills", []),
                "education": candidate.get("education", []),
                "projects": candidate.get("projects", []),
                "certifications": candidate.get("certifications", []),
                "strengths": narrative["strengths"],
                "skill_gaps": narrative["skill_gaps"],
                "reason": narrative["reason"],
                "recruiter_summary": narrative["recruiter_summary"],
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, row in enumerate(results):
        row["rank"] = i + 1
        if i < len(results) - 1:
            row["why_above_next"] = explain_rank_difference(row, results[i + 1])
        else:
            row["why_above_next"] = "Lowest rank in this run."

    pairwise: list[str] = []
    for i in range(len(results) - 1):
        pairwise.append(explain_rank_difference(results[i], results[i + 1]))

    payload = {
        "job": {
            "title": jd.get("title"),
            "required_skills": jd.get("required_skills"),
            "preferred_skills": jd.get("preferred_skills"),
            "min_experience_years": jd.get("min_experience_years"),
            "scoring_weights": {
                "similarity": WEIGHT_SIMILARITY,
                "skills": WEIGHT_SKILLS,
                "experience": WEIGHT_EXPERIENCE,
                "education": WEIGHT_EDUCATION,
            },
        },
        "llm": llm_status(),
        "pairwise_explanations": pairwise,
        "candidates": results,
    }

    if output_dir is not None:
        ensure_dir(output_dir)
        csv_path = output_dir / "ranked_candidates.csv"
        json_path = output_dir / "ranked_candidates.json"
        csv_rows = [
            {
                "Rank": r["rank"],
                "Candidate": r["candidate"],
                "File": r["file"],
                "Score": r["score"],
                "Similarity": r["similarity"],
                "Skills": r["skills_score"],
                "Experience": r["experience_score"],
                "Education": r["education_score"],
                "Matched Skills": ", ".join(r["matched_skills"]),
                "Missing Skills": ", ".join(r["missing_skills"]),
                "Reason": r["reason"],
                "Why Above Next": r.get("why_above_next", ""),
                "Recruiter Summary": r["recruiter_summary"],
            }
            for r in results
        ]
        write_csv(csv_rows, csv_path)
        write_json(payload, json_path)
        payload["csv_path"] = str(csv_path)
        payload["json_path"] = str(json_path)

    return payload
