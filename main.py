"""Resume Screening Agent — end-to-end CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    JD_PATH,
    OPENAI_API_KEY,
    OUTPUT_DIR,
    RESUME_DIR,
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_SIMILARITY,
    WEIGHT_SKILLS,
)
from extract import extract_text, list_resume_files, load_job_description
from llm import generate_recruiter_summary, parse_job_description, parse_resume
from ranking import EmbeddingEngine, score_candidate
from utils import ensure_dir, write_csv, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen resumes against a job description and rank candidates."
    )
    parser.add_argument("--resumes", type=Path, default=RESUME_DIR, help="Folder of PDF/DOCX/TXT resumes")
    parser.add_argument("--jd", type=Path, default=JD_PATH, help="Path to job description file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of resumes to process (0 = all)",
    )
    return parser


def run(resume_dir: Path, jd_path: Path, output_dir: Path, limit: int = 0) -> None:
    print("=== Resume Screening Agent ===")
    print(f"JD:      {jd_path}")
    print(f"Resumes: {resume_dir}")
    print(f"Output:  {output_dir}")
    print(
        "Weights: "
        f"similarity={WEIGHT_SIMILARITY:.0%}, "
        f"skills={WEIGHT_SKILLS:.0%}, "
        f"experience={WEIGHT_EXPERIENCE:.0%}, "
        f"education={WEIGHT_EDUCATION:.0%}"
    )
    if OPENAI_API_KEY:
        print("LLM:     OpenAI enabled")
    else:
        print("LLM:     disabled (heuristic extraction); set OPENAI_API_KEY for richer summaries")

    jd_text = load_job_description(jd_path)
    jd = parse_job_description(jd_text)
    print(f"\nRole: {jd.get('title')}")
    print(f"Required skills ({len(jd.get('required_skills', []))}): {', '.join(jd.get('required_skills', [])[:12])}")

    files = list_resume_files(resume_dir)
    if limit > 0:
        files = files[:limit]
    print(f"Screening {len(files)} resume(s)...\n")

    engine = EmbeddingEngine()
    results: list[dict] = []

    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path.name}")
        text = extract_text(path)
        if not text.strip():
            print(f"  skip (empty text): {path.name}")
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
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

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
            "Recruiter Summary": r["recruiter_summary"],
        }
        for r in results
    ]
    write_csv(csv_rows, csv_path)
    write_json(
        {
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
            "candidates": results,
        },
        json_path,
    )

    print("\n=== Rankings ===")
    for r in results:
        print(f"#{r['rank']:>2}  {r['candidate']:<28}  score={r['score']:.1f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run(args.resumes, args.jd, args.output, args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
