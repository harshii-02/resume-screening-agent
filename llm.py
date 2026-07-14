"""LLM-backed (with heuristic fallback) structured extraction and summaries."""

from __future__ import annotations

import json
import re
from typing import Any

from config import (
    GROQ_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    resolve_llm_provider,
)
from utils import canonicalize_skills, extract_years_of_experience, guess_candidate_name

COMMON_SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C#",
    "Go",
    "Rust",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "Kafka",
    "FastAPI",
    "Flask",
    "Django",
    "Spring",
    "Node.js",
    "React",
    "Vue",
    "Angular",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "Git",
    "CI/CD",
    "Linux",
    "REST API",
    "GraphQL",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Hugging Face",
    "Transformers",
    "LLM",
    "OpenAI",
    "LangChain",
    "Apache Spark",
    "Airflow",
    "Terraform",
    "Ansible",
    "Jenkins",
    "Power BI",
    "Tableau",
    "Excel",
]


def _heuristic_skills(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for skill in COMMON_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(skill)
    return canonicalize_skills(found)


def _section_lines(text: str, headers: list[str]) -> list[str]:
    lines = text.splitlines()
    collecting = False
    collected: list[str] = []
    header_re = re.compile(r"^[#*\-\s]*(" + "|".join(headers) + r")\s*[:#]?\s*$", re.I)
    next_header_re = re.compile(
        r"^[#*\-\s]*(skills?|experience|education|projects?|certifications?|summary|"
        r"work history|employment|objective|contact|requirements?|required|"
        r"preferred|qualifications?|responsibilities|must[\s-]?have|"
        r"nice[\s-]?to[\s-]?have|optional)\s*[:#]?\s*$",
        re.I,
    )
    for line in lines:
        stripped = line.strip()
        if header_re.match(stripped):
            collecting = True
            continue
        if collecting and next_header_re.match(stripped) and not header_re.match(stripped):
            break
        if collecting and stripped:
            collected.append(stripped.strip(" -\t"))
    return collected


def heuristic_parse_resume(filename: str, text: str) -> dict[str, Any]:
    skills = _heuristic_skills(text)
    experience_lines = _section_lines(text, ["experience", "work experience", "work history", "employment"])
    education_lines = _section_lines(text, ["education", "academics"])
    project_lines = _section_lines(text, ["projects", "project"])
    cert_lines = _section_lines(text, ["certifications", "certificates", "licenses"])
    years = extract_years_of_experience(text)

    return {
        "name": guess_candidate_name(filename, text),
        "skills": skills,
        "experience_years": years,
        "experience": experience_lines[:8] or (["Experience mentioned in resume"] if years else []),
        "education": education_lines[:5] or ["Not specified"],
        "projects": project_lines[:6],
        "certifications": cert_lines[:6],
        "summary": "",
    }


def _skills_from_section(text: str, headers: list[str]) -> list[str]:
    """Collect skills listed under a titled section (Requirements / Preferred)."""
    lines = _section_lines(text, headers)
    found: list[str] = []
    for line in lines:
        cleaned = line.strip(" -\t•*")
        if not cleaned or cleaned.lower() in {"requirements", "preferred", "qualifications"}:
            continue
        # Whole-line skill (e.g. "FastAPI") or comma-separated list
        for part in re.split(r"[,;/|]", cleaned):
            part = part.strip()
            if not part:
                continue
            found.extend(_heuristic_skills(part))
            if 1 <= len(part.split()) <= 4:
                found.extend(canonicalize_skills([part]))
    # Prefer section hits that appear in the known skill vocabulary
    known = set(canonicalize_skills(COMMON_SKILLS))
    return [s for s in canonicalize_skills(found) if s in known]


def heuristic_parse_jd(text: str) -> dict[str, Any]:
    required = _skills_from_section(text, ["requirements", "required", "must have", "must-have"])
    preferred = _skills_from_section(text, ["preferred", "nice to have", "nice-to-have", "optional"])

    # Fallback: scan full text if no explicit Requirements section
    if not required:
        required = _heuristic_skills(text)
        required = [s for s in required if s not in set(preferred)]

    years = extract_years_of_experience(text)
    title_match = re.search(r"(?im)^(?:job\s*title|role|position)\s*[:\-]\s*(.+)$", text)
    title = title_match.group(1).strip() if title_match else text.splitlines()[0].strip()

    return {
        "title": title,
        "required_skills": required,
        "preferred_skills": preferred,
        "min_experience_years": years,
        "education_keywords": _education_keywords(text),
        "raw_text": text,
    }


def _education_keywords(text: str) -> list[str]:
    keywords = []
    lowered = text.lower()
    for term in [
        "bachelor",
        "master",
        "phd",
        "b.tech",
        "b.e",
        "m.tech",
        "mba",
        "computer science",
        "information technology",
        "software engineering",
        "data science",
    ]:
        if term in lowered:
            keywords.append(term)
    return keywords


def llm_status() -> str:
    provider, model = resolve_llm_provider()
    if provider is None:
        return "disabled (heuristic fallback); set GROQ_API_KEY or OPENAI_API_KEY"
    return f"{provider} / {model}"


def _llm_client():
    provider, model = resolve_llm_provider()
    if provider is None:
        return None, ""
    try:
        from openai import OpenAI

        if provider == "groq":
            client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
            return client, model
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client, model or OPENAI_MODEL
    except Exception:
        return None, ""


def _chat_json(system: str, user: str) -> dict[str, Any] | None:
    client, model = _llm_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001 — fall back gracefully
        print(f"[warn] LLM call failed ({exc}); using heuristic extraction.")
        return None


def explain_rank_difference(higher: dict[str, Any], lower: dict[str, Any]) -> str:
    """Deterministic explanation of why candidate A ranks above candidate B."""
    gaps = []
    for key, label in [
        ("score", "overall score"),
        ("similarity", "embedding similarity"),
        ("skills_score", "skill match"),
        ("experience_score", "experience fit"),
        ("education_score", "education fit"),
    ]:
        delta = float(higher.get(key, 0)) - float(lower.get(key, 0))
        if abs(delta) >= 0.5:
            gaps.append(f"{label} {delta:+.1f}")
    matched_extra = sorted(set(higher.get("matched_skills", [])) - set(lower.get("matched_skills", [])))
    detail = "; ".join(gaps[:4]) if gaps else "narrow margin on combined score"
    skills_note = ""
    if matched_extra:
        skills_note = f" Extra matched skills vs lower rank: {', '.join(matched_extra[:4])}."
    return (
        f"{higher.get('candidate')} ranks above {lower.get('candidate')} primarily due to {detail}."
        f"{skills_note}"
    )


def parse_resume(filename: str, text: str) -> dict[str, Any]:
    system = (
        "You extract structured candidate facts from resumes. "
        "Return strict JSON with keys: name, skills (array of strings), "
        "experience_years (number), experience (array of short strings), "
        "education (array), projects (array), certifications (array)."
    )
    user = f"Resume filename: {filename}\n\nResume text:\n{text[:12000]}"
    parsed = _chat_json(system, user)
    if not parsed:
        return heuristic_parse_resume(filename, text)

    base = heuristic_parse_resume(filename, text)
    skills = canonicalize_skills([str(s) for s in parsed.get("skills", [])] or base["skills"])
    try:
        years = float(parsed.get("experience_years", base["experience_years"]) or 0)
    except (TypeError, ValueError):
        years = base["experience_years"]

    return {
        "name": str(parsed.get("name") or base["name"]),
        "skills": skills,
        "experience_years": years,
        "experience": [str(x) for x in parsed.get("experience", [])] or base["experience"],
        "education": [str(x) for x in parsed.get("education", [])] or base["education"],
        "projects": [str(x) for x in parsed.get("projects", [])] or base["projects"],
        "certifications": [str(x) for x in parsed.get("certifications", [])] or base["certifications"],
        "summary": "",
    }


def parse_job_description(text: str) -> dict[str, Any]:
    system = (
        "You extract hiring requirements from a job description. "
        "Return strict JSON with keys: title, required_skills (array), "
        "preferred_skills (array), min_experience_years (number), "
        "education_keywords (array of lowercase strings)."
    )
    parsed = _chat_json(system, f"Job description:\n{text[:8000]}")
    if not parsed:
        return heuristic_parse_jd(text)

    base = heuristic_parse_jd(text)
    required = canonicalize_skills(
        [str(s) for s in parsed.get("required_skills", [])] or base["required_skills"]
    )
    preferred = canonicalize_skills([str(s) for s in parsed.get("preferred_skills", [])])
    try:
        years = float(parsed.get("min_experience_years", base["min_experience_years"]) or 0)
    except (TypeError, ValueError):
        years = base["min_experience_years"]

    return {
        "title": str(parsed.get("title") or base["title"]),
        "required_skills": required,
        "preferred_skills": preferred,
        "min_experience_years": years,
        "education_keywords": [str(x).lower() for x in parsed.get("education_keywords", [])]
        or base["education_keywords"],
        "raw_text": text,
    }


def generate_recruiter_summary(
    candidate: dict[str, Any],
    jd: dict[str, Any],
    matched: list[str],
    missing: list[str],
    score: float,
) -> dict[str, Any]:
    """Produce strengths, skill gaps, reason, and recruiter summary."""
    strengths = matched[:6] or candidate.get("skills", [])[:4]
    strengths = [s.title() if len(s) <= 3 else s.title() for s in strengths]
    missing_pretty = [m.title() if len(m) <= 3 else m.title() for m in missing[:8]]

    fallback_reason = (
        f"Overall fit score {score:.1f}/100. Matched {len(matched)} required skills"
        + (f"; gaps: {', '.join(missing_pretty[:4])}." if missing_pretty else ".")
    )
    fallback_summary = (
        ("Highly recommended. " if score >= 80 else "Recommended with caveats. " if score >= 65 else "Partial fit. ")
        + (f"Strong in {', '.join(strengths[:3])}. " if strengths else "")
        + (f"Needs exposure to {', '.join(missing_pretty[:3])}." if missing_pretty else "Covers most required skills.")
    )

    system = (
        "You are a technical recruiter. Return JSON with keys: "
        "strengths (array of short phrases), skill_gaps (array), "
        "reason (1-2 sentences explaining ranking fit), "
        "recruiter_summary (2-4 short sentences)."
    )
    user = (
        f"Role: {jd.get('title')}\n"
        f"Required skills: {jd.get('required_skills')}\n"
        f"Candidate: {candidate.get('name')}\n"
        f"Candidate skills: {candidate.get('skills')}\n"
        f"Years experience: {candidate.get('experience_years')}\n"
        f"Education: {candidate.get('education')}\n"
        f"Matched: {matched}\nMissing: {missing}\nScore: {score}\n"
    )
    parsed = _chat_json(system, user)
    if not parsed:
        return {
            "strengths": strengths,
            "skill_gaps": missing_pretty,
            "reason": fallback_reason,
            "recruiter_summary": fallback_summary,
        }

    return {
        "strengths": [str(x) for x in parsed.get("strengths", [])] or strengths,
        "skill_gaps": [str(x) for x in parsed.get("skill_gaps", [])] or missing_pretty,
        "reason": str(parsed.get("reason") or fallback_reason),
        "recruiter_summary": str(parsed.get("recruiter_summary") or fallback_summary),
    }
