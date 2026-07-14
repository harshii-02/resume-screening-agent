"""Unit tests for scoring helpers (no network / model download required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranking import education_score, experience_score, overall_score, skill_match_score
from utils import canonicalize_skills, extract_years_of_experience


def test_skill_match_perfect():
    score, matched, missing = skill_match_score(
        ["Python", "FastAPI", "SQL"],
        ["python", "fastapi", "sql"],
    )
    assert score == 100.0
    assert missing == []
    assert set(matched) == {"python", "fastapi", "sql"}


def test_skill_match_partial():
    score, matched, missing = skill_match_score(["Python", "Git"], ["python", "docker", "git"])
    assert abs(score - (200 / 3)) < 0.01
    assert "docker" in missing
    assert "python" in matched


def test_experience_meets_requirement():
    assert experience_score(3, 3) == 100.0
    assert experience_score(1.5, 3) == 50.0


def test_education_keywords():
    assert education_score(["B.S. Computer Science"], ["computer science", "bachelor"]) >= 60


def test_overall_weights():
    # Uses config defaults 0.5 / 0.25 / 0.15 / 0.10
    total = overall_score(100, 100, 100, 100)
    assert total == 100.0


def test_years_extraction():
    assert extract_years_of_experience("with 4 years of Python experience") == 4.0
    assert extract_years_of_experience("5+ years experience building APIs") == 5.0


def test_canonicalize_aliases():
    assert "node.js" in canonicalize_skills(["NodeJS", "nodejs"])
    assert canonicalize_skills(["JS", "JavaScript"]) == ["javascript"]


if __name__ == "__main__":
    test_skill_match_perfect()
    test_skill_match_partial()
    test_experience_meets_requirement()
    test_education_keywords()
    test_overall_weights()
    test_years_extraction()
    test_canonicalize_aliases()
    print("All tests passed.")
