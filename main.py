"""Resume Screening Agent — end-to-end CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import screen_resumes
from config import (
    JD_PATH,
    OUTPUT_DIR,
    RESUME_DIR,
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_SIMILARITY,
    WEIGHT_SKILLS,
)
from llm import llm_status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen resumes against a job description and rank candidates."
    )
    parser.add_argument(
        "--resumes",
        type=Path,
        default=RESUME_DIR,
        help="Folder of PDF/DOCX/TXT resumes, or a single resume file",
    )
    parser.add_argument("--jd", type=Path, default=JD_PATH, help="Path to job description file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of resumes to process (0 = all)",
    )
    return parser


def run(resume_path: Path, jd_path: Path, output_dir: Path, limit: int = 0) -> None:
    print("=== Resume Screening Agent ===")
    print(f"JD:      {jd_path}")
    print(f"Resumes: {resume_path}")
    print(f"Output:  {output_dir}")
    print(
        "Weights: "
        f"similarity={WEIGHT_SIMILARITY:.0%}, "
        f"skills={WEIGHT_SKILLS:.0%}, "
        f"experience={WEIGHT_EXPERIENCE:.0%}, "
        f"education={WEIGHT_EDUCATION:.0%}"
    )
    print(f"LLM:     {llm_status()}")

    def _progress(idx: int, total: int, name: str) -> None:
        print(f"[{idx}/{total}] {name}")

    payload = screen_resumes(
        resume_path,
        jd_path,
        output_dir=output_dir,
        limit=limit,
        progress_callback=_progress,
    )

    print(f"\nRole: {payload['job'].get('title')}")
    req = payload["job"].get("required_skills") or []
    print(f"Required skills ({len(req)}): {', '.join(req[:12])}")

    print("\n=== Rankings ===")
    for r in payload["candidates"]:
        print(f"#{r['rank']:>2}  {r['candidate']:<28}  score={r['score']:.1f}")

    if payload.get("pairwise_explanations"):
        print("\n=== Why this order ===")
        for line in payload["pairwise_explanations"][:5]:
            print(f"- {line}")

    print(f"\nWrote {payload.get('csv_path')}")
    print(f"Wrote {payload.get('json_path')}")


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
