# Resume Screening Agent

An end-to-end AI agent that screens a folder of resumes against a job description, ranks candidates with explainable scores, and exports **CSV + JSON** results.

It combines:

- **File parsing** (PDF / DOCX / TXT)
- **Structured extraction** (skills, experience, education, projects, certifications)
- **Embeddings + cosine similarity** (`all-MiniLM-L6-v2`)
- **Weighted ranking** (similarity, skill match, experience, education)
- **Optional LLM** summaries, skill gaps, and recruiter reasons (OpenAI)

Works **without an API key** using heuristic extraction; add `OPENAI_API_KEY` for richer LLM reasoning.

---

## Architecture

```
Job Description
        │
        ▼
   Read JD text
        │
        ▼
 Read resume folder (PDF/DOCX/TXT)
        │
        ▼
   Extract plain text
        │
        ▼
 LLM / heuristic extraction
 (skills, experience, education, projects)
        │
        ▼
 Generate embeddings (SentenceTransformers)
        │
        ▼
 Cosine similarity + skill/exp/edu scores
        │
        ▼
 Overall weighted score → ranking
        │
        ▼
 CSV + JSON output (+ recruiter summary)
```

---

## Scoring formula

| Component            | Weight | How it is computed                                      |
| -------------------- | ------ | ------------------------------------------------------- |
| Embedding similarity | 50%    | Cosine similarity between JD and resume embeddings      |
| Skill match          | 25%    | `% of required JD skills found in candidate skills`     |
| Experience           | 15%    | Candidate years vs JD minimum (capped at 100)           |
| Education            | 10%    | Keyword overlap (e.g. Computer Science, Bachelor, …)    |

```
Final score (0–100) =
  0.50 × Similarity
+ 0.25 × Skill Match
+ 0.15 × Experience
+ 0.10 × Education
```

Weights are configurable via environment variables.

---

## Folder structure

```
resume-screening-agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py              # CLI entrypoint
├── config.py            # Paths, model, scoring weights
├── utils.py             # Helpers (skills, I/O, name guess)
├── extract.py           # PDF / DOCX / TXT extraction
├── llm.py               # OpenAI + heuristic parsing/summaries
├── ranking.py           # Embeddings + scoring
├── JD/
│   └── job_description.txt
├── sample_resumes/      # 13 samples (TXT + DOCX + PDF)
└── outputs/             # Sample ranked CSV/JSON committed for reviewers
    ├── ranked_candidates.csv
    └── ranked_candidates.json
```

---

## Tech stack

| Area        | Library / model                          |
| ----------- | ---------------------------------------- |
| Language    | Python 3.10+                             |
| LLM (opt.)  | OpenAI (`gpt-4o-mini` by default)        |
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF         | PyMuPDF                                  |
| DOCX        | python-docx                              |
| Ranking     | scikit-learn cosine similarity           |
| Tabular I/O | pandas                                   |

---

## Installation

```bash
# 1. Clone
git clone <your-repo-url>
cd resume-screening-agent

# 2. Create virtualenv
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

First run downloads the SentenceTransformers model (~80MB).

---

## Environment variables

Copy the example file and edit:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

| Variable            | Required | Default                                      | Description                          |
| ------------------- | -------- | -------------------------------------------- | ------------------------------------ |
| `OPENAI_API_KEY`    | No       | _(empty)_                                    | Enables LLM extraction & summaries   |
| `OPENAI_MODEL`      | No       | `gpt-4o-mini`                                | Chat model name                      |
| `RESUME_DIR`        | No       | `sample_resumes`                             | Resume folder                        |
| `JD_PATH`           | No       | `JD/job_description.txt`                     | Job description file                 |
| `OUTPUT_DIR`        | No       | `outputs`                                    | Where CSV/JSON are written           |
| `EMBEDDING_MODEL`   | No       | `sentence-transformers/all-MiniLM-L6-v2`     | Local embedding model                |
| `WEIGHT_SIMILARITY` | No       | `0.50`                                       | Score weight                         |
| `WEIGHT_SKILLS`     | No       | `0.25`                                       | Score weight                         |
| `WEIGHT_EXPERIENCE` | No       | `0.15`                                       | Score weight                         |
| `WEIGHT_EDUCATION`  | No       | `0.10`                                       | Score weight                         |

**API key setup**

1. Create a key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Put it in `.env` as `OPENAI_API_KEY=sk-...`
3. Re-run the agent

Without a key, the agent still ranks resumes using heuristics + embeddings.

---

## How to run

```bash
# Screen the bundled sample resumes against the sample JD
python main.py

# Custom paths
python main.py --resumes path/to/resumes --jd path/to/jd.txt --output outputs

# Limit to first N files (handy for demos)
python main.py --limit 5
```

---

## Sample input

**Job description** (`JD/job_description.txt`):

```text
Python Developer

Requirements
Python
FastAPI
SQL
Docker
Git
Machine Learning
```

**Resumes:** drop PDF / DOCX / TXT files into `sample_resumes/` (13 samples included — mostly TXT, plus `alice_chen.docx` and `john_rivera.pdf` to demonstrate all parsers).

---

## Sample output

### Console

```text
=== Rankings ===
# 1  Chris Patel                   score=94.2
# 2  Alice Chen                    score=91.5
# 3  Priya Nair                    score=88.0
...
```

### CSV (`outputs/ranked_candidates.csv`)

| Rank | Candidate   | Score | Similarity | Skills | Experience | Reason |
| ---- | ----------- | ----- | ---------- | ------ | ---------- | ------ |
| 1    | Chris Patel | 94.2  | …          | …      | …          | …      |
| 2    | Alice Chen  | 91.5  | …          | …      | …          | …      |

### JSON (`outputs/ranked_candidates.json`)

```json
{
  "job": {
    "title": "Python Developer",
    "required_skills": ["python", "fastapi", "sql", "docker", "git", "machine learning"]
  },
  "candidates": [
    {
      "rank": 1,
      "candidate": "Chris Patel",
      "score": 94.2,
      "matched_skills": ["python", "fastapi", "sql", "docker", "git", "machine learning"],
      "missing_skills": [],
      "strengths": ["Python", "FastAPI", "Machine Learning"],
      "skill_gaps": [],
      "reason": "…",
      "recruiter_summary": "…"
    }
  ]
}
```

---

## Features

### Resume parsing
- Formats: **PDF**, **DOCX**, **TXT**
- Extracts: skills, experience, education, projects, certifications

### Similarity
- Local embeddings with `all-MiniLM-L6-v2`
- Cosine similarity between resume and JD

### Bonus outputs
- Skill gap analysis (missing required skills)
- Strengths list
- Recruiter summary + explainable ranking reason

---

## Tradeoffs

| Choice                         | Why                                                         | Tradeoff                                      |
| ------------------------------ | ----------------------------------------------------------- | --------------------------------------------- |
| Local MiniLM embeddings        | Free, fast, reproducible, no quota                          | Less semantic nuance than larger cloud models |
| Heuristic fallback without LLM | Project runs offline / without paid keys                    | Weaker name/section parsing on messy resumes  |
| Weighted score formula         | Transparent for reviewers; easy to tune                     | Weights are domain heuristics, not learned    |
| Batch folder processing        | Matches “10+ resumes in one run” evaluation need            | No interactive UI in v1                       |

---

## Future improvements

- Web UI for upload + live ranking table
- Cross-encoder re-ranking for top-k candidates
- Bias / fairness checks on names and demographics
- Vector store for historical JD/resume search
- Groq / local LLM backends for free inference
- Unit tests + GitHub Actions CI
- Larger PDF/DOCX sample packs and OCR for scanned resumes

---

## License

MIT — use freely for demos, learning, and interviews.
