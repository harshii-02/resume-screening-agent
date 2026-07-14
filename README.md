# Resume Screening Agent

> **One job:** My agent takes a **job description + a folder (or file) of resumes** and produces a **ranked shortlist with scores, skill gaps, and reasons**.

End-to-end AI agent for the Rooman / HireAI 24-hour challenge. Reviewers can run a **CLI** (primary) or optional **Streamlit UI**.

It combines:

- **File parsing** (PDF / DOCX / TXT) via PyMuPDF + python-docx  
- **LLM structured extraction** (skills, experience, education, projects) — Groq or OpenAI  
- **Heuristic fallback** if no API key (still fully runnable)  
- **Embeddings + cosine similarity** (`all-MiniLM-L6-v2`)  
- **Weighted ranking** + pairwise “why A above B” explanations  
- **CSV + JSON** outputs  

---

## Quick start (Windows — foolproof)

```powershell
cd resume-screening-agent
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python main.py
```

**Must activate `.venv`** before `python main.py` or you will get `ModuleNotFoundError: fitz`.

### Optional LLM brain (recommended)

1. Free key: [Groq Console](https://console.groq.com/keys) → put in `.env` as `GROQ_API_KEY=...`  
2. Or OpenAI → `OPENAI_API_KEY=...`  
3. Re-run `python main.py`

Without a key, embeddings + heuristics still produce a complete ranked shortlist.

### Optional web UI

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Opens at `http://localhost:8501`. **Reviewers do not need the UI** — `python main.py` alone is enough to score the submission.

---

## Agent loop (Input → Think → Act → Output)

```
User input: JD file + resume folder/file
        │
        ▼
Tool: extract text (PDF/DOCX/TXT)
        │
        ▼
Think: LLM system prompts extract skills/exp/education
       (heuristic fallback if no API key)
        │
        ▼
Think: SentenceTransformers embeddings + cosine similarity
       + skill/experience/education component scores
        │
        ▼
Act: rank candidates, write pairwise explanations
        │
        ▼
Output: ranked_candidates.csv + ranked_candidates.json
        (+ Streamlit table if using the UI)
```

Core orchestration lives in `agent.py` (shared by CLI and UI). System prompts live in `llm.py`.

---

## Scoring method (rubric deliverable)

| Component            | Weight | Method |
| -------------------- | ------ | ------ |
| Embedding similarity | 50%    | Cosine similarity of MiniLM embeddings of structured resume profile vs JD |
| Skill match          | 25%    | % of **required** JD skills present in extracted candidate skills |
| Experience           | 15%    | Candidate years vs JD minimum (capped at 100) |
| Education            | 10%    | Keyword overlap (CS, Bachelor, Master, …) |

```
Final score =
  0.50 × Similarity
+ 0.25 × Skill Match
+ 0.15 × Experience
+ 0.10 × Education
```

**Why this model mix**

- **MiniLM (`all-MiniLM-L6-v2`)** — fast local NLP similarity; reproducible for reviewers; no embedding API quota.  
- **Deterministic skill/exp/edu scores** — explainable and auditable (important for hiring).  
- **LLM (Groq/OpenAI)** — better structured extraction + recruiter narrative; optional so the demo never blocks on billing.  

Pairwise explanations answer: *why Candidate A ranked above Candidate B* using score-component deltas.

---

## Folder structure

```
resume-screening-agent/
├── README.md
├── setup.ps1
├── requirements.txt
├── .env.example
├── main.py              # CLI entrypoint
├── app.py               # Streamlit UI (optional)
├── agent.py             # Shared screening loop
├── config.py
├── extract.py           # PDF / DOCX / TXT tools
├── llm.py               # System prompts + Groq/OpenAI + heuristics
├── ranking.py           # Embeddings + scoring
├── utils.py
├── tests/test_scoring.py
├── JD/job_description.txt
├── sample_resumes/      # 13 samples (TXT + DOCX + PDF)
└── outputs/
    ├── ranked_candidates.csv
    └── ranked_candidates.json
```

---

## Installation (manual)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python tests/test_scoring.py
python main.py
```

First run downloads the MiniLM model (~80MB).

### Environment variables

| Variable | Required | Default | Description |
| -------- | -------- | ------- | ----------- |
| `GROQ_API_KEY` | Optional | empty | Free LLM via Groq |
| `OPENAI_API_KEY` | Optional | empty | OpenAI LLM |
| `LLM_PROVIDER` | Optional | `auto` | `auto` / `groq` / `openai` |
| `RESUME_DIR` | Optional | `sample_resumes` | Resume folder |
| `JD_PATH` | Optional | `JD/job_description.txt` | JD file |
| `OUTPUT_DIR` | Optional | `outputs` | Output folder |
| `WEIGHT_*` | Optional | see scoring table | Tune formula |

---

## How to run

```powershell
# Sample demo (13 resumes)
python main.py

# Custom folder or single file
python main.py --resumes path\to\resumes --jd path\to\jd.txt
python main.py --resumes path\to\one_resume.pdf --jd JD\job_description.txt

# UI
streamlit run app.py
```

---

## Sample output

| Rank | Candidate | Score |
| ---- | --------- | ----- |
| 1 | Alice Chen | ~90 |
| 2 | Priya Nair | ~88 |
| 3 | Chris Patel | ~88 |
| … | … | … |

See committed files:

- `outputs/ranked_candidates.csv`
- `outputs/ranked_candidates.json` (includes `pairwise_explanations`)

---

## Agent-specific deliverables checklist

- [x] Job description (`JD/job_description.txt`)  
- [x] Folder of 10+ sample resumes (`sample_resumes/`)  
- [x] Ranked CSV + JSON (`outputs/`)  
- [x] Scoring method note (this README)  

---

## Tradeoffs & limitations

| Choice | Why | Tradeoff / limit |
| ------ | --- | ---------------- |
| Local MiniLM embeddings | Free, fast, reproducible for reviewers | Less semantic depth than large cloud embedding models |
| Weighted transparent score | Auditable hiring decisions; easy to defend in interview | Weights are heuristics, not learned from labeled hire data |
| Optional LLM (Groq/OpenAI) | Demo never blocks without a paid key | Extraction quality drops for messy/scanned resumes without LLM |
| No OCR | Keeps deps light | Image-only / scanned PDFs extract poorly |
| Streamlit optional UI | Recruiter-friendly demo without rewriting core agent | CLI remains the authoritative path for automated review |
| Batch screening (not chat Q&A) | Matches Resume Screening job | Not an interactive multi-turn chatbot |

### What I’d improve with more time

- Cross-encoder re-rank of top-k candidates  
- OCR for scanned PDFs  
- Bias checks on names/demographics  
- Learned weights from recruiter feedback  
- CI that runs `tests/test_scoring.py` + smoke `main.py --limit 2`  

---

## License

MIT
