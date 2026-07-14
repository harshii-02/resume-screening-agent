"""Streamlit UI for recruiters — optional; CLI remains the primary reviewer path."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import screen_resumes
from config import JD_PATH, RESUME_DIR
from llm import llm_status

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("Resume Screening Agent")
st.caption(
    "Upload a JD + resumes (or use bundled samples). "
    "Ranks candidates with NLP similarity + explainable scores."
)
st.write(f"**LLM status:** `{llm_status()}`")

mode = st.radio("Input source", ["Bundled samples", "Upload files"], horizontal=True)

if mode == "Bundled samples":
    jd_path = JD_PATH
    resume_path = RESUME_DIR
    st.info(f"Using `{jd_path.name}` and `{resume_path.name}/`")
else:
    jd_file = st.file_uploader("Job description (.txt / .pdf / .docx)", type=["txt", "pdf", "docx"])
    resume_files = st.file_uploader(
        "Resumes (.pdf / .docx / .txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if not jd_file or not resume_files:
        st.stop()
    tmpdir = Path(tempfile.mkdtemp(prefix="rsa_"))
    jd_path = tmpdir / jd_file.name
    jd_path.write_bytes(jd_file.getvalue())
    resume_path = tmpdir / "resumes"
    resume_path.mkdir()
    for rf in resume_files:
        (resume_path / rf.name).write_bytes(rf.getvalue())

if st.button("Screen candidates", type="primary"):
    progress = st.progress(0.0, text="Starting…")

    def _cb(idx: int, total: int, name: str) -> None:
        progress.progress(idx / max(total, 1), text=f"[{idx}/{total}] {name}")

    with st.spinner("Running agent…"):
        payload = screen_resumes(resume_path, jd_path, output_dir=None, progress_callback=_cb)
    progress.empty()

    rows = payload["candidates"]
    if not rows:
        st.error("No candidates scored.")
        st.stop()

    st.subheader(f"Results — {payload['job'].get('title', 'Role')}")
    table = pd.DataFrame(
        [
            {
                "Rank": r["rank"],
                "Candidate": r["candidate"],
                "Score": r["score"],
                "Similarity": r["similarity"],
                "Skills": r["skills_score"],
                "Experience": r["experience_score"],
                "Education": r["education_score"],
                "Matched": ", ".join(r["matched_skills"]),
                "Missing": ", ".join(r["missing_skills"]),
                "Reason": r["reason"],
            }
            for r in rows
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Why this ranking order")
    for line in payload.get("pairwise_explanations", [])[:8]:
        st.markdown(f"- {line}")

    st.download_button(
        "Download JSON",
        data=json.dumps(payload, indent=2),
        file_name="ranked_candidates.json",
        mime="application/json",
    )
    st.download_button(
        "Download CSV",
        data=table.to_csv(index=False),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )
