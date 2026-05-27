from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.keywords import extract_keywords, keyword_overlap
from app.models import AnalysisResponse

MODEL_NAME = "all-MiniLM-L6-v2"

# Section headers to split resume into logical parts for per-section scoring
SECTION_PATTERNS = {
    "skills": re.compile(r"\b(skills?|technologies|tech stack|tools?|languages?)\b", re.I),
    "experience": re.compile(r"\b(experience|work history|employment|positions?)\b", re.I),
    "education": re.compile(r"\b(education|degree|university|college|academic)\b", re.I),
    "projects": re.compile(r"\b(projects?|portfolio|side projects?)\b", re.I),
}


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def is_model_loaded() -> bool:
    try:
        _get_model()
        return True
    except Exception:
        return False


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _score_to_grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def _extract_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {k: [] for k in SECTION_PATTERNS}
    sections["general"] = []
    current = "general"
    for line in lines:
        matched = False
        for section, pattern in SECTION_PATTERNS.items():
            if pattern.search(line):
                current = section
                matched = True
                break
        sections[current].append(line)
    return {k: " ".join(v).strip() for k, v in sections.items() if v}


def _section_scores(resume_text: str, job_text: str) -> dict[str, float]:
    resume_sections = _extract_sections(resume_text)
    scores = {}
    job_emb = _embed([job_text])
    for section, content in resume_sections.items():
        if not content.strip():
            continue
        section_emb = _embed([content])
        sim = float(cosine_similarity(section_emb, job_emb)[0][0])
        scores[section] = round(sim * 100, 1)
    return scores


def _build_recommendations(
    match_score: float,
    missing_keywords: list[str],
    section_scores: dict[str, float],
) -> list[str]:
    recs = []
    if missing_keywords:
        top_missing = missing_keywords[:5]
        recs.append(
            f"Add these missing keywords to strengthen your resume: {', '.join(top_missing)}"
        )
    if match_score < 60:
        recs.append(
            "Your resume has low semantic alignment with the job description. "
            "Consider rewriting your summary to mirror the role's language."
        )
    weak_sections = [s for s, score in section_scores.items() if score < 50]
    for section in weak_sections:
        recs.append(f"Strengthen your '{section}' section — it has low relevance to this role.")
    if not recs:
        recs.append("Your resume is a strong match. Tailor your cover letter to highlight your top skills.")
    return recs


def analyze(resume_text: str, job_description: str) -> AnalysisResponse:
    embeddings = _embed([resume_text, job_description])
    resume_emb, job_emb = embeddings[0:1], embeddings[1:2]
    similarity = float(cosine_similarity(resume_emb, job_emb)[0][0])
    match_score = round(similarity * 100, 2)

    resume_kws = extract_keywords(resume_text)
    job_kws = extract_keywords(job_description)
    matched, missing, extra = keyword_overlap(resume_kws, job_kws)

    sec_scores = _section_scores(resume_text, job_description)
    recommendations = _build_recommendations(match_score, missing, sec_scores)

    return AnalysisResponse(
        match_score=match_score,
        grade=_score_to_grade(match_score),
        matched_keywords=matched,
        missing_keywords=missing,
        extra_keywords=extra,
        recommendations=recommendations,
        section_scores=sec_scores,
    )
