import pytest
from app.keywords import extract_keywords, keyword_overlap
from app.analyzer import _score_to_grade, analyze
from app.parser import clean_text

SAMPLE_RESUME = """
Software Engineer with 3 years of experience in Python, machine learning, and FastAPI.
Skilled in Docker, Kubernetes, PostgreSQL, and REST APIs.
Built NLP pipelines using sentence-transformers, spaCy, and scikit-learn.
Education: B.S. Computer Science, State University 2021.
Projects: real-time data pipeline, recommendation system using collaborative filtering.
"""

SAMPLE_JD = """
We are looking for a Python Software Engineer with experience in machine learning
and NLP. Requirements include FastAPI, Docker, PostgreSQL, sentence-transformers,
scikit-learn, and REST API design. Knowledge of transformer models a plus.
"""

UNRELATED_JD = """
Seeking a licensed plumber with 5 years of pipe fitting experience.
Must know water heater installation and building codes. CDL license preferred.
"""


class TestGrading:
    def test_grade_a(self):
        assert _score_to_grade(90) == "A"

    def test_grade_b(self):
        assert _score_to_grade(78) == "B"

    def test_grade_c(self):
        assert _score_to_grade(62) == "C"

    def test_grade_f(self):
        assert _score_to_grade(30) == "F"


class TestKeywords:
    def test_extract_returns_list(self):
        kws = extract_keywords(SAMPLE_RESUME)
        assert isinstance(kws, list)
        assert len(kws) > 0

    def test_bigrams_captured(self):
        kws = extract_keywords(SAMPLE_RESUME)
        assert "machine learning" in kws

    def test_overlap(self):
        resume_kws = extract_keywords(SAMPLE_RESUME)
        job_kws = extract_keywords(SAMPLE_JD)
        matched, missing, extra = keyword_overlap(resume_kws, job_kws)
        assert isinstance(matched, list)
        assert "python" in matched or "fastapi" in matched

    def test_no_stopwords_in_keywords(self):
        kws = extract_keywords(SAMPLE_RESUME)
        stopwords = {"the", "and", "or", "is", "a", "with"}
        assert not any(k in stopwords for k in kws)


class TestCleanText:
    def test_collapses_whitespace(self):
        result = clean_text("hello   world\n\nfoo")
        assert "  " not in result

    def test_strips(self):
        result = clean_text("  hello  ")
        assert result == "hello"


class TestAnalyzer:
    def test_high_score_for_matching_resume(self):
        result = analyze(SAMPLE_RESUME, SAMPLE_JD)
        assert result.match_score > 60, f"Expected high score, got {result.match_score}"

    def test_low_score_for_unrelated(self):
        result = analyze(SAMPLE_RESUME, UNRELATED_JD)
        assert result.match_score < 70, f"Expected low score, got {result.match_score}"

    def test_response_structure(self):
        result = analyze(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(result.matched_keywords, list)
        assert isinstance(result.missing_keywords, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.section_scores, dict)
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_recommendations_present(self):
        result = analyze(SAMPLE_RESUME, SAMPLE_JD)
        assert len(result.recommendations) >= 1
