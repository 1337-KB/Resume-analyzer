# AI Resume Analyzer

An AI-powered tool that compares resumes against job descriptions using semantic embeddings and NLP. Instead of simple keyword matching, it understands meaning — so "built REST APIs with Python" will match a job that asks for "backend development experience."

---

## Features

- **Semantic similarity scoring** using `sentence-transformers` (cosine similarity on 384-dim embeddings)
- **Per-section scoring** — breaks your resume into Skills, Experience, Education, and Projects and scores each section independently
- **Keyword gap analysis** — identifies which job keywords are missing from your resume
- **Actionable recommendations** — tells you exactly what to improve and where
- **PDF resume support** — upload a PDF directly or pass raw text
- **REST API** — built with FastAPI, fully documented at `/docs`
- **Dockerized** — runs anywhere with a single command

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Similarity | scikit-learn (cosine similarity) |
| PDF parsing | pdfplumber |
| Validation | Pydantic v2 |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
resume-analyzer/
├── app/
│   ├── main.py          # FastAPI app and route definitions
│   ├── analyzer.py      # Core ML logic — embeddings, similarity, section scoring
│   ├── keywords.py      # Keyword extraction and overlap analysis
│   ├── parser.py        # PDF text extraction and text cleaning
│   └── models.py        # Pydantic request/response schemas
├── tests/
│   └── test_analyzer.py # 14 unit tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Getting Started

### Option 1 — Run Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the server**
```bash
uvicorn app.main:app --reload
```

**3. Open the interactive docs**
```
http://localhost:8000/docs
```

---

### Option 2 — Run with Docker

```bash
docker compose up --build
```

The model is pre-downloaded during the Docker build, so the container starts up immediately with no cold-start delay.

---

## API Reference

### `GET /health`
Check that the server is up and the model is loaded.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /analyze`
Analyze resume text against a job description.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with 3 years experience in FastAPI, Docker, and machine learning...",
    "job_description": "Looking for a Python engineer with FastAPI, Docker, and NLP experience..."
  }'
```

---

### `POST /analyze/pdf`
Upload a PDF resume with a job description.

```bash
curl -X POST http://localhost:8000/analyze/pdf \
  -F "resume=@/path/to/resume.pdf" \
  -F "job_description=Looking for a Python engineer with FastAPI and NLP experience..."
```

---

### Sample Response

```json
{
  "match_score": 82.4,
  "grade": "B",
  "matched_keywords": ["python", "fastapi", "docker", "machine learning", "rest api"],
  "missing_keywords": ["kubernetes", "redis", "ci cd"],
  "extra_keywords": ["tableau", "excel"],
  "recommendations": [
    "Add these missing keywords to strengthen your resume: kubernetes, redis, ci cd",
    "Strengthen your 'projects' section — it has low relevance to this role."
  ],
  "section_scores": {
    "skills": 88.1,
    "experience": 79.4,
    "education": 61.2,
    "projects": 47.8,
    "general": 74.0
  }
}
```

### Score Grading

| Score | Grade | Meaning |
|---|---|---|
| 85 – 100 | A | Excellent match |
| 75 – 84 | B | Strong match |
| 60 – 74 | C | Moderate match — some gaps |
| 45 – 59 | D | Weak match — significant gaps |
| 0 – 44 | F | Poor match |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: **14 passed**

Tests cover grading logic, keyword extraction, stopword filtering, bigram detection, text cleaning, semantic scoring (high match vs. unrelated), and full response structure validation.

---

## Notes

- **First run** downloads the `all-MiniLM-L6-v2` model (~90MB) from HuggingFace and caches it locally. Subsequent runs load from cache instantly.
- **AMD GPU users** — `sentence-transformers` runs on CPU by default, which is fast enough for inference. ROCm/DirectML GPU acceleration can be enabled by installing a ROCm-compatible PyTorch build separately.
- **PDF file limit** — the `/analyze/pdf` endpoint enforces a 5MB maximum file size.
- **Windows symlink warning** — HuggingFace Hub shows a symlink warning on Windows. This is cosmetic and does not affect functionality. To suppress it, enable Developer Mode in Windows Settings or run Python as Administrator.
