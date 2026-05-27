from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.analyzer import analyze, is_model_loaded
from app.models import AnalysisRequest, AnalysisResponse, HealthResponse
from app.parser import clean_text, extract_text_from_pdf

app = FastAPI(
    title="AI Resume Analyzer",
    description="Semantic resume-to-job-description matching using sentence transformers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=is_model_loaded())


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_text(request: AnalysisRequest):
    try:
        return analyze(
            clean_text(request.resume_text),
            clean_text(request.job_description),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/pdf", response_model=AnalysisResponse)
async def analyze_pdf(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., min_length=50),
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_bytes = await resume.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {e}")

    if len(resume_text) < 50:
        raise HTTPException(status_code=422, detail="Could not extract enough text from the PDF")

    try:
        return analyze(clean_text(resume_text), clean_text(job_description))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
