from pydantic import BaseModel, Field
from typing import List, Optional


class AnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Full resume text")
    job_description: str = Field(..., min_length=50, description="Full job description text")


class KeywordMatch(BaseModel):
    keyword: str
    in_resume: bool
    in_job: bool


class AnalysisResponse(BaseModel):
    match_score: float = Field(..., description="Semantic similarity score 0-100")
    grade: str = Field(..., description="Letter grade based on score")
    matched_keywords: List[str]
    missing_keywords: List[str]
    extra_keywords: List[str]
    recommendations: List[str]
    section_scores: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
