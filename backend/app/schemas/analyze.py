from pydantic import BaseModel
from typing import List

class AnalyzeResponse(BaseModel):
    match_score: float
    resume_skills: List[str]
    job_skills: List[str]
    missing_skills: List[str]
    resume_preview: str
