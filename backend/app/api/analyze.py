from fastapi import APIRouter, UploadFile, File, Form
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.model import get_model
from app.services.pdf import extract_text
from app.services.skills import extract_skills
from app.schemas.analyze import AnalyzeResponse

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    model = get_model()

    resume_text = extract_text(resume.file)

    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    score = cosine_similarity([resume_emb], [job_emb])[0][0]

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    return AnalyzeResponse(
        match_score=round(float(score), 3),
        resume_skills=resume_skills,
        job_skills=job_skills,
        missing_skills=list(set(job_skills) - set(resume_skills)),
        resume_preview=resume_text[:300]
    )
