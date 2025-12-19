from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

SKILLS = [
    "Python", "JavaScript", "TypeScript", "C++",
    "React", "FastAPI", "SQL", "Docker", "AWS",
    "PyTorch", "Machine Learning", "Git", "Linux"
]

def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return " ".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception:
        return ""
    
def extract_skills(text: str):
    text_lower = text.lower()
    found = []

    for skill in SKILLS:
        if skill.lower() in text_lower:
            found.append(skill)

    return sorted(set(found))



@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    resume_text = extract_text_from_pdf(resume.file)

    if not resume_text:
        return HTTPException(status_code=400, detail = "Could not extract text from resume")

    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    missing_skills = list(set(job_skills) - set(resume_skills))



    similarity = cosine_similarity(
        [resume_emb], [job_emb]
    )[0][0]

    return {
        "match_score": round(float(similarity), 3),
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "missing_skills": missing_skills,
        "resume_preview": resume_text[:300]
    }
