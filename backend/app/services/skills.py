SKILLS = [
    "Python", "JavaScript", "TypeScript", "C++",
    "React", "FastAPI", "SQL", "Docker", "AWS",
    "PyTorch", "Machine Learning", "Git", "Linux"
]

def extract_skills(text: str):
    text = text.lower()
    return sorted({s for s in SKILLS if s.lower() in text})
