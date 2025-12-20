export async function analyzeResume(resume: File, jobDescription: string) {
  const formData = new FormData();
  formData.append("resume", resume);
  formData.append("job_description", jobDescription);

  const res = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Failed to analyze");
  }

  return res.json();
}
