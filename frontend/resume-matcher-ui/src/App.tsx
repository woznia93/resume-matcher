import { useState } from "react";

function App() {
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();

    if (!resume) {
      setError("Please upload a resume PDF.");
      return;
    }

    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("resume", resume);
      formData.append("job_description", jobDescription);

      const res = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Server error");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Resume ↔ Job Description Matcher</h1>

      <div style={{ marginBottom: "1rem" }}>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setResume(e.target.files?.[0] || null)}
        />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <textarea
          rows={10}
          cols={60}
          placeholder="Paste job description here"
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />
      </div>

      <button type="button" onClick={submit} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: "2rem" }}>
          <h2>Results</h2>
          <p><strong>Match Score:</strong> {result.match_score}</p>
          <p><strong>Resume Skills:</strong> {result.resume_skills.join(", ")}</p>
          <p><strong>Job Skills:</strong> {result.job_skills.join(", ")}</p>
          <p><strong>Missing Skills:</strong> {result.missing_skills.join(", ")}</p>
          <p><strong>Resume Preview:</strong> {result.resume_preview}</p>
        </div>
      )}
    </div>
  );
}

export default App;
