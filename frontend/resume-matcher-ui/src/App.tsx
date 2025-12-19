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
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white overflow-x-hidden">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-80 h-80 bg-red-600/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-red-600/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 w-full h-full flex flex-col items-center justify-center px-4 py-8">
        <div className="w-full max-w-2xl">
          <div className="text-center mb-16">
            <h1 className="text-5xl md:text-6xl font-black mb-4 leading-tight">
              <span className="bg-gradient-to-r from-red-400 via-red-500 to-red-600 bg-clip-text text-transparent drop-shadow-lg">
                Resume Matcher
              </span>
            </h1>
            <p className="text-lg text-gray-300 max-w-xl mx-auto mb-2 font-light">
              Discover your perfect job fit instantly
            </p>
            <p className="text-sm text-gray-500">
              AI-powered analysis of your resume against job requirements
            </p>
          </div>

          <div className="mb-12 bg-gray-900/60 backdrop-blur-xl border border-red-500/20 rounded-3xl p-8 md:p-10 shadow-2xl shadow-black/50 hover:border-red-500/40 transition-colors duration-300">
            <div className="mb-8">
              <label className="block text-base font-semibold text-gray-200 mb-3 flex items-center justify-center">
                Upload Resume (PDF)
              </label>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setResume(e.target.files?.[0] || null)}
                className="w-full px-4 py-4 bg-gray-800/50 border-2 border-dashed border-gray-600 hover:border-red-500 rounded-2xl text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-bold file:bg-gradient-to-r file:from-red-600 file:to-red-700 file:text-white hover:file:from-red-700 hover:file:to-red-800 cursor-pointer transition-all duration-300 file:cursor-pointer"
              />
              {resume && (
                <div className="mt-3 p-3 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center gap-2 justify-center">
                  <span className="text-green-400">✓</span>
                  <span className="text-sm text-green-300 font-medium">{resume.name}</span>
                </div>
              )}
            </div>

            <div className="mb-8">
              <label className="block text-base font-semibold text-gray-200 mb-3 flex items-center justify-center">
                Paste Job Description
              </label>
              <textarea
                rows={7}
                placeholder="Paste the complete job description here..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                className="w-full px-4 py-4 bg-gray-800/50 border-2 border-gray-600 rounded-2xl text-white placeholder-gray-500 focus:outline-none focus:border-red-500 focus:ring-2 focus:ring-red-500/20 transition-all duration-300 resize-none"
              />
            </div>

            <button
              type="button"
              onClick={submit}
              disabled={loading}
              className="w-full bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 disabled:from-gray-700 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-2xl transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-red-500/50 disabled:scale-100 disabled:cursor-not-allowed disabled:shadow-none flex items-center justify-center gap-3 text-lg"
            >
              {loading ? (
                <>
                  <span className="inline-flex h-5 w-5 animate-spin rounded-full border-3 border-white border-t-red-300"></span>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>Analyze Resume</>
              )}
            </button>

            {error && (
              <div className="mt-6 p-4 bg-red-900/20 border-2 border-red-500/40 rounded-2xl">
                <p className="text-red-300 text-sm font-medium">{error}</p>
              </div>
            )}
          </div>

          {result && (
            <div className="space-y-8 animate-fade-in">
              <div className="bg-gray-900/60 backdrop-blur-xl border border-red-500/20 rounded-3xl p-8 md:p-10 shadow-2xl shadow-black/50">
                <div className="text-center mb-10">
                  <h2 className="text-4xl font-black bg-gradient-to-r from-red-400 to-red-600 bg-clip-text text-transparent">
                    Your Analysis
                  </h2>
                </div>

                <div className="mb-10 p-8 text-center bg-gradient-to-br from-red-600/15 to-red-700/15 border-2 border-red-500/30 rounded-3xl">
                  <p className="text-sm uppercase tracking-widest text-gray-400 font-bold mb-2">
                    Overall Match Score
                  </p>
                  <p className="text-7xl md:text-8xl font-black text-transparent bg-gradient-to-r from-red-400 to-red-600 bg-clip-text mb-6">
                    {(result.match_score * 100).toFixed(0)}%
                  </p>
                  <div className="w-full bg-gray-700/30 rounded-full h-3 overflow-hidden border border-red-500/30">
                    <div
                      className="bg-gradient-to-r from-red-500 to-red-600 h-full transition-all duration-1000 ease-out shadow-lg shadow-red-500/60"
                      style={{ width: `${(result.match_score * 100).toFixed(1)}%` }}
                    ></div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="bg-gray-800/50 border border-green-500/20 rounded-2xl p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">Your Skills</h3>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {result.resume_skills.map((skill: string, idx: number) => (
                        <span
                          key={idx}
                          className="px-4 py-2 bg-green-500/20 border border-green-500/50 text-green-300 text-sm font-medium rounded-full hover:bg-green-500/30 transition-colors"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gray-800/50 border border-blue-500/20 rounded-2xl p-6">
                    <h3 className="text-lg font-bold text-blue-400 mb-4">Required Skills</h3>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {result.job_skills.map((skill: string, idx: number) => (
                        <span
                          key={idx}
                          className="px-4 py-2 bg-blue-500/20 border border-blue-500/50 text-blue-300 text-sm font-medium rounded-full hover:bg-blue-500/30 transition-colors"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>

                  {result.missing_skills.length > 0 && (
                    <div className="bg-gray-800/50 border border-orange-500/20 rounded-2xl p-6">
                      <h3 className="text-lg font-bold text-orange-400 mb-4">Skills to Learn</h3>
                      <div className="flex flex-wrap gap-2 justify-center">
                        {result.missing_skills.map((skill: string, idx: number) => (
                          <span
                            key={idx}
                            className="px-4 py-2 bg-orange-500/20 border border-orange-500/50 text-orange-300 text-sm font-medium rounded-full hover:bg-orange-500/30 transition-colors"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="bg-gray-800/50 border border-gray-600/30 rounded-2xl p-6">
                    <h3 className="text-lg font-bold text-gray-200 mb-4">Resume Summary</h3>
                    <p className="text-gray-300 leading-relaxed text-center">{result.resume_preview}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="h-8"></div>
        </div>
      </div>
    </div>
  );
}

export default App;
