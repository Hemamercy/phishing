import { useState } from "react";
import type { FormEvent } from "react";

interface PredictionResponse {
  url: string;
  prediction: string;
}

function App() {
  const [url, setUrl] = useState<string>("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data: PredictionResponse = await response.json();
      setResult(data.prediction);
    } catch (error) {
      console.error("Error:", error);
      setResult("Error contacting backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-8">
      <h1 className="text-3xl font-bold mb-8 text-blue-700">
        Phishing URL Detector
      </h1>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col sm:flex-row items-center gap-4 mb-6 w-full max-w-md"
      >
        <input
          type="text"
          placeholder="Enter a URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
        <button
          type="submit"
          className="px-6 py-2 bg-blue-600 text-white rounded-md font-semibold hover:bg-blue-700 transition-colors"
        >
          Check
        </button>
      </form>

      {loading && <p className="text-gray-500 text-lg">Checking...</p>}

      {result && !loading && (
        <h2 className="mt-6 text-xl font-medium text-gray-800">
          Prediction: <span className="font-bold text-blue-600">{result}</span>
        </h2>
      )}
    </div>
  );
}

export default App;
