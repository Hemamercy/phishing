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
    <div style={{ padding: "2rem" }}>
      <h1>Phishing URL Detector</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter a URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={{ width: "300px", marginRight: "10px" }}
        />
        <button type="submit">Check</button>
      </form>

      {loading && <p>Checking...</p>}

      {result && !loading && (
        <h2 style={{ marginTop: "20px" }}>
          Prediction: <span>{result}</span>
        </h2>
      )}
    </div>
  );
}

export default App;
