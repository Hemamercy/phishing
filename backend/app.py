from flask import Flask, request, jsonify
import joblib
import pandas as pd
from phishing import extract_advanced_features, normalize_url
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load ensemble model
artifact = joblib.load("models/ensemble.joblib")
model = artifact['model']
scaler = artifact['scaler']
selector = artifact['feature_selector']
all_features = artifact['all_features']

# Label mapping
LABEL_MAP = {0: "benign", 1: "phishing", 2: "defacement", 3: "malware"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract features
    norm_url = normalize_url(url)
    features = extract_advanced_features(norm_url)
    X = pd.DataFrame([features], columns=all_features)

    # Apply preprocessing
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    # Predict
    pred = model.predict(X_selected)[0]
    label = LABEL_MAP[pred]

    return jsonify({"url": url, "prediction": label})


if __name__ == "__main__":
    app.run(debug=True)
