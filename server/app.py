from flask import Flask, request, jsonify, render_template
from .feature_builders import build_inference_row
from .model_io import predict_label

app = Flask(__name__, template_folder="../templates", static_folder="../static")

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/predict")
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    df = build_inference_row(text)
    label, proba = predict_label(df)
    confidence = round(proba * 100, 1)
    if confidence >= 85:
        insight = "High confidence: clear sentiment, recommended for automated analysis"
    elif confidence >= 60:
        insight = "Medium confidence: mixed signals, consider human review"
    else:
        insight = "Low confidence: unclear sentiment, human review recommended"
    return jsonify({"label": label, "probability": proba, "confidence": confidence, "insight": insight})

# (Optional) keep form POST for your existing templates/result.html
@app.post("/predict")
def form_predict():
    review = request.form.get("input", "")
    df = build_inference_row(review)
    label, proba = predict_label(df)
    confidence = round(proba * 100, 1)
    if confidence >= 85:
        insight = "High confidence: clear sentiment, recommended for automated analysis"
    elif confidence >= 60:
        insight = "Medium confidence: mixed signals, consider human review"
    else:
        insight = "Low confidence: unclear sentiment, human review recommended"
    return render_template("result.html", prediction=label, confidence=f"{confidence:.1f}%", insight=insight)

if __name__ == "__main__":
    # Run from repo root: python server/app.py
    app.run(host="0.0.0.0", port=5001, debug=True)
