from flask import Flask, render_template, request
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean(text):
    text = text.lower()                         # make all text lowercase
    text = re.sub(r'[^a-z\s]', '', text)        # remove special chars, punctuation, numbers
    tokens = text.split()                       # split text by words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

app = Flask(__name__)

# Load your trained model
model = joblib.load("models/logreg_sentiment.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        review = request.form.get("input")

        # Build the required input format
        input_data = {
            'cleaned_text': [clean(review)],
            'has_upvotes': [0],
            'review_length': [len(review.split())],
            'exclamation_count': [review.count('!')],
            'question_count': [review.count('?')],
            'ngram_great app': [0],
            'ngram_good app': [0],
            'ngram_easy use': [0],
            'ngram_love app': [0],
            'ngram_pro version': [0],
            'ngram_google calendar': [0],
            'ngram_free version': [0],
            'ngram_use app': [0],
            'ngram_like app': [0],
            'ngram_doesnt work': [0],
            'ngram_really like app': [0],
            'ngram_app easy use': [0],
            'ngram_buy pro version': [0],
            'ngram_using app years': [0],
            'ngram_paid pro version': [0],
            'ngram_really good app': [0],
            'ngram_simple easy use': [0],
            'ngram_used app years': [0],
            'ngram_sync google calendar': [0],
            'ngram_todo list app': [0]
        }

        input_df = pd.DataFrame(input_data)
        prediction = model.predict(input_df)[0]
        label = "Positive" if prediction == 1 else "Negative"
        
        # calculate confidence
        prediction_proba = model.predict_proba(input_df)[0]
        confidence = max(prediction_proba) * 100
        
        # insight from predicted confidence 
        if confidence >= 85:
            insight = "High confidence: clear sentiment, recommended for automated analysis"
        elif confidence >= 60:
            insight = "Medium confidence: mixed signals, consider human review"
        else:
            insight = "Low confidence: unclear sentiment, human review recommended for analysis"
            
        return render_template("result.html", prediction=label, confidence=f"{confidence:.1f}%", insight=insight)

    return "Something went wrong."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)