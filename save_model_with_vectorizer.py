# save_model_with_vectorizer.py - Run this once to fix model format
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load your existing model
try:
    with open('models/logreg_sentiment.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # If it's just the model, you'll need to recreate the vectorizer
    # This assumes you have your preprocessing code available
    print("Loaded existing model successfully")
    print("Type:", type(model_data))
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("You may need to retrain with vectorizer included")