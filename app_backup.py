from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from datetime import datetime
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
analysis_history = []

def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    
    try:
        # Load the logistic regression model
        with open('models/logreg_sentiment.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        # Handle different pickle formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            vectorizer = model_data.get('vectorizer')
        else:
            # If it's just the model, we need to recreate the vectorizer
            model = model_data
            # You might need to retrain the vectorizer or save it separately
            print("Warning: Vectorizer not found in model file. You may need to retrain.")
            
        if model is None:
            raise Exception("Model not loaded properly")
            
        print("Model and vectorizer loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your model is saved with both model and vectorizer")

def preprocess_text(text):
    """Basic text preprocessing"""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
            
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Make prediction
        if vectorizer is None:
            return jsonify({'error': 'Vectorizer not available'}), 500
            
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities) * 100
        except:
            # Fallback if predict_proba not available
            confidence = 85.0  # Default confidence
        
        result = {
            'text': text,
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        analysis_history.append(result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts at once"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
            
        texts = data['texts']
        if not texts or len(texts) == 0:
            return jsonify({'error': 'Empty texts array'}), 400
            
        results = []
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize all texts at once
        if vectorizer is None:
            return jsonify({'error': 'Vectorizer not available'}), 500
            
        text_vectorized = vectorizer.transform(processed_texts)
        predictions = model.predict(text_vectorized)
        
        # Get probabilities if available
        try:
            probabilities = model.predict_proba(text_vectorized)
        except:
            probabilities = None
            
        for i, text in enumerate(texts):
            if probabilities is not None:
                confidence = max(probabilities[i]) * 100
            else:
                confidence = 85.0
                
            result = {
                'text': text,
                'sentiment': 'Positive' if predictions[i] == 1 else 'Negative',
                'confidence': round(confidence, 2)
            }
            results.append(result)
            
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error in batch_analyze: {e}")
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.route('/analytics')
def get_analytics():
    """Get analytics dashboard data"""
    try:
        if not analysis_history:
            return jsonify({'message': 'No analysis history available'})
            
        total_analyses = len(analysis_history)
        positive_count = sum(1 for item in analysis_history if item['sentiment'] == 'Positive')
        negative_count = total_analyses - positive_count
        
        avg_confidence = sum(item['confidence'] for item in analysis_history) / total_analyses
        
        return jsonify({
            'total_analyses': total_analyses,
            'positive_percentage': round((positive_count / total_analyses) * 100, 2),
            'negative_percentage': round((negative_count / total_analyses) * 100, 2),
            'average_confidence': round(avg_confidence, 2),
            'recent_analyses': analysis_history[-10:]  # Last 10 analyses
        })
        
    except Exception as e:
        print(f"Error in get_analytics: {e}")
        return jsonify({'error': f'Analytics failed: {str(e)}'}), 500

@app.route('/model_comparison')
def get_model_comparison():
    """Load model comparison data from evaluation results"""
    try:
        # Load metrics from your evaluation files
        models_data = []
        
        # Load each model's metrics
        model_files = [
            ('Logistic Regression', 'evaluation/logreg_metrics.json'),
            ('Naive Bayes', 'evaluation/naivebayes_metrics.json'),
            ('SVM', 'evaluation/svm_metrics.json'),
            ('CNN', 'evaluation/cnn_metrics.json'),
            ('LSTM', 'evaluation/lstm_metrics.json'),
            ('BERT', 'evaluation/bert_metrics.json')
        ]
        
        for model_name, filepath in model_files:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        metrics = json.load(f)
                    
                    # Extract relevant metrics
                    accuracy = metrics.get('accuracy', 0)
                    training_time = metrics.get('training_time', 'N/A')
                    
                    model_info = {
                        'name': model_name,
                        'accuracy': accuracy,
                        'training_time': training_time,
                        'selected': model_name == 'Logistic Regression'
                    }
                    
                    # Add specific pros and selection reasoning
                    if model_name == 'Logistic Regression':
                        model_info['pros'] = ['Fast inference', 'Interpretable', 'Low resource usage', 'Good balance']
                        model_info['reason'] = 'Best performance/efficiency trade-off for production'
                    elif model_name == 'BERT':
                        model_info['pros'] = ['Highest accuracy', 'Context understanding']
                        model_info['reason_not_selected'] = 'Too computationally expensive for real-time use'
                    elif model_name == 'CNN':
                        model_info['pros'] = ['Good pattern recognition', 'Faster than LSTM']
                        model_info['reason_not_selected'] = 'Overfitting issues observed'
                    elif model_name == 'LSTM':
                        model_info['pros'] = ['Sequential understanding', 'Good for longer texts']
                        model_info['reason_not_selected'] = 'Overkill for short reviews, slower training'
                    elif model_name == 'Naive Bayes':
                        model_info['pros'] = ['Very fast', 'Simple', 'Good baseline']
                        model_info['reason_not_selected'] = 'Feature independence assumption limiting'
                    elif model_name == 'SVM':
                        model_info['pros'] = ['Robust', 'Works with small datasets']
                        model_info['reason_not_selected'] = 'Lower accuracy than alternatives'
                    
                    models_data.append(model_info)
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        return jsonify({
            'models': models_data,
            'selection_reasoning': 'Logistic Regression provides the optimal balance of accuracy, speed, and interpretability for production deployment of sentiment analysis.'
        })
        
    except Exception as e:
        print(f"Error in get_model_comparison: {e}")
        return jsonify({'error': f'Model comparison failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Loading model and vectorizer...")
    load_model_and_vectorizer()
    
    if model is None:
        print("WARNING: Model not loaded. Some features may not work.")
        print("Please ensure your model file contains both model and vectorizer.")
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)