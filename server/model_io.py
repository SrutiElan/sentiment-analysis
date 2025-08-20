import joblib
from pathlib import Path
from typing import Tuple
import pandas as pd

_MODEL_PATH = Path("models/logreg_sentiment.pkl")
_PIPELINE = None  # lazy-loaded sklearn pipeline

def load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = joblib.load(_MODEL_PATH)
    return _PIPELINE

def predict_proba(df: pd.DataFrame) -> float:
    pipe = load_pipeline()
    # pipe should be a sklearn Pipeline ending with a classifier
    proba_fn = getattr(pipe, "predict_proba", None)
    if proba_fn is None:
        # Some classifiers lack predict_proba
        scores = pipe.decision_function(df)
        # map arbitrary score to [0,1] via logistic-ish transform if needed
        import numpy as np
        p = 1 / (1 + np.exp(-scores))[0]
        return float(p)
    return float(proba_fn(df)[0, 1])

def predict_label(df: pd.DataFrame) -> str:
    p = predict_proba(df)
    return "Positive" if p >= 0.5 else "Negative", p
