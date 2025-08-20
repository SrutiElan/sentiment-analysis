import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# EXACT feature names used by your model during training:
NGRAM_COLUMNS = [
    "ngram_great app", "ngram_good app", "ngram_easy use", "ngram_love app",
    "ngram_pro version", "ngram_google calendar", "ngram_free version",
    "ngram_use app", "ngram_like app", "ngram_doesnt work",
    "ngram_really like app", "ngram_app easy use", "ngram_buy pro version",
    "ngram_using app years", "ngram_paid pro version", "ngram_really good app",
    "ngram_simple easy use", "ngram_used app years",
    "ngram_sync google calendar", "ngram_todo list app"
]

STOP = set(ENGLISH_STOP_WORDS)

_word_re = re.compile(r"[^a-z\s]+")

def clean_text(text: str) -> str:
    text = text.lower()
    text = _word_re.sub(" ", text)
    tokens = [w for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)

def _ngrams_present(cleaned: str) -> dict:
    words = cleaned.split()
    bigrams = [" ".join(pair) for pair in zip(words, words[1:])]
    trigrams = [" ".join(tri) for tri in zip(words, words[1:], words[2:])]
    bag = set(bigrams) | set(trigrams)

    flags = {}
    for col in NGRAM_COLUMNS:
        phrase = col.replace("ngram_", "")
        flags[col] = 1 if phrase in bag else 0
    return flags

def build_inference_row(raw_text: str) -> pd.DataFrame:
    cleaned = clean_text(raw_text)
    # basic numeric features
    exclam = raw_text.count("!")
    quest = raw_text.count("?")
    length = len(cleaned.split())

    base = {
        "cleaned_text": [cleaned],
        "has_upvotes": [0],  # unknown at inference time
        "review_length": [length],
        "exclamation_count": [exclam],
        "question_count": [quest],
    }
    base.update({k: [v] for k, v in _ngrams_present(cleaned).items()})
    return pd.DataFrame(base)
