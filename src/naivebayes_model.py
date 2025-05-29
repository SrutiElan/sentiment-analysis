import json
import os
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import LinearSVC
import seaborn as sns

# load 
df = pd.read_csv("data/processed/preprocessed_reviews.csv")
X = df[["cleaned_text", "exclamation_count", "question_count", "has_upvotes"]]
y = df["sentiment_binary"]

# training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# naivebayes requires that text keeps everything ≥ 0
preprocessor = ColumnTransformer([
    # TF-IDF on text always ≥0
    ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"), "cleaned_text"),
    # rescale numeric counts to [0,1]
    ("num", MinMaxScaler(), ["exclamation_count", "question_count", "has_upvotes"])
])

# 4. Pipeline with MultinomialNB
nb_pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", MultinomialNB(alpha=1.0))  # alpha is laplace smoothing
])

# training
nb_pipeline.fit(X_train, y_train)
# testing
y_pred_nb = nb_pipeline.predict(X_test)


# save model
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "nb_sentiment.pkl")
joblib.dump(nb_pipeline, MODEL_PATH)


print(f"Trained model saved to {MODEL_PATH}")


# evaluation metrics
acc   = accuracy_score(y_test, y_pred_nb)
prec  = precision_score(y_test, y_pred_nb)
rec   = recall_score(y_test, y_pred_nb)
f1    = f1_score(y_test, y_pred_nb)
print(classification_report(y_test, y_pred_nb, target_names=["Neg","Pos"], digits=4))

# create evaluation directory
os.makedirs("evaluation", exist_ok=True)

# save numeric metrics to JSON
metrics = {
    "model": "naive bayes",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
}
with open("evaluation/naivebayes_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)


# confusion matrix
cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(5, 4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg","Pos"],
            yticklabels=["Neg","Pos"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation/naivebayes_confusion_matrix.png")
plt.close()


