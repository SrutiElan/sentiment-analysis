import pandas as pd
import matplotlib.pyplot as plt

import json

model_names = ["bert", "logreg", "lstm", "cnn", "naivebayes", "svm"]

metrics_dict = {}
for name in model_names:
    with open(f"evaluation/{name}_metrics.json") as f:
        metrics_dict[name] = json.load(f)

df = pd.DataFrame.from_dict(metrics_dict, orient='index')

df = df.rename(
    index={"bert" : "BERT",
            "logreg": "Logistic Regression", 
            "lstm" : "LSTM",
            "cnn" : "CNN",
           "naivebayes": "Naive Bayes", 
           "svm": "SVM"},
    columns={"f1": "macro F1 score"}
)




df = df[['accuracy', 'precision', 'recall', 'macro F1 score', 'time']]
# creating latex table
with open('evaluation/table.tex', 'w') as f:
   f.write(df.to_latex())
print(df.to_latex())

df = df[['accuracy', 'precision', 'recall', 'macro F1 score']]
df_transposed = df.transpose()

ax = df_transposed.plot.bar(rot=0)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison on Key Metrics")
plt.tight_layout()
plt.show()
