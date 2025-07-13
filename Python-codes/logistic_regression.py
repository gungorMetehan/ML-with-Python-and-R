# Info on Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
pip install ucimlrepo

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# data manipulation
heart_disease = fetch_ucirepo(id = 45)
df = pd.DataFrame(data = heart_disease.data.features)
df["target"] = heart_disease.data.targets

# dealing with missing values
df.dropna(inplace = True)

# features and the target
X = df.drop("target", axis = 1).values
y = df["target"].values

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# pipeline: StandardScaler + LogisticRegression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(C = 1, max_iter = 100, random_state = 42, solver = "lbfgs"))
])

# training
pipeline.fit(X_train, y_train)

# y_pred
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# accuracy, confusion matrix and AUC score
accuracy = round(accuracy_score(y_test, y_pred), 2)
conf_matrix = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

y_proba_all = pipeline.predict_proba(X_test)  # shape: (n_samples, n_classes)
auc_score = round(roc_auc_score(y_test, y_proba_all, multi_class = 'ovr'), 2)

# ROC curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

## number of classes
n_classes = len(df["target"].unique())

## binarizing y_test to (n, n_classes) shape
y_test_bin = label_binarize(y_test, classes=range(n_classes))
y_score = pipeline.predict_proba(X_test)  # (n_samples, n_classes)

## computing ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

## data visualization: multi-class (5 classes) ROC curve (one-vs-rest)
plt.figure(figsize = (10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label = f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label = "Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve (One-vs-Rest)")
plt.legend(loc = "lower right")
plt.grid(True)
plt.show()