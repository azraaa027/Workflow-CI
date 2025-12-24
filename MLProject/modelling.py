import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="titanic_preprocessed.csv")
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()

# Load & prepare data
df = pd.read_csv(args.dataset)
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs")
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ✅ Manual logging (wajib untuk *Skilled*)
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("C", args.C)
mlflow.log_param("max_iter", args.max_iter)
mlflow.log_param("solver", "lbfgs")

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(model, artifact_path="model")

print("✅ CI Training selesai")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")