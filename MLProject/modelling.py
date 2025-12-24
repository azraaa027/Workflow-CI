# modelling.py (versi kompatibel dengan mlflow run)
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ MLflow otomatis set tracking URI ke ./mlruns/
mlflow.set_experiment("titanic-ci")

df = pd.read_csv("titanic_preprocessed.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Autolog → hasilkan artefak lengkap
mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc)
    
    print(f"✅ CI training selesai. Accuracy: {acc:.4f}")