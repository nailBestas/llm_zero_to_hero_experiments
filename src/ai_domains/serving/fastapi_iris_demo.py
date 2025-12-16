from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class IrisInput(BaseModel):
    features: List[float]  # 4 boyutlu: sepal_len, sepal_wid, petal_len, petal_wid


app = FastAPI(title="Iris Classifier Demo")

# Modeli uygulama start-up'ında eğitelim (küçük dataset, sorun değil).
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
test_acc = clf.score(X_test, y_test)


@app.get("/")
def root():
    return {
        "message": "Iris classifier API - demo",
        "test_accuracy": test_acc,
        "classes": iris.target_names.tolist(),
    }


@app.post("/predict")
def predict(payload: IrisInput):
    if len(payload.features) != 4:
        return {"error": "features must be a list of 4 floats"}

    pred = clf.predict([payload.features])[0]
    proba = clf.predict_proba([payload.features])[0].tolist()
    return {
        "pred_class_idx": int(pred),
        "pred_class_name": iris.target_names[pred],
        "proba": proba,
    }
