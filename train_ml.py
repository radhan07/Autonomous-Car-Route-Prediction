# train_model.py
import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

DATA_PATH = "ml_training_data.jsonl"
MODEL_PATH = os.getenv("ML_MODEL_PATH", "ml_route_model.pkl")

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows

def extract_supervised(rows):
    """
    We expect rows like:
      {"features": {...}, "meta": {...}}
    We need a numeric target.
    Priority:
      1) meta.user_score in [0,1] if present,
      2) features.score if present,
      else skip row.
    """
    X, y = [], []
    for r in rows:
        feats = r.get("features") or {}
        meta = r.get("meta") or {}
        target = None

        # 1) user-annotated
        if "user_score" in meta and isinstance(meta["user_score"], (int, float)):
            target = float(meta["user_score"])

        # 2) or a precomputed score (if your pipeline logged it)
        elif "score" in feats and isinstance(feats["score"], (int, float)):
            target = float(feats["score"])

        if feats and target is not None:
            X.append(feats)
            y.append(target)

    return X, y

def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data file: {DATA_PATH}")
        return

    rows = load_jsonl(DATA_PATH)
    X_dicts, y = extract_supervised(rows)

    if not X_dicts:
        print("No supervised examples found. Add a 'meta.user_score' or 'features.score' to your logs first.")
        return

    pipe = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X_dicts, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print("R2:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
