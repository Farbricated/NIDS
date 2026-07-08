"""
Train and compare multiple ML models for network intrusion detection
on the NSL-KDD dataset. Saves the best model + preprocessing artifacts
to models_store/ for use by the API.

Run: python train_model.py

Outputs to models_store/metadata.json:
  - all_results: accuracy/precision/recall/f1 per model (single train/test split)
  - cv_scores: 5-fold cross-validation mean±std per model
  - classification_report: per-class precision/recall/F1 for best model
  - training_date: ISO timestamp
  - dataset_hash: SHA-256 of KDDTrain.txt for reproducibility
  - library_versions: scikit-learn, xgboost, numpy, pandas versions
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data.columns import COLUMNS, CATEGORICAL_COLS, map_attack_category

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models_store"
MODEL_DIR.mkdir(exist_ok=True)


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _library_versions() -> dict:
    """Collect key library versions for the metadata record."""
    import sklearn
    import numpy as np_
    import pandas as pd_
    versions = {
        "scikit-learn": sklearn.__version__,
        "numpy": np_.__version__,
        "pandas": pd_.__version__,
    }
    if HAS_XGB:
        import xgboost as xgb
        versions["xgboost"] = xgb.__version__
    try:
        import joblib as jl
        versions["joblib"] = jl.__version__
    except Exception:
        pass
    return versions


def load_data():
    train = pd.read_csv(DATA_DIR / "KDDTrain.txt", names=COLUMNS)
    test = pd.read_csv(DATA_DIR / "KDDTest.txt", names=COLUMNS)
    train.drop(columns=["difficulty"], inplace=True)
    test.drop(columns=["difficulty"], inplace=True)
    train["category"] = train["label"].apply(map_attack_category)
    test["category"] = test["label"].apply(map_attack_category)
    train.drop(columns=["label"], inplace=True)
    test.drop(columns=["label"], inplace=True)
    return train, test


def build_preprocessors(train_df: pd.DataFrame):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(train_df[col])
        encoders[col] = le
    return encoders


def apply_preprocessors(df: pd.DataFrame, encoders: dict, unseen_label="__unseen__"):
    df = df.copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        df[col] = df[col].apply(lambda v: v if v in known else known.__iter__().__next__())
        df[col] = le.transform(df[col])
    return df


def main():
    print("Loading NSL-KDD dataset...")
    train_df, test_df = load_data()
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print("Class distribution (train):")
    print(train_df["category"].value_counts())

    # Dataset hash for reproducibility
    dataset_hash = _file_sha256(DATA_DIR / "KDDTrain.txt")
    print(f"\nDataset SHA-256: {dataset_hash[:16]}...")

    encoders = build_preprocessors(train_df)
    train_df = apply_preprocessors(train_df, encoders)
    test_df = apply_preprocessors(test_df, encoders)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["category"])
    y_test = label_encoder.transform(
        test_df["category"].apply(lambda c: c if c in label_encoder.classes_ else "Unknown")
        if "Unknown" in label_encoder.classes_ else test_df["category"]
    )

    feature_cols = [c for c in train_df.columns if c != "category"]
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=20, n_jobs=-1, random_state=42
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=150, max_depth=8,
            eval_metric="mlogloss", n_jobs=-1, random_state=42
        )

    results = {}
    cv_scores = {}
    trained_models = {}

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test_scaled)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(infer_time, 4),
        }
        trained_models[name] = model
        print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  train_time={train_time:.2f}s")

        # k-fold CV (use f1_weighted + accuracy)
        print(f"  Running 5-fold CV for {name}...")
        cv_result = cross_validate(
            model.__class__(**model.get_params()),
            X_train_scaled, y_train,
            cv=cv,
            scoring=["accuracy", "f1_weighted"],
            n_jobs=-1,
        )
        cv_scores[name] = {
            "cv_accuracy_mean": round(float(np.mean(cv_result["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_result["test_accuracy"])), 4),
            "cv_f1_mean": round(float(np.mean(cv_result["test_f1_weighted"])), 4),
            "cv_f1_std": round(float(np.std(cv_result["test_f1_weighted"])), 4),
        }
        print(
            f"  CV accuracy={cv_scores[name]['cv_accuracy_mean']:.4f}"
            f"±{cv_scores[name]['cv_accuracy_std']:.4f}  "
            f"F1={cv_scores[name]['cv_f1_mean']:.4f}"
            f"±{cv_scores[name]['cv_f1_std']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    y_pred_best = best_model.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred_best, target_names=label_encoder.classes_,
        zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred_best).tolist()

    # Feature importance (if available)
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: -x[1])[:15]
        )

    # Save artifacts
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "categorical_encoders.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")

    for name, model in trained_models.items():
        joblib.dump(model, MODEL_DIR / f"model_{name}.pkl")

    metadata = {
        "best_model": best_name,
        "all_results": results,
        "cv_scores": cv_scores,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": label_encoder.classes_.tolist(),
        "feature_importance": feature_importance,
        "feature_columns": feature_cols,
        # Reproducibility metadata
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "library_versions": _library_versions(),
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    print(json.dumps(results, indent=2))
    print("\n5-fold Cross-Validation Results:")
    print(json.dumps(cv_scores, indent=2))


if __name__ == "__main__":
    main()
