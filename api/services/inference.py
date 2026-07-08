"""
Loads trained model artifacts once at startup and exposes predict functions
used by the API routers.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models_store"

SEVERITY_MAP = {
    "Normal": "Info",
    "Probe": "Low",
    "DoS": "High",
    "R2L": "Medium",
    "U2R": "Critical",
    "Unknown": "Medium",
}


class InferenceService:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "best_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.encoders = joblib.load(MODEL_DIR / "categorical_encoders.pkl")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
        with open(MODEL_DIR / "metadata.json") as f:
            self.metadata = json.load(f)

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self.encoders.items():
            known = set(le.classes_)
            fallback = le.classes_[0]
            df[col] = df[col].apply(lambda v: v if v in known else fallback)
            df[col] = le.transform(df[col])
        return df

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        df = df[self.feature_cols].copy()
        df = self._encode_categoricals(df)
        return self.scaler.transform(df.values)

    def predict_single(self, record: dict) -> dict:
        df = pd.DataFrame([record])
        X = self._prepare(df)
        proba = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        category = self.label_encoder.classes_[pred_idx]
        confidence = float(proba[pred_idx])
        class_probs = {
            cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba)
        }
        return {
            "predicted_category": category,
            "confidence": round(confidence, 4),
            "severity": SEVERITY_MAP.get(category, "Medium"),
            "class_probabilities": {k: round(v, 4) for k, v in class_probs.items()},
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(df)
        proba = self.model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        categories = self.label_encoder.classes_[pred_idx]
        confidences = proba[np.arange(len(proba)), pred_idx]

        result = df.copy()
        result["predicted_category"] = categories
        result["confidence"] = np.round(confidences, 4)
        result["severity"] = result["predicted_category"].map(
            lambda c: SEVERITY_MAP.get(c, "Medium")
        )
        return result

    def get_metadata(self) -> dict:
        return self.metadata


# Singleton instance loaded once at API startup
inference_service = InferenceService()
