"""
Loads trained model artifacts once at startup and exposes predict + SHAP functions
used by the API routers.
"""
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
        t0 = time.perf_counter()
        self.model = joblib.load(MODEL_DIR / "best_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.encoders = joblib.load(MODEL_DIR / "categorical_encoders.pkl")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
        with open(MODEL_DIR / "metadata.json") as f:
            self.metadata = json.load(f)
        elapsed = time.perf_counter() - t0
        logger.info(
            "InferenceService loaded",
            extra={
                "model": self.metadata.get("best_model", "unknown"),
                "features": len(self.feature_cols),
                "load_time_ms": round(elapsed * 1000, 1),
            },
        )
        # SHAP explainer — cached after first use
        self._shap_explainer = None

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
        t0 = time.perf_counter()
        proba = self.model.predict_proba(X)[0]
        latency = time.perf_counter() - t0
        pred_idx = int(np.argmax(proba))
        category = self.label_encoder.classes_[pred_idx]
        confidence = float(proba[pred_idx])
        class_probs = {
            cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba)
        }
        logger.debug(
            "predict_single",
            extra={"category": category, "confidence": round(confidence, 4), "latency_ms": round(latency * 1000, 2)},
        )
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

    def _get_shap_explainer(self):
        """Lazily initialise and cache SHAP TreeExplainer."""
        if self._shap_explainer is not None:
            return self._shap_explainer
        import shap  # type: ignore[import]

        model_type = type(self.model).__name__
        logger.info("Initialising SHAP explainer", extra={"model_type": model_type})
        try:
            # TreeExplainer works natively with RF, XGBoost, Decision Tree
            self._shap_explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback to KernelExplainer for other models (slower)
            logger.warning("TreeExplainer failed, falling back to KernelExplainer")
            # Use a small background dataset for speed
            bg = np.zeros((10, len(self.feature_cols)))
            self._shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, bg
            )
        return self._shap_explainer

    def get_shap_values(self, record: dict) -> dict:
        """
        Return SHAP feature attributions for a single record.

        Returns dict with predicted_category, confidence, severity,
        base_value (expected model output), and top-15 shap_values by magnitude.
        """
        import shap  # noqa: F401 — imported here to surface ImportError cleanly

        df = pd.DataFrame([record])
        X = self._prepare(df)

        # Standard prediction
        pred_result = self.predict_single(record)

        explainer = self._get_shap_explainer()
        t0 = time.perf_counter()
        shap_vals = explainer.shap_values(X)
        latency = time.perf_counter() - t0
        logger.info("SHAP computed", extra={"latency_ms": round(latency * 1000, 1)})

        # shap_vals shape varies by model:
        # - For multi-class: list of arrays (one per class) OR 3D array
        # - For binary: 2D array
        pred_idx = self.label_encoder.transform([pred_result["predicted_category"]])[0]

        sv_for_class: np.ndarray
        if isinstance(shap_vals, list):
            # Multi-class: list[n_classes] each of shape (1, n_features)
            sv_for_class = np.array(shap_vals[pred_idx]).flatten()
            base_val = float(explainer.expected_value[pred_idx]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # 3D: (1, n_features, n_classes)
            sv_for_class = np.asarray(shap_vals[0, :, pred_idx]).flatten()
            ev = explainer.expected_value
            base_val = float(ev[pred_idx]) if hasattr(ev, "__len__") else float(ev)
        else:
            # 2D: (1, n_features) — binary or single-class SHAP
            sv_for_class = np.array(shap_vals).flatten()
            ev = explainer.expected_value
            base_val = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

        # Map back to feature names and sort by magnitude
        feature_shap = dict(zip(self.feature_cols, sv_for_class.tolist()))
        top15 = dict(
            sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        )
        # Round for JSON cleanliness
        top15 = {k: round(v, 6) for k, v in top15.items()}

        return {
            "predicted_category": pred_result["predicted_category"],
            "confidence": pred_result["confidence"],
            "severity": pred_result["severity"],
            "base_value": round(base_val, 6),
            "shap_values": top15,
        }


# Singleton instance loaded once at API startup
inference_service = InferenceService()
