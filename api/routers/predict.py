"""
Prediction endpoints with API key auth, rate limiting, and file upload validation.
"""
import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import verify_api_key
from api.models.schemas import FlowFeatures, PredictionResponse
from api.services import alert_store
from api.services.inference import inference_service
from data.columns import COLUMNS

logger = logging.getLogger(__name__)

# ── Rate limiting (graceful fallback if slowapi not installed) ─────────────────
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    _RATE_LIMIT_AVAILABLE = False
    logger.warning("slowapi not installed — rate limiting disabled")

router = APIRouter(prefix="/predict", tags=["prediction"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_single(flow: FlowFeatures):
    """Classify a single network flow record."""
    record = flow.model_dump()
    src_ip = record.pop("src_ip", "0.0.0.0")
    dst_ip = record.pop("dst_ip", "0.0.0.0")

    result = inference_service.predict_single(record)

    alert_store.log_alert(
        src_ip=src_ip,
        dst_ip=dst_ip,
        predicted_category=result["predicted_category"],
        confidence=result["confidence"],
        severity=result["severity"],
    )

    logger.info(
        "Single prediction",
        extra={
            "src_ip": src_ip,
            "category": result["predicted_category"],
            "confidence": result["confidence"],
            "severity": result["severity"],
        },
    )
    return result


@router.post("/batch", dependencies=[Depends(verify_api_key)])
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV file (NSL-KDD schema, no header, 41 or 43 columns)
    or a CSV with a header matching the feature schema.

    Limits: 10 MB max, CSV/text content-type only.
    """
    # ── File validation ──────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in (
        "text/csv", "text/plain", "application/csv",
        "application/octet-stream",  # some browsers send this for .txt
    ):
        raise HTTPException(
            415,
            f"Unsupported content type '{file.content_type}'. Upload a CSV file.",
        )

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Max is 10 MB.",
        )

    # ── Parse CSV ────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(contents), header=None)
        if df.shape[1] in (41, 42, 43):
            names = COLUMNS[: df.shape[1]]
            df.columns = names
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    for col in ("label", "difficulty", "category"):
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [c for c in inference_service.feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            400, f"Uploaded file is missing required columns: {missing[:5]}..."
        )

    result_df = inference_service.predict_batch(df)

    alerts_generated = 0
    for _, row in result_df.iterrows():
        if row["predicted_category"] != "Normal":
            alert_store.log_alert(
                src_ip="batch-upload",
                dst_ip="batch-upload",
                predicted_category=row["predicted_category"],
                confidence=float(row["confidence"]),
                severity=row["severity"],
            )
            alerts_generated += 1

    logger.info(
        "Batch prediction complete",
        extra={
            "rows": len(result_df),
            "alerts_generated": alerts_generated,
            "upload_filename": file.filename,
        },
    )

    summary = {
        "total_rows": len(result_df),
        "predictions": result_df["predicted_category"].value_counts().to_dict(),
        "severity_breakdown": result_df["severity"].value_counts().to_dict(),
        "alerts_generated": alerts_generated,
        "results_preview": result_df.head(50).to_dict(orient="records"),
    }
    return summary
