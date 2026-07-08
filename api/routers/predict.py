import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.models.schemas import FlowFeatures, PredictionResponse
from api.services import alert_store
from api.services.inference import inference_service
from data.columns import COLUMNS

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
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
    return result


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV file (NSL-KDD schema, no header, 41 or 43 columns)
    or a CSV with a header matching the feature schema.
    """
    contents = await file.read()
    try:
        # Try headerless NSL-KDD raw format first
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

    summary = {
        "total_rows": len(result_df),
        "predictions": result_df["predicted_category"].value_counts().to_dict(),
        "severity_breakdown": result_df["severity"].value_counts().to_dict(),
        "alerts_generated": alerts_generated,
        "results_preview": result_df.head(50).to_dict(orient="records"),
    }
    return summary
