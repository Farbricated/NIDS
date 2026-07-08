"""
SHAP-based feature attribution endpoint.

POST /predict/explain
  - Runs SHAP TreeExplainer on the best trained model
  - Returns top contributing features for a given prediction
  - Pairs naturally with Groq LLM explanations:
      SHAP  = "which features drove this prediction"
      Groq  = "what those features mean in plain English"

Requires: pip install shap
"""
import logging

from fastapi import APIRouter, HTTPException

from api.models.schemas import FlowFeatures
from api.services.inference import inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["explainability"])


@router.post("/explain")
def explain_prediction(flow: FlowFeatures):
    """
    Return SHAP-based feature attributions for a single network flow prediction.

    Response schema:
    {
        "predicted_category": "DoS",
        "confidence": 0.92,
        "severity": "High",
        "base_value": 0.12,
        "shap_values": {"feature_name": shap_contribution, ...}   # top 15 by |value|
    }
    """
    record = flow.model_dump()
    record.pop("src_ip", None)
    record.pop("dst_ip", None)

    try:
        result = inference_service.get_shap_values(record)
    except ImportError:
        raise HTTPException(
            503,
            "SHAP is not installed. Run: pip install shap",
        )
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        raise HTTPException(500, f"SHAP computation error: {type(exc).__name__}")

    return result
