from fastapi import APIRouter

from api.services.inference import inference_service

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def get_model_metrics():
    """Return training/evaluation metadata: model comparison, confusion matrix, feature importance."""
    return inference_service.get_metadata()
