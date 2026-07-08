from fastapi import APIRouter, Query

from api.services import alert_store

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts(limit: int = Query(default=100, le=1000)):
    return alert_store.get_recent_alerts(limit=limit)


@router.get("/stats")
def alert_stats():
    return alert_store.get_alert_stats()


@router.delete("")
def clear_alerts():
    alert_store.clear_alerts()
    return {"status": "cleared"}
