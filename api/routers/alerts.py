from fastapi import APIRouter, HTTPException, Query

from api.services import alert_store
from api.services import llm_explainer

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts(limit: int = Query(default=100, le=1000)):
    return alert_store.get_recent_alerts(limit=limit)


@router.get("/stats")
def alert_stats():
    return alert_store.get_alert_stats()


@router.get("/{alert_id}/explain")
def explain_alert(alert_id: int):
    """
    Return a plain-English LLM explanation of why an alert looks malicious.

    - First call: generates explanation via Groq, caches it in SQLite.
    - Subsequent calls: returns the cached explanation instantly (no Groq call).
    - Returns 404 if alert_id doesn't exist.
    - Returns 503 if GROQ_API_KEY is not configured.
    - Returns 502 if the Groq API call itself fails.
    """
    alert = alert_store.get_alert_by_id(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    # Return cached explanation if already generated
    if alert.get("explanation"):
        return {"alert_id": alert_id, "explanation": alert["explanation"], "cached": True}

    # Call Groq LLM
    try:
        explanation = llm_explainer.explain_alert(alert)
    except RuntimeError as exc:
        # Missing API key or missing package — config error
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Network error, rate limit, etc.
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {exc}",
        ) from exc

    # Persist and return
    alert_store.set_alert_explanation(alert_id, explanation)
    return {"alert_id": alert_id, "explanation": explanation, "cached": False}


@router.delete("")
def clear_alerts():
    alert_store.clear_alerts()
    return {"status": "cleared"}
