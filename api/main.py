"""
NIDS FastAPI application — entrypoint with lifespan, middleware, and routing.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.logging_config import setup_logging
from api.routers import alerts, metrics, predict
from api.services import alert_store

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# ── Prometheus metrics (optional — graceful fallback if not installed) ────────
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    REQUEST_COUNT = Counter(
        "nids_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "nids_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path"],
    )
    PREDICTION_COUNTER = Counter(
        "nids_predictions_total",
        "Total predictions by category",
        ["category"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — /metrics/prometheus disabled")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NIDS API starting up — initialising database")
    alert_store.init_db()
    logger.info("Database initialised successfully")
    yield
    logger.info("NIDS API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NIDS API",
    description="Network Intrusion Detection System — inference & alerting API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.environ.get(
    "NIDS_ALLOWED_ORIGINS",
    "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Request logging + Prometheus middleware ───────────────────────────────────
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency = time.perf_counter() - start

    method = request.method
    path = request.url.path
    status = response.status_code

    logger.info(
        "HTTP request",
        extra={
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(latency * 1000, 2),
        },
    )

    if _PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(latency)

    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/metrics/prometheus", tags=["observability"], include_in_schema=_PROMETHEUS_AVAILABLE)
def prometheus_metrics():
    """Expose Prometheus-compatible metrics (requires prometheus_client package)."""
    if not _PROMETHEUS_AVAILABLE:
        from fastapi import HTTPException
        raise HTTPException(503, "prometheus_client not installed")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(predict.router)
app.include_router(alerts.router)
app.include_router(metrics.router)

# Import and include the SHAP explain router
try:
    from api.routers import explain as explain_router
    app.include_router(explain_router.router)
    logger.info("SHAP explain router loaded")
except ImportError as e:
    logger.warning("SHAP router not loaded (missing shap package?): %s", e)
