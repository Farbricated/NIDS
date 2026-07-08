from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import alerts, metrics, predict
from api.services import alert_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    alert_store.init_db()
    yield


app = FastAPI(
    title="NIDS API",
    description="Network Intrusion Detection System — inference & alerting API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


app.include_router(predict.router)
app.include_router(alerts.router)
app.include_router(metrics.router)
