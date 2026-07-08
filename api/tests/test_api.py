import io

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient used as a context manager so that FastAPI's
    lifespan handler fires (calling alert_store.init_db() and creating the
    `alerts` table). Without the context manager the lifespan never runs,
    causing 'no such table: alerts' on a clean checkout (e.g. CI).
    """
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_single_normal_flow(client):
    payload = {
        "duration": 0, "protocol_type": "tcp", "service": "ftp_data", "flag": "SF",
        "src_bytes": 491, "dst_bytes": 0, "count": 2, "srv_count": 2,
        "same_srv_rate": 1.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_category" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["severity"] in ("Info", "Low", "Medium", "High", "Critical")


def test_predict_single_dos_like_flow(client):
    payload = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "count": 200, "srv_count": 200,
        "serror_rate": 1.0, "srv_serror_rate": 1.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_category"] in ("DoS", "Probe", "R2L", "U2R", "Normal")


def test_predict_missing_fields_uses_defaults(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200


def test_predict_invalid_field_type(client):
    r = client.post("/predict", json={"duration": "not-a-number"})
    assert r.status_code == 422


def test_alerts_endpoint_returns_list(client):
    r = client.get("/alerts")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_alerts_stats_shape(client):
    r = client.get("/alerts/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_alerts" in body
    assert "by_category" in body
    assert "by_severity" in body


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "best_model" in body
    assert "all_results" in body
    assert "confusion_matrix" in body


def test_batch_predict_with_sample_csv(client, tmp_path):
    sample_path = tmp_path / "sample.csv"
    with open("data/raw/KDDTest.txt") as src:
        lines = [next(src) for _ in range(20)]
    sample_path.write_text("".join(lines))

    with open(sample_path, "rb") as f:
        r = client.post("/predict/batch", files={"file": ("sample.csv", f, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == 20
    assert "predictions" in body


def test_batch_predict_rejects_bad_columns(client):
    bad_csv = io.BytesIO(b"a,b,c\n1,2,3\n")
    r = client.post("/predict/batch", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert r.status_code == 400


def test_predict_explain_endpoint(client):
    """SHAP-based feature attribution endpoint returns expected keys."""
    payload = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 491, "dst_bytes": 0, "count": 2, "srv_count": 2,
    }
    r = client.post("/predict/explain", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_category" in body
    assert "shap_values" in body
    assert isinstance(body["shap_values"], dict)
    assert len(body["shap_values"]) > 0
