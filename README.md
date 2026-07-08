# Network Intrusion Detection System (NIDS)

A machine-learning-based Network Intrusion Detection System with a FastAPI
inference backend and a Streamlit dashboard, trained on the NSL-KDD dataset.
Built as a near-production reference project — containerized, tested, and
CI-integrated.

## Architecture

```
Streamlit Dashboard  ──HTTP──▶  FastAPI (/predict, /predict/batch, /alerts, /metrics)
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                 Feature Prep   ML Model (best of  Alert Engine
                 (scaler +      4 compared:        (severity +
                 encoders)      LR/DT/RF/XGBoost)  SQLite log)
```

- **API layer** (`api/`): FastAPI service exposing prediction, batch upload,
  alert history, and model metrics endpoints. Stateless except for the
  SQLite alert log.
- **ML layer** (`train_model.py`): trains and compares 4 classifiers
  (Logistic Regression, Decision Tree, Random Forest, XGBoost) on NSL-KDD,
  auto-selects the best by weighted F1, and persists all artifacts.
- **Dashboard** (`dashboard/`): Streamlit client with 3 views — simulated
  live monitor, batch CSV analysis, and a model/alert analytics report.
- **Storage**: trained model artifacts (`models_store/*.pkl`), model
  metadata (`metadata.json`), and alert history (`alerts.db`, SQLite).

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — an improved version
of the classic KDD Cup 1999 dataset that removes duplicate records and
redundant easy examples. 41 flow-level features per record (protocol,
byte counts, error rates, host-based statistics), attacks grouped into
4 categories: **DoS, Probe, R2L, U2R** (plus Normal).

The official test set intentionally includes attack subtypes not seen in
training, so ~75-78% accuracy is the expected, literature-consistent range
for classical ML on this benchmark — it tests generalization, not just
memorization.

## Project Structure

```
nids-project/
├── api/
│   ├── main.py                # FastAPI app entrypoint
│   ├── routers/                # predict, alerts, metrics endpoints
│   ├── services/                # inference engine, SQLite alert store
│   ├── models/schemas.py       # Pydantic request/response models
│   └── tests/                  # pytest suite (API + unit tests)
├── dashboard/
│   └── app.py                   # Streamlit UI (3 tabs)
├── data/
│   ├── columns.py               # NSL-KDD schema + attack category mapping
│   └── raw/                     # KDDTrain.txt, KDDTest.txt
├── models_store/                # trained model artifacts (generated)
├── train_model.py               # training + model comparison script
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml     # lint → train → test → docker build
```

## Running Locally (without Docker)

```bash
pip install -r requirements.txt

# 1. Train the model (downloads artifacts into models_store/)
python train_model.py

# 2. Start the API
uvicorn api.main:app --reload --port 8000

# 3. In another terminal, start the dashboard
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501　·　API docs: http://localhost:8000/docs

## Running with Docker Compose

```bash
python train_model.py          # generates models_store/ (not baked into image build context churn)
docker compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

## Running Tests

```bash
pytest api/tests/ -v
ruff check .
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Classify a single flow record |
| POST | `/predict/batch` | Upload CSV, classify all rows |
| GET | `/alerts` | Recent alert log |
| GET | `/alerts/stats` | Aggregated alert stats (by category/severity/source) |
| DELETE | `/alerts` | Clear alert log |
| GET | `/metrics` | Model comparison results, confusion matrix, feature importance |

## Model Comparison (example run)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.762 | 0.753 | 0.762 | 0.714 |
| Decision Tree | 0.760 | 0.802 | 0.760 | 0.714 |
| Random Forest | 0.748 | 0.817 | 0.748 | 0.705 |
| **XGBoost (selected)** | **0.772** | **0.817** | **0.772** | **0.725** |

*(Exact numbers may vary slightly by run/seed; re-run `train_model.py` to regenerate.)*

## Design Notes / Talking Points

- **API/UI separation**: the dashboard is a thin HTTP client — any other
  client (CLI, mobile, SIEM plugin) could call the same API.
- **Model selection is automatic and explainable**: all 4 models' metrics
  are persisted, not just the winner, so the choice is auditable.
- **Severity mapping**: prediction categories map to a severity scale
  (Info/Low/Medium/High/Critical) — a simple example of turning ML output
  into SOC-actionable triage.
- **Graceful handling of unseen categorical values** at inference time
  (protocol/service/flag not seen during training) avoids runtime crashes
  on live traffic.
- **CI pipeline** mirrors a real MLOps flow: lint → train → test → build
  container images.

## Future Enhancements

- Real packet capture via Scapy/CICFlowMeter for true live traffic (vs.
  simulated replay), with root/admin privileges.
- Model retraining pipeline triggered on new labeled data.
- Authentication on the API (JWT) before any public exposure.
- Deep learning baseline (LSTM/Autoencoder) for comparison against classical ML.
