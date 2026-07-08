.PHONY: dev test lint docker-up docker-down train install pre-commit

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

pre-commit:
	pip install pre-commit
	pre-commit install

# ── Development ────────────────────────────────────────────────────────────────
train:
	python train_model.py

dev: train
	@echo "Starting NIDS dashboard (it will auto-start the backend API)..."
	@echo "  Dashboard: http://localhost:8501"
	@echo "  API docs:  http://localhost:8000/docs (once backend is up)"
	streamlit run dashboard/app.py

# ── Testing ────────────────────────────────────────────────────────────────────
test:
	pytest api/tests/ -v --cov=api --cov-report=term-missing --cov-report=xml

lint:
	ruff check .
	mypy api/ --ignore-missing-imports

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-up: train
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
