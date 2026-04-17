.PHONY: install install-dev test lint format dashboard notebook clean

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ app/ notebooks/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=html

# ── Run ───────────────────────────────────────────────────────────────────────

dashboard:
	streamlit run app/dashboard.py --server.port 8501

notebook:
	jupyter lab notebooks/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/cache/*.parquet 2>/dev/null || true
