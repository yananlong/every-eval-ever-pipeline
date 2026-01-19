.PHONY: api worker fmt lint install

# Start the FastAPI server
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start the worker
worker:
	python -m worker.main

# Format code
fmt:
	ruff format .

# Lint code
lint:
	ruff check .

# Install dependencies
install:
	pip install -e ".[dev]"
