# every-eval-ever-pipeline

Minimal backend scaffold for ingesting aggregate + instance evaluation JSONL into DuckDB.

## Quick start

```bash
pip install -e ".[dev]"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Backend endpoints

- `GET /health`
- `POST /ingest/aggregate?jsonl_path=...`
  - Defaults to `data/simulated/public_leaderboards/aggregate_rows_200.jsonl`
- `POST /ingest/instance?jsonl_path=...`
  - Defaults to `data/simulated/public_leaderboards/instance_rows_200.jsonl`
- `GET /stats`
- `GET /metrics/top-models?metric_kind=...&metric_name=...&limit=...`
- `GET /join-integrity`

## Simulation Tooling

Synthetic data for backend testing is available in `tools/sim_data/`:

- `generate_public_leaderboard_sample.py`: Scrapes diverse public leaderboards to populate the backend with high-volume aggregate data.
- `generate_backend_sim_data.py`: Generates "pathology" fixtures (collisions, ambiguities) and linked instance-level JSONLs from the real `EEE_datastore` for code integrity validation.

## Why this backend shape

- Stores aggregate runs and per-metric rows in normalized tables for SQL queries.
- Supports deterministic join key preference (`evaluation_result_id`) across aggregate + instance schemas.
- Provides fallback join behavior (`evaluation_name`) when `evaluation_result_id` is missing.
- Captures optional dedupe identity fields (`dedupe_identity.*`) for idempotent ingestion.
