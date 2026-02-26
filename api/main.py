"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from shared.config import settings
from shared.duckdb_backend import DuckDBBackend

app = FastAPI(
    title="Every Eval Ever Pipeline",
    description="Middleware for ingesting and publishing AI evaluation results",
    version="0.1.0",
)

backend = DuckDBBackend(settings.duckdb_path)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"ok": True}


@app.post("/ingest/aggregate")
async def ingest_aggregate(jsonl_path: str | None = None) -> dict:
    """Ingest aggregate JSONL records (eval.schema.json shape) into DuckDB."""
    path = Path(jsonl_path or settings.default_aggregate_jsonl_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Aggregate file not found: {path}")

    try:
        stats = backend.ingest_aggregate_jsonl(path)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    return {"ok": True, "path": str(path), **stats}


@app.post("/ingest/instance")
async def ingest_instance(jsonl_path: str | None = None) -> dict:
    """Ingest instance-level JSONL records (instance_level_eval.schema.json shape) into DuckDB."""
    path = Path(jsonl_path or settings.default_instance_jsonl_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Instance file not found: {path}")

    try:
        stats = backend.ingest_instance_jsonl(path)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    return {"ok": True, "path": str(path), **stats}


@app.get("/stats")
async def get_stats() -> dict:
    """Return backend table-level stats."""
    return {"ok": True, "stats": backend.stats()}


@app.get("/metrics/top-models")
async def top_model_metrics(
    metric_kind: str | None = Query(default=None, description="Filter by metric_kind"),
    metric_name: str | None = Query(default=None, description="Filter by metric_name"),
    limit: int = Query(default=20, ge=1, le=500),
) -> dict:
    """Return top model averages for a given metric slice."""
    rows = backend.top_model_metrics(metric_kind=metric_kind, metric_name=metric_name, limit=limit)
    return {"ok": True, "rows": rows}


@app.get("/join-integrity")
async def join_integrity() -> dict:
    """Report deterministic linkage coverage across aggregate and instance tables."""
    return {"ok": True, "join_integrity": backend.join_integrity()}
