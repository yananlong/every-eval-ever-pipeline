"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Every Eval Ever Pipeline",
    description="Middleware for ingesting and publishing AI evaluation results",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"ok": True}
