"""DuckDB-backed storage and query helpers for aggregate + instance eval data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return cleaned


class DuckDBBackend:
    """Simple backend storage optimized for ingestion/idempotency and SQL analytics."""

    def __init__(self, db_path: str = "data/backend.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                evaluation_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                retrieved_timestamp TEXT NOT NULL,
                source_name TEXT,
                source_type TEXT,
                source_organization_name TEXT,
                source_organization_url TEXT,
                evaluator_relationship TEXT,
                model_id TEXT NOT NULL,
                model_name TEXT,
                model_developer TEXT,
                eval_library_name TEXT,
                eval_library_version TEXT,
                run_fingerprint TEXT,
                content_hash TEXT,
                hash_algorithm TEXT,
                canonicalization_version TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                evaluation_id TEXT NOT NULL,
                metric_index INTEGER NOT NULL,
                evaluation_result_id TEXT,
                result_join_id TEXT NOT NULL,
                join_key_source TEXT NOT NULL,
                evaluation_name TEXT NOT NULL,
                metric_id TEXT,
                metric_name TEXT,
                metric_kind TEXT,
                metric_unit TEXT,
                metric_parameters_json TEXT,
                lower_is_better BOOLEAN,
                score_type TEXT,
                min_score DOUBLE,
                max_score DOUBLE,
                score DOUBLE NOT NULL,
                source_dataset_name TEXT,
                source_type TEXT,
                source_ref TEXT,
                metric_config_json TEXT,
                score_details_json TEXT,
                PRIMARY KEY (evaluation_id, metric_index)
            )
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_join
            ON evaluation_metrics (evaluation_id, result_join_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_kind
            ON evaluation_metrics (metric_kind)
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instance_evaluations (
                evaluation_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                evaluation_name TEXT NOT NULL,
                evaluation_result_id TEXT,
                result_join_id TEXT NOT NULL,
                join_key_source TEXT NOT NULL,
                model_id TEXT NOT NULL,
                score DOUBLE,
                is_correct BOOLEAN,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, sample_id, result_join_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instance_join
            ON instance_evaluations (evaluation_id, result_join_id)
            """
        )

    def _source_reference(self, source_data: dict[str, Any]) -> str | None:
        source_type = source_data.get("source_type")
        if source_type == "url":
            urls = source_data.get("url")
            if isinstance(urls, list):
                return ", ".join(str(x) for x in urls)
        if source_type == "hf_dataset":
            repo = source_data.get("hf_repo")
            split = source_data.get("hf_split")
            if repo and split:
                return f"{repo}:{split}"
            if repo:
                return str(repo)
        dataset_name = source_data.get("dataset_name")
        return str(dataset_name) if dataset_name is not None else None

    def _result_join_id(self, evaluation_result_id: str | None, evaluation_name: str) -> tuple[str, str]:
        if evaluation_result_id:
            return evaluation_result_id, "evaluation_result_id"
        return f"name:{_normalize_name(evaluation_name)}", "evaluation_name_fallback"

    def ingest_aggregate_jsonl(self, jsonl_path: str | Path) -> dict[str, int]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)

        runs = 0
        metrics = 0

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue

                payload = json.loads(raw_line)
                evaluation_id = str(payload.get("evaluation_id", "")).strip()
                if not evaluation_id:
                    continue

                dedupe = payload.get("dedupe_identity") or {}
                if not isinstance(dedupe, dict):
                    dedupe = {}

                source_meta = payload.get("source_metadata") or {}
                if not isinstance(source_meta, dict):
                    source_meta = {}

                model_info = payload.get("model_info") or {}
                if not isinstance(model_info, dict):
                    model_info = {}

                eval_library = payload.get("eval_library") or {}
                if not isinstance(eval_library, dict):
                    eval_library = {}

                try:
                    run_metrics = 0
                    self.conn.begin()
                    self.conn.execute(
                        "DELETE FROM evaluation_metrics WHERE evaluation_id = ?",
                        [evaluation_id],
                    )
                    self.conn.execute(
                        "DELETE FROM evaluation_runs WHERE evaluation_id = ?",
                        [evaluation_id],
                    )

                    self.conn.execute(
                        """
                        INSERT INTO evaluation_runs (
                            evaluation_id,
                            schema_version,
                            retrieved_timestamp,
                            source_name,
                            source_type,
                            source_organization_name,
                            source_organization_url,
                            evaluator_relationship,
                            model_id,
                            model_name,
                            model_developer,
                            eval_library_name,
                            eval_library_version,
                            run_fingerprint,
                            content_hash,
                            hash_algorithm,
                            canonicalization_version,
                            raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            evaluation_id,
                            str(payload.get("schema_version", "")),
                            str(payload.get("retrieved_timestamp", "")),
                            source_meta.get("source_name"),
                            source_meta.get("source_type"),
                            source_meta.get("source_organization_name"),
                            source_meta.get("source_organization_url"),
                            source_meta.get("evaluator_relationship"),
                            model_info.get("id"),
                            model_info.get("name"),
                            model_info.get("developer"),
                            eval_library.get("name"),
                            eval_library.get("version"),
                            dedupe.get("run_fingerprint"),
                            dedupe.get("content_hash"),
                            dedupe.get("hash_algorithm"),
                            dedupe.get("canonicalization_version"),
                            json.dumps(payload, ensure_ascii=False),
                        ],
                    )

                    result_items = payload.get("evaluation_results")
                    if not isinstance(result_items, list):
                        result_items = []

                    for metric_index, result in enumerate(result_items):
                        if not isinstance(result, dict):
                            continue

                        metric_cfg = result.get("metric_config") or {}
                        if not isinstance(metric_cfg, dict):
                            metric_cfg = {}

                        score_details = result.get("score_details") or {}
                        if not isinstance(score_details, dict):
                            score_details = {}

                        score = _to_float(score_details.get("score"))
                        if score is None:
                            continue

                        evaluation_name = str(result.get("evaluation_name", "")).strip()
                        if not evaluation_name:
                            continue

                        evaluation_result_id = result.get("evaluation_result_id")
                        if evaluation_result_id is not None:
                            evaluation_result_id = str(evaluation_result_id)

                        result_join_id, join_key_source = self._result_join_id(
                            evaluation_result_id,
                            evaluation_name,
                        )

                        source_data = result.get("source_data") or {}
                        if not isinstance(source_data, dict):
                            source_data = {}

                        metric_parameters = metric_cfg.get("metric_parameters")
                        metric_parameters_json = None
                        if isinstance(metric_parameters, dict):
                            metric_parameters_json = json.dumps(
                                metric_parameters,
                                sort_keys=True,
                                ensure_ascii=False,
                            )

                        self.conn.execute(
                            """
                            INSERT INTO evaluation_metrics (
                                evaluation_id,
                                metric_index,
                                evaluation_result_id,
                                result_join_id,
                                join_key_source,
                                evaluation_name,
                                metric_id,
                                metric_name,
                                metric_kind,
                                metric_unit,
                                metric_parameters_json,
                                lower_is_better,
                                score_type,
                                min_score,
                                max_score,
                                score,
                                source_dataset_name,
                                source_type,
                                source_ref,
                                metric_config_json,
                                score_details_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                evaluation_id,
                                metric_index,
                                evaluation_result_id,
                                result_join_id,
                                join_key_source,
                                evaluation_name,
                                metric_cfg.get("metric_id"),
                                metric_cfg.get("metric_name"),
                                metric_cfg.get("metric_kind"),
                                metric_cfg.get("metric_unit"),
                                metric_parameters_json,
                                metric_cfg.get("lower_is_better"),
                                metric_cfg.get("score_type"),
                                _to_float(metric_cfg.get("min_score")),
                                _to_float(metric_cfg.get("max_score")),
                                score,
                                source_data.get("dataset_name"),
                                source_data.get("source_type"),
                                self._source_reference(source_data),
                                json.dumps(metric_cfg, ensure_ascii=False),
                                json.dumps(score_details, ensure_ascii=False),
                            ],
                        )
                        run_metrics += 1

                    self.conn.commit()
                    metrics += run_metrics
                    runs += 1
                except Exception:
                    self.conn.rollback()
                    raise

        return {"runs_ingested": runs, "metrics_ingested": metrics}

    def ingest_instance_jsonl(self, jsonl_path: str | Path) -> dict[str, int]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)

        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue

                payload = json.loads(raw_line)
                evaluation_id = str(payload.get("evaluation_id", "")).strip()
                sample_id = str(payload.get("sample_id", "")).strip()
                evaluation_name = str(payload.get("evaluation_name", "")).strip()
                model_id = str(payload.get("model_id", "")).strip()

                if not all([evaluation_id, sample_id, evaluation_name, model_id]):
                    continue

                evaluation_result_id = payload.get("evaluation_result_id")
                if evaluation_result_id is not None:
                    evaluation_result_id = str(evaluation_result_id)

                result_join_id, join_key_source = self._result_join_id(
                    evaluation_result_id,
                    evaluation_name,
                )

                evaluation_obj = payload.get("evaluation") or {}
                if not isinstance(evaluation_obj, dict):
                    evaluation_obj = {}

                score = _to_float(evaluation_obj.get("score"))
                is_correct = evaluation_obj.get("is_correct")
                if not isinstance(is_correct, bool):
                    is_correct = None

                self.conn.execute(
                    """
                    DELETE FROM instance_evaluations
                    WHERE evaluation_id = ? AND sample_id = ? AND result_join_id = ?
                    """,
                    [evaluation_id, sample_id, result_join_id],
                )

                self.conn.execute(
                    """
                    INSERT INTO instance_evaluations (
                        evaluation_id,
                        sample_id,
                        evaluation_name,
                        evaluation_result_id,
                        result_join_id,
                        join_key_source,
                        model_id,
                        score,
                        is_correct,
                        raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evaluation_id,
                        sample_id,
                        evaluation_name,
                        evaluation_result_id,
                        result_join_id,
                        join_key_source,
                        model_id,
                        score,
                        is_correct,
                        json.dumps(payload, ensure_ascii=False),
                    ],
                )
                rows += 1

        return {"instance_rows_ingested": rows}

    def stats(self) -> dict[str, int]:
        runs_count = int(self.conn.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0])
        metrics_count = int(
            self.conn.execute("SELECT COUNT(*) FROM evaluation_metrics").fetchone()[0]
        )
        models_count = int(
            self.conn.execute("SELECT COUNT(DISTINCT model_id) FROM evaluation_runs").fetchone()[0]
        )
        metric_kind_count = int(
            self.conn.execute(
                "SELECT COUNT(DISTINCT metric_kind) FROM evaluation_metrics WHERE metric_kind IS NOT NULL"
            ).fetchone()[0]
        )
        instance_count = int(
            self.conn.execute("SELECT COUNT(*) FROM instance_evaluations").fetchone()[0]
        )
        return {
            "evaluation_runs": runs_count,
            "evaluation_metrics": metrics_count,
            "models": models_count,
            "metric_kinds": metric_kind_count,
            "instance_rows": instance_count,
        }

    def top_model_metrics(
        self,
        metric_kind: str | None = None,
        metric_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                r.model_id,
                COALESCE(m.metric_name, m.evaluation_name) AS metric_name,
                COALESCE(m.metric_kind, 'unknown') AS metric_kind,
                AVG(m.score) AS avg_score,
                COUNT(*) AS observations
            FROM evaluation_metrics m
            JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
            WHERE (? IS NULL OR m.metric_kind = ?)
              AND (? IS NULL OR COALESCE(m.metric_name, m.evaluation_name) = ?)
            GROUP BY 1, 2, 3
            ORDER BY avg_score DESC
            LIMIT ?
            """,
            [metric_kind, metric_kind, metric_name, metric_name, limit],
        ).fetchall()

        return [
            {
                "model_id": row[0],
                "metric_name": row[1],
                "metric_kind": row[2],
                "avg_score": float(row[3]),
                "observations": int(row[4]),
            }
            for row in rows
        ]

    def join_integrity(self) -> dict[str, int]:
        metrics_total = int(
            self.conn.execute("SELECT COUNT(*) FROM evaluation_metrics").fetchone()[0]
        )
        metrics_with_result_id = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM evaluation_metrics WHERE evaluation_result_id IS NOT NULL"
            ).fetchone()[0]
        )
        instance_total = int(
            self.conn.execute("SELECT COUNT(*) FROM instance_evaluations").fetchone()[0]
        )
        instance_with_result_id = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM instance_evaluations WHERE evaluation_result_id IS NOT NULL"
            ).fetchone()[0]
        )

        matched_instances = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations i
                JOIN evaluation_metrics m
                  ON i.evaluation_id = m.evaluation_id
                 AND i.result_join_id = m.result_join_id
                """
            ).fetchone()[0]
        )

        return {
            "aggregate_metrics_total": metrics_total,
            "aggregate_metrics_with_evaluation_result_id": metrics_with_result_id,
            "instance_rows_total": instance_total,
            "instance_rows_with_evaluation_result_id": instance_with_result_id,
            "instance_rows_joined_to_aggregate": matched_instances,
        }
