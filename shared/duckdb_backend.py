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


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return cleaned


def _name_join_id(evaluation_name: str) -> str:
    return f"name:{_normalize_name(evaluation_name)}"


_UUID_FILE_RE = re.compile(
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(?:_samples)?(?:\.jsonl?)?$",
    re.IGNORECASE,
)


def _file_link_key(file_path: str | Path | None) -> str | None:
    if file_path is None:
        return None

    filename = Path(str(file_path)).name.strip()
    if not filename:
        return None

    match = _UUID_FILE_RE.search(filename)
    if match:
        return f"uuid:{match.group('uuid').lower()}"

    return f"name:{filename.lower()}"


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
                detailed_results_file_path TEXT,
                detailed_results_file_key TEXT,
                detailed_results_total_rows INTEGER,
                raw_json TEXT NOT NULL
            )
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_detailed_results_key
            ON evaluation_runs (detailed_results_file_key)
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                evaluation_id TEXT NOT NULL,
                metric_index INTEGER NOT NULL,
                evaluation_result_id TEXT,
                result_join_id TEXT NOT NULL,
                name_join_id TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_metrics_name_join
            ON evaluation_metrics (evaluation_id, name_join_id)
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
                name_join_id TEXT,
                join_key_source TEXT NOT NULL,
                original_evaluation_id TEXT,
                evaluation_id_validation_status TEXT,
                ingest_source_path TEXT,
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
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instance_name_join
            ON instance_evaluations (evaluation_id, name_join_id)
            """
        )

        # Backward-compatible migration path for databases created before name_join_id existed.
        self.conn.execute(
            """
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS detailed_results_file_path TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS detailed_results_file_key TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS detailed_results_total_rows INTEGER
            """
        )
        self.conn.execute(
            """
            ALTER TABLE evaluation_metrics
            ADD COLUMN IF NOT EXISTS name_join_id TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS name_join_id TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS original_evaluation_id TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS evaluation_id_validation_status TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS ingest_source_path TEXT
            """
        )

        runs_missing_detailed_results = self.conn.execute(
            """
            SELECT evaluation_id, raw_json
            FROM evaluation_runs
            WHERE detailed_results_file_key IS NULL
              AND raw_json LIKE '%detailed_evaluation_results%'
            """
        ).fetchall()
        for evaluation_id, raw_json in runs_missing_detailed_results:
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue

            (
                detailed_results_file_path,
                detailed_results_file_key,
                detailed_results_total_rows,
            ) = self._detailed_results_info(payload)
            self.conn.execute(
                """
                UPDATE evaluation_runs
                SET detailed_results_file_path = ?,
                    detailed_results_file_key = ?,
                    detailed_results_total_rows = ?
                WHERE evaluation_id = ?
                """,
                [
                    detailed_results_file_path,
                    detailed_results_file_key,
                    detailed_results_total_rows,
                    evaluation_id,
                ],
            )

        metrics_missing_name_join = self.conn.execute(
            """
            SELECT evaluation_id, metric_index, evaluation_name
            FROM evaluation_metrics
            WHERE name_join_id IS NULL
            """
        ).fetchall()
        for evaluation_id, metric_index, evaluation_name in metrics_missing_name_join:
            self.conn.execute(
                """
                UPDATE evaluation_metrics
                SET name_join_id = ?
                WHERE evaluation_id = ? AND metric_index = ?
                """,
                [_name_join_id(evaluation_name), evaluation_id, metric_index],
            )

        instances_missing_name_join = self.conn.execute(
            """
            SELECT evaluation_id, sample_id, result_join_id, evaluation_name
            FROM instance_evaluations
            WHERE name_join_id IS NULL
            """
        ).fetchall()
        for evaluation_id, sample_id, result_join_id, evaluation_name in instances_missing_name_join:
            self.conn.execute(
                """
                UPDATE instance_evaluations
                SET name_join_id = ?
                WHERE evaluation_id = ? AND sample_id = ? AND result_join_id = ?
                """,
                [_name_join_id(evaluation_name), evaluation_id, sample_id, result_join_id],
            )

        self.conn.execute(
            """
            UPDATE instance_evaluations
            SET original_evaluation_id = evaluation_id
            WHERE original_evaluation_id IS NULL
            """
        )
        self.conn.execute(
            """
            UPDATE instance_evaluations
            SET evaluation_id_validation_status = 'legacy_unvalidated'
            WHERE evaluation_id_validation_status IS NULL
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

    def _join_ids(
        self,
        evaluation_result_id: str | None,
        evaluation_name: str,
    ) -> tuple[str, str, str]:
        name_join_id = _name_join_id(evaluation_name)
        if evaluation_result_id:
            return evaluation_result_id, name_join_id, "evaluation_result_id"
        return name_join_id, name_join_id, "evaluation_name_fallback"

    def _detailed_results_info(
        self, payload: dict[str, Any]
    ) -> tuple[str | None, str | None, int | None]:
        detailed_results = payload.get("detailed_evaluation_results") or {}
        if not isinstance(detailed_results, dict):
            return None, None, None

        file_path = detailed_results.get("file_path")
        if file_path is not None:
            file_path = str(file_path).strip() or None

        return (
            file_path,
            _file_link_key(file_path),
            _to_int(detailed_results.get("total_rows")),
        )

    def _expected_evaluation_id_for_instance_path(
        self, path: Path
    ) -> tuple[str | None, str]:
        file_link_key = _file_link_key(path)
        if file_link_key is None:
            return None, "no_matching_detailed_results"

        rows = self.conn.execute(
            """
            SELECT evaluation_id
            FROM evaluation_runs
            WHERE detailed_results_file_key = ?
            ORDER BY evaluation_id
            """,
            [file_link_key],
        ).fetchall()
        if len(rows) == 1:
            return str(rows[0][0]), "matched_detailed_results"
        if len(rows) > 1:
            return None, "ambiguous_detailed_results"
        return None, "no_matching_detailed_results"

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

                model_id = str(model_info.get("id", "")).strip()
                if not model_id:
                    raise ValueError(
                        f"Row with evaluation_id={evaluation_id!r} is missing required"
                        " field model_info.id"
                    )

                try:
                    (
                        detailed_results_file_path,
                        detailed_results_file_key,
                        detailed_results_total_rows,
                    ) = self._detailed_results_info(payload)
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
                            detailed_results_file_path,
                            detailed_results_file_key,
                            detailed_results_total_rows,
                            raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            model_id,
                            model_info.get("name"),
                            model_info.get("developer"),
                            eval_library.get("name"),
                            eval_library.get("version"),
                            dedupe.get("run_fingerprint"),
                            dedupe.get("content_hash"),
                            dedupe.get("hash_algorithm"),
                            dedupe.get("canonicalization_version"),
                            detailed_results_file_path,
                            detailed_results_file_key,
                            detailed_results_total_rows,
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

                        result_join_id, name_join_id, join_key_source = self._join_ids(
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
                                name_join_id,
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
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                evaluation_id,
                                metric_index,
                                evaluation_result_id,
                                result_join_id,
                                name_join_id,
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
                        metrics += 1

                    runs += 1
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise

        return {"runs_ingested": runs, "metrics_ingested": metrics}

    def ingest_instance_jsonl(self, jsonl_path: str | Path) -> dict[str, Any]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)

        rows = 0
        validated_rows = 0
        repaired_rows = 0
        filled_rows = 0
        unvalidated_rows = 0
        expected_evaluation_id, file_link_lookup_status = (
            self._expected_evaluation_id_for_instance_path(path)
        )
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue

                payload = json.loads(raw_line)
                original_evaluation_id = str(payload.get("evaluation_id", "")).strip()
                sample_id = str(payload.get("sample_id", "")).strip()
                evaluation_name = str(payload.get("evaluation_name", "")).strip()
                model_id = str(payload.get("model_id", "")).strip()

                evaluation_id = original_evaluation_id
                if expected_evaluation_id is not None:
                    if original_evaluation_id == expected_evaluation_id:
                        evaluation_id_validation_status = "validated_from_detailed_results"
                    elif original_evaluation_id:
                        evaluation_id = expected_evaluation_id
                        evaluation_id_validation_status = "repaired_from_detailed_results"
                    else:
                        evaluation_id = expected_evaluation_id
                        evaluation_id_validation_status = "filled_from_detailed_results"
                elif file_link_lookup_status == "ambiguous_detailed_results":
                    evaluation_id_validation_status = "payload_only_ambiguous_file_link"
                else:
                    evaluation_id_validation_status = "payload_only_unvalidated"

                if not all([evaluation_id, sample_id, evaluation_name, model_id]):
                    continue

                evaluation_result_id = payload.get("evaluation_result_id")
                if evaluation_result_id is not None:
                    evaluation_result_id = str(evaluation_result_id)

                result_join_id, name_join_id, join_key_source = self._join_ids(
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

                delete_evaluation_ids = {evaluation_id}
                if original_evaluation_id and original_evaluation_id != evaluation_id:
                    delete_evaluation_ids.add(original_evaluation_id)
                for delete_evaluation_id in delete_evaluation_ids:
                    self.conn.execute(
                        """
                        DELETE FROM instance_evaluations
                        WHERE evaluation_id = ? AND sample_id = ? AND result_join_id = ? AND name_join_id = ?
                        """,
                        [delete_evaluation_id, sample_id, result_join_id, name_join_id],
                    )

                self.conn.execute(
                    """
                    INSERT INTO instance_evaluations (
                        evaluation_id,
                        sample_id,
                        evaluation_name,
                        evaluation_result_id,
                        result_join_id,
                        name_join_id,
                        join_key_source,
                        original_evaluation_id,
                        evaluation_id_validation_status,
                        ingest_source_path,
                        model_id,
                        score,
                        is_correct,
                        raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evaluation_id,
                        sample_id,
                        evaluation_name,
                        evaluation_result_id,
                        result_join_id,
                        name_join_id,
                        join_key_source,
                        original_evaluation_id or None,
                        evaluation_id_validation_status,
                        str(path),
                        model_id,
                        score,
                        is_correct,
                        json.dumps(payload, ensure_ascii=False),
                    ],
                )
                if evaluation_id_validation_status == "validated_from_detailed_results":
                    validated_rows += 1
                elif evaluation_id_validation_status == "repaired_from_detailed_results":
                    repaired_rows += 1
                elif evaluation_id_validation_status == "filled_from_detailed_results":
                    filled_rows += 1
                else:
                    unvalidated_rows += 1
                rows += 1

        return {
            "instance_rows_ingested": rows,
            "instance_rows_validated": validated_rows,
            "instance_rows_repaired": repaired_rows,
            "instance_rows_filled_from_file_link": filled_rows,
            "instance_rows_unvalidated": unvalidated_rows,
            "file_link_lookup_status": file_link_lookup_status,
            "expected_evaluation_id": expected_evaluation_id,
        }

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
        source_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            WITH agg AS (
                SELECT
                    r.model_id,
                    r.source_name,
                    COALESCE(m.metric_name, m.evaluation_name) AS metric_name,
                    COALESCE(m.metric_kind, 'unknown') AS metric_kind,
                    AVG(m.score) AS avg_score,
                    COUNT(*) AS observations,
                    COALESCE(BOOL_OR(m.lower_is_better), FALSE) AS lower_is_better
                FROM evaluation_metrics m
                JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
                WHERE (? IS NULL OR m.metric_kind = ?)
                  AND (? IS NULL OR COALESCE(m.metric_name, m.evaluation_name) = ?)
                  AND (? IS NULL OR r.source_name = ?)
                GROUP BY 1, 2, 3, 4
            )
            SELECT *
            FROM agg
            ORDER BY CASE WHEN lower_is_better THEN avg_score ELSE -avg_score END ASC NULLS LAST
            LIMIT ?
            """,
            [metric_kind, metric_kind, metric_name, metric_name, source_name, source_name, limit],
        ).fetchall()

        return [
            {
                "model_id": row[0],
                "source_name": row[1],
                "metric_name": row[2],
                "metric_kind": row[3],
                "avg_score": float(row[4]),
                "observations": int(row[5]),
                "lower_is_better": bool(row[6]),
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
        runs_with_detailed_results = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM evaluation_runs
                WHERE detailed_results_file_path IS NOT NULL
                """
            ).fetchone()[0]
        )
        instance_rows_validated = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations
                WHERE evaluation_id_validation_status = 'validated_from_detailed_results'
                """
            ).fetchone()[0]
        )
        instance_rows_repaired = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations
                WHERE evaluation_id_validation_status IN (
                    'repaired_from_detailed_results',
                    'filled_from_detailed_results'
                )
                """
            ).fetchone()[0]
        )
        instance_rows_unvalidated = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations
                WHERE evaluation_id_validation_status IN (
                    'payload_only_unvalidated',
                    'payload_only_ambiguous_file_link',
                    'legacy_unvalidated'
                )
                """
            ).fetchone()[0]
        )

        matched_instances_exact = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT i.evaluation_id, i.sample_id, i.result_join_id
                    FROM instance_evaluations i
                    JOIN evaluation_metrics m
                      ON i.evaluation_id = m.evaluation_id
                     AND i.evaluation_result_id IS NOT NULL
                     AND m.evaluation_result_id = i.evaluation_result_id
                )
                """
            ).fetchone()[0]
        )

        matched_instances_name_fallback = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT i.evaluation_id, i.sample_id, i.result_join_id
                    FROM instance_evaluations i
                    JOIN evaluation_metrics m
                      ON i.evaluation_id = m.evaluation_id
                     AND i.name_join_id = m.name_join_id
                     AND NOT (
                        i.evaluation_result_id IS NOT NULL
                        AND m.evaluation_result_id IS NOT NULL
                     )
                )
                """
            ).fetchone()[0]
        )

        return {
            "aggregate_metrics_total": metrics_total,
            "aggregate_metrics_with_evaluation_result_id": metrics_with_result_id,
            "aggregate_runs_with_detailed_results": runs_with_detailed_results,
            "instance_rows_total": instance_total,
            "instance_rows_with_evaluation_result_id": instance_with_result_id,
            "instance_rows_validated_from_detailed_results": instance_rows_validated,
            "instance_rows_repaired_via_detailed_results": instance_rows_repaired,
            "instance_rows_without_trusted_file_link_validation": instance_rows_unvalidated,
            "instance_rows_joined_by_evaluation_result_id": matched_instances_exact,
            "instance_rows_joined_by_name_fallback": matched_instances_name_fallback,
            "instance_rows_joined_to_aggregate": (
                matched_instances_exact + matched_instances_name_fallback
            ),
        }

    def orphan_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            WITH coverage AS (
                SELECT
                    r.evaluation_id,
                    r.source_name,
                    r.model_id,
                    r.detailed_results_file_path,
                    r.detailed_results_total_rows,
                    COUNT(i.sample_id) AS ingested_instance_rows,
                    COALESCE(
                        SUM(
                            CASE
                                WHEN i.evaluation_id_validation_status IN (
                                    'repaired_from_detailed_results',
                                    'filled_from_detailed_results'
                                ) THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ) AS repaired_instance_rows
                FROM evaluation_runs r
                LEFT JOIN instance_evaluations i
                  ON i.evaluation_id = r.evaluation_id
                WHERE r.detailed_results_file_path IS NOT NULL
                GROUP BY 1, 2, 3, 4, 5
            ),
            issues AS (
                SELECT
                    evaluation_id,
                    source_name,
                    model_id,
                    detailed_results_file_path,
                    detailed_results_total_rows,
                    ingested_instance_rows,
                    repaired_instance_rows,
                    CASE
                        WHEN detailed_results_total_rows IS NULL THEN NULL
                        ELSE GREATEST(detailed_results_total_rows - ingested_instance_rows, 0)
                    END AS missing_instance_rows,
                    CASE
                        WHEN ingested_instance_rows = 0 THEN 'missing_ingested_instances'
                        WHEN repaired_instance_rows > 0 THEN 'repaired_instance_evaluation_id'
                        WHEN detailed_results_total_rows IS NOT NULL
                             AND ingested_instance_rows < detailed_results_total_rows
                            THEN 'partial_instance_ingest'
                        ELSE NULL
                    END AS issue_type
                FROM coverage
            )
            SELECT
                evaluation_id,
                source_name,
                model_id,
                detailed_results_file_path,
                detailed_results_total_rows,
                ingested_instance_rows,
                missing_instance_rows,
                repaired_instance_rows,
                issue_type
            FROM issues
            WHERE issue_type IS NOT NULL
            ORDER BY
                CASE issue_type
                    WHEN 'missing_ingested_instances' THEN 0
                    WHEN 'partial_instance_ingest' THEN 1
                    WHEN 'repaired_instance_evaluation_id' THEN 2
                    ELSE 3
                END,
                evaluation_id
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return [
            {
                "evaluation_id": row[0],
                "source_name": row[1],
                "model_id": row[2],
                "detailed_results_file_path": row[3],
                "detailed_results_total_rows": (
                    int(row[4]) if row[4] is not None else None
                ),
                "ingested_instance_rows": int(row[5]),
                "missing_instance_rows": int(row[6]) if row[6] is not None else None,
                "repaired_instance_rows": int(row[7]),
                "issue_type": row[8],
            }
            for row in rows
        ]

    def identifier_issues(self, limit: int = 100) -> dict[str, Any]:
        summary = {
            "aggregate_metrics_missing_evaluation_result_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM evaluation_metrics
                    WHERE evaluation_result_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "aggregate_metrics_missing_metric_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM evaluation_metrics
                    WHERE metric_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "aggregate_metrics_missing_metric_kind": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM evaluation_metrics
                    WHERE metric_kind IS NULL
                    """
                ).fetchone()[0]
            ),
            "instance_rows_missing_evaluation_result_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_result_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "instance_rows_repaired_evaluation_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_id_validation_status = 'repaired_from_detailed_results'
                    """
                ).fetchone()[0]
            ),
            "instance_rows_filled_evaluation_id_from_file_link": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_id_validation_status = 'filled_from_detailed_results'
                    """
                ).fetchone()[0]
            ),
            "instance_rows_without_trusted_file_link_validation": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_id_validation_status IN (
                        'payload_only_unvalidated',
                        'payload_only_ambiguous_file_link',
                        'legacy_unvalidated'
                    )
                    """
                ).fetchone()[0]
            ),
        }

        repaired_rows = self.conn.execute(
            """
            SELECT
                ingest_source_path,
                sample_id,
                evaluation_name,
                model_id,
                original_evaluation_id,
                evaluation_id,
                evaluation_id_validation_status
            FROM instance_evaluations
            WHERE evaluation_id_validation_status IN (
                'repaired_from_detailed_results',
                'filled_from_detailed_results'
            )
            ORDER BY ingest_source_path, sample_id
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        unvalidated_rows = self.conn.execute(
            """
            SELECT
                ingest_source_path,
                sample_id,
                evaluation_name,
                model_id,
                original_evaluation_id,
                evaluation_id,
                evaluation_id_validation_status
            FROM instance_evaluations
            WHERE evaluation_id_validation_status IN (
                'payload_only_unvalidated',
                'payload_only_ambiguous_file_link',
                'legacy_unvalidated'
            )
            ORDER BY
                CASE evaluation_id_validation_status
                    WHEN 'payload_only_ambiguous_file_link' THEN 0
                    WHEN 'payload_only_unvalidated' THEN 1
                    ELSE 2
                END,
                ingest_source_path,
                sample_id
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        aggregate_metric_rows = self.conn.execute(
            """
            SELECT
                evaluation_id,
                metric_index,
                evaluation_name,
                metric_id,
                metric_name,
                metric_kind
            FROM evaluation_metrics
            WHERE evaluation_result_id IS NULL
               OR metric_id IS NULL
               OR metric_kind IS NULL
            ORDER BY evaluation_id, metric_index
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return {
            "summary": summary,
            "repaired_instance_examples": [
                {
                    "ingest_source_path": row[0],
                    "sample_id": row[1],
                    "evaluation_name": row[2],
                    "model_id": row[3],
                    "original_evaluation_id": row[4],
                    "stored_evaluation_id": row[5],
                    "evaluation_id_validation_status": row[6],
                }
                for row in repaired_rows
            ],
            "unvalidated_instance_examples": [
                {
                    "ingest_source_path": row[0],
                    "sample_id": row[1],
                    "evaluation_name": row[2],
                    "model_id": row[3],
                    "original_evaluation_id": row[4],
                    "stored_evaluation_id": row[5],
                    "evaluation_id_validation_status": row[6],
                }
                for row in unvalidated_rows
            ],
            "aggregate_metric_examples": [
                {
                    "evaluation_id": row[0],
                    "metric_index": int(row[1]),
                    "evaluation_name": row[2],
                    "metric_id": row[3],
                    "metric_name": row[4],
                    "metric_kind": row[5],
                }
                for row in aggregate_metric_rows
            ],
        }
