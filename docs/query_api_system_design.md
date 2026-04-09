# Query API System Design

> Based on the current `every_eval_ever` package schemas (`eval.schema.json` v0.2.2 and `instance_level_eval.schema.json` v0.2.2), cross-checked against `every_eval_ever` commit `899a12e6d7b73913656be13a37836e7aaa310280` and the live Hugging Face datastore `evaleval/EEE_datastore` raw repo commit `2f19bdd29fe04c4a979223493a15728adf135787` on April 9, 2026. The raw datastore repo currently contains 6,270 aggregate JSON files, 41,616 metric results, and 5,497 sample rows across 395 sample JSONLs. The Hugging Face datasets-server/viewer subset currently exposes only 6,087 aggregate rows and should not be treated as interchangeable with the raw repo. Unless explicitly stated otherwise, the counts below refer to raw repo contents because that is the authoritative ingest source.

## Overview

The backend already stores enough data to support model discovery, source-aware comparison, metric normalization, instance drilldown, temporal analysis, and data quality checks. The problem is not storage; it is that the API currently exposes only a minimal analytical surface: one fixed ranking query plus one join-quality read. This design condenses the next step into a small, practical query API that reflects current data quality rather than idealized schema semantics.

Phase 1 should use DuckDB as the analytical store and query-serving engine. That is the right tradeoff for speed, simplicity, and local OLAP performance. The long-term design, however, must assume that evaluation data arrives continuously from automated scrapers and evaluation bots. The serving layer should therefore be designed so that a streaming ingestion path can feed the same query model over time.

Current state:

| Signal | Current state |
|--------|---------------|
| API surface | 6 endpoints; `GET /metrics/top-models` is the only ranking endpoint and `GET /join-integrity` is the only join-quality read |
| Aggregate data | 6,270 aggregate runs and 41,616 metric rows in the raw datastore repo; the Hugging Face datasets-server/viewer subset currently exposes only 6,087 aggregate rows |
| Instance data | 5,497 sample rows across 395 sample JSONLs in 8 sample-bearing benchmark configs; 395 aggregate rows include `detailed_evaluation_results` references |
| Source provenance | 5,738 `documentation` rows and 532 `evaluation_run` rows; `source_name` is populated on all observed aggregate rows |
| Models | 5,386 distinct model IDs; 233 appear in more than one source |
| Metric identifiers | `metric_id`, `metric_name`, `metric_unit`, and `evaluation_result_id` are populated on 4,797 / 41,616 metric rows; `metric_kind` is populated on 4,484 / 41,616 |
| Join identifiers | All observed sample rows omit `evaluation_result_id`; 231 sample-linked aggregate runs already have at least one `evaluation_result_id`, so exact and fallback joins must coexist. `name_join_id` helps only when `evaluation_id` is already correct. |
| Reference integrity | Most linked sample files reuse the aggregate `evaluation_id`, but at least one live `detailed_evaluation_results` reference currently points to a sample file whose `evaluation_id` does not match the aggregate row |
| Score scales | Mixed and not directly comparable |
| Timestamps | `retrieved_timestamp` is universal, but top-level `evaluation_timestamp` appears on only 279 / 6,270 rows and per-result `evaluation_timestamp` on 599 / 41,616 metric rows |
| Ingestion model | Bulk ingest endpoints today; long term needs continuous submissions from automated producers |

The design should follow six rules:

- Prefer `metric_id` when present, but treat `(source_name, evaluation_name)` as the minimum safe fallback metric identity.
- Default to source-local normalization before any cross-evaluation or cross-source comparison.
- Move repeated parsing and disambiguation into ingestion, not query time.
- Treat `name_join_id` as a metric-selective fallback inside a trusted `evaluation_id`, not as a repair mechanism for malformed file linkage.
- Expose only the endpoints that can be defended with current data quality.
- Make sparsity explicit: some endpoints will return thin results until aliasing, repeated scrapes, and real instance data exist.

## Requirements

The query API must support the following product requirements:

- Discover what data exists: sources, models, evaluations, and model-to-evaluation coverage.
- Compare models safely within a source and, where possible, across sources.
- Normalize scores in a way that respects heterogeneous scales and `lower_is_better`.
- Drill down from aggregate metrics to instances once real instance data is available.
- Audit the data for name collisions, outliers, orphaned rows, identifier issues, and duplicate runs.
- Leave room for later correlation, efficiency, and temporal analyses without forcing premature API commitments.
- Accept continuous submissions from automated scrapers and evaluation bots without changing query semantics.

The design must also preserve the harder join patterns that appear later in the roadmap:

| Join pattern | Why it is hard | Design implication |
|-------------|----------------|--------------------|
| Cross-source ranking comparison | The same `evaluation_name` can mean different things in different sources, and model IDs do not reliably align | Require `evaluation_name_registry`, `model_aliases`, and explicit normalization |
| Pairwise model correlation across evaluations | Requires joining normalized scores across shared canonical evaluation identities | Keep metric identity source-aware and normalization queryable |
| Aggregate-to-instance consistency checks | Live data mixes rows with and without `evaluation_result_id`, some sample files only cover one metric within a multi-metric aggregate run, and at least one linked sample file has a malformed `evaluation_id` | Preserve `(evaluation_id, result_join_id)` as the preferred exact join key and `(evaluation_id, name_join_id)` as the metric-selective fallback path; quarantine or repair broken `evaluation_id` references |
| Efficiency analysis such as cost-per-accuracy | Requires joining instance token and latency fields back to normalized aggregate scores | Extract structured instance fields at ingestion time |
| Name-collision and integrity audits | Requires comparing metrics across sources without assuming global uniqueness | Keep source information available on hot analytical paths |

The API and storage layer must satisfy these technical requirements:

- `(evaluation_id, result_join_id)` remains the preferred exact join key between `evaluation_metrics` and `instance_evaluations`, with `evaluation_result_id` as the preferred stable component when present.
- `(evaluation_id, name_join_id)` must also be materialized as a fallback join path for mixed ID/no-ID data, but only as a metric-selective join inside a validated `evaluation_id`.
- Analytical queries must not rely on `evaluation_name` alone across sources.
- Queries should prefer `metric_id` and `metric_kind` when present, but degrade safely to source-scoped display names when they are absent.
- Queries must tolerate `source_name = NULL` by using an explicit missing-source bucket or explicit filtering, not by collapsing to `source_organization_name`.
- The system must avoid repeated JSON extraction from `raw_json` for hot analytical paths.
- Ingestion must detect malformed instance `evaluation_id` references and quarantine or repair them using trusted file linkage metadata before advertising aggregate-to-instance coverage.
- Query logic must not infer that a sample file covers every aggregate metric in a run; instance coverage is often metric-selective.
- Normalization must be explicit and method-scoped, not implicit in endpoint behavior.
- The ingestion contract must be decoupled from the serving schema so Phase 1 DuckDB storage can later sit behind a streaming materialization layer.
- Ingestion must be idempotent and tolerate duplicate, late, and out-of-order submissions from scrapers and bots.
- The system must preserve parsed `evaluation_timestamp` and `retrieved_timestamp` from the schema plus internal `ingested_at` as distinct timestamps, and it should preserve per-result `evaluation_timestamp` when present.
- API responses must be emitted as UTF-8 `application/json`; if clients render model or evaluation names into HTML, escaping belongs at render time rather than in the API payload.

The current data imposes these hard constraints:

- The raw datastore repo and the Hugging Face datasets-server/viewer subset diverge today, so counts and coverage derived from one surface must be labeled as such.
- Cross-source comparison is now materially possible because 233 models span multiple sources, which makes source-unsafe endpoints more dangerous, not less.
- `metric_kind`, `metric_name`, and `metric_id` are useful on a meaningful minority of rows, but they are still too incomplete to be treated as globally reliable.
- `min_score` and `max_score` are not reliable theoretical bounds, so naive min-max normalization is not acceptable.
- Instance-level analytics are now possible for some benchmarks, but coverage is still sparse relative to aggregate coverage.
- Temporal analysis is still limited because true evaluation timestamps are sparse even though retrieval timestamps exist.
- Aggregate-to-instance linkage must handle mixed populations where aggregate rows may have `evaluation_result_id` but matching sample rows do not; in the current raw repo, sample rows never populate `evaluation_result_id`.
- Some sample files cover only one metric in a multi-metric aggregate run, so aggregate-to-instance joins are metric-selective rather than run-wide.
- `name_join_id` does not repair malformed file linkage by itself; at least one live linked sample file currently has a mismatched `evaluation_id`.
- DuckDB is a good Phase 1 serving store, but continuous writes should be serialized or micro-batched rather than issued concurrently by many producers.

## Timeline

| Phase | Goal | Deliverables |
|------|------|--------------|
| Phase 1 | Ship the DuckDB-backed query API | DuckDB as the serving store; keep `/join-integrity`; add `/sources`, `/models`, `/evaluations`, `/models/{model_id}/evaluations`, `/models/multi-source`, `/quality/name-collisions`, `/quality/orphan-runs`, `/quality/identifier-issues`, `/quality/outliers` |
| Phase 2 | Make DuckDB ingestion streaming-compatible | Serialized or micro-batched ingest path; denormalized `source_name`; parsed timestamps; extracted token and latency fields; idempotent dedup; benchmark-driven selective indexes only if profiling justifies them |
| Phase 3 | Enable safe normalized comparison | `evaluation_name_registry`, `model_aliases`, `/metrics/normalized`, and `/compare/rankings` with explicit normalization and source-scoped semantics |
| Phase 4 | Support continuous ingest and advanced analysis | Automated scraper and bot submissions through a durable ingest layer feeding materialized query tables; then temporal history, correlation, efficiency, and aggregate-vs-instance consistency checks |

Phase gates:

- Do not ship cross-source ranking endpoints without source-aware metric identity and explicit normalization.
- Do not ship min-max normalization until curated scale bounds exist in a registry.
- Do not ship performance or token analytics until those fields are extracted at ingestion time.
- Do not promise temporal trend endpoints until data spans multiple scrape dates or evaluation dates.
- Do not allow many scrapers or bots to write directly to DuckDB concurrently; funnel writes through a single ingestion service or micro-batch worker.
- Preserve a stable ingestion contract so the serving layer can evolve beyond DuckDB without changing producer behavior.

## Implementation

The implementation should stay small and opinionated. The current DuckDB prototype is the starting point, not a blank slate.

Data model changes:

| Area | Change | Why |
|------|--------|-----|
| `evaluation_metrics` | Retain and standardize denormalized `source_name`, parsed `result_evaluation_timestamp TIMESTAMP`, and `name_join_id` in the serving schema | The current prototype already materializes `name_join_id`; Phase 1 should preserve that fallback path, avoid repeated joins for most analytical queries, and keep per-result timing |
| `evaluation_runs` | Add parsed `retrieved_at TIMESTAMP`, top-level `evaluation_timestamp TIMESTAMP`, and `ingested_at TIMESTAMP` | Separate record creation time, evaluation time, and system ingestion time while preserving current schema semantics |
| `instance_evaluations` | Retain `name_join_id`; extract `total_tokens`, `input_tokens`, `output_tokens`, `latency_ms`, `time_to_first_token_ms`, `interaction_type` | The current prototype already materializes `name_join_id`; Phase 1 should make instance analytics queryable without JSON parsing and preserve fallback linkage when sample rows omit `evaluation_result_id` |
| New table | `evaluation_name_registry` with canonical name, metric family, verified scale bounds, and `lower_is_better` | Required for safe cross-source comparison and verified min-max normalization |
| New table | `model_aliases` for canonical model identity | Required for non-trivial cross-source model matching |

Ingestion architecture:

- In Phase 1, DuckDB is the primary analytical store and query engine.
- Producers should submit data to an application-owned ingestion service, not write to the DuckDB file directly.
- The ingestion service should validate payloads, compute stable dedup keys, stamp `ingested_at`, serialize writes into DuckDB, and quarantine or repair malformed aggregate-to-instance references before materializing analytical tables.
- In the long term, the ingestion service should write to a durable append-only landing layer, such as a queue or event log, and background workers should materialize serving tables from that stream.
- Query endpoints should read from materialized analytical tables, not directly from raw ingestion events.

Index changes:

- Keep the existing `(evaluation_id, result_join_id)` indexes for aggregate-to-instance joins.
- Keep the existing `(evaluation_id, name_join_id)` indexes for mixed exact/fallback aggregate-to-instance joins.
- Add explicit single-column indexes only for proven highly selective equality filters such as `evaluation_runs(model_id)` or `evaluation_runs(source_name)`.
- Re-check candidate indexes with `EXPLAIN ANALYZE`; DuckDB ART indexes do not materially improve join-heavy, aggregation-heavy, or sorting-heavy endpoints.

Normalization policy:

- When a registry-approved `metric_id` exists, use it ahead of display-name matching.
- Default to percentile rank within `(source_name, evaluation_name)` for display and comparison.
- Offer z-score as an explicit analytical mode.
- Offer min-max normalization only when scale bounds come from verified `evaluation_name_registry` entries.
- Reject any endpoint behavior that silently aggregates heterogeneous raw scores into a composite metric.

Endpoint plan:

| Method | Path | Notes |
|--------|------|-------|
| GET | `/sources` | Distinct sources with counts |
| GET | `/models` | Filterable model listing with source counts |
| GET | `/evaluations` | Evaluation listing scoped by source |
| GET | `/models/{model_id}/evaluations` | Coverage view for one model |
| GET | `/models/multi-source` | Useful immediately, but results will be sparse |
| GET | `/join-integrity` | Keep the current prototype coverage read; useful immediately for mixed exact/fallback linkage debugging |
| GET | `/metrics/normalized` | Phase 3; requires explicit `normalization=percentile|zscore|minmax` |
| GET | `/compare/rankings` | Phase 3; only valid for source-safe evaluation identities |
| GET | `/quality/name-collisions` | Detect reused names with conflicting semantics |
| GET | `/quality/orphan-runs` | Now useful immediately because real instance data exists, though coverage is still partial; include malformed `evaluation_id` reference cases, not only missing sample files |
| GET | `/quality/identifier-issues` | Identifier completeness and encoding audit |
| GET | `/quality/outliers` | Source-local z-score outlier detection |

Future analytical endpoints that the design should explicitly accommodate:

| Method | Path | Dependency |
|--------|------|------------|
| POST | `/analysis/correlation-matrix` | Canonical evaluation identity plus normalized scores |
| GET | `/analysis/rank-flips` | Cross-source ranking support and model aliasing |
| GET | `/analysis/efficiency` | Real instance data plus extracted token fields |
| GET | `/quality/aggregate-instance-consistency` | Stable aggregate-to-instance joins, repair/quarantine of malformed references, and metric-family awareness |

Query safety rules:

- Always group or partition by `COALESCE(source_name, '__missing_source__')` when source identity matters; never substitute `source_organization_name` for missing source identity.
- Prefer `metric_id` over display name when it is present and registry-approved.
- Never treat `evaluation_name` as globally unique.
- Treat `name_join_id` fallback as metric-selective inside a validated `evaluation_id`; do not assume a sample file covers sibling metrics in the same aggregate run.
- Detect and surface malformed `evaluation_id` references explicitly; do not silently coalesce them by filename or source name at query time.
- Never use row-level `min_score` and `max_score` as authoritative scale bounds.
- Emit UTF-8 `application/json` responses, preserve raw model and evaluation strings in the payload, and URL-encode evaluation identifiers in requests.
- Return explicit errors for unsupported modes, especially unverified min-max normalization and temporal queries without sufficient history.
- Treat advanced analytical joins as a first-class design constraint even when the initial endpoint set is smaller.

This plan intentionally stages the more complex joins and analyses rather than ignoring them. The correct sequence is to make metric identity safe, reduce ingestion-time ambiguity, and then expose a smaller set of endpoints that can be trusted while preserving the join structure needed for later phases.
