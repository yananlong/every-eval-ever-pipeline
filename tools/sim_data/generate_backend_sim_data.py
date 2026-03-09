#!/usr/bin/env python3
"""
Generate backend-oriented simulation data using real Every Eval Ever rows.

The generator pulls aggregate rows from the Hugging Face datasets-server API,
then emits:
  1) Baseline fixtures (aggregate JSON + synthetic instance JSONL)
  2) Pathology fixtures for backend validation of:
     - dedupe identity behavior
     - metric identity ambiguity
     - deterministic aggregate/instance linkage
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

HF_API_ROOT = "https://huggingface.co/api/datasets"
DATASETS_SERVER_ROOT = "https://datasets-server.huggingface.co"
DEFAULT_DATASET = "evaleval/EEE_datastore"
DEFAULT_CONFIGS = ["global-mmlu-lite", "hfopenllm_v2", "reward-bench"]


@dataclass
class WrittenFixture:
    bucket: str
    scenario: str
    aggregate_relpath: str
    instance_relpath: str | None
    instance_rows: int
    expected_backend_action: str
    notes: str


def _request_json(url: str, timeout: int = 30) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        # Fallback to curl because some local Python SSL trust stores are missing.
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "--fail",
                "--max-time",
                str(timeout),
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)


def get_dataset_sha(dataset: str) -> str | None:
    encoded = quote(dataset, safe="/")
    payload = _request_json(f"{HF_API_ROOT}/{encoded}")
    sha = payload.get("sha")
    return sha if isinstance(sha, str) else None


def list_configs(dataset: str) -> list[str]:
    encoded = quote(dataset, safe="")
    payload = _request_json(f"{DATASETS_SERVER_ROOT}/splits?dataset={encoded}")
    splits = payload.get("splits", [])
    configs: list[str] = []
    for item in splits:
        config = item.get("config")
        if isinstance(config, str) and config not in configs:
            configs.append(config)
    return configs


def fetch_rows(dataset: str, config: str, split: str, limit: int) -> list[dict[str, Any]]:
    encoded_dataset = quote(dataset, safe="")
    encoded_config = quote(config, safe="")
    encoded_split = quote(split, safe="")
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        page_len = min(100, limit - len(rows))
        url = (
            f"{DATASETS_SERVER_ROOT}/rows?dataset={encoded_dataset}"
            f"&config={encoded_config}&split={encoded_split}"
            f"&offset={offset}&length={page_len}"
        )
        payload = _request_json(url)
        page_rows = payload.get("rows", [])
        if not page_rows:
            break
        for wrapped in page_rows:
            row = wrapped.get("row")
            if isinstance(row, dict):
                rows.append(row)
        offset += page_len
    return rows


def sanitize_component(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    return cleaned or "unknown"


def parse_path_components(row: dict[str, Any]) -> tuple[str, str, str]:
    evaluation_id = row.get("evaluation_id")
    benchmark = "unknown_benchmark"
    if isinstance(evaluation_id, str) and "/" in evaluation_id:
        benchmark = evaluation_id.split("/", 1)[0]

    model_id = (
        row.get("model_info", {}).get("id")
        if isinstance(row.get("model_info"), dict)
        else None
    )
    developer = "unknown_developer"
    model_name = "unknown_model"
    if isinstance(model_id, str) and "/" in model_id:
        developer, model_name = model_id.split("/", 1)
    elif isinstance(model_id, str):
        model_name = model_id

    return (
        sanitize_component(benchmark),
        sanitize_component(developer),
        sanitize_component(model_name),
    )


def _json_stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def coerce_schema_compatibility(row: dict[str, Any]) -> dict[str, Any]:
    """
    Light coercion so generated fixtures can pass the current schema:
    - ensure eval_library exists
    - force additional_details values to strings
    """
    out = copy.deepcopy(row)

    eval_library = out.get("eval_library")
    if not isinstance(eval_library, dict):
        out["eval_library"] = {"name": "unknown", "version": "unknown"}
    else:
        eval_library.setdefault("name", "unknown")
        eval_library.setdefault("version", "unknown")

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            rewritten: dict[str, Any] = {}
            for key, val in node.items():
                if key == "additional_details" and isinstance(val, dict):
                    rewritten[key] = {
                        inner_key: _json_stringify(inner_val)
                        for inner_key, inner_val in val.items()
                    }
                else:
                    rewritten[key] = visit(val)
            return rewritten
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    out = visit(out)

    # Source rows frequently store score_details.uncertainty as null.
    # Current schema expects an object when the key exists.
    results = out.get("evaluation_results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            score_details = result.get("score_details")
            if isinstance(score_details, dict) and score_details.get("uncertainty") is None:
                score_details.pop("uncertainty", None)

    return out


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def score_to_probability(result: dict[str, Any]) -> float:
    metric_cfg = result.get("metric_config", {})
    score = _numeric_or_none(result.get("score_details", {}).get("score"))
    min_score = _numeric_or_none(metric_cfg.get("min_score"))
    max_score = _numeric_or_none(metric_cfg.get("max_score"))
    lower_is_better = bool(metric_cfg.get("lower_is_better", False))

    if score is None:
        probability = 0.5
    elif min_score is not None and max_score is not None and max_score > min_score:
        probability = (score - min_score) / (max_score - min_score)
    else:
        # Fallback for missing range metadata.
        probability = max(0.0, min(1.0, score))

    probability = max(0.0, min(1.0, probability))
    if lower_is_better:
        probability = 1.0 - probability
    return probability


def synthesize_instances(
    aggregate: dict[str, Any], instances_per_aggregate: int, rng: random.Random
) -> list[dict[str, Any]]:
    results = aggregate.get("evaluation_results", [])
    model_id = aggregate.get("model_info", {}).get("id", "unknown/unknown")
    evaluation_id = aggregate.get("evaluation_id", "unknown_eval_id")
    schema_version = "0.2.1"
    if not isinstance(results, list) or not results:
        return []

    rows: list[dict[str, Any]] = []
    for i in range(instances_per_aggregate):
        result = results[i % len(results)]
        evaluation_name = str(result.get("evaluation_name", "unknown_eval"))
        probability = score_to_probability(result)
        is_correct = rng.random() < probability
        gold = "A"
        pred = "A" if is_correct else "B"
        prompt = (
            f"[{evaluation_name}] Simulated prompt #{i}. "
            "Answer with a single option."
        )
        sample_hash_input = f"{prompt}|{gold}"
        sample_hash = hashlib.sha256(sample_hash_input.encode("utf-8")).hexdigest()
        latency_ms = 120 + int(rng.random() * 850)
        input_tokens = 60 + int(rng.random() * 200)
        output_tokens = 3 + int(rng.random() * 40)

        row = {
            "schema_version": schema_version,
            "evaluation_id": str(evaluation_id),
            "model_id": str(model_id),
            "evaluation_name": evaluation_name,
            "sample_id": f"sim_{i:06d}",
            "sample_hash": sample_hash,
            "interaction_type": "single_turn",
            "input": {
                "raw": prompt,
                "formatted": None,
                "reference": [gold],
                "choices": ["A", "B", "C", "D"],
            },
            "output": {"raw": [pred], "reasoning_trace": None},
            "messages": None,
            "answer_attribution": [
                {
                    "turn_idx": 0,
                    "source": "output.raw",
                    "extracted_value": pred,
                    "extraction_method": "exact_match",
                    "is_terminal": True,
                }
            ],
            "evaluation": {
                "score": 1.0 if is_correct else 0.0,
                "is_correct": is_correct,
                "num_turns": None,
                "tool_calls_count": None,
            },
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_cache_write": None,
                "input_tokens_cache_read": None,
                "reasoning_tokens": None,
            },
            "performance": {
                "latency_ms": float(latency_ms),
                "time_to_first_token_ms": float(25 + int(rng.random() * 120)),
                "generation_time_ms": float(max(20, latency_ms - 15)),
                "additional_details": None,
            },
            "error": None,
            "metadata": {
                "source": "simulated_from_real",
                "scenario": "baseline",
            },
        }
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def perturb_first_score(row: dict[str, Any], delta: float) -> None:
    results = row.get("evaluation_results", [])
    if not isinstance(results, list) or not results:
        return
    score = _numeric_or_none(results[0].get("score_details", {}).get("score"))
    if score is None:
        return
    score_details = results[0].setdefault("score_details", {})
    score_details["score"] = round(score + delta, 6)


def make_pathology_rows(base_rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str, str]]:
    """
    Returns tuples:
      (scenario_name, aggregate_row, expected_backend_action, notes)
    """
    if not base_rows:
        return []

    rows: list[tuple[str, dict[str, Any], str, str]] = []

    # P1: same semantic content but volatile fields/order changed.
    p1 = copy.deepcopy(base_rows[0])
    if isinstance(p1.get("evaluation_results"), list):
        p1["evaluation_results"] = list(reversed(p1["evaluation_results"]))
    p1["evaluation_id"] = f"{p1.get('evaluation_id', 'unknown')}_rescrape"
    p1["retrieved_timestamp"] = str(datetime.now(UTC).timestamp())
    rows.append(
        (
            "dedupe_same_content_diff_volatile_fields",
            p1,
            "dedupe_drop_if_fingerprint_enabled",
            "Equivalent run with changed volatile identifiers and reordered results.",
        )
    )

    # P2: same evaluation_id but different score content.
    p2_source = base_rows[1] if len(base_rows) > 1 else base_rows[0]
    p2 = copy.deepcopy(p2_source)
    perturb_first_score(p2, delta=0.01)
    rows.append(
        (
            "evaluation_id_collision_different_scores",
            p2,
            "quarantine_conflict_if_idempotency_enforced",
            "Same evaluation_id as baseline but modified score payload.",
        )
    )

    # P3: duplicate evaluation_name entries with different semantics.
    p3_source = base_rows[2] if len(base_rows) > 2 else base_rows[0]
    p3 = copy.deepcopy(p3_source)
    results = p3.get("evaluation_results")
    if isinstance(results, list) and results:
        duplicate = copy.deepcopy(results[0])
        duplicate_cfg = duplicate.setdefault("metric_config", {})
        duplicate_cfg["evaluation_description"] = (
            "Simulated ambiguity: F1 score for same evaluation_name"
        )
        perturb_first_score({"evaluation_results": [duplicate]}, delta=-0.02)
        results.append(duplicate)
    rows.append(
        (
            "ambiguous_metric_identity_same_evaluation_name",
            p3,
            "ambiguous_join_without_metric_id",
            "Same evaluation_name appears twice with different metric descriptions.",
        )
    )

    return rows


def write_fixture(
    out_dir: Path,
    bucket: str,
    scenario: str,
    aggregate_row: dict[str, Any],
    instances_per_aggregate: int,
    rng: random.Random,
    expected_backend_action: str,
    notes: str,
) -> WrittenFixture:
    file_uuid = str(uuid.uuid4())
    benchmark, developer, model_name = parse_path_components(aggregate_row)

    aggregate_relpath = (
        Path(bucket)
        / "data"
        / benchmark
        / developer
        / model_name
        / f"{file_uuid}.json"
    )
    instance_relpath = (
        Path(bucket)
        / "data"
        / benchmark
        / developer
        / model_name
        / f"{file_uuid}.jsonl"
    )

    aggregate_abspath = out_dir / aggregate_relpath
    instance_abspath = out_dir / instance_relpath

    instance_rows = synthesize_instances(aggregate_row, instances_per_aggregate, rng)
    checksum = write_jsonl(instance_abspath, instance_rows)

    aggregate_row = copy.deepcopy(aggregate_row)
    aggregate_row["detailed_evaluation_results"] = {
        "format": "jsonl",
        "file_path": str(instance_relpath),
        "hash_algorithm": "sha256",
        "checksum": checksum,
        "total_rows": len(instance_rows),
        "additional_details": {"simulated_scenario": scenario},
    }

    write_json(aggregate_abspath, aggregate_row)

    return WrittenFixture(
        bucket=bucket,
        scenario=scenario,
        aggregate_relpath=str(aggregate_relpath),
        instance_relpath=str(instance_relpath),
        instance_rows=len(instance_rows),
        expected_backend_action=expected_backend_action,
        notes=notes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate backend simulation data from real EEE dataset rows."
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write fixtures.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id.")
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config names. Defaults to global-mmlu-lite,hfopenllm_v2,reward-bench.",
    )
    parser.add_argument(
        "--rows-per-config",
        type=int,
        default=8,
        help="How many real aggregate rows to pull per config.",
    )
    parser.add_argument(
        "--instances-per-aggregate",
        type=int,
        default=25,
        help="How many synthetic instance rows to create per aggregate fixture.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--schema-compatible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply light coercion for current schema compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    available_configs = set(list_configs(args.dataset))
    requested_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [c for c in requested_configs if c in available_configs]
    if not configs:
        raise SystemExit(
            "None of the requested configs are available in the dataset. "
            f"Requested={requested_configs}, Available={sorted(available_configs)}"
        )

    source_rows: list[dict[str, Any]] = []
    source_pull_log: list[dict[str, Any]] = []
    for config in configs:
        pulled = fetch_rows(args.dataset, config=config, split="train", limit=args.rows_per_config)
        source_rows.extend(pulled)
        source_pull_log.append({"config": config, "rows_pulled": len(pulled)})

    if args.schema_compatible:
        source_rows = [coerce_schema_compatibility(row) for row in source_rows]

    written: list[WrittenFixture] = []
    for row in source_rows:
        fixture = write_fixture(
            out_dir=out_dir,
            bucket="baseline",
            scenario="baseline_real",
            aggregate_row=row,
            instances_per_aggregate=args.instances_per_aggregate,
            rng=rng,
            expected_backend_action="accept",
            notes="Real pulled aggregate row with synthetic instance drilldown rows.",
        )
        written.append(fixture)

    pathology_rows = make_pathology_rows(source_rows)
    for scenario, row, expected_action, notes in pathology_rows:
        fixture = write_fixture(
            out_dir=out_dir,
            bucket="pathology",
            scenario=scenario,
            aggregate_row=row,
            instances_per_aggregate=max(8, args.instances_per_aggregate // 2),
            rng=rng,
            expected_backend_action=expected_action,
            notes=notes,
        )
        written.append(fixture)

    dataset_sha = get_dataset_sha(args.dataset)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "dataset": args.dataset,
        "dataset_sha": dataset_sha,
        "configs": configs,
        "rows_per_config": args.rows_per_config,
        "instances_per_aggregate": args.instances_per_aggregate,
        "schema_compatible": args.schema_compatible,
        "source_pull_log": source_pull_log,
        "counts": {
            "baseline_aggregate_files": sum(1 for x in written if x.bucket == "baseline"),
            "pathology_aggregate_files": sum(1 for x in written if x.bucket == "pathology"),
            "instance_rows_total": sum(x.instance_rows for x in written),
        },
        "fixtures": [
            {
                "bucket": item.bucket,
                "scenario": item.scenario,
                "aggregate_path": item.aggregate_relpath,
                "instance_path": item.instance_relpath,
                "instance_rows": item.instance_rows,
                "expected_backend_action": item.expected_backend_action,
                "notes": item.notes,
            }
            for item in written
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"Generated fixture pack at: {out_dir}")
    print(
        "Counts: "
        f"{manifest['counts']['baseline_aggregate_files']} baseline aggregates, "
        f"{manifest['counts']['pathology_aggregate_files']} pathology aggregates, "
        f"{manifest['counts']['instance_rows_total']} instance rows."
    )


if __name__ == "__main__":
    main()
