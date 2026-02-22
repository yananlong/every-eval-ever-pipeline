#!/usr/bin/env python3
"""
Generate ~N aggregate rows for backend development by scraping public leaderboards.

Current stable public sources:
1) Global MMLU Lite (Kaggle benchmark API)
2) RewardBench v1 (Hugging Face Space CSV)

Output:
- data/simulated/public_leaderboards/aggregate_rows_<N>.jsonl
- data/simulated/public_leaderboards/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GLOBAL_MMLU_URL = "https://www.kaggle.com/api/v1/benchmarks/cohere-labs/global-mmlu-lite/leaderboard"
REWARDBENCH_V1_CSV = "https://huggingface.co/spaces/allenai/reward-bench/resolve/main/leaderboard/final-rbv1-data.csv"


def fetch_text(url: str, timeout_sec: int = 90) -> str:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "-sS",
            "--fail",
            "--max-time",
            str(timeout_sec),
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def fetch_json(url: str, timeout_sec: int = 90) -> dict[str, Any]:
    return json.loads(fetch_text(url, timeout_sec=timeout_sec))


def sanitize_component(raw: str | None) -> str:
    if not raw:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-").lower()
    return cleaned or "unknown"


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def normalize_model_id(model: str) -> tuple[str, str, str]:
    """
    Returns (model_id, developer, model_name)
    model_id should follow developer/model when possible.
    """
    if "/" in model:
        developer, model_name = model.split("/", 1)
        return f"{developer}/{model_name}", developer, model_name
    developer = "unknown"
    model_name = model
    return f"{developer}/{model_name}", developer, model_name


def convert_global_mmlu() -> list[dict[str, Any]]:
    payload = fetch_json(GLOBAL_MMLU_URL)
    rows = payload.get("rows", [])
    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    for row in rows:
        model_slug = row.get("modelVersionSlug")
        if not isinstance(model_slug, str) or not model_slug:
            continue

        model_display_name = row.get("modelVersionName") or model_slug
        model_id, developer, model_name = normalize_model_id(model_slug)

        eval_results: list[dict[str, Any]] = []
        for task in row.get("taskResults", []):
            task_name = task.get("benchmarkTaskName")
            if not isinstance(task_name, str) or not task_name:
                continue

            result_obj = task.get("result", {})
            numeric_result = result_obj.get("numericResult") or result_obj.get("numericResultNullable") or {}
            score = parse_float(numeric_result.get("value"))
            if score is None:
                continue

            confidence_interval = parse_float(numeric_result.get("confidenceInterval"))
            uncertainty = None
            if confidence_interval is not None and confidence_interval > 0:
                uncertainty = {
                    "confidence_interval": {
                        "lower": round(-confidence_interval, 6),
                        "upper": round(confidence_interval, 6),
                        "method": "reported",
                    }
                }

            score_details = {"score": round(score, 6)}
            if uncertainty is not None:
                score_details["uncertainty"] = uncertainty

            eval_results.append(
                {
                    "evaluation_name": task_name,
                    "source_data": {
                        "dataset_name": "global-mmlu-lite",
                        "source_type": "url",
                        "url": ["https://www.kaggle.com/datasets/cohere-labs/global-mmlu-lite"],
                    },
                    "metric_config": {
                        "evaluation_description": f"Global MMLU Lite - {task_name}",
                        "lower_is_better": False,
                        "score_type": "continuous",
                        "min_score": 0.0,
                        "max_score": 1.0,
                    },
                    "score_details": score_details,
                }
            )

        if not eval_results:
            continue

        converted.append(
            {
                "schema_version": "0.2.1",
                "evaluation_id": f"global-mmlu-lite/{sanitize_component(model_id)}/{now_ts}",
                "retrieved_timestamp": now_ts,
                "source_metadata": {
                    "source_name": "Global MMLU Lite Leaderboard",
                    "source_type": "documentation",
                    "source_organization_name": "kaggle",
                    "source_organization_url": "https://www.kaggle.com",
                    "evaluator_relationship": "third_party",
                },
                "eval_library": {"name": "unknown", "version": "unknown"},
                "model_info": {
                    "name": model_display_name,
                    "id": model_id,
                    "developer": developer,
                    "inference_platform": "unknown",
                },
                "evaluation_results": eval_results,
                "_sim_source": "global_mmlu_lite",
            }
        )

    return converted


def extract_model_name_from_html(html_string: str) -> str:
    match = re.search(r">([^<]+)<", html_string)
    if match:
        return re.sub(r"\s*[\*⚠️]+$", "", match.group(1).strip()).strip()
    return re.sub(r"\s*[\*⚠️]+$", "", html_string).strip()


def parse_rewardbench_score(raw: str) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    # RewardBench CSV is often 0-100.
    return round(value / 100.0, 6) if value > 1 else round(value, 6)


def convert_rewardbench_v1() -> list[dict[str, Any]]:
    text = fetch_text(REWARDBENCH_V1_CSV)
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    metric_columns = [
        ("Score", "Overall RewardBench Score"),
        ("Chat", "Chat score"),
        ("Chat Hard", "Chat Hard score"),
        ("Safety", "Safety score"),
        ("Reasoning", "Reasoning score"),
        ("Prior Sets (0.5 weight)", "Prior sets score (0.5 weight)"),
    ]

    for row in rows:
        raw_model = row.get("Model", "")
        model_name = extract_model_name_from_html(raw_model)
        if not model_name or model_name.lower() == "random":
            continue

        model_id, developer, clean_model_name = normalize_model_id(model_name)
        eval_results: list[dict[str, Any]] = []
        for metric_name, metric_desc in metric_columns:
            score = parse_rewardbench_score(row.get(metric_name, ""))
            if score is None:
                continue
            eval_results.append(
                {
                    "evaluation_name": metric_name,
                    "source_data": {
                        "dataset_name": "RewardBench",
                        "source_type": "url",
                        "url": ["https://huggingface.co/spaces/allenai/reward-bench"],
                    },
                    "metric_config": {
                        "evaluation_description": metric_desc,
                        "lower_is_better": False,
                        "score_type": "continuous",
                        "min_score": 0.0,
                        "max_score": 1.0,
                    },
                    "score_details": {"score": score},
                }
            )

        if not eval_results:
            continue

        converted.append(
            {
                "schema_version": "0.2.1",
                "evaluation_id": f"reward-bench/{sanitize_component(model_id)}/{now_ts}",
                "retrieved_timestamp": now_ts,
                "source_metadata": {
                    "source_name": "RewardBench",
                    "source_type": "documentation",
                    "source_organization_name": "Allen Institute for AI",
                    "source_organization_url": "https://allenai.org",
                    "evaluator_relationship": "third_party",
                },
                "eval_library": {"name": "unknown", "version": "unknown"},
                "model_info": {
                    "name": clean_model_name,
                    "id": model_id,
                    "developer": developer,
                    "inference_platform": "unknown",
                    "additional_details": {
                        "model_type": row.get("Model Type", "unknown"),
                    },
                },
                "evaluation_results": eval_results,
                "_sim_source": "rewardbench_v1",
            }
        )

    return converted


def sample_rows_by_source(
    rows_by_source: dict[str, list[dict[str, Any]]],
    target_rows: int,
    min_per_source: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining_pool: list[dict[str, Any]] = []

    for source, rows in rows_by_source.items():
        shuffled = rows[:]
        rng.shuffle(shuffled)
        guaranteed = shuffled[: min(min_per_source, len(shuffled))]
        selected.extend(guaranteed)
        remaining_pool.extend(shuffled[len(guaranteed) :])

    if len(selected) > target_rows:
        rng.shuffle(selected)
        return selected[:target_rows]

    slots = target_rows - len(selected)
    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[:slots])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ~N aggregate rows from public leaderboards for backend testing."
    )
    parser.add_argument("--target-rows", type=int, default=200)
    parser.add_argument("--min-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="data/simulated/public_leaderboards",
        help="Repo-relative output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    source_rows = {
        "global_mmlu_lite": convert_global_mmlu(),
        "rewardbench_v1": convert_rewardbench_v1(),
    }

    all_count = sum(len(v) for v in source_rows.values())
    if all_count == 0:
        raise SystemExit("No rows fetched from public leaderboard sources.")

    target = min(args.target_rows, all_count)
    sampled = sample_rows_by_source(
        rows_by_source=source_rows,
        target_rows=target,
        min_per_source=args.min_per_source,
        rng=rng,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / f"aggregate_rows_{target}.jsonl"

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts_by_source: dict[str, int] = {}
    for row in sampled:
        source_name = str(row.get("_sim_source", "unknown"))
        counts_by_source[source_name] = counts_by_source.get(source_name, 0) + 1

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_rows_requested": args.target_rows,
        "target_rows_written": target,
        "seed": args.seed,
        "sources": {
            key: {
                "rows_fetched": len(value),
            }
            for key, value in source_rows.items()
        },
        "counts_by_source_in_output": counts_by_source,
        "output_file": str(output_jsonl),
        "notes": "Rows are scraped from public leaderboards and normalized to eval-style aggregate records.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {target} rows to {output_jsonl}")
    print(f"Counts by source: {counts_by_source}")


if __name__ == "__main__":
    main()
