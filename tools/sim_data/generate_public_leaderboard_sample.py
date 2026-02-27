#!/usr/bin/env python3
"""
Generate ~N aggregate rows for backend development by scraping public leaderboards.

This script intentionally pulls from diverse public leaderboard sources so backend
prototypes can be tested against heterogeneous schemas and metric conventions.

Current sources (11 total, usually >=10 available in one run):
1) Global MMLU Lite (Kaggle benchmark API)
2) RewardBench v1 (HF Space CSV)
3) LMSYS Arena leaderboard snapshot (HF Space CSV)
4) LMSYS Arena-Hard Auto leaderboard (HF Space CSV)
5) TabArena all-tasks leaderboard (HF Space CSV)
6) TabArena binary-tasks leaderboard (HF Space CSV)
7) UGI leaderboard (HF Space CSV)
8) Hebrew transcription leaderboard benchmark (HF Space CSV)
9) Open PT LLM leaderboard (HF Space JSON)
10) Open LLM Leaderboard contents (HF datasets-server)
11) BigCode models leaderboard community summaries (HF Space JSON files)

Output:
- data/simulated/public_leaderboards/aggregate_rows_<N>.jsonl
- data/simulated/public_leaderboards/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

GLOBAL_MMLU_URL = "https://www.kaggle.com/api/v1/benchmarks/cohere-labs/global-mmlu-lite/leaderboard"
REWARDBENCH_V1_CSV = "https://huggingface.co/spaces/allenai/reward-bench/resolve/main/leaderboard/final-rbv1-data.csv"
LMARENA_LATEST_CSV = "https://huggingface.co/spaces/lmarena-ai/arena-leaderboard/resolve/main/leaderboard_table_20250804.csv"
LMARENA_HARD_AUTO_CSV = "https://huggingface.co/spaces/lmarena-ai/arena-leaderboard/resolve/main/arena_hard_auto_leaderboard_v0.1.csv"
TABARENA_ALL_CSV = "https://huggingface.co/spaces/TabArena/leaderboard/resolve/main/data/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv"
TABARENA_BINARY_CSV = "https://huggingface.co/spaces/TabArena/leaderboard/resolve/main/data/imputation_no/splits_all/tasks_binary/datasets_all/website_leaderboard.csv"
UGI_CSV = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"
IVRIT_BENCHMARK_CSV = "https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard/resolve/main/benchmark.csv"
OPEN_PT_LLM_JSON = "https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard/resolve/main/external_models_results.json"
OPEN_LLM_CONTENTS_ROWS_API = (
    "https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents"
    "&config=default&split=train&offset={offset}&length={length}"
)
BIGCODE_TREE_API = "https://huggingface.co/api/spaces/bigcode/bigcode-models-leaderboard/tree/main?recursive=true"
BIGCODE_RAW_PREFIX = "https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/resolve/main/"


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


def fetch_json(url: str, timeout_sec: int = 90) -> Any:
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


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"true", "false", "none", "null", "nan", "n/a"}:
        return None

    # Skip ranges/interval-ish forms like "+64/-51" or "(1.9, 2.0)".
    if "/" in text and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text):
        return None
    if text.startswith("(") and text.endswith(")"):
        return None

    if text.endswith("%"):
        text = text[:-1].strip()

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text)

    return None


def normalize_model_id(model: str) -> tuple[str, str, str]:
    """
    Returns (model_id, developer, model_name).
    """
    if "/" in model:
        developer, model_name = model.split("/", 1)
        return f"{developer}/{model_name}", developer, model_name
    developer = "unknown"
    model_name = model
    return f"{developer}/{model_name}", developer, model_name


def extract_model_name_from_html(html_string: str) -> str:
    match = re.search(r">([^<]+)</a>", html_string)
    if match:
        return re.sub(r"\s*[\*⚠️]+$", "", match.group(1).strip()).strip()
    return re.sub(r"\s*[\*⚠️]+$", "", html_string).strip()


def extract_model_name(raw: Any) -> str:
    if raw is None:
        return "unknown-model"
    text = str(raw).replace("\ufeff", "").strip()
    if not text:
        return "unknown-model"

    md_match = re.match(r"^\[([^\]]+)\]\([^\)]+\)$", text)
    if md_match:
        text = md_match.group(1).strip()

    if "<a " in text or "</a>" in text:
        text = extract_model_name_from_html(text)
        text = re.sub(r"<[^>]+>", "", text).strip()

    text = re.sub(r"\s{2,}", " ", text).strip()
    return text or "unknown-model"


def infer_lower_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    hints = [
        "⬇️",
        "lower",
        "error",
        "mae",
        "rmse",
        "mse",
        "rank",
        "latency",
        "time",
        "cost",
        "improvability",
    ]
    return any(hint in lowered for hint in hints)


def clean_evaluation_name(metric_name: str) -> str:
    return re.sub(r"\s+", " ", metric_name.strip())


def infer_metric_unit(metric_name: str, score: float, lower_is_better: bool) -> str:
    lowered = metric_name.lower()
    lowered_norm = lowered.replace("₂", "2")
    if "kg" in lowered_norm or "co2" in lowered_norm:
        return "kg"
    if "rank" in lowered_norm:
        return "rank"
    if "char" in lowered_norm:
        return "characters"
    if "count" in lowered_norm:
        return "count"
    if "ms" in lowered or "millisecond" in lowered:
        return "ms"
    if "sec" in lowered or "latency" in lowered or "time" in lowered:
        return "seconds"
    if "token" in lowered:
        return "tokens"
    if "elo" in lowered:
        return "points"
    if lower_is_better and any(x in lowered for x in ["rmse", "mae", "mse", "error", "loss"]):
        return "points"

    abs_score = abs(float(score))
    if abs_score <= 1.0:
        return "proportion"
    if abs_score <= 100.0:
        return "percent"
    return "points"


def infer_metric_identity(
    *,
    source_key: str,
    evaluation_name: str,
    metric_label: str | None,
    score: float,
    lower_is_better: bool,
) -> dict[str, Any]:
    raw_name = clean_evaluation_name(metric_label or evaluation_name)
    lowered = raw_name.lower()
    compact = re.sub(r"\s+", "", lowered)

    metric_id = sanitize_component(raw_name).replace("-", "_")
    metric_name = raw_name
    metric_kind = "leaderboard_score"
    metric_parameters: dict[str, Any] = {}
    metric_unit = infer_metric_unit(raw_name, score, lower_is_better)
    lowered_norm = lowered.replace("₂", "2")

    pass_match = (
        re.search(r"pass\s*[@_ ]\s*(\d+)", lowered)
        or re.search(r"passat(\d+)", compact)
        or re.search(r"pass_at_(\d+)", compact)
    )
    if pass_match:
        k = int(pass_match.group(1))
        metric_id = "pass_at_k"
        metric_name = f"pass@{k}"
        metric_kind = "pass_rate"
        metric_parameters = {"k": k}
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
        return {
            "metric_id": metric_id,
            "metric_name": metric_name,
            "metric_kind": metric_kind,
            "metric_unit": metric_unit,
            "metric_parameters": metric_parameters,
        }

    if source_key in {"lmarena_hard_auto", "lmarena_latest"} and lowered in {"score", "arena score"}:
        metric_id = "arena_elo"
        metric_kind = "elo"
        metric_unit = "points"
        return {
            "metric_id": metric_id,
            "metric_name": metric_name,
            "metric_kind": metric_kind,
            "metric_unit": metric_unit,
            "metric_parameters": metric_parameters,
        }

    if source_key == "rewardbench_v1":
        metric_kind = "preference_rate"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
        if lowered == "score":
            metric_id = "rewardbench_overall"
        else:
            metric_id = f"rewardbench_{sanitize_component(raw_name).replace('-', '_')}"
        return {
            "metric_id": metric_id,
            "metric_name": metric_name,
            "metric_kind": metric_kind,
            "metric_unit": metric_unit,
            "metric_parameters": metric_parameters,
        }

    if "elo" in lowered_norm:
        metric_id = "elo"
        metric_kind = "elo"
        metric_unit = "points"
    elif "rank" in lowered_norm:
        metric_id = "rank"
        metric_kind = "rank"
        metric_unit = "rank"
    elif "time" in lowered_norm or "latency" in lowered_norm:
        metric_id = "latency"
        metric_kind = "latency"
        metric_unit = "seconds"
    elif "co2" in lowered_norm or "carbon" in lowered_norm or "emission" in lowered_norm:
        metric_id = "emissions"
        metric_kind = "emissions"
        metric_unit = "kg"
    elif any(x in lowered_norm for x in ["error", "improvability", "imputed", "percent_error", "error_pct"]):
        metric_id = "error_rate" if any(x in lowered_norm for x in ["%", "percent", "pct", "rate"]) else "error"
        metric_kind = "error_rate" if "rate" in metric_id else "error"
        metric_unit = "percent" if any(x in lowered_norm for x in ["%", "percent", "pct"]) else "points"
    elif any(
        x in lowered_norm
        for x in [
            "sensitive info",
            "hazard",
            "safety",
            "offensive",
            "hate",
            "nsfw",
            "dark_score",
        ]
    ):
        metric_id = "safety_risk"
        metric_kind = "safety_risk"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
    elif any(
        x in lowered_norm
        for x in [
            "political lean",
            "federal-unitary",
            "democratic-autocratic",
            "security-freedom",
            "nationalism-internationalism",
            "militarist-pacifist",
            "assimilationist-multiculturalist",
            "collectivize-privatize",
            "planned-laissezfaire",
            "isolationism-globalism",
            "irreligious-religious",
            "progressive-traditional",
            "acceleration-bioconservative",
        ]
    ):
        metric_id = "ideology_axis"
        metric_kind = "ideology_axis"
        metric_unit = "points"
    elif any(
        x in lowered_norm
        for x in [
            "writing",
            "dialogue_percentage",
            "verb_to_noun_ratio",
            "adjective_adverb_percentage",
            "readability",
            "originality",
            "semantic",
            "lexical",
            "rec score",
            "show rec",
            "dipl",
            "govt",
            "econ",
            "scty",
        ]
    ):
        metric_id = "quality_score"
        metric_kind = "quality_score"
        metric_unit = "points"
    elif any(
        x in lowered_norm
        for x in [
            "ifeval",
            "bbh",
            "math",
            "gpqa",
            "musr",
            "challenge",
            "exams",
            "rte",
            "nli",
            "tweetsent",
            "offensive",
            "hate_speech",
            "sts",
        ]
    ):
        metric_id = "accuracy"
        metric_kind = "accuracy"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
    elif any(x in lowered for x in ["accuracy", " acc", "acc ", "mmlu"]):
        metric_id = "accuracy"
        metric_kind = "accuracy"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
    elif "f1" in lowered:
        metric_kind = "f1"
        if "macro" in lowered:
            metric_id = "f1_macro"
        elif "micro" in lowered:
            metric_id = "f1_micro"
        elif "weighted" in lowered:
            metric_id = "f1_weighted"
        else:
            metric_id = "f1"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
    elif "auroc" in lowered or "roc-auc" in lowered or "roc_auc" in lowered or "auc" == compact:
        metric_id = "auroc"
        metric_kind = "auroc"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
    elif "rmse" in lowered:
        metric_id = "rmse"
        metric_kind = "rmse"
        metric_unit = "points"
    elif "mae" in lowered:
        metric_id = "mae"
        metric_kind = "mae"
        metric_unit = "points"
    elif "mse" in lowered:
        metric_id = "mse"
        metric_kind = "mse"
        metric_unit = "points"
    elif "wer" in lowered:
        metric_id = "word_error_rate"
        metric_kind = "word_error_rate"
        metric_unit = "proportion" if abs(float(score)) <= 1.0 else "percent"
    elif "mt-bench" in lowered:
        metric_id = "mt_bench_score"
        metric_kind = "quality_score"
        metric_unit = "points"

    return {
        "metric_id": metric_id,
        "metric_name": metric_name,
        "metric_kind": metric_kind,
        "metric_unit": metric_unit,
        "metric_parameters": metric_parameters,
    }


def build_evaluation_result_id(
    *,
    source_name: str,
    model_id: str,
    result: dict[str, Any],
) -> str:
    metric_config = result.get("metric_config", {})
    source_data = result.get("source_data", {})
    canonical_payload = {
        "source_name": source_name,
        "model_id": model_id,
        "dataset_name": source_data.get("dataset_name"),
        "evaluation_name": result.get("evaluation_name"),
        "metric_id": metric_config.get("metric_id"),
        "metric_name": metric_config.get("metric_name"),
        "metric_kind": metric_config.get("metric_kind"),
        "metric_unit": metric_config.get("metric_unit"),
        "metric_parameters": metric_config.get("metric_parameters", {}),
    }
    canonical_text = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    return f"er_{digest}"


def make_metric(
    *,
    source_key: str,
    evaluation_name: str,
    source_dataset_name: str,
    source_url: str,
    score: float,
    lower_is_better: bool,
    metric_label: str | None = None,
) -> dict[str, Any]:
    metric_identity = infer_metric_identity(
        source_key=source_key,
        evaluation_name=evaluation_name,
        metric_label=metric_label,
        score=score,
        lower_is_better=lower_is_better,
    )
    min_score = min(0.0, float(score))
    max_score = max(1.0, float(score))
    return {
        "evaluation_name": clean_evaluation_name(evaluation_name),
        "source_data": {
            "dataset_name": source_dataset_name,
            "source_type": "url",
            "url": [source_url],
        },
        "metric_config": {
            "evaluation_description": f"{source_dataset_name} - {clean_evaluation_name(evaluation_name)}",
            "lower_is_better": lower_is_better,
            "score_type": "continuous",
            "min_score": round(min_score, 6),
            "max_score": round(max_score, 6),
            "metric_id": metric_identity["metric_id"],
            "metric_name": metric_identity["metric_name"],
            "metric_kind": metric_identity["metric_kind"],
            "metric_unit": metric_identity["metric_unit"],
            "metric_parameters": metric_identity["metric_parameters"],
        },
        "score_details": {"score": round(score, 6)},
    }


def make_aggregate_record(
    *,
    source_key: str,
    source_name: str,
    source_url: str,
    source_org_name: str,
    source_org_url: str,
    model_name: str,
    eval_results: list[dict[str, Any]],
    now_ts: str,
    row_idx: int,
    additional_model_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_id, developer, _ = normalize_model_id(model_name)
    model_info: dict[str, Any] = {
        "name": model_name,
        "id": model_id,
        "developer": developer,
        "inference_platform": "unknown",
    }
    if additional_model_details:
        model_info["additional_details"] = additional_model_details

    record = {
        "schema_version": "0.2.1",
        "evaluation_id": f"{source_key}/{sanitize_component(model_id)}/{now_ts}-{row_idx}",
        "retrieved_timestamp": now_ts,
        "source_metadata": {
            "source_name": source_name,
            "source_type": "documentation",
            "source_organization_name": source_org_name,
            "source_organization_url": source_org_url,
            "evaluator_relationship": "third_party",
        },
        "eval_library": {"name": "unknown", "version": "unknown"},
        "model_info": model_info,
        "evaluation_results": eval_results,
    }
    for result in record["evaluation_results"]:
        if "evaluation_result_id" not in result:
            result["evaluation_result_id"] = build_evaluation_result_id(
                source_name=source_name,
                model_id=model_id,
                result=result,
            )
    return record


def should_skip_metric_column(column_name: str) -> bool:
    raw = column_name.strip()
    lowered = raw.lower()

    exact_skip = {
        "",
        "#",
        "model",
        "model link",
        "link",
        "key",
        "date",
        "release date",
        "test date",
        "knowledge cutoff date",
        "organization",
        "license",
        "prompt template",
        "engine",
        "fullname",
        "name",
        "status",
        "main_language",
        "model_type",
        "type",
        "typename",
        "precision",
        "t",
        "weight type",
        "architecture",
        "model sha",
        "hub license",
        "hub ❤️",
        "available on the hub",
        "moe",
        "flagged",
        "chat template",
        "official providers",
        "upload to hub date",
        "submission date",
        "generation",
        "base model",
    }
    if lowered in exact_skip:
        return True

    keyword_skip = [
        "link",
        "date",
        "template",
        "license",
        "organization",
        "model sha",
        "provider",
        "hub",
        "available",
        "chat template",
        "params",
        "parameter",
    ]
    return any(keyword in lowered for keyword in keyword_skip)


def first_non_empty(row: dict[str, Any], candidates: list[str]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def convert_csv_source(
    *,
    source_key: str,
    source_name: str,
    source_org_name: str,
    source_org_url: str,
    source_url: str,
    model_field_candidates: list[str],
    include_metric_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    text = fetch_text(source_url)
    rows = list(csv.DictReader(io.StringIO(text)))
    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    include_set = set(include_metric_columns) if include_metric_columns else None

    for row_idx, row in enumerate(rows):
        model_raw = first_non_empty(row, model_field_candidates)
        if not model_raw:
            continue

        model_name = extract_model_name(model_raw)
        if model_name.lower() in {"random", "n/a", "none"}:
            continue

        eval_results: list[dict[str, Any]] = []
        for key, value in row.items():
            key = key.replace("\ufeff", "").strip()
            if include_set is not None and key not in include_set:
                continue
            if should_skip_metric_column(key):
                continue

            score = parse_numeric(value)
            if score is None:
                continue

            eval_results.append(
                make_metric(
                    source_key=source_key,
                    evaluation_name=key,
                    source_dataset_name=source_name,
                    source_url=source_url,
                    score=score,
                    lower_is_better=infer_lower_is_better(key),
                )
            )

        if not eval_results:
            continue

        converted.append(
            make_aggregate_record(
                source_key=source_key,
                source_name=source_name,
                source_url=source_url,
                source_org_name=source_org_name,
                source_org_url=source_org_url,
                model_name=model_name,
                eval_results=eval_results,
                now_ts=now_ts,
                row_idx=row_idx,
            )
        )

    return converted


def convert_global_mmlu() -> list[dict[str, Any]]:
    payload = fetch_json(GLOBAL_MMLU_URL)
    rows = payload.get("rows", [])
    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        model_slug = row.get("modelVersionSlug")
        if not isinstance(model_slug, str) or not model_slug:
            continue

        model_display_name = row.get("modelVersionName") or model_slug

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
            metric = make_metric(
                source_key="global_mmlu_lite",
                evaluation_name=task_name,
                source_dataset_name="global-mmlu-lite",
                source_url="https://www.kaggle.com/datasets/cohere-labs/global-mmlu-lite",
                score=score,
                lower_is_better=False,
                metric_label="accuracy",
            )
            if confidence_interval is not None and confidence_interval > 0:
                metric["score_details"]["uncertainty"] = {
                    "confidence_interval": {
                        "lower": round(-confidence_interval, 6),
                        "upper": round(confidence_interval, 6),
                        "method": "reported",
                    }
                }
            eval_results.append(metric)

        if not eval_results:
            continue

        converted.append(
            make_aggregate_record(
                source_key="global_mmlu_lite",
                source_name="Global MMLU Lite Leaderboard",
                source_url="https://www.kaggle.com/datasets/cohere-labs/global-mmlu-lite",
                source_org_name="Kaggle",
                source_org_url="https://www.kaggle.com",
                model_name=model_display_name,
                eval_results=eval_results,
                now_ts=now_ts,
                row_idx=row_idx,
            )
        )

    return converted


def parse_rewardbench_score(raw: Any) -> float | None:
    """Parse a RewardBench score, normalizing percentage-like values to [0, 1]."""
    value = parse_numeric(raw)
    if value is None:
        return None
    # RewardBench CSV values are typically reported as percentages in [0, 100].
    # Heuristic:
    # - If value <= 1, assume it is already normalized to [0, 1].
    # - If 1 < value <= 100, treat it as a percentage and divide by 100.
    # - If value > 100, return it as-is rather than implicitly normalizing.
    if 1 < value <= 100:
        return round(value / 100.0, 6)
    return round(value, 6)


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

    for row_idx, row in enumerate(rows):
        raw_model = row.get("Model", "")
        model_name = extract_model_name_from_html(raw_model)
        if not model_name or model_name.lower() == "random":
            continue

        eval_results: list[dict[str, Any]] = []
        for metric_name, metric_desc in metric_columns:
            score = parse_rewardbench_score(row.get(metric_name, ""))
            if score is None:
                continue

            metric = make_metric(
                source_key="rewardbench_v1",
                evaluation_name=metric_name,
                source_dataset_name="RewardBench",
                source_url="https://huggingface.co/spaces/allenai/reward-bench",
                score=score,
                lower_is_better=False,
            )
            metric["metric_config"]["evaluation_description"] = metric_desc
            metric["metric_config"]["min_score"] = 0.0
            metric["metric_config"]["max_score"] = 1.0
            eval_results.append(metric)

        if not eval_results:
            continue

        converted.append(
            make_aggregate_record(
                source_key="rewardbench_v1",
                source_name="RewardBench",
                source_url="https://huggingface.co/spaces/allenai/reward-bench",
                source_org_name="Allen Institute for AI",
                source_org_url="https://allenai.org",
                model_name=model_name,
                eval_results=eval_results,
                now_ts=now_ts,
                row_idx=row_idx,
                additional_model_details={"model_type": row.get("Model Type", "unknown")},
            )
        )

    return converted


def convert_open_pt_llm() -> list[dict[str, Any]]:
    payload = fetch_json(OPEN_PT_LLM_JSON)
    if not isinstance(payload, list):
        return []

    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    for row_idx, row in enumerate(payload):
        if not isinstance(row, dict):
            continue

        model_name = extract_model_name(row.get("model") or row.get("name") or "unknown-model")
        metrics = row.get("result_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        eval_results: list[dict[str, Any]] = []
        for metric_name, raw_score in metrics.items():
            score = parse_numeric(raw_score)
            if score is None:
                continue
            eval_results.append(
                make_metric(
                    source_key="open_pt_llm_leaderboard",
                    evaluation_name=metric_name,
                    source_dataset_name="Open PT LLM Leaderboard",
                    source_url="https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard",
                    score=score,
                    lower_is_better=False,
                )
            )

        for metric_name in ["result_metrics_average", "result_metrics_npm"]:
            score = parse_numeric(row.get(metric_name))
            if score is None:
                continue
            eval_results.append(
                make_metric(
                    source_key="open_pt_llm_leaderboard",
                    evaluation_name=metric_name,
                    source_dataset_name="Open PT LLM Leaderboard",
                    source_url="https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard",
                    score=score,
                    lower_is_better=False,
                )
            )

        if not eval_results:
            continue

        converted.append(
            make_aggregate_record(
                source_key="open_pt_llm_leaderboard",
                source_name="Open PT LLM Leaderboard",
                source_url="https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard",
                source_org_name="eduagarcia",
                source_org_url="https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard",
                model_name=model_name,
                eval_results=eval_results,
                now_ts=now_ts,
                row_idx=row_idx,
                additional_model_details={
                    "status": row.get("status"),
                    "main_language": row.get("main_language"),
                    "model_type": row.get("model_type"),
                },
            )
        )

    return converted


def fetch_open_llm_contents_rows(max_rows: int = 250, page_size: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < max_rows:
        length = min(page_size, max_rows - len(rows))
        payload = fetch_json(OPEN_LLM_CONTENTS_ROWS_API.format(offset=offset, length=length), timeout_sec=45)
        chunk = payload.get("rows", []) if isinstance(payload, dict) else []
        if not chunk:
            break

        added = 0
        for item in chunk:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
                added += 1
                if len(rows) >= max_rows:
                    break

        if added == 0:
            break
        offset += len(chunk)

    return rows


def convert_open_llm_contents() -> list[dict[str, Any]]:
    rows = fetch_open_llm_contents_rows(max_rows=250)
    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        model_name = extract_model_name(row.get("fullname") or row.get("Model") or "unknown-model")

        eval_results: list[dict[str, Any]] = []
        for key, value in row.items():
            if should_skip_metric_column(str(key)):
                continue
            score = parse_numeric(value)
            if score is None:
                continue
            eval_results.append(
                make_metric(
                    source_key="open_llm_leaderboard_contents",
                    evaluation_name=str(key),
                    source_dataset_name="Open LLM Leaderboard Contents",
                    source_url="https://huggingface.co/datasets/open-llm-leaderboard/contents",
                    score=score,
                    lower_is_better=infer_lower_is_better(str(key)),
                )
            )

        if not eval_results:
            continue

        converted.append(
            make_aggregate_record(
                source_key="open_llm_leaderboard_contents",
                source_name="Open LLM Leaderboard Contents",
                source_url="https://huggingface.co/datasets/open-llm-leaderboard/contents",
                source_org_name="open-llm-leaderboard",
                source_org_url="https://huggingface.co/open-llm-leaderboard",
                model_name=model_name,
                eval_results=eval_results,
                now_ts=now_ts,
                row_idx=row_idx,
            )
        )

    return converted


def fetch_bigcode_summary_paths(max_files: int = 40) -> list[str]:
    tree = fetch_json(BIGCODE_TREE_API, timeout_sec=60)
    if not isinstance(tree, list):
        return []

    paths: list[str] = []
    for entry in tree:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        size = entry.get("size", 0) or 0
        if not isinstance(path, str):
            continue

        lowered = path.lower()
        if not lowered.startswith("community_results/"):
            continue
        if not lowered.endswith(".json"):
            continue
        if "/generations_" in lowered or "/metrics_" in lowered:
            continue
        if size <= 0 or size > 4000:
            continue

        paths.append(path)

    paths.sort()
    return paths[:max_files]


def convert_bigcode_community() -> list[dict[str, Any]]:
    paths = fetch_bigcode_summary_paths(max_files=40)
    now_ts = str(time.time())
    converted: list[dict[str, Any]] = []

    for row_idx, path in enumerate(paths):
        payload = fetch_json(BIGCODE_RAW_PREFIX + path, timeout_sec=45)
        if not isinstance(payload, dict):
            continue

        model_name = extract_model_name(payload.get("meta", {}).get("model") if isinstance(payload.get("meta"), dict) else None)
        results = payload.get("results")
        if not isinstance(results, list):
            continue

        eval_results: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            task_name = item.get("task")
            score = parse_numeric(item.get("pass@1"))
            if not isinstance(task_name, str) or score is None:
                continue
            metric = make_metric(
                source_key="bigcode_models_leaderboard",
                evaluation_name=task_name,
                source_dataset_name="BigCode Models Leaderboard",
                source_url="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard",
                score=score,
                lower_is_better=False,
                metric_label="pass@1",
            )
            metric["metric_config"]["min_score"] = 0.0
            metric["metric_config"]["max_score"] = 1.0
            eval_results.append(metric)

        if not eval_results:
            continue

        converted.append(
            make_aggregate_record(
                source_key="bigcode_models_leaderboard",
                source_name="BigCode Models Leaderboard",
                source_url="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard",
                source_org_name="BigCode",
                source_org_url="https://huggingface.co/bigcode",
                model_name=model_name,
                eval_results=eval_results,
                now_ts=now_ts,
                row_idx=row_idx,
            )
        )

    return converted


def convert_lmarena_latest() -> list[dict[str, Any]]:
    return convert_csv_source(
        source_key="lmarena_latest",
        source_name="LMSYS Arena Leaderboard (latest snapshot)",
        source_org_name="LMSYS",
        source_org_url="https://lmarena.ai",
        source_url=LMARENA_LATEST_CSV,
        model_field_candidates=["Model", "key"],
        include_metric_columns=["MT-bench (score)", "MMLU"],
    )


def convert_lmarena_hard_auto() -> list[dict[str, Any]]:
    return convert_csv_source(
        source_key="lmarena_hard_auto",
        source_name="LMSYS Arena Hard Auto",
        source_org_name="LMSYS",
        source_org_url="https://lmarena.ai",
        source_url=LMARENA_HARD_AUTO_CSV,
        model_field_candidates=["model", "Model"],
        include_metric_columns=["score"],
    )


def convert_tabarena_all() -> list[dict[str, Any]]:
    return convert_csv_source(
        source_key="tabarena_all",
        source_name="TabArena (all tasks)",
        source_org_name="TabArena",
        source_org_url="https://huggingface.co/spaces/TabArena/leaderboard",
        source_url=TABARENA_ALL_CSV,
        model_field_candidates=["Model", "TypeName"],
    )


def convert_tabarena_binary() -> list[dict[str, Any]]:
    return convert_csv_source(
        source_key="tabarena_binary",
        source_name="TabArena (binary tasks)",
        source_org_name="TabArena",
        source_org_url="https://huggingface.co/spaces/TabArena/leaderboard",
        source_url=TABARENA_BINARY_CSV,
        model_field_candidates=["Model", "TypeName"],
    )


def convert_ugi() -> list[dict[str, Any]]:
    return convert_csv_source(
        source_key="ugi_leaderboard",
        source_name="UGI Leaderboard",
        source_org_name="DontPlanToEnd",
        source_org_url="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard",
        source_url=UGI_CSV,
        model_field_candidates=["\ufeffauthor/model_name", "author/model_name"],
    )


def convert_ivrit_benchmark() -> list[dict[str, Any]]:
    return convert_csv_source(
        source_key="ivrit_asr_leaderboard",
        source_name="Hebrew Transcription Leaderboard",
        source_org_name="ivrit-ai",
        source_org_url="https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard",
        source_url=IVRIT_BENCHMARK_CSV,
        model_field_candidates=["model", "engine"],
        include_metric_columns=[
            "ivrit-ai/eval-d1",
            "ivrit-ai/eval-whatsapp",
            "ivrit-ai/saspeech",
            "google/fleurs/he",
            "mozilla-foundation/common_voice_17_0/he",
            "imvladikon/hebrew_speech_kan",
        ],
    )


def sample_rows_by_source(
    rows_by_source: dict[str, list[dict[str, Any]]],
    target_rows: int,
    min_per_source: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pools = {source: rows[:] for source, rows in rows_by_source.items() if rows}
    for rows in pools.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    selected_count: dict[str, int] = {source: 0 for source in pools}

    sources = list(pools.keys())

    # First pass: ensure source coverage when possible.
    if target_rows >= len(sources):
        for source in sources:
            if pools[source]:
                selected.append(pools[source].pop())
                selected_count[source] += 1

    # Second pass: try to guarantee min_per_source per source.
    for source in sources:
        needed = max(0, min(min_per_source, len(rows_by_source[source])) - selected_count[source])
        while needed > 0 and pools[source] and len(selected) < target_rows:
            selected.append(pools[source].pop())
            selected_count[source] += 1
            needed -= 1

    if len(selected) >= target_rows:
        rng.shuffle(selected)
        return selected[:target_rows]

    remaining_pool: list[dict[str, Any]] = []
    for rows in pools.values():
        remaining_pool.extend(rows)

    rng.shuffle(remaining_pool)
    slots = target_rows - len(selected)
    selected.extend(remaining_pool[:slots])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ~N aggregate rows from public leaderboards for backend testing."
    )
    parser.add_argument("--target-rows", type=int, default=200)
    parser.add_argument("--min-per-source", type=int, default=10)
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

    source_builders: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = [
        ("global_mmlu_lite", convert_global_mmlu),
        ("rewardbench_v1", convert_rewardbench_v1),
        ("lmarena_latest", convert_lmarena_latest),
        ("lmarena_hard_auto", convert_lmarena_hard_auto),
        ("tabarena_all", convert_tabarena_all),
        ("tabarena_binary", convert_tabarena_binary),
        ("ugi_leaderboard", convert_ugi),
        ("ivrit_asr_leaderboard", convert_ivrit_benchmark),
        ("open_pt_llm_leaderboard", convert_open_pt_llm),
        ("open_llm_leaderboard_contents", convert_open_llm_contents),
        ("bigcode_models_leaderboard", convert_bigcode_community),
    ]

    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_status: dict[str, dict[str, Any]] = {}

    for source_key, builder in source_builders:
        try:
            rows = builder()
            source_rows[source_key] = rows
            source_status[source_key] = {
                "status": "ok",
                "rows_fetched": len(rows),
            }
        except Exception as exc:  # pragma: no cover - defensive for flaky network sources.
            source_rows[source_key] = []
            source_status[source_key] = {
                "status": "error",
                "rows_fetched": 0,
                "error": str(exc),
            }

    non_empty_sources = {k: v for k, v in source_rows.items() if v}
    all_count = sum(len(v) for v in non_empty_sources.values())
    if all_count == 0:
        raise SystemExit("No rows fetched from public leaderboard sources.")

    target = min(args.target_rows, all_count)
    sampled = sample_rows_by_source(
        rows_by_source=non_empty_sources,
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
        evaluation_id = str(row.get("evaluation_id", ""))
        source_name = evaluation_id.split("/", 1)[0] if "/" in evaluation_id else "unknown"
        counts_by_source[source_name] = counts_by_source.get(source_name, 0) + 1

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_rows_requested": args.target_rows,
        "target_rows_written": target,
        "seed": args.seed,
        "min_per_source": args.min_per_source,
        "sources_attempted": len(source_builders),
        "sources_available_with_rows": len(non_empty_sources),
        "sources": source_status,
        "counts_by_source_in_output": counts_by_source,
        "output_file": str(output_jsonl),
        "notes": (
            "Rows are scraped from multiple public leaderboard sources and normalized "
            "to eval-style aggregate records."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {target} rows to {output_jsonl}")
    print(f"Sources available: {len(non_empty_sources)} / {len(source_builders)}")
    print(f"Counts by source: {counts_by_source}")


if __name__ == "__main__":
    main()
