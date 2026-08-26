#!/usr/bin/env python3
"""Run frozen Rubric curation and blind LLM judging on saved trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV_ROOT = ROOT / "environments/ShopSimulator/shop_env"
if str(SHOP_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOP_ENV_ROOT))

from web_agent_site.engine.engine import load_products
from web_agent_site.engine.goal import get_goals

from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.comparison import (
    compare_evaluation_runs,
)
from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    rubric_ids,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.metrics import (
    compute_deterministic_metrics,
)
from shopping_grpo.evaluation.model_client import (
    OpenAIJSONClient,
)
from shopping_grpo.evaluation.pipeline import (
    evaluate_trajectories,
)
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_rubric_curator_messages,
    build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (
    build_not_judged_result,
)
from shopping_grpo.evaluation.rubric import (
    build_task_facts,
    extract_rubric_candidates,
    materialize_rubric_bundle,
)
from shopping_grpo.evaluation.trajectory import (
    normalize_trajectory,
)
from shopping_grpo.local_env import load_project_env

RUBRIC_VERSION = "final240-v2.2-flash-r1"
RUNNER_VERSION = "existing-trajectory-judge-runner-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate frozen Rubrics and run blind LLM-as-a-Judge over existing "
            "Base/SFT/GRPO trajectories."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/evaluation/tasks.jsonl",
    )
    parser.add_argument(
        "--benchmark-slices",
        type=Path,
        default=ROOT / "data/evaluation/slices.jsonl",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV_ROOT / "data/items_eval_train.json",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=TRAJECTORIES_JSONL",
        help="Repeat for Base, SFT and GRPO.",
    )
    parser.add_argument(
        "--actor-manifest",
        action="append",
        default=[],
        metavar="LABEL=RUN_MANIFEST_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rubric-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=20)
    parser.add_argument("--rubric-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--force-finalize", action="store_true")
    return parser.parse_args()


def _parse_label_paths(values: Iterable[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = str(value).partition("=")
        label = label.strip()
        raw_path = raw_path.strip()
        if not separator or not label or not raw_path:
            raise SystemExit(f"{option} requires LABEL=PATH")
        if label in result:
            raise SystemExit(f"duplicate label for {option}: {label}")
        result[label] = Path(raw_path).expanduser().resolve()
    return result


def _load_rows(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def _index_unique(
    rows: Iterable[Mapping],
    *,
    key: str,
    label: str,
) -> dict[Any, dict]:
    indexed: dict[Any, dict] = {}
    for row in rows:
        if key not in row:
            raise ValueError(f"{label} row is missing {key}")
        value = row[key]
        if value in indexed:
            raise ValueError(f"duplicate {label} {key}={value!r}")
        indexed[value] = dict(row)
    return indexed


def build_task_facts_for_task_ids(
    *,
    task_ids: Iterable[int],
    all_products: list[Mapping],
    goals: list[Mapping],
) -> list[dict]:
    """Resolve raw product task IDs exactly as ShopSimulator Environment v2.4."""

    first_goal_by_asin: dict[str, Mapping] = {}
    for goal in goals:
        first_goal_by_asin.setdefault(str(goal.get("asin")), goal)
    result = []
    for raw_task_id in task_ids:
        task_id = int(raw_task_id)
        if task_id < 0 or task_id >= len(all_products):
            raise IndexError(f"task_id {task_id} is outside the product pool")
        product = all_products[task_id]
        asin = str(product.get("asin"))
        goal = first_goal_by_asin.get(asin)
        if not isinstance(goal, Mapping):
            raise TypeError(f"task_id {task_id} has no goal for ASIN {asin!r}")
        query = str(goal.get("instruction_text") or "").strip()
        if not query:
            raise ValueError(f"task_id {task_id} has no instruction_text")
        result.append(
            build_task_facts(
                task_id=task_id,
                query=query,
                target_product=product,
                instruction_record={
                    "instruction": query,
                    "attributes": goal.get("attributes") or [],
                    "instruction_options": goal.get("goal_options") or [],
                },
                reward_goal=goal,
            )
        )
    return result


def _load_task_facts(products_path: Path, task_ids: list[int]) -> list[dict]:
    all_products, _, product_prices, _ = load_products(products_path)
    goals = get_goals(all_products, product_prices)
    return build_task_facts_for_task_ids(
        task_ids=task_ids,
        all_products=all_products,
        goals=goals,
    )


def _client_from_prefix(
    prefix: str,
    *,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> OpenAIJSONClient:
    model = os.environ.get(f"{prefix}_MODEL")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    missing = [
        name
        for name, value in {
            f"{prefix}_MODEL": model,
            f"{prefix}_BASE_URL": base_url,
            f"{prefix}_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError("missing environment variables: " + ", ".join(missing))
    return OpenAIJSONClient(
        model=str(model),
        base_url=str(base_url),
        api_key=str(api_key),
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        response_format_json=True,
        thinking=False,
    )


def _repair_messages(
    messages: list[Mapping],
    *,
    response: Mapping,
    error: Exception,
) -> list[dict]:
    return [
        *deepcopy(messages),
        {
            "role": "assistant",
            "content": json.dumps(response, ensure_ascii=False, sort_keys=True),
        },
        {
            "role": "user",
            "content": (
                "The JSON failed the frozen schema validation: "
                f"{type(error).__name__}: {error}. "
                "Return one corrected strict JSON object only. Do not add "
                "Markdown or explanation."
            ),
        },
    ]


def _complete_validated(
    *,
    client: OpenAIJSONClient,
    messages: list[Mapping],
    validate: Callable[[Mapping], dict],
    validation_retries: int,
) -> tuple[dict, dict]:
    active_messages = deepcopy(messages)
    calls = []
    last_error: Exception | None = None
    for validation_attempt in range(validation_retries + 1):
        response = client.complete_json(active_messages)
        calls.append(response["metadata"])
        try:
            validated = validate(response["result"])
        except (ContractValidationError, ValueError, TypeError) as exc:
            last_error = exc
            if validation_attempt >= validation_retries:
                raise
            active_messages = _repair_messages(
                messages,
                response=response["result"],
                error=exc,
            )
            continue
        return validated, {
            "validation_repairs": validation_attempt,
            "calls": calls,
        }
    raise RuntimeError(f"validation failed: {last_error}")


def _checkpoint_index(path: Path, *, key: str) -> dict[Any, dict]:
    if not path.is_file():
        return {}
    return _index_unique(iter_jsonl(path), key=key, label=path.name)


def _run_parallel(
    *,
    label: str,
    items: list[Any],
    workers: int,
    worker: Callable[[Any], tuple[dict, dict]],
    checkpoint_path: Path,
    metadata_path: Path,
    row_key: Callable[[dict], Any],
) -> tuple[dict[Any, dict], list[dict]]:
    existing = _checkpoint_index(checkpoint_path, key="_checkpoint_key")
    failures: list[dict] = []
    pending = [item for item in items if str(item[0]) not in existing]
    total = len(items)
    completed = len(existing)
    print(f"{label}: resuming with {completed}/{total} completed", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            checkpoint_key = str(item[0])
            try:
                row, metadata = future.result()
            except Exception as exc:  # noqa: BLE001 - persisted for resumability
                failures.append(
                    {
                        "checkpoint_key": checkpoint_key,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                checkpoint_row = {"_checkpoint_key": checkpoint_key, **row}
                append_jsonl_fsync(checkpoint_path, checkpoint_row)
                append_jsonl_fsync(
                    metadata_path,
                    {
                        "_checkpoint_key": checkpoint_key,
                        "result_key": row_key(row),
                        **metadata,
                    },
                )
                existing[checkpoint_key] = checkpoint_row
            completed += 1
            if completed == total or completed % 10 == 0:
                print(
                    f"{label}: {completed}/{total}; failures={len(failures)}",
                    flush=True,
                )
    return existing, failures


def _strip_checkpoint_fields(rows: Iterable[Mapping]) -> list[dict]:
    return [
        {key: deepcopy(value) for key, value in row.items() if key != "_checkpoint_key"}
        for row in rows
    ]


def _load_actor(label: str, manifest_path: Path | None) -> dict:
    if manifest_path is None:
        return {"label": label, "model": label, "tokenizer": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actor = manifest.get("actor")
    if not isinstance(actor, Mapping):
        raise TypeError(f"manifest has no actor object: {manifest_path}")
    return dict(actor)


def _metadata_totals(path: Path) -> dict:
    totals = {
        "results": 0,
        "provider_calls": 0,
        "latency_seconds": 0.0,
        "retry_wait_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "validation_repairs": 0,
    }
    if not path.is_file():
        return totals
    for row in iter_jsonl(path):
        totals["results"] += 1
        totals["validation_repairs"] += int(row.get("validation_repairs", 0))
        for call in row.get("calls") or []:
            if not isinstance(call, Mapping):
                continue
            totals["provider_calls"] += 1
            totals["latency_seconds"] += float(call.get("latency_seconds", 0.0))
            totals["retry_wait_seconds"] += float(call.get("retry_wait_seconds", 0.0))
            usage = call.get("usage") or {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[name] += int(usage.get(name, 0) or 0)
    return totals


def main() -> None:
    args = parse_args()
    if args.rubric_workers < 1 or args.judge_workers < 1:
        raise SystemExit("worker counts must be positive")
    load_project_env()
    run_paths = _parse_label_paths(args.run, option="--run")
    manifest_paths = _parse_label_paths(
        args.actor_manifest,
        option="--actor-manifest",
    )
    unknown_manifests = sorted(set(manifest_paths) - set(run_paths))
    if unknown_manifests:
        raise SystemExit("actor manifests reference unknown runs: " + ", ".join(unknown_manifests))

    output = args.output.expanduser().resolve()
    checkpoints = output / "checkpoints"
    calls_dir = output / "calls"
    runs_dir = output / "runs"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    calls_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = _load_rows(args.benchmark)
    task_ids = [int(row["task_id"]) for row in benchmark_rows]
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("benchmark contains duplicate task IDs")
    expected_set = set(task_ids)
    task_slices = {int(row["task_id"]): row for row in _load_rows(args.benchmark_slices)}
    if set(task_slices) != expected_set:
        raise SystemExit("benchmark slices do not match benchmark task IDs")

    facts = _load_task_facts(args.products, task_ids)
    facts_by_task = {int(row["task_id"]): row for row in facts}
    candidates_by_task = {
        task_id: extract_rubric_candidates(facts_by_task[task_id]) for task_id in task_ids
    }

    raw_runs: dict[str, list[dict]] = {}
    source_audit: dict[str, dict] = {}
    for label, path in run_paths.items():
        rows = _load_rows(path)
        indexed = _index_unique(rows, key="task_id", label=f"{label} trajectory")
        included = [indexed[task_id] for task_id in task_ids if task_id in indexed]
        included_ids = {int(row["task_id"]) for row in included}
        excluded_ids = sorted(int(task_id) for task_id in indexed if task_id not in expected_set)
        for row in included:
            normalized = normalize_trajectory(row)
            query = facts_by_task[int(row["task_id"])]["query"]
            if query not in str(normalized.get("actor_query") or ""):
                raise ValueError(
                    f"{label} task {row['task_id']} actor query does not match "
                    "the current frozen environment goal"
                )
        raw_runs[label] = included
        source_audit[label] = {
            "source": str(path),
            "source_sha256": sha256_file(path),
            "source_rows": len(rows),
            "included_rows": len(included),
            "missing_task_ids": sorted(expected_set - included_ids),
            "excluded_task_ids": excluded_ids,
        }

    flash_client = _client_from_prefix(
        "SHOPPING_TEACHER",
        max_tokens=args.rubric_max_tokens,
        timeout=args.timeout,
        retries=args.transport_retries,
    )
    pro_client = _client_from_prefix(
        "SHOPPING_JUDGE",
        max_tokens=args.judge_max_tokens,
        timeout=args.timeout,
        retries=args.transport_retries,
    )

    rubric_checkpoint = checkpoints / "rubrics.jsonl"
    rubric_calls = calls_dir / "rubrics.jsonl"

    def curate(item: tuple[int, dict]) -> tuple[dict, dict]:
        task_id, task_facts = item
        candidates = candidates_by_task[task_id]
        messages = build_rubric_curator_messages(
            task_id=task_id,
            query=task_facts["query"],
            candidates=candidates["candidates"],
        )

        def validate(response: Mapping) -> dict:
            return materialize_rubric_bundle(
                task_facts=task_facts,
                candidates=candidates,
                curator_response=response,
                curator_model=flash_client.model,
                curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                rubric_version=RUBRIC_VERSION,
            )

        return _complete_validated(
            client=flash_client,
            messages=messages,
            validate=validate,
            validation_retries=args.validation_retries,
        )

    rubric_rows, rubric_failures = _run_parallel(
        label="Rubric curation",
        items=[(task_id, facts_by_task[task_id]) for task_id in task_ids],
        workers=args.rubric_workers,
        worker=curate,
        checkpoint_path=rubric_checkpoint,
        metadata_path=rubric_calls,
        row_key=lambda row: row["task_id"],
    )
    write_json_atomic(
        output / "rubric_failures.json",
        {"failures": rubric_failures},
        force=True,
    )
    if rubric_failures and not args.force_finalize:
        raise SystemExit(f"Rubric curation has {len(rubric_failures)} failures; rerun to resume")

    rubrics_by_task: dict[int, dict] = {}
    for checkpoint_row in rubric_rows.values():
        bundle = _strip_checkpoint_fields([checkpoint_row])[0]
        validated = validate_rubric_bundle(bundle)
        task_id = int(validated["task_id"])
        if str(validated["query"]) != str(facts_by_task[task_id]["query"]):
            raise ValueError(f"Rubric query mismatch for task {task_id}")
        rubrics_by_task[task_id] = validated
    missing_rubrics = sorted(expected_set - set(rubrics_by_task))
    if missing_rubrics and not args.force_finalize:
        raise SystemExit(f"missing Rubrics for tasks: {missing_rubrics}")
    write_jsonl_atomic(
        output / "rubrics.jsonl",
        (rubrics_by_task[task_id] for task_id in task_ids if task_id in rubrics_by_task),
        force=True,
    )

    all_judge_failures: dict[str, list[dict]] = {}
    evaluations_by_label: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    actors: dict[str, dict] = {}

    for label, trajectories in raw_runs.items():
        normalized_metrics = {}
        for raw in trajectories:
            normalized = normalize_trajectory(raw)
            metrics = compute_deterministic_metrics(normalized)
            normalized_metrics[normalized["trajectory_id"]] = (
                normalized,
                metrics,
            )
        checkpoint_path = checkpoints / f"judges-{label}.jsonl"
        metadata_path = calls_dir / f"judges-{label}.jsonl"

        def judge(item: tuple[str, tuple[dict, dict]]) -> tuple[dict, dict]:
            trajectory_id, pair = item
            normalized, metrics = pair
            task_id = int(normalized["task_id"])
            if metrics["validity"].get("infrastructure_invalid"):
                row = build_not_judged_result(
                    task_id=task_id,
                    trajectory_id=trajectory_id,
                    reason="infrastructure_invalid",
                )
                return row, {
                    "validation_repairs": 0,
                    "calls": [],
                    "not_judged_reason": "infrastructure_invalid",
                }
            rubric = rubrics_by_task[task_id]
            messages = build_trajectory_judge_messages(
                normalized=normalized,
                rubric_bundle=rubric,
                deterministic_metrics=metrics,
            )
            allowed_event_ids = [
                str(event["event_id"])
                for event in normalized.get("events") or []
                if isinstance(event, Mapping) and event.get("event_id")
            ]

            def validate(response: Mapping) -> dict:
                return validate_judge_result(
                    response,
                    rubric_ids=rubric_ids(rubric),
                    expected_task_id=task_id,
                    expected_trajectory_id=trajectory_id,
                    allowed_event_ids=allowed_event_ids,
                )

            try:
                return _complete_validated(
                    client=pro_client,
                    messages=messages,
                    validate=validate,
                    validation_retries=args.validation_retries,
                )
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 400 and "Content Exists Risk" in body:
                    row = build_not_judged_result(
                        task_id=task_id,
                        trajectory_id=trajectory_id,
                        reason="judge_provider_content_filter",
                    )
                    return row, {
                        "validation_repairs": 0,
                        "calls": [],
                        "not_judged_reason": "judge_provider_content_filter",
                        "provider_http_status": 400,
                    }
                raise

        judge_rows, judge_failures = _run_parallel(
            label=f"{label} Judge",
            items=list(normalized_metrics.items()),
            workers=args.judge_workers,
            worker=judge,
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            row_key=lambda row: row["trajectory_id"],
        )
        all_judge_failures[label] = judge_failures
        if judge_failures and not args.force_finalize:
            write_json_atomic(
                output / "judge_failures.json",
                all_judge_failures,
                force=True,
            )
            raise SystemExit(f"{label} Judge has {len(judge_failures)} failures; rerun to resume")
        canonical_judges = _strip_checkpoint_fields(judge_rows.values())
        canonical_judges.sort(key=lambda row: int(row["task_id"]))
        write_jsonl_atomic(
            output / f"judges-{label}.jsonl",
            canonical_judges,
            force=True,
        )
        actor = _load_actor(label, manifest_paths.get(label))
        actors[label] = actor
        artifacts = evaluate_trajectories(
            expected_task_ids=task_ids,
            trajectories=trajectories,
            actor=actor,
            task_slices=task_slices,
            rubric_bundles=(
                rubrics_by_task[task_id] for task_id in task_ids if task_id in rubrics_by_task
            ),
            judge_results=canonical_judges,
        )
        run_output = runs_dir / label
        run_output.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(
            run_output / "evaluations.jsonl",
            artifacts["evaluations"],
            force=True,
        )
        write_json_atomic(
            run_output / "summary.json",
            artifacts["summary"],
            force=True,
        )
        evaluations_by_label[label] = artifacts["evaluations"]
        summaries[label] = artifacts["summary"]

    write_json_atomic(
        output / "judge_failures.json",
        all_judge_failures,
        force=True,
    )
    if len(evaluations_by_label) >= 2:
        comparison = compare_evaluation_runs(
            expected_task_ids=task_ids,
            runs=evaluations_by_label,
            task_slices=task_slices,
        )
    else:
        comparison = {
            "schema_version": "single-run-evaluation-comparison-v1",
            "expected_tasks": len(task_ids),
            "runs": list(evaluations_by_label),
            "pairwise": {},
            "note": "Single-run Judge execution; pairwise comparison is not applicable.",
        }
    write_json_atomic(output / "comparison.json", comparison, force=True)
    write_json_atomic(output / "source_audit.json", source_audit, force=True)

    call_totals = {
        "rubrics": _metadata_totals(rubric_calls),
        "judges": {
            label: _metadata_totals(calls_dir / f"judges-{label}.jsonl") for label in run_paths
        },
    }
    output_files = [
        output / "rubrics.jsonl",
        output / "comparison.json",
        output / "source_audit.json",
        *[runs_dir / label / "evaluations.jsonl" for label in run_paths],
        *[runs_dir / label / "summary.json" for label in run_paths],
        *[output / f"judges-{label}.jsonl" for label in run_paths],
    ]
    manifest = {
        "schema_version": "existing-trajectory-judge-run-v1",
        "runner_version": RUNNER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "path": str(args.benchmark.resolve()),
            "sha256": sha256_file(args.benchmark),
            "tasks": len(task_ids),
        },
        "protocol": {
            "missing_tasks_count_as_failures": True,
            "rubric_prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
            "judge_prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "rubric_workers": args.rubric_workers,
            "judge_workers": args.judge_workers,
            "validation_retries": args.validation_retries,
            "judge_blind_to_reward_and_gold": True,
        },
        "models": {
            "rubric_curator": flash_client.model,
            "trajectory_judge": pro_client.model,
            "actors": actors,
        },
        "source_audit": source_audit,
        "call_totals": call_totals,
        "summaries": summaries,
        "outputs": {
            str(path.relative_to(output)): sha256_file(path)
            for path in output_files
            if path.is_file()
        },
    }
    write_json_atomic(output / "run_manifest.json", manifest, force=True)
    print(json.dumps({"output": str(output), "call_totals": call_totals}))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    main()
