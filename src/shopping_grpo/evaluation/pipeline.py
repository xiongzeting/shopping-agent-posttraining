"""Unified ShopBench-LH v2 post-rollout evaluation pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256

from shopping_grpo.evaluation.contracts import RUBRIC_SCHEMA_VERSION
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation,
    build_not_judged_result,
    summarize_evaluations,
)
from shopping_grpo.evaluation.trajectory import normalize_trajectory


PIPELINE_VERSION = "shopbench-lh-v2-unified-pipeline-v1"


def _index_unique(
    rows: Iterable[Mapping],
    *,
    key: str,
    label: str,
) -> dict[object, Mapping]:
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} rows must be objects")
        if key not in row:
            raise ValueError(f"{label} row is missing {key}")
        value = row[key]
        if value in indexed:
            raise ValueError(f"duplicate {label} {key}={value!r}")
        indexed[value] = row
    return indexed


def _fallback_rubric(task_id: int, query: str) -> dict:
    """Represent unavailable LLM curation without inventing requirements."""

    query = str(query or "").strip()
    if not query:
        raise ValueError(f"trajectory for task_id {task_id} has no actor query")
    query_hash = sha256(query.encode("utf-8")).hexdigest()
    task_hash = sha256(f"{task_id}\0{query}".encode("utf-8")).hexdigest()
    return {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_version": "rubric-not-curated-v1",
        "task_id": int(task_id),
        "query": query,
        "generation": {
            "extractor_version": "not_run",
            "curator_model": "not_run",
            "curator_prompt_version": "not_run",
            "task_data_hash": task_hash,
            "query_hash": query_hash,
        },
        "rubrics": [],
    }


def evaluate_trajectories(
    *,
    expected_task_ids: Iterable[int],
    trajectories: Iterable[Mapping],
    actor: Mapping,
    task_slices: Mapping[int, Mapping] | None = None,
    rubric_bundles: Iterable[Mapping] | None = None,
    judge_results: Iterable[Mapping] | None = None,
) -> dict:
    """Normalize and score one formal attempt per expected task."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    expected_set = set(expected)
    raw_by_task = _index_unique(
        trajectories,
        key="task_id",
        label="trajectory",
    )
    unexpected = sorted(int(task_id) for task_id in raw_by_task if int(task_id) not in expected_set)
    if unexpected:
        raise ValueError(f"unexpected trajectory task_ids: {unexpected}")
    rubrics = _index_unique(
        rubric_bundles or [],
        key="task_id",
        label="rubric",
    )
    judges = _index_unique(
        judge_results or [],
        key="trajectory_id",
        label="judge",
    )

    normalized_rows = []
    metrics_rows = []
    rubric_rows = []
    judge_rows = []
    evaluations = []
    for task_id in expected:
        source = raw_by_task.get(task_id)
        if source is None:
            continue
        if int(source.get("attempt_index", 0) or 0) != 0:
            raise ValueError("formal Benchmark v2.1 evaluation requires attempt_index=0")
        normalized = normalize_trajectory(source)
        metrics = compute_deterministic_metrics(normalized)
        rubric = rubrics.get(task_id) or _fallback_rubric(task_id, normalized.get("actor_query"))
        judge = judges.get(normalized["trajectory_id"])
        if judge is None:
            invalid = metrics["validity"].get("infrastructure_invalid")
            reason = "infrastructure_invalid" if invalid else "judge_disabled"
            judge = build_not_judged_result(
                task_id=task_id,
                trajectory_id=normalized["trajectory_id"],
                reason=reason,
            )
        evaluation = assemble_task_evaluation(
            actor=actor,
            normalized_trajectory=normalized,
            deterministic_metrics=metrics,
            rubric_bundle=rubric,
            judge_result=judge,
        )
        normalized_rows.append(normalized)
        metrics_rows.append(metrics)
        judge_rows.append(judge)
        evaluations.append(evaluation)
        rubric_rows.append(
            {
                "task_id": task_id,
                "trajectory_id": normalized["trajectory_id"],
                "rubric_bundle": rubric,
                "assessments": judge.get("rubric_assessments") or [],
            }
        )

    summary = summarize_evaluations(
        expected_task_ids=expected,
        evaluations=evaluations,
        task_slices=task_slices,
    )
    summary["pipeline_version"] = PIPELINE_VERSION
    summary["judge_mode"] = "provided" if judges else "disabled"
    summary["rubric_mode"] = "provided" if rubrics else "reward_constraints_only"
    return {
        "normalized": normalized_rows,
        "deterministic_metrics": metrics_rows,
        "rubric_assessments": rubric_rows,
        "judges": judge_rows,
        "evaluations": evaluations,
        "summary": summary,
    }
