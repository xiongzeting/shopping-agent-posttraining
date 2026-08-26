"""Assemble four evaluation sections and paired-ready summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from statistics import median

from shopping_grpo.evaluation.contracts import (
    CONTRACT_VERSION,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    rubric_ids,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.metrics import DETERMINISTIC_METRICS_VERSION
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION


EVALUATION_RESULT_VERSION = "shopping-per-task-evaluation-v1"
EVALUATION_SUMMARY_VERSION = "shopping-evaluation-summary-v1"


def build_not_judged_result(
    *,
    task_id: int,
    trajectory_id: str,
    reason: str,
) -> dict:
    """Represent an infrastructure-invalid trajectory without fake zero scores."""

    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": int(task_id),
        "trajectory_id": str(trajectory_id),
        "judge_status": "not_judged",
        "not_judged_reason": str(reason),
        "rubric_assessments": [],
        "dimension_scores": {},
        "errors": {
            "primary": (
                "infrastructure_invalid" if str(reason) == "infrastructure_invalid" else None
            ),
            "secondary": [],
            "evidence_event_ids": [],
        },
        "overall_diagnosis": (
            "轨迹因基础设施无效而未进行需求和轨迹质量评分。"
            if str(reason) == "infrastructure_invalid"
            else "本次运行未启用离线 Judge，不生成需求判断或轨迹质量分数。"
        ),
    }


def _reward_rubric_disagreement(
    *,
    metrics: Mapping,
    rubric_bundle: Mapping,
    judge_result: Mapping,
) -> dict:
    if judge_result.get("judge_status") != "valid":
        return {"value": False, "reasons": []}
    hardness = {item["rubric_id"]: item["hardness"] for item in rubric_bundle["rubrics"]}
    statuses = {item["rubric_id"]: item["status"] for item in judge_result["rubric_assessments"]}
    reward = metrics.get("reward_and_outcome")
    reward = reward if isinstance(reward, Mapping) else {}
    reward_type = reward.get("reward_type")
    purchase_success = reward.get("purchase_success") is True
    reasons = []

    violated = sorted(rubric_id for rubric_id, status in statuses.items() if status == "violated")
    violated_hard = sorted(rubric_id for rubric_id in violated if hardness.get(rubric_id) == "hard")
    if purchase_success and violated:
        reasons.append(
            {
                "type": "reward_success_with_rubric_violation",
                "rubric_ids": violated,
            }
        )
    if purchase_success and violated_hard:
        reasons.append(
            {
                "type": "reward_success_with_hard_rubric_violation",
                "rubric_ids": violated_hard,
            }
        )

    applicable_hard = {
        rubric_id: status
        for rubric_id, status in statuses.items()
        if hardness.get(rubric_id) == "hard" and status != "not_applicable"
    }
    all_hard_satisfied = bool(applicable_hard) and all(
        status == "satisfied" for status in applicable_hard.values()
    )
    if reward_type == "wrong_purchase" and all_hard_satisfied:
        reasons.append(
            {
                "type": "wrong_purchase_with_all_hard_rubrics_satisfied",
                "rubric_ids": sorted(applicable_hard),
            }
        )
    applicable = {
        rubric_id: status for rubric_id, status in statuses.items() if status != "not_applicable"
    }
    if (
        reward_type == "partial_alternative_purchase"
        and applicable
        and all(status == "satisfied" for status in applicable.values())
    ):
        reasons.append(
            {
                "type": "partial_reward_with_all_rubrics_satisfied",
                "rubric_ids": sorted(applicable),
            }
        )
    return {"value": bool(reasons), "reasons": reasons}


def assemble_task_evaluation(
    *,
    actor: Mapping,
    normalized_trajectory: Mapping,
    deterministic_metrics: Mapping,
    rubric_bundle: Mapping,
    judge_result: Mapping,
) -> dict:
    """Join one Actor run without mixing its four evaluation sections."""

    if normalized_trajectory.get("schema_version") != NORMALIZED_TRAJECTORY_VERSION:
        raise ValueError("unsupported normalized trajectory schema")
    if deterministic_metrics.get("schema_version") != DETERMINISTIC_METRICS_VERSION:
        raise ValueError("unsupported deterministic metrics schema")
    task_id = int(normalized_trajectory["task_id"])
    trajectory_id = str(normalized_trajectory["trajectory_id"])
    rubric = validate_rubric_bundle(
        rubric_bundle,
        expected_task_id=task_id,
    )
    judge = validate_judge_result(
        judge_result,
        rubric_ids=rubric_ids(rubric),
        expected_task_id=task_id,
        expected_trajectory_id=trajectory_id,
        allowed_event_ids=[
            event["event_id"]
            for event in normalized_trajectory.get("events") or []
            if isinstance(event, Mapping) and event.get("event_id")
        ],
    )
    if int(deterministic_metrics.get("task_id")) != task_id:
        raise ValueError("metrics task_id does not match trajectory")
    if str(deterministic_metrics.get("trajectory_id")) != trajectory_id:
        raise ValueError("metrics trajectory_id does not match trajectory")

    validity = deterministic_metrics.get("validity")
    validity = validity if isinstance(validity, Mapping) else {}
    if validity.get("infrastructure_invalid") and judge["judge_status"] != "not_judged":
        raise ValueError("infrastructure-invalid trajectories must use judge_status=not_judged")
    disagreement = _reward_rubric_disagreement(
        metrics=deterministic_metrics,
        rubric_bundle=rubric,
        judge_result=judge,
    )
    requirement_constraints = deterministic_metrics.get("requirement_constraints")
    if not isinstance(requirement_constraints, Mapping):
        raise ValueError("deterministic metrics are missing requirement_constraints")
    return {
        "schema_version": EVALUATION_RESULT_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "actor": deepcopy(dict(actor)),
        "reward_and_terminal": {
            "metrics": deepcopy(deterministic_metrics["reward_and_outcome"]),
            "terminal": deepcopy(normalized_trajectory.get("terminal") or {}),
        },
        "requirement_rubric": {
            "rubric_version": rubric["rubric_version"],
            "rubrics": deepcopy(rubric["rubrics"]),
            "assessments": deepcopy(judge["rubric_assessments"]),
            "reward_constraint_version": requirement_constraints.get("query_constraint_version"),
            "reward_constraint_results": deepcopy(
                requirement_constraints.get("constraint_results") or []
            ),
            "reward_constraint_summary": deepcopy(
                requirement_constraints.get("constraint_summary") or {}
            ),
            "reward_rubric_disagreement": disagreement["value"],
            "disagreement_reasons": disagreement["reasons"],
        },
        "trajectory_quality": {
            "judge_status": judge["judge_status"],
            "not_judged_reason": judge.get("not_judged_reason"),
            "dimension_scores": deepcopy(judge["dimension_scores"]),
            "errors": deepcopy(judge["errors"]),
            "overall_diagnosis": judge["overall_diagnosis"],
        },
        "deterministic": {
            key: deepcopy(value)
            for key, value in deterministic_metrics.items()
            if key
            not in {
                "schema_version",
                "evaluation_contract",
                "trajectory_id",
                "task_id",
                "reward_and_outcome",
                "requirement_constraints",
            }
        },
        "artifacts": {
            "normalized_trajectory_schema": NORMALIZED_TRAJECTORY_VERSION,
            "deterministic_metrics_schema": DETERMINISTIC_METRICS_VERSION,
            "rubric_schema": rubric["schema_version"],
            "judge_schema": judge["schema_version"],
        },
    }


def _mean(total: float, denominator: int) -> float:
    return total / denominator if denominator else 0.0


def _distribution(values: list[float]) -> dict:
    return {
        "tasks": len(values),
        "median": median(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }


def summarize_evaluations(
    *,
    expected_task_ids: Iterable[int],
    evaluations: Iterable[Mapping],
    task_slices: Mapping[int, Mapping] | None = None,
) -> dict:
    """Summarize four panels without producing a composite score."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    expected_set = set(expected)
    by_task = {}
    unexpected = []
    for record in evaluations:
        if record.get("schema_version") != EVALUATION_RESULT_VERSION:
            raise ValueError("unsupported per-task evaluation schema")
        task_id = int(record["task_id"])
        if task_id not in expected_set:
            unexpected.append(task_id)
            continue
        if task_id in by_task:
            raise ValueError(f"duplicate evaluation for task_id {task_id}")
        by_task[task_id] = record
    if unexpected:
        raise ValueError(f"unexpected task_ids: {sorted(set(unexpected))}")

    denominator = len(expected)
    missing = sorted(expected_set - set(by_task))
    reward_type_counts = Counter()
    strict_successes = []
    purchase_successes = 0
    reward_valid = 0
    total_reward = 0.0
    total_terminal_utility = 0.0
    total_weighted_score = 0.0
    rubric_status = Counter()
    rubric_status_by_hardness = defaultdict(Counter)
    reward_constraint_status = Counter()
    reward_constraint_status_by_role = defaultdict(Counter)
    reward_constraint_status_by_type = defaultdict(Counter)
    disagreement_tasks = []
    judge_status = Counter()
    dimension_distributions = {name: Counter() for name in JUDGE_DIMENSIONS}
    primary_errors = Counter()
    secondary_errors = Counter()
    primary_error_task_ids = defaultdict(list)
    secondary_error_task_ids = defaultdict(list)
    total_steps = 0
    total_attempts = 0
    total_guards = 0
    total_duplicate_actions = 0
    total_duplicate_searches = 0
    truncated_tasks = 0
    successful_steps = []
    failed_steps = []
    context_usage_ratios = []
    raw_observation_tokens = 0
    visible_observation_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    model_latency_seconds = 0.0
    tool_latency_seconds = 0.0
    trajectory_duration_seconds = 0.0
    token_tasks = 0
    timing_tasks = 0
    infrastructure_invalid_tasks = []

    for task_id, record in by_task.items():
        reward = record["reward_and_terminal"]["metrics"]
        reward_type_counts[str(reward.get("reward_type") or "unknown")] += 1
        if reward.get("strict_gold_success") is True:
            strict_successes.append(task_id)
        purchase_successes += reward.get("purchase_success") is True
        reward_valid += reward.get("reward_valid") is True
        total_reward += float(reward.get("final_reward", 0.0) or 0.0)
        total_terminal_utility += float(reward.get("terminal_utility", 0.0) or 0.0)
        total_weighted_score += float(reward.get("weighted_score", 0.0) or 0.0)

        rubric = record["requirement_rubric"]
        hardness = {item["rubric_id"]: item["hardness"] for item in rubric["rubrics"]}
        for assessment in rubric["assessments"]:
            status = assessment["status"]
            rubric_status[status] += 1
            rubric_status_by_hardness[hardness.get(assessment["rubric_id"], "unknown")][status] += 1
        for constraint in rubric.get("reward_constraint_results") or []:
            if not isinstance(constraint, Mapping):
                continue
            status = str(constraint.get("status") or "unknown")
            role = str(constraint.get("role") or "unknown")
            constraint_type = str(constraint.get("constraint_type") or "unknown")
            reward_constraint_status[status] += 1
            reward_constraint_status_by_role[role][status] += 1
            reward_constraint_status_by_type[constraint_type][status] += 1
        if rubric["reward_rubric_disagreement"]:
            disagreement_tasks.append(task_id)

        quality = record["trajectory_quality"]
        judge_status[quality["judge_status"]] += 1
        if quality["judge_status"] == "valid":
            for name in JUDGE_DIMENSIONS:
                dimension_distributions[name][int(quality["dimension_scores"][name]["score"])] += 1
            primary = quality["errors"].get("primary")
            if primary:
                primary_errors[str(primary)] += 1
                primary_error_task_ids[str(primary)].append(task_id)
            for secondary in quality["errors"].get("secondary") or []:
                secondary_errors[str(secondary)] += 1
                secondary_error_task_ids[str(secondary)].append(task_id)

        deterministic = record["deterministic"]
        actions = deterministic["actions_and_efficiency"]
        repetition = deterministic["repetition"]
        legality = deterministic["legality"]
        context = deterministic["context"]
        validity = deterministic["validity"]
        total_steps += int(actions.get("executed_tool_steps", 0))
        step_value = int(actions.get("executed_tool_steps", 0))
        if reward.get("strict_gold_success") is True:
            successful_steps.append(step_value)
        else:
            failed_steps.append(step_value)
        total_attempts += int(actions.get("action_attempts", 0))
        total_guards += int(legality.get("guard_rejection_count", 0))
        total_duplicate_actions += int(repetition.get("duplicate_canonical_action_count", 0))
        total_duplicate_searches += int(repetition.get("duplicate_search_query_count", 0))
        truncated_tasks += bool(context.get("any_observation_truncated"))
        raw_observation_tokens += int(context.get("raw_observation_tokens", 0) or 0)
        visible_observation_tokens += int(context.get("visible_observation_tokens", 0) or 0)
        usage_ratio = context.get("max_context_usage_ratio")
        if isinstance(usage_ratio, (int, float)) and not isinstance(usage_ratio, bool):
            context_usage_ratios.append(float(usage_ratio))
        task_total_tokens = context.get("total_tokens")
        task_completion_tokens = context.get("completion_tokens")
        if isinstance(task_total_tokens, (int, float)) and not isinstance(task_total_tokens, bool):
            total_tokens += int(task_total_tokens)
            completion_tokens += int(task_completion_tokens or 0)
            token_tasks += 1
        timing = deterministic.get("timing")
        timing = timing if isinstance(timing, Mapping) else {}
        duration = timing.get("trajectory_duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            trajectory_duration_seconds += float(duration)
            model_latency_seconds += float(timing.get("model_latency_seconds") or 0.0)
            tool_latency_seconds += float(timing.get("tool_latency_seconds") or 0.0)
            timing_tasks += 1
        if validity.get("infrastructure_invalid"):
            infrastructure_invalid_tasks.append(task_id)

    valid_judges = judge_status.get("valid", 0)
    dimension_summary = {}
    for name, distribution in dimension_distributions.items():
        score_total = sum(score * count for score, count in distribution.items())
        dimension_summary[name] = {
            "score_counts": {str(score): distribution.get(score, 0) for score in (0, 1, 2)},
            "mean_score_among_valid_judges": _mean(score_total, valid_judges),
        }
    summary = {
        "schema_version": EVALUATION_SUMMARY_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": denominator,
        "completed_evaluations": len(by_task),
        "missing_task_ids": missing,
        "reward_and_terminal": {
            "strict_gold_successes": len(strict_successes),
            "strict_gold_task_ids": sorted(strict_successes),
            "gold_purchase_rate": _mean(len(strict_successes), denominator),
            "purchase_successes": purchase_successes,
            "purchase_success_rate": _mean(purchase_successes, denominator),
            "reward_valid_tasks": reward_valid,
            "reward_valid_rate": _mean(reward_valid, denominator),
            "reward_type_counts": dict(sorted(reward_type_counts.items())),
            "total_final_reward": total_reward,
            "mean_final_reward_fixed_denominator": _mean(total_reward, denominator),
            "mean_terminal_utility_fixed_denominator": _mean(total_terminal_utility, denominator),
            "mean_weighted_score_fixed_denominator": _mean(total_weighted_score, denominator),
        },
        "requirement_rubric": {
            "status_counts": dict(sorted(rubric_status.items())),
            "status_counts_by_hardness": {
                hardness: dict(sorted(counts.items()))
                for hardness, counts in sorted(rubric_status_by_hardness.items())
            },
            "reward_rubric_disagreement_tasks": len(disagreement_tasks),
            "reward_rubric_disagreement_task_ids": sorted(disagreement_tasks),
            "reward_constraint_status_counts": dict(sorted(reward_constraint_status.items())),
            "reward_constraint_status_counts_by_role": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(reward_constraint_status_by_role.items())
            },
            "reward_constraint_status_counts_by_type": {
                constraint_type: dict(sorted(counts.items()))
                for constraint_type, counts in sorted(reward_constraint_status_by_type.items())
            },
        },
        "trajectory_quality": {
            "judge_status_counts": dict(sorted(judge_status.items())),
            "judge_coverage_rate": _mean(valid_judges, denominator),
            "dimensions": dimension_summary,
            "primary_error_counts": dict(sorted(primary_errors.items())),
            "primary_error_task_ids": {
                error: sorted(task_ids)
                for error, task_ids in sorted(primary_error_task_ids.items())
            },
            "secondary_error_counts": dict(sorted(secondary_errors.items())),
            "secondary_error_task_ids": {
                error: sorted(task_ids)
                for error, task_ids in sorted(secondary_error_task_ids.items())
            },
        },
        "deterministic": {
            "average_executed_steps_fixed_denominator": _mean(total_steps, denominator),
            "average_action_attempts_fixed_denominator": _mean(total_attempts, denominator),
            "executed_steps_successful_tasks": _distribution(successful_steps),
            "executed_steps_failed_tasks": _distribution(failed_steps),
            "total_guard_rejections": total_guards,
            "total_duplicate_canonical_actions": total_duplicate_actions,
            "total_duplicate_search_queries": total_duplicate_searches,
            "tasks_with_observation_truncation": truncated_tasks,
            "observation_tokens": {
                "raw": raw_observation_tokens,
                "visible": visible_observation_tokens,
                "compression_ratio_visible_over_raw": (
                    visible_observation_tokens / raw_observation_tokens
                    if raw_observation_tokens
                    else None
                ),
            },
            "context_budget": {
                "tasks_with_usage_ratio": len(context_usage_ratios),
                "max_usage_ratio": max(context_usage_ratios, default=None),
            },
            "tokens": {
                "tasks_with_provider_usage": token_tasks,
                "completion_tokens": completion_tokens if token_tasks else None,
                "total_tokens": total_tokens if token_tasks else None,
            },
            "timing": {
                "tasks_with_timing": timing_tasks,
                "trajectory_duration_seconds": (
                    trajectory_duration_seconds if timing_tasks else None
                ),
                "model_latency_seconds": model_latency_seconds if timing_tasks else None,
                "tool_latency_seconds": tool_latency_seconds if timing_tasks else None,
            },
            "infrastructure_invalid_tasks": len(infrastructure_invalid_tasks),
            "infrastructure_invalid_task_ids": sorted(infrastructure_invalid_tasks),
        },
    }
    if task_slices is not None:
        summary["stratified"] = _summarize_evaluation_slices(
            expected_task_ids=expected,
            evaluations=by_task,
            task_slices=task_slices,
        )
    return summary


def _summarize_evaluation_slices(
    *,
    expected_task_ids: list[int],
    evaluations: Mapping[int, Mapping],
    task_slices: Mapping[int, Mapping],
) -> dict:
    normalized_slices = {int(task_id): row for task_id, row in task_slices.items()}
    if set(normalized_slices) != set(expected_task_ids):
        raise ValueError("task_slices must cover every expected task exactly once")
    groups = {
        "suite": defaultdict(list),
        "domain": defaultdict(list),
        "challenge_slice": defaultdict(list),
    }
    for task_id in expected_task_ids:
        row = normalized_slices[task_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"task_slices[{task_id}] must be an object")
        suite = row.get("suite")
        domain = row.get("domain")
        challenge_slice = row.get("challenge_slice")
        if suite not in {"core", "challenge"}:
            raise ValueError(f"task_slices[{task_id}] has an invalid suite")
        if not isinstance(domain, str) or not domain:
            raise ValueError(f"task_slices[{task_id}] has an invalid domain")
        groups["suite"][suite].append(task_id)
        groups["domain"][domain].append(task_id)
        if suite == "challenge":
            if not isinstance(challenge_slice, str) or not challenge_slice:
                raise ValueError(f"task_slices[{task_id}] has an invalid challenge_slice")
            groups["challenge_slice"][challenge_slice].append(task_id)
    summarized = {
        group_name: {
            bucket_name: summarize_evaluations(
                expected_task_ids=task_ids,
                evaluations=[
                    evaluations[task_id] for task_id in task_ids if task_id in evaluations
                ],
            )
            for bucket_name, task_ids in sorted(buckets.items())
        }
        for group_name, buckets in groups.items()
    }
    for bucket in summarized["challenge_slice"].values():
        bucket["statistical_scope"] = "descriptive_only"
        bucket["statistical_note"] = (
            "Challenge slices contain 10 tasks each; use them for diagnosis, "
            "not standalone significance claims."
        )
    return summarized
