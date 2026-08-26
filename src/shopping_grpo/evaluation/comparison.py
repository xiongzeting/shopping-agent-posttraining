"""Simple paired Baseline/SFT/GRPO diagnostics without a composite score."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import combinations

from shopping_grpo.evaluation.contracts import CONTRACT_VERSION, JUDGE_DIMENSIONS
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION


COMPARISON_SCHEMA_VERSION = "shopping-paired-model-comparison-v1"


def _index_run(*, label: str, evaluations: Iterable[Mapping], expected: set[int]):
    result = {}
    for record in evaluations:
        if record.get("schema_version") != EVALUATION_RESULT_VERSION:
            raise ValueError(f"{label} contains an unsupported evaluation schema")
        task_id = int(record["task_id"])
        if task_id not in expected:
            raise ValueError(f"{label} contains unexpected task_id {task_id}")
        if task_id in result:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        result[task_id] = record
    return result


def _hard_violation_count(record: Mapping) -> int | None:
    quality = record["trajectory_quality"]
    if quality.get("judge_status") != "valid":
        return None
    rubric = record["requirement_rubric"]
    hardness = {item["rubric_id"]: item["hardness"] for item in rubric["rubrics"]}
    return sum(
        assessment["status"] == "violated" and hardness.get(assessment["rubric_id"]) == "hard"
        for assessment in rubric["assessments"]
    )


def _reward_constraint_status_count(record: Mapping, status: str) -> int:
    return sum(
        isinstance(item, Mapping) and item.get("status") == status
        for item in record["requirement_rubric"].get("reward_constraint_results", [])
    )


def _dimension_score(record: Mapping, name: str) -> int | None:
    quality = record["trajectory_quality"]
    if quality.get("judge_status") != "valid":
        return None
    return int(quality["dimension_scores"][name]["score"])


def _delta_summary(deltas: list[float], *, lower_is_better: bool) -> dict:
    if lower_is_better:
        improved = sum(delta < 0 for delta in deltas)
        worsened = sum(delta > 0 for delta in deltas)
    else:
        improved = sum(delta > 0 for delta in deltas)
        worsened = sum(delta < 0 for delta in deltas)
    return {
        "paired_tasks": len(deltas),
        "mean_delta_target_minus_source": sum(deltas) / len(deltas) if deltas else 0.0,
        "improved_tasks": improved,
        "unchanged_tasks": sum(delta == 0 for delta in deltas),
        "worsened_tasks": worsened,
    }


def _pairwise(
    expected_task_ids: list[int],
    source_label: str,
    source: Mapping[int, Mapping],
    target_label: str,
    target: Mapping[int, Mapping],
) -> dict:
    expected_ids = sorted(expected_task_ids)
    completed_pair_ids = sorted(set(source) & set(target))
    strict_transitions = Counter()
    reward_type_transitions = Counter()
    disagreement_transitions = Counter()
    hard_deltas = []
    reward_constraint_fail_deltas = []
    reward_constraint_unverifiable_deltas = []
    dimension_deltas = {name: [] for name in JUDGE_DIMENSIONS}
    step_deltas = []
    guard_deltas = []
    duplicate_action_deltas = []

    for task_id in expected_ids:
        left = source.get(task_id)
        right = target.get(task_id)
        left_reward = left["reward_and_terminal"]["metrics"] if left else {}
        right_reward = right["reward_and_terminal"]["metrics"] if right else {}
        left_success = bool(left_reward.get("strict_gold_success"))
        right_success = bool(right_reward.get("strict_gold_success"))
        strict_transitions[
            f"{'success' if left_success else 'failure'}_to_"
            f"{'success' if right_success else 'failure'}"
        ] += 1
        reward_type_transitions[
            f"{left_reward.get('reward_type', 'missing')} -> "
            f"{right_reward.get('reward_type', 'missing')}"
        ] += 1

        if left is None or right is None:
            continue
        left_disagreement = bool(left["requirement_rubric"]["reward_rubric_disagreement"])
        right_disagreement = bool(right["requirement_rubric"]["reward_rubric_disagreement"])
        disagreement_transitions[
            f"{'disagreement' if left_disagreement else 'aligned'}_to_"
            f"{'disagreement' if right_disagreement else 'aligned'}"
        ] += 1
        left_hard = _hard_violation_count(left)
        right_hard = _hard_violation_count(right)
        if left_hard is not None and right_hard is not None:
            hard_deltas.append(right_hard - left_hard)
        reward_constraint_fail_deltas.append(
            _reward_constraint_status_count(right, "fail")
            - _reward_constraint_status_count(left, "fail")
        )
        reward_constraint_unverifiable_deltas.append(
            _reward_constraint_status_count(right, "unverifiable")
            - _reward_constraint_status_count(left, "unverifiable")
        )
        for name in JUDGE_DIMENSIONS:
            left_score = _dimension_score(left, name)
            right_score = _dimension_score(right, name)
            if left_score is not None and right_score is not None:
                dimension_deltas[name].append(right_score - left_score)
        left_deterministic = left["deterministic"]
        right_deterministic = right["deterministic"]
        step_deltas.append(
            right_deterministic["actions_and_efficiency"]["executed_tool_steps"]
            - left_deterministic["actions_and_efficiency"]["executed_tool_steps"]
        )
        guard_deltas.append(
            right_deterministic["legality"]["guard_rejection_count"]
            - left_deterministic["legality"]["guard_rejection_count"]
        )
        duplicate_action_deltas.append(
            right_deterministic["repetition"]["duplicate_canonical_action_count"]
            - left_deterministic["repetition"]["duplicate_canonical_action_count"]
        )

    return {
        "source": source_label,
        "target": target_label,
        "paired_tasks": len(expected_ids),
        "both_completed_tasks": len(completed_pair_ids),
        "reward_and_terminal": {
            "strict_success_transitions": dict(sorted(strict_transitions.items())),
            "reward_type_transitions": dict(sorted(reward_type_transitions.items())),
        },
        "requirement_rubric": {
            "hard_violation_delta": _delta_summary(hard_deltas, lower_is_better=True),
            "reward_rubric_disagreement_transitions": dict(
                sorted(disagreement_transitions.items())
            ),
            "reward_constraint_fail_delta": _delta_summary(
                reward_constraint_fail_deltas, lower_is_better=True
            ),
            "reward_constraint_unverifiable_delta": _delta_summary(
                reward_constraint_unverifiable_deltas, lower_is_better=True
            ),
        },
        "trajectory_quality": {
            name: _delta_summary(deltas, lower_is_better=False)
            for name, deltas in dimension_deltas.items()
        },
        "deterministic": {
            "executed_tool_steps": _delta_summary(step_deltas, lower_is_better=True),
            "guard_rejections": _delta_summary(guard_deltas, lower_is_better=True),
            "duplicate_canonical_actions": _delta_summary(
                duplicate_action_deltas, lower_is_better=True
            ),
        },
    }


def compare_evaluation_runs(
    *,
    expected_task_ids: Iterable[int],
    runs: Mapping[str, Iterable[Mapping]],
    task_slices: Mapping[int, Mapping] | None = None,
) -> dict:
    """Compare each model pair on identical task IDs with simple deltas."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    if len(runs) < 2:
        raise ValueError("at least two model runs are required")
    expected_set = set(expected)
    indexed = {
        str(label): _index_run(label=str(label), evaluations=rows, expected=expected_set)
        for label, rows in runs.items()
    }
    labels = list(indexed)
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": len(expected),
        "models": {},
        "pairwise": {
            f"{source}_to_{target}": _pairwise(
                expected, source, indexed[source], target, indexed[target]
            )
            for source, target in combinations(labels, 2)
        },
    }
    for label in labels:
        successes = sum(
            bool(record["reward_and_terminal"]["metrics"].get("strict_gold_success"))
            for record in indexed[label].values()
        )
        result["models"][label] = {
            "completed_evaluations": len(indexed[label]),
            "missing_task_ids": sorted(expected_set - set(indexed[label])),
            "strict_gold_successes": successes,
            "strict_gold_success_rate": successes / len(expected) if expected else 0.0,
            "missing_tasks_count_as_failures": True,
        }

    if task_slices is not None:
        normalized_slices = {int(task_id): row for task_id, row in task_slices.items()}
        if set(normalized_slices) != expected_set:
            raise ValueError("task_slices must cover every expected task exactly once")
        groups = {"suite": {}, "domain": {}, "challenge_slice": {}}
        for task_id in expected:
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
            groups["suite"].setdefault(suite, []).append(task_id)
            groups["domain"].setdefault(domain, []).append(task_id)
            if suite == "challenge":
                if not isinstance(challenge_slice, str) or not challenge_slice:
                    raise ValueError(f"task_slices[{task_id}] has an invalid challenge_slice")
                groups["challenge_slice"].setdefault(challenge_slice, []).append(task_id)
        result["stratified"] = {
            group_name: {
                bucket_name: compare_evaluation_runs(
                    expected_task_ids=task_ids,
                    runs={
                        label: [
                            indexed[label][task_id]
                            for task_id in task_ids
                            if task_id in indexed[label]
                        ]
                        for label in labels
                    },
                )
                for bucket_name, task_ids in sorted(buckets.items())
            }
            for group_name, buckets in groups.items()
        }
    return result
