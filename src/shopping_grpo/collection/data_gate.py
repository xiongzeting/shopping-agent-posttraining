"""Corpus-level data gate for base-friendly successful Teacher trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

DATA_GATE_VERSION = "shopping-teacher-data-gate-v1"
RETRIEVAL_BUCKETS = ("rank1", "rank2_5", "rank6_20", "rank21_150", "missing")
DEFAULT_POLICY = {
    "target_rows": 1000,
    "retrieval": {
        "rank1_max_share": 0.60,
        "rank2_5_min_share": 0.20,
        "rank6_20_min_share": 0.10,
        "rank21_150_min_share": 0.06,
        "missing_min_share": 0.02,
    },
    "coverage_min_share": {
        "search_reformulation": 0.20,
        "candidate_comparison": 0.25,
        "multiple_options": 0.20,
        "guard_recovery": 0.05,
        "medium_or_long": 0.35,
        "long": 0.10,
        "loop_recovery": 0.20,
        "evidence_verification": 0.35,
        "variant_selection": 0.20,
        "explicit_terminal_buy": 1.00,
        "clean_critical_actions": 1.00,
    },
    "caps": {
        "exact_action_sequence_max_share": 0.12,
        "eight_step_max_share": 0.30,
    },
}


def retrieval_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    rank = int(rank)
    if rank > 150:
        return "missing"
    if rank < 1:
        raise ValueError(f"retrieval rank must be positive, got {rank}")
    if rank <= 1:
        return "rank1"
    if rank <= 5:
        return "rank2_5"
    if rank <= 20:
        return "rank6_20"
    return "rank21_150"


def _step_parameters(step: dict) -> dict:
    parameters = step.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    function = (step.get("tool_call") or {}).get("function") or {}
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return arguments if isinstance(arguments, dict) else {}


def action_sequence(trajectory: dict) -> tuple[str, ...]:
    return tuple(str(step.get("tool_name") or "") for step in trajectory.get("steps") or [])


def trajectory_coverage(trajectory: dict) -> dict[str, bool]:
    sequence = action_sequence(trajectory)
    queries = []
    candidates = []
    for step in trajectory.get("steps") or []:
        parameters = _step_parameters(step)
        if step.get("tool_name") == "search_products":
            query = " ".join(str(parameters.get("query") or "").split()).casefold()
            if query:
                queries.append(query)
        elif step.get("tool_name") == "open_product":
            asin = str(parameters.get("asin") or "").strip()
            if asin:
                candidates.append(asin)
    length = len(sequence)
    evidence_actions = {
        "view_features",
        "view_description",
        "view_attributes",
        "view_reviews",
    }
    local_repeat_loop = False
    previous_occurrences = {}
    steps = trajectory.get("steps") or []
    initial_state = (trajectory.get("initial_result") or {}).get("observation_state")
    for index, step in enumerate(steps):
        tool_name = str(step.get("tool_name") or "")
        if tool_name not in {"search_products", "open_product", "select_option", "buy_now"}:
            continue
        parameters = _step_parameters(step)
        if tool_name != "buy_now" and not parameters:
            continue
        signature = json.dumps(
            [tool_name, parameters],
            ensure_ascii=False,
            sort_keys=True,
        ).casefold()
        pre_state = (
            initial_state
            if index == 0
            else ((steps[index - 1].get("result") or {}).get("observation_state"))
        )
        for previous_index, previous_state in reversed(
            previous_occurrences.get(signature, [])
        ):
            if index - previous_index > 3:
                break
            if pre_state is None or previous_state is None or pre_state == previous_state:
                local_repeat_loop = True
                break
        previous_occurrences.setdefault(signature, []).append((index, pre_state))
    return {
        "search_reformulation": len(set(queries)) >= 2,
        "candidate_comparison": len(set(candidates)) >= 2,
        "multiple_options": sequence.count("select_option") >= 2,
        "guard_recovery": bool(trajectory.get("blocked_tool_calls")),
        "medium_or_long": length > 10,
        "long": length > 20,
        "loop_recovery": len(set(queries)) >= 2
        and (
            len(set(candidates)) >= 2
            or "back_to_search" in sequence
            or "next_page" in sequence
        ),
        "evidence_verification": any(name in evidence_actions for name in sequence),
        "variant_selection": "select_option" in sequence,
        "explicit_terminal_buy": bool(sequence) and sequence[-1] == "buy_now",
        "clean_critical_actions": not local_repeat_loop,
    }


def _minimum_count(total: int, share: float) -> int:
    return math.ceil(int(total) * float(share))


def _maximum_count(total: int, share: float) -> int:
    return math.floor(int(total) * float(share))


def audit_data_gate(
    trajectories: list[dict],
    *,
    retrieval_ranks: dict[int, int | None],
    policy: dict | None = None,
) -> dict:
    """Return a deterministic pass/fail report for the final Teacher corpus."""

    effective = json.loads(json.dumps(policy or DEFAULT_POLICY))
    target = int(effective["target_rows"])
    rank_counts = Counter()
    coverage_counts = Counter()
    sequences = Counter()
    lengths = Counter()
    task_ids = set()
    missing_rank_metadata = []

    for trajectory in trajectories:
        task_id = int(trajectory["task_id"])
        task_ids.add(task_id)
        if task_id not in retrieval_ranks:
            missing_rank_metadata.append(task_id)
        else:
            rank_counts[retrieval_bucket(retrieval_ranks[task_id])] += 1
        coverage_counts.update(
            name for name, present in trajectory_coverage(trajectory).items() if present
        )
        sequence = action_sequence(trajectory)
        sequences[sequence] += 1
        lengths[len(sequence)] += 1

    deficits = {}
    if len(trajectories) != target:
        deficits["target_rows"] = {"required": target, "actual": len(trajectories)}
    if len(task_ids) != len(trajectories):
        deficits["unique_task_ids"] = {
            "required": len(trajectories),
            "actual": len(task_ids),
        }
    if missing_rank_metadata:
        deficits["retrieval_rank_metadata"] = {
            "required": len(trajectories),
            "actual": len(trajectories) - len(missing_rank_metadata),
            "missing_task_ids": sorted(set(missing_rank_metadata)),
        }

    retrieval_policy = effective["retrieval"]
    rank1_max = _maximum_count(target, retrieval_policy["rank1_max_share"])
    if rank_counts["rank1"] > rank1_max:
        deficits["rank1_max"] = {"required_max": rank1_max, "actual": rank_counts["rank1"]}
    for bucket in RETRIEVAL_BUCKETS[1:]:
        key = f"{bucket}_min_share"
        minimum = _minimum_count(target, retrieval_policy[key])
        if rank_counts[bucket] < minimum:
            deficits[f"{bucket}_min"] = {
                "required_min": minimum,
                "actual": rank_counts[bucket],
            }

    coverage_minimums = {}
    for name, share in effective["coverage_min_share"].items():
        minimum = _minimum_count(target, share)
        coverage_minimums[name] = minimum
        if coverage_counts[name] < minimum:
            deficits[f"coverage.{name}"] = {
                "required_min": minimum,
                "actual": coverage_counts[name],
            }

    caps = effective["caps"]
    sequence_cap = _maximum_count(target, caps["exact_action_sequence_max_share"])
    top_sequence_count = sequences.most_common(1)[0][1] if sequences else 0
    if top_sequence_count > sequence_cap:
        deficits["exact_action_sequence_max"] = {
            "required_max": sequence_cap,
            "actual": top_sequence_count,
        }
    eight_step_cap = _maximum_count(target, caps["eight_step_max_share"])
    if lengths[8] > eight_step_cap:
        deficits["eight_step_max"] = {
            "required_max": eight_step_cap,
            "actual": lengths[8],
        }

    top_sequences = [
        {
            "count": count,
            "sha256": hashlib.sha256("\u2192".join(sequence).encode()).hexdigest(),
            "sequence": list(sequence),
        }
        for sequence, count in sequences.most_common(10)
    ]
    return {
        "schema_version": DATA_GATE_VERSION,
        "status": "passed" if not deficits else "failed",
        "policy": effective,
        "rows": len(trajectories),
        "unique_task_ids": len(task_ids),
        "retrieval_rank_counts": dict(sorted(rank_counts.items())),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "coverage_minimums": coverage_minimums,
        "length_histogram": dict(sorted(lengths.items())),
        "unique_action_sequences": len(sequences),
        "top_sequence_count": top_sequence_count,
        "top_sequences": top_sequences,
        "deficits": deficits,
    }
