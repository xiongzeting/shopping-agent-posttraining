"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    # Ray/DataProto may pass a numpy object array; boolean evaluation of such
    # an array raises an ambiguous-truth-value error.
    if shopping_infos is None or len(shopping_infos) == 0:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "total",
        "efficiency",
        "penalty_overlong",
        "penalty_unfinished",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    infrastructure_invalid = []
    reward_unverifiable = []
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    match_scores = []
    evidence_coverage = []
    partial_purchase = []
    reward_type_counts = {}
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        terminal_utilities.append(
            float(reward.get("terminal_utility", reward["total"]))
        )
        purchase_success.append(
            float(bool(reward.get("purchase_success", reward["full"])))
        )
        sampling_invalid.append(
            float(
                bool(
                    reward.get(
                        "sampling_invalid",
                        info.get("infrastructure_invalid")
                        or info.get("reward_unverifiable"),
                    )
                )
            )
        )
        match_scores.append(float(reward.get("match_score", reward["r_att"])))
        evidence_coverage.append(
            float(reward.get("evidence_coverage", 0.0))
        )
        partial_purchase.append(
            float(info.get("reward_type") == "partial_alternative_purchase")
        )
        reward_type = str(info.get("reward_type") or "unknown")
        reward_type_counts[reward_type] = reward_type_counts.get(reward_type, 0) + 1

    def mean(values):
        return sum(values) / len(values)

    metrics = {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/terminal_utility_min": min(terminal_utilities),
        "reward/terminal_utility_mean": mean(terminal_utilities),
        "reward/terminal_utility_max": max(terminal_utilities),
        "reward/purchase_success_rate": mean(purchase_success),
        "reward/partial_purchase_rate": mean(partial_purchase),
        "reward/match_score_mean": mean(match_scores),
        "reward/evidence_coverage_mean": mean(evidence_coverage),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/overlong_mean": mean(rewards["penalty_overlong"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
    }
    # Keep terminal classifications visible in the GRPO log.  This is the
    # first diagnostic needed when a gate reports 0/4 successful trajectories:
    # it distinguishes genuine model failure from reward/infra misclassification.
    for reward_type, count in sorted(reward_type_counts.items()):
        metrics[f"reward/type_count/{reward_type}"] = float(count)
    return metrics


def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return terminal utility, success metrics, and explicit invalid reasons."""
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing terminal_utility"
            ) from exc
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping terminal_utility at index {index} is not finite"
            )
        raw_purchase_success = info["reward"].get("purchase_success")
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise ValueError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        if bool(info.get("reward_unverifiable")):
            reasons.append("reward_unverifiable")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if str(info.get("reward_type") or info["reward"].get("reward_type") or "") == "assistant_final":
            reasons.append("assistant_final")
        if reward_sampling_invalid and not reasons:
            reasons.append("reward_sampling_invalid")
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(
            any(reason != "assistant_final" for reason in reasons)
        )
        invalid_reasons.append(tuple(reasons))
    return (
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
    require_purchase_success: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with useful reward spread.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    utility_values = (
        terminal_utilities if terminal_utilities is not None else seq_rewards
    )
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_utility,
        raw_success,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            utility_values,
            success_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        utility = float(raw_utility)
        if not math.isfinite(utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["terminal_utilities"].append(utility)
        group["purchase_success"].append(bool(raw_success))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        utilities = group["terminal_utilities"]
        utility_min = min(utilities)
        utility_max = max(utilities)
        reward_range = utility_max - utility_min
        utility_varying = reward_range > tolerance
        has_sampling_invalid = any(group["sampling_invalid"])
        has_assistant_final = "assistant_final" in group["sampling_invalid_reasons"]
        # assistant_final is a valid model failure with a negative learning
        # signal. Only genuinely unverifiable trajectories are removed.
        forbidden_reasons = {"reward_unverifiable"}
        discard_indices = {
            index
            for index, reasons_for_row in zip(
                group["indices"],
                # Reconstruct row-level reasons from the aligned input.  A
                # forbidden terminal is discarded individually, never retried.
                [reason_values[index] for index in group["indices"]],
            )
            if forbidden_reasons.intersection(str(reason) for reason in reasons_for_row)
        }
        valid_indices = [index for index in group["indices"] if index not in discard_indices]
        valid_positions = [group["indices"].index(index) for index in valid_indices]
        valid_utilities = [utilities[position] for position in valid_positions]
        valid_successes = [group["purchase_success"][position] for position in valid_positions]
        valid_reward_range = (
            max(valid_utilities) - min(valid_utilities) if valid_utilities else 0.0
        )
        valid_has_purchase_success = any(valid_successes)
        has_purchase_success = any(group["purchase_success"])
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        if discard_indices:
            # GRPO keeps group-relative advantages and veRL's actor iterator
            # requires complete group-sized batches.  Remove the entire
            # affected group (without retrying it) rather than sending a
            # partial 3/4 group that can produce a non-divisible actor batch.
            drop_reason = "forbidden_trajectory_in_group"
        elif not valid_indices:
            drop_reason = "all_trajectories_invalid"
        elif require_purchase_success and not valid_has_purchase_success:
            drop_reason = "no_purchase_success"
        elif valid_reward_range <= tolerance:
            drop_reason = "low_reward_variation"
        elif has_sampling_invalid and not discard_indices:
            drop_reason = "sampling_invalid"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(valid_indices),
                "discard_indices": tuple(discard_indices),
                "invalid_trajectory_count": len(discard_indices),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(valid_utilities),
                "purchase_success": tuple(valid_successes),
                "has_purchase_success": valid_has_purchase_success,
                "utility_min": utility_min,
                "utility_max": utility_max,
                "reward_range": reward_range,
                "reward_tolerance": tolerance,
                "reward_varying": utility_varying,
                "sampling_invalid": has_sampling_invalid,
                "assistant_final": has_assistant_final,
                "discarded_invalid_trajectories": bool(discard_indices),
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                # An unverifiable trajectory is permanently discarded. Even
                # if the remaining trajectories later fail the variation
                # gate, do not regenerate the discarded slot.
                "retryable": (
                    not discard_indices
                    and drop_reason not in {
                        "all_trajectories_invalid",
                        "sampling_invalid",
                        "forbidden_trajectory_in_group",
                    }
                ),
                "kept": keep,
            }
        )

    kept_index_set = {
        index
        for group in groups
        if group["kept"]
        for index in group["indices"]
    }
    trajectory_indices = [index for index in range(len(uids)) if index in kept_index_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "low_reward_variation_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_utility_group_count": sum(
            bool(group["terminal_utilities"])
            and max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            bool(group["purchase_success"])
            and all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "discarded_invalid_trajectory_count": sum(
            group["invalid_trajectory_count"] for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not group["has_purchase_success"] for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
        "retryable_dropped_group_indices": tuple(
            index for index, group in enumerate(groups)
            if not group["kept"] and group["retryable"]
        ),
    }
    return trajectory_indices, stats
