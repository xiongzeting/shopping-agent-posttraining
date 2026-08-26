"""Two-gate contract for preparing and admitting GRPO probe tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product as cartesian_product
import math
import re
from typing import Any
import unicodedata


def _nonempty(value: object) -> bool:
    return bool(str(value or "").strip())


def _option_values(raw_values: object) -> list[str]:
    values = []
    if not isinstance(raw_values, list):
        return values
    for raw in raw_values:
        value = raw.get("value") if isinstance(raw, Mapping) else raw
        if _nonempty(value) and str(value) not in values:
            values.append(str(value))
    return values


def _candidate_selections(product: Mapping, required: Mapping) -> list[dict]:
    axes = product.get("customization_options") or {}
    if not isinstance(axes, Mapping):
        return [dict(required)]
    names = []
    values_by_axis = []
    for name, raw_values in axes.items():
        values = _option_values(raw_values)
        if values:
            names.append(str(name))
            values_by_axis.append(values)
    selections = [dict(required), {}]
    combinations = 1
    for values in values_by_axis:
        combinations *= len(values)
    if combinations <= 256:
        for values in cartesian_product(*values_by_axis):
            selection = dict(zip(names, values))
            selection.update(required)
            selections.append(selection)
    return selections


def _required_selection(required_options: object) -> dict[str, str]:
    selected = {}
    if not isinstance(required_options, Mapping):
        return selected
    for canonical_axis, raw in required_options.items():
        if isinstance(raw, Mapping):
            axis = raw.get("source_axis") or canonical_axis
            value = raw.get("value")
        else:
            axis = canonical_axis
            value = raw
        if _nonempty(axis) and _nonempty(value):
            selected[str(axis)] = str(value)
    return selected


def _finite_price(value: object) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def _normalized_option_value_collisions(product: Mapping) -> list[dict]:
    collisions = []
    axes = product.get("customization_options") or {}
    if not isinstance(axes, Mapping):
        return collisions
    for axis, entries in axes.items():
        normalized_values: dict[str, set[str]] = {}
        if not isinstance(entries, list):
            continue
        for entry in entries:
            raw = entry.get("value") if isinstance(entry, Mapping) else entry
            value = str(raw or "")
            normalized = re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", value).casefold().replace("/", "|"),
            )
            if normalized:
                normalized_values.setdefault(normalized, set()).add(value)
        for normalized, raw_values in normalized_values.items():
            if len(raw_values) > 1:
                collisions.append(
                    {"axis": str(axis), "normalized": normalized, "values": sorted(raw_values)}
                )
    return collisions


def _pricing_option_range_mismatch(product: Mapping) -> dict | None:
    pricing = [
        price
        for price in (_finite_price(value) for value in product.get("pricing") or [])
        if price is not None
    ]
    option_prices = []
    axes = product.get("customization_options") or {}
    if isinstance(axes, Mapping):
        for entries in axes.values():
            for entry in entries or []:
                if isinstance(entry, Mapping):
                    price = _finite_price(entry.get("price"))
                    if price is not None:
                        option_prices.append(price)
    if not pricing or not option_prices:
        return None
    pricing_range = [min(pricing), max(pricing)]
    option_range = [min(option_prices), max(option_prices)]
    if pricing_range == option_range:
        return None
    return {"pricing_range": pricing_range, "option_price_range": option_range}


def validate_probe_task_data(
    task_id: int,
    product: object,
    *,
    compile_reward_features: Callable[[object, object], dict],
    resolve_variant_price: Callable[[dict, object], dict],
    allowed_source_tags: Sequence[str] = ("train",),
) -> dict[str, Any]:
    """Gate 1: determine whether a source task is safe to probe."""

    reasons = []
    if not isinstance(product, Mapping):
        return {"task_id": int(task_id), "accepted": False, "reasons": ["task_not_object"]}
    allowed_tags = {str(tag) for tag in allowed_source_tags}
    source_tag = str(product.get("tag") or "")
    if source_tag not in allowed_tags:
        reasons.append(f"source_tag_not_allowed:{source_tag or 'missing'}")
    for field in ("asin", "title", "shop_name", "category"):
        if not _nonempty(product.get(field)):
            reasons.append(f"missing_{field}")
    value_collisions = _normalized_option_value_collisions(product)
    if value_collisions:
        reasons.append("normalized_option_value_collision")
    price_range_mismatch = _pricing_option_range_mismatch(product)
    if price_range_mismatch:
        reasons.append("pricing_option_range_mismatch")
    instructions = product.get("instructions")
    if not isinstance(instructions, list) or len(instructions) != 1 or not isinstance(instructions[0], Mapping):
        reasons.append("instruction_contract_invalid")
        instruction = {}
    else:
        instruction = instructions[0]
        if not _nonempty(instruction.get("instruction")):
            reasons.append("instruction_text_missing")
        if str(instruction.get("asin") or "") != str(product.get("asin") or ""):
            reasons.append("instruction_target_asin_mismatch")

    reward_features = {}
    try:
        reward_features = compile_reward_features(instruction, product)
    except Exception as exc:
        reasons.append(f"reward_feature_compile_error:{exc.__class__.__name__}")
    if reward_features:
        if reward_features.get("reward_feature_version") != "shopping-reward-features-v2":
            reasons.append("reward_feature_version_invalid")
        if not _nonempty(reward_features.get("category")):
            reasons.append("reward_category_missing")
        unresolved = reward_features.get("unresolved_option_requirements") or []
        if unresolved:
            reasons.append("required_options_unresolved")
        contract = reward_features.get("query_constraint_contract")
        if not isinstance(contract, Mapping) or not isinstance(contract.get("constraints"), list):
            reasons.append("query_constraint_contract_invalid")

    required_options = reward_features.get("required_options_by_key") or {}
    required_selection = _required_selection(required_options)
    price_resolution = None
    if not reasons or all(reason == "query_constraint_contract_invalid" for reason in reasons):
        for selection in _candidate_selections(product, required_selection):
            try:
                resolution = resolve_variant_price(dict(product), selection)
            except Exception as exc:
                reasons.append(f"price_resolution_error:{exc.__class__.__name__}")
                break
            price = resolution.get("price") if isinstance(resolution, Mapping) else None
            if (
                isinstance(resolution, Mapping)
                and resolution.get("status") == "pass"
                and isinstance(price, (int, float))
                and math.isfinite(float(price))
                and float(price) >= 0
            ):
                price_resolution = {
                    "status": "pass",
                    "price": float(price),
                    "method": str(resolution.get("method") or ""),
                    "selection": selection,
                }
                break
        if price_resolution is None and not any(reason.startswith("price_resolution_error:") for reason in reasons):
            reasons.append("no_verifiable_variant_price")

    return {
        "task_id": int(task_id),
        "accepted": not reasons,
        "reasons": reasons,
        "reward_feature_version": reward_features.get("reward_feature_version"),
        "required_options": required_options,
        "price_resolution": price_resolution,
        "normalized_option_value_collisions": value_collisions,
        "price_range_mismatch": price_range_mismatch,
    }


def normalize_probe_trajectory(row: object) -> dict[str, Any]:
    """Normalize evaluator or veRL-style trajectory diagnostics for Gate 2."""

    if not isinstance(row, Mapping):
        return {"valid": False, "invalid_reasons": ["trajectory_not_object"]}
    reasons = []
    try:
        task_id = int(row.get("task_id"))
    except (TypeError, ValueError):
        task_id = None
        reasons.append("task_id_missing")
    try:
        attempt_index = int(row.get("attempt_index", 0))
    except (TypeError, ValueError):
        attempt_index = 0
        reasons.append("attempt_index_invalid")

    terminal = row.get("terminal_result") or {}
    shopping = row.get("shopping") or {}
    reward = (
        terminal.get("reward_detail")
        if isinstance(terminal, Mapping)
        else None
    )
    if not isinstance(reward, Mapping) and isinstance(shopping, Mapping):
        reward = shopping.get("reward")
    if not isinstance(reward, Mapping):
        reward = row.get("reward")
    if not isinstance(reward, Mapping):
        reward = {}
        reasons.append("reward_detail_missing")

    infrastructure_invalid = bool(
        row.get("infrastructure_invalid")
        or (shopping.get("infrastructure_invalid") if isinstance(shopping, Mapping) else False)
        or row.get("error")
        or row.get("release_error")
    )
    if infrastructure_invalid:
        reasons.append("infrastructure_invalid")
    if row.get("done") is not True:
        reasons.append("trajectory_not_done")
    reward_valid = reward.get("reward_valid")
    sampling_invalid = reward.get("sampling_invalid")
    if reward_valid is not True:
        reasons.append("reward_invalid")
    if sampling_invalid is True:
        reasons.append("sampling_invalid")
    try:
        terminal_utility = float(reward.get("terminal_utility"))
        if not math.isfinite(terminal_utility):
            raise ValueError
    except (TypeError, ValueError):
        terminal_utility = None
        reasons.append("terminal_utility_missing_or_nonfinite")
    purchase_success = bool(reward.get("purchase_success"))
    reward_type = str(reward.get("reward_type") or "")
    termination_reason = str(
        reward.get("termination_reason")
        or (terminal.get("termination_reason") if isinstance(terminal, Mapping) else "")
        or row.get("termination_reason")
        or ""
    )
    expected_purchase_success = reward_type in {
        "gold_purchase",
        "valid_alternative_purchase",
    }
    if purchase_success != expected_purchase_success:
        reasons.append("purchase_success_reward_type_mismatch")
    if termination_reason and reward_type and termination_reason != reward_type:
        reasons.append("termination_reward_type_mismatch")
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "terminal_utility": terminal_utility,
        "purchase_success": purchase_success,
        "reward_type": reward_type,
        "termination_reason": termination_reason,
        "valid": not reasons,
        "invalid_reasons": sorted(set(reasons)),
    }


def decide_grpo_admission(
    rows: Sequence[object],
    *,
    rollout_n: int = 4,
    max_rounds: int = 3,
    reward_tolerance: float = 0.025,
) -> dict[str, Any]:
    """Gate 2: accept, re-probe, or reject a task after grouped rollouts."""

    if rollout_n < 2 or max_rounds < 1:
        raise ValueError("rollout_n must be >=2 and max_rounds must be >=1")
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise ValueError("reward_tolerance must be finite and non-negative")
    normalized = [normalize_probe_trajectory(row) for row in rows]
    task_ids = {row["task_id"] for row in normalized if row.get("task_id") is not None}
    if len(task_ids) != 1:
        return {
            "decision": "reject",
            "reason": "mixed_or_missing_task_ids",
            "rounds": [],
        }
    task_id = next(iter(task_ids))
    grouped: dict[int, list[dict]] = {}
    for row in normalized:
        round_index = row["attempt_index"] // rollout_n
        grouped.setdefault(round_index, []).append(row)

    round_reports = []
    accepted_round = None
    for round_index in range(max_rounds):
        group = grouped.get(round_index, [])
        reasons = sorted(
            {
                reason
                for row in group
                for reason in row.get("invalid_reasons", [])
            }
        )
        if len(group) != rollout_n:
            reasons.append("incomplete_rollout_group")
        attempt_indices = [row["attempt_index"] for row in group]
        expected_indices = list(
            range(round_index * rollout_n, (round_index + 1) * rollout_n)
        )
        if sorted(attempt_indices) != expected_indices:
            reasons.append("attempt_indices_incomplete_or_duplicate")
        utilities = [row["terminal_utility"] for row in group if row.get("terminal_utility") is not None]
        reward_min = min(utilities) if utilities else None
        reward_max = max(utilities) if utilities else None
        reward_range = reward_max - reward_min if reward_min is not None else None
        reward_mean = sum(utilities) / len(utilities) if utilities else None
        reward_std = (
            math.sqrt(
                sum((value - reward_mean) ** 2 for value in utilities)
                / len(utilities)
            )
            if utilities
            else None
        )
        flat = reward_range is None or reward_range <= reward_tolerance
        if not reasons and flat:
            reasons.append("reward_group_near_constant")
        valid_varying = not reasons and not flat
        purchase_successes = sum(row.get("purchase_success", False) for row in group)
        if not valid_varying:
            signal_class = "invalid_or_low_variation"
        elif 0 < purchase_successes < rollout_n:
            signal_class = "mixed_outcome_frontier"
        elif purchase_successes == 0:
            signal_class = "no_gold_trajectory"
        else:
            signal_class = "all_success_reward_varying"
        round_reports.append(
            {
                "round_index": round_index,
                "trajectory_count": len(group),
                "reward_min": reward_min,
                "reward_max": reward_max,
                "reward_range": reward_range,
                "reward_std": reward_std,
                "reward_tolerance": reward_tolerance,
                "purchase_successes": purchase_successes,
                "valid_varying": valid_varying,
                "signal_class": signal_class,
                "reasons": sorted(set(reasons)),
            }
        )
        # A group with no strict-gold trajectory can only rank different failure
        # modes.  Do not admit it to GRPO: there is no positive behavior for the
        # group-relative objective to reinforce.
        if valid_varying and signal_class != "no_gold_trajectory":
            accepted_round = round_index
            break

    if accepted_round is not None:
        decision = "accept"
        reason = "valid_reward_variation"
        accepted_signal_class = round_reports[accepted_round]["signal_class"]
        probe_role = {
            "mixed_outcome_frontier": "frontier",
            "all_success_reward_varying": "regression_guard",
        }[accepted_signal_class]
    elif any(report["signal_class"] == "no_gold_trajectory" for report in round_reports):
        if len(grouped) < max_rounds:
            decision = "reprobe"
            reason = "no_gold_trajectory_yet"
            probe_role = "unresolved"
        else:
            decision = "reject"
            reason = "three_attempts_without_gold_or_valid_purchase"
            probe_role = "quarantine"
    elif len(grouped) < max_rounds:
        decision = "reprobe"
        reason = "attempt_invalid_or_low_reward_variation"
        probe_role = "unresolved"
    else:
        decision = "reject"
        reason = "three_attempts_without_valid_reward_variation"
        probe_role = "quarantine"
    return {
        "task_id": task_id,
        "decision": decision,
        "reason": reason,
        "probe_role": probe_role,
        "accepted_round": accepted_round,
        "attempts_observed": min(len(grouped), max_rounds),
        "max_attempts": max_rounds,
        "eligible_for_more_sampling": decision == "reprobe",
        "quarantine": decision == "reject",
        "reward_tolerance": reward_tolerance,
        "rounds": round_reports,
    }
