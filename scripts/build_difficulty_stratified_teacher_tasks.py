#!/usr/bin/env python3
"""Build leak-free Teacher task pools proportional to source difficulty bands."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

try:
    from build_evaluation_benchmark import (
        _challenge_eligible,
        _challenge_rank,
        _family_id,
        _feature_row,
        _semantic_training_overlap,
        _write_atomic,
    )
except ModuleNotFoundError:  # Imported as scripts.* by unit tests.
    from scripts.build_evaluation_benchmark import (
        _challenge_eligible,
        _challenge_rank,
        _family_id,
        _feature_row,
        _semantic_training_overlap,
        _write_atomic,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "data"
    / "items_eval_train.json"
)
BANDS = ("lt10", "10to15", "15to18", "ge18")
DEFAULT_STRATEGY_CYCLE = (
    "search_reformulation",
    "candidate_comparison",
    "evidence_verification",
    "multi_option",
    "focused_verification",
    "price_semantics",
    "search_reformulation",
    "candidate_comparison",
    "evidence_verification",
    "multi_option",
    "focused_verification",
    "search_reformulation",
    "candidate_comparison",
    "evidence_verification",
    "price_semantics",
    "multi_option",
    "focused_verification",
    "search_reformulation",
    "candidate_comparison",
    "evidence_verification",
)
REPAIR_STRATEGY_CYCLE = (
    "loop_recovery",
    "near_miss_rejection",
    "loop_recovery",
    "near_miss_rejection",
    "terminal_tool_commit",
    "loop_recovery",
    "near_miss_rejection",
    "option_grounding",
    "loop_recovery",
    "near_miss_rejection",
    "loop_recovery",
    "near_miss_rejection",
    "terminal_tool_commit",
    "loop_recovery",
    "near_miss_rejection",
    "option_grounding",
    "loop_recovery",
    "near_miss_rejection",
    "terminal_tool_commit",
    "loop_recovery",
)
STRATEGY_PROXIES = {
    "loop_recovery": "search_reformulation",
    "near_miss_rejection": "candidate_comparison",
    "terminal_tool_commit": "focused_verification",
    "option_grounding": "multi_option",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-sft", type=Path, default=ROOT / "data/sft/all.jsonl")
    parser.add_argument(
        "--held-out", type=Path, default=ROOT / "data/evaluation/tasks.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument(
        "--strategy-profile",
        choices=("default", "repair"),
        default="default",
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Additional task or trajectory JSONL files whose task IDs are excluded.",
    )
    parser.add_argument("--seed", default="fresh-sft-difficulty-proportional-v1")
    return parser.parse_args()


def _task_ids(path: Path) -> set[int]:
    return {
        int(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _band(score: float) -> str:
    if score < 10:
        return "lt10"
    if score < 15:
        return "10to15"
    if score < 18:
        return "15to18"
    return "ge18"


def _stable(seed: str, band: str, task_id: int) -> str:
    return hashlib.sha256(f"{seed}:{band}:{task_id}".encode()).hexdigest()


def _proportional_quotas(counts: Counter, total: int) -> dict[str, int]:
    denominator = sum(counts.values())
    exact = {band: total * counts[band] / denominator for band in BANDS}
    quotas = {band: math.floor(exact[band]) for band in BANDS}
    remaining = total - sum(quotas.values())
    order = sorted(BANDS, key=lambda band: (-(exact[band] % 1), BANDS.index(band)))
    for band in order[:remaining]:
        quotas[band] += 1
    return quotas


def _interleave_proportional(rows_by_band: dict[str, list[dict]]) -> list[dict]:
    positioned = []
    for band_index, band in enumerate(BANDS):
        rows = rows_by_band.get(band, [])
        denominator = max(len(rows), 1)
        for row_index, row in enumerate(rows):
            positioned.append((row_index / denominator, band_index, row_index, row))
    return [item[-1] for item in sorted(positioned)]


def _eligible_strategy(strategy: str, feature: dict) -> bool:
    strategy = STRATEGY_PROXIES.get(strategy, strategy)
    if strategy == "focused_verification":
        return True
    return _challenge_eligible(strategy, feature)


def _strategy_rank(strategy: str, feature: dict, seed: str, band: str) -> tuple:
    strategy = STRATEGY_PROXIES.get(strategy, strategy)
    if strategy == "focused_verification":
        return (_stable(seed + ":focused", band, feature["task_id"]),)
    return (
        *_challenge_rank(strategy, feature),
        _stable(seed + ":strategy", band, feature["task_id"]),
    )


def _select_strategy_balanced(
    candidates: list[dict],
    *,
    limit: int,
    seed: str,
    band: str,
    selected_families: set[str],
    strategy_cycle: tuple[str, ...] = DEFAULT_STRATEGY_CYCLE,
) -> list[dict]:
    queues = {
        strategy: iter(
            sorted(
                (row for row in candidates if _eligible_strategy(strategy, row)),
                key=lambda row: _strategy_rank(strategy, row, seed, band),
            )
        )
        for strategy in set(strategy_cycle)
    }
    selected = []
    exhausted = set()
    cycle_index = 0
    while len(selected) < limit and len(exhausted) < len(queues):
        strategy = strategy_cycle[cycle_index % len(strategy_cycle)]
        cycle_index += 1
        if strategy in exhausted:
            continue
        queue = queues[strategy]
        while True:
            try:
                feature = next(queue)
            except StopIteration:
                exhausted.add(strategy)
                break
            if feature["family_id"] in selected_families:
                continue
            selected_families.add(feature["family_id"])
            selected.append({**feature, "teacher_strategy": strategy})
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.total < 1:
        raise SystemExit("--total must be at least 1")
    if args.candidate_multiplier < 1:
        raise SystemExit("--candidate-multiplier must be at least 1")

    products = json.loads(args.source.read_text(encoding="utf-8"))
    source_counts = Counter()
    category_frequency = Counter(
        str(product.get("category") or "").split("›")[-1]
        for product in products
        if product.get("tag") == "train"
    )
    source_features = {}
    for task_id, product in enumerate(products):
        if product.get("tag") != "train":
            continue
        feature = _feature_row(task_id, product, category_frequency)
        source_features[task_id] = feature
        source_counts[_band(feature["difficulty_score"])] += 1
    quotas = _proportional_quotas(source_counts, args.total)

    base_ids = _task_ids(args.base_sft)
    held_out_ids = _task_ids(args.held_out)
    extra_excluded_ids = set().union(*(_task_ids(path) for path in args.exclude_jsonl))
    excluded_ids = base_ids | held_out_ids | extra_excluded_ids
    excluded_families = {
        _family_id(products[task_id])
        for task_id in excluded_ids
        if 0 <= task_id < len(products)
    }
    held_out_by_category: dict[str, list[dict]] = defaultdict(list)
    for task_id in held_out_ids:
        if task_id in source_features:
            feature = source_features[task_id]
        elif 0 <= task_id < len(products):
            feature = _feature_row(task_id, products[task_id], category_frequency)
        else:
            continue
        held_out_by_category[feature["category_leaf"]].append(feature)

    candidates: dict[str, list[dict]] = defaultdict(list)
    for task_id, feature in source_features.items():
        if task_id in excluded_ids or feature["family_id"] in excluded_families:
            continue
        if _semantic_training_overlap(feature, held_out_by_category):
            continue
        candidates[_band(feature["difficulty_score"])].append(feature)

    selected_families = set(excluded_families)
    strategy_cycle = (
        REPAIR_STRATEGY_CYCLE
        if args.strategy_profile == "repair"
        else DEFAULT_STRATEGY_CYCLE
    )
    output_counts = {}
    selected_by_band = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        limit = quotas[band] * args.candidate_multiplier
        selected_features = _select_strategy_balanced(
            candidates[band],
            limit=limit,
            seed=args.seed,
            band=band,
            selected_families=selected_families,
            strategy_cycle=strategy_cycle,
        )
        selected = [
            {
                "task_id": feature["task_id"],
                "difficulty_band": band,
                "difficulty_score": feature["difficulty_score"],
                "teacher_strategy": feature["teacher_strategy"],
            }
            for feature in selected_features
        ]
        if len(selected) < quotas[band]:
            raise SystemExit(
                f"insufficient {band} candidates: {len(selected)}/{quotas[band]}"
            )
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ).encode("utf-8")
        _write_atomic(args.output_dir / f"{band}.jsonl", payload)
        output_counts[band] = len(selected)
        selected_by_band[band] = selected

    interleaved = _interleave_proportional(selected_by_band)
    combined_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in interleaved
    ).encode("utf-8")
    _write_atomic(args.output_dir / "tasks.jsonl", combined_payload)

    print(
        json.dumps(
            {
                "source_counts": dict(source_counts),
                "target_accepted": quotas,
                "candidate_counts": output_counts,
                "excluded_task_ids": len(excluded_ids),
                "excluded_families": len(excluded_families),
                "strategy_profile": args.strategy_profile,
                "combined_candidates": len(interleaved),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
