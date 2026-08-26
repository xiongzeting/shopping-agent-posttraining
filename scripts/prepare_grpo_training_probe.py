#!/usr/bin/env python3
"""Prepare the offline-gated candidate pool for the GRPO training probe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

try:
    from build_difficulty_stratified_teacher_tasks import (
        BANDS,
        DEFAULT_STRATEGY_CYCLE,
        _band,
        _select_strategy_balanced,
    )
    from build_evaluation_benchmark import (
        _feature_row,
        _semantic_training_overlap,
        _sha256_file,
        _write_atomic,
    )
    from build_grpo_dev_probe import (
        ROOT,
        DEFAULT_SOURCE,
        _default_jsonl_exclusions,
        _default_parquet_exclusions,
        _existing,
        _jsonl_bytes,
        _jsonl_task_ids,
        _parquet_task_ids,
        _parse_quotas,
        _stream_json_array,
    )
except ModuleNotFoundError:
    from scripts.build_difficulty_stratified_teacher_tasks import (
        BANDS,
        DEFAULT_STRATEGY_CYCLE,
        _band,
        _select_strategy_balanced,
    )
    from scripts.build_evaluation_benchmark import (
        _feature_row,
        _semantic_training_overlap,
        _sha256_file,
        _write_atomic,
    )
    from scripts.build_grpo_dev_probe import (
        ROOT,
        DEFAULT_SOURCE,
        _default_jsonl_exclusions,
        _default_parquet_exclusions,
        _existing,
        _jsonl_bytes,
        _jsonl_task_ids,
        _parquet_task_ids,
        _parse_quotas,
        _stream_json_array,
    )


SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.insert(0, str(SHOP_ENV))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.grpo.probe_gates import validate_probe_task_data
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.variant_price import resolve_variant_price


DEFAULT_OUTPUT = ROOT / "data/grpo/training-probe-v1"
DEFAULT_DEV_PROBE = ROOT / "data/grpo/dev-probe-v1.1/tasks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-out", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument("--dev-probe", type=Path, default=DEFAULT_DEV_PROBE)
    parser.add_argument("--quotas", default="80,360,280,80")
    parser.add_argument("--seed", default="grpo-training-probe-v1-20260810")
    parser.add_argument("--max-domain-share", type=float, default=0.25)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--exclude-parquet", type=Path, action="append", default=[])
    return parser.parse_args()


def _select_with_domain_cap(
    candidates: dict[str, list[dict]],
    quotas: dict[str, int],
    *,
    seed: str,
    excluded_families: set[str],
    max_domain_share: float,
) -> list[dict]:
    total = sum(quotas.values())
    domain_cap = max(1, math.floor(total * max_domain_share))
    domain_counts = Counter()
    selected_families = set(excluded_families)
    selected = []
    for band in BANDS:
        proposal_families = set(excluded_families)
        proposals = _select_strategy_balanced(
            candidates[band],
            limit=min(len(candidates[band]), max(quotas[band] * 4, quotas[band])),
            seed=seed,
            band=band,
            selected_families=proposal_families,
            strategy_cycle=DEFAULT_STRATEGY_CYCLE,
        )
        band_rows = []
        for row in proposals:
            domain = str(row.get("domain") or "")
            if row["family_id"] in selected_families or domain_counts[domain] >= domain_cap:
                continue
            selected_families.add(row["family_id"])
            domain_counts[domain] += 1
            band_rows.append({**row, "difficulty_band": band})
            if len(band_rows) == quotas[band]:
                break
        if len(band_rows) != quotas[band]:
            raise SystemExit(
                f"insufficient {band} candidates under domain cap: {len(band_rows)}/{quotas[band]}"
            )
        selected.extend(band_rows)
    return selected


def main() -> None:
    args = parse_args()
    quotas = _parse_quotas(args.quotas)
    if sum(quotas.values()) != 800 or any(quotas[band] % 4 for band in BANDS):
        raise SystemExit("training probe quotas must total 800 and each band must be divisible by 4")
    if not 0 < args.max_domain_share <= 1:
        raise SystemExit("--max-domain-share must be in (0, 1]")
    required_paths = (args.source, args.held_out, args.dev_probe)
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"required file does not exist: {path}")

    jsonl_exclusions = _existing(
        [*_default_jsonl_exclusions(), args.dev_probe, *args.exclude_jsonl]
    )
    parquet_exclusions = _existing(
        [*_default_parquet_exclusions(), *args.exclude_parquet]
    )
    exclusion_details = []
    excluded_ids = set()
    for path in jsonl_exclusions:
        ids = _jsonl_task_ids(path)
        excluded_ids.update(ids)
        exclusion_details.append(
            {"path": str(path.relative_to(ROOT)), "kind": "jsonl", "task_ids": len(ids), "sha256": _sha256_file(path)}
        )
    for path in parquet_exclusions:
        ids = _parquet_task_ids(path)
        excluded_ids.update(ids)
        exclusion_details.append(
            {"path": str(path.relative_to(ROOT)), "kind": "parquet", "task_ids": len(ids), "sha256": _sha256_file(path)}
        )
    held_out_ids = _jsonl_task_ids(args.held_out)
    excluded_ids.update(held_out_ids)

    category_frequency = Counter()
    source_rows = train_rows = 0
    for product in _stream_json_array(args.source):
        source_rows += 1
        if product.get("tag") == "train":
            train_rows += 1
            leaf = str(product.get("category") or "").split("›")[-1]
            category_frequency[leaf] += 1

    features = []
    excluded_families = set()
    held_out_by_category = defaultdict(list)
    for task_id, product in enumerate(_stream_json_array(args.source)):
        if product.get("tag") != "train":
            continue
        feature = _feature_row(task_id, product, category_frequency)
        features.append(feature)
        if task_id in excluded_ids:
            excluded_families.add(feature["family_id"])
        if task_id in held_out_ids:
            held_out_by_category[feature["category_leaf"]].append(feature)

    candidate_features = {}
    semantic_rejections = 0
    for feature in features:
        if feature["task_id"] in excluded_ids or feature["family_id"] in excluded_families:
            continue
        if _semantic_training_overlap(feature, held_out_by_category):
            semantic_rejections += 1
            continue
        candidate_features[feature["task_id"]] = feature

    gated_candidates = defaultdict(list)
    data_gate_rejected = []
    gate_by_id = {}
    for task_id, product in enumerate(_stream_json_array(args.source)):
        feature = candidate_features.get(task_id)
        if feature is None:
            continue
        gate = validate_probe_task_data(
            task_id,
            product,
            compile_reward_features=compile_reward_features,
            resolve_variant_price=resolve_variant_price,
        )
        if gate["accepted"]:
            gate_by_id[task_id] = gate
            gated_candidates[_band(feature["difficulty_score"])].append(feature)
        else:
            data_gate_rejected.append(gate)

    selected = _select_with_domain_cap(
        gated_candidates,
        quotas,
        seed=args.seed,
        excluded_families=excluded_families,
        max_domain_share=args.max_domain_share,
    )
    selected_by_id = {row["task_id"]: row for row in selected}
    full_products = {}
    for task_id, product in enumerate(_stream_json_array(args.source)):
        if task_id in selected_by_id:
            full_products[task_id] = product

    tasks = []
    full_rows = []
    for row in selected:
        gate = gate_by_id[row["task_id"]]
        metadata = {
            "task_id": row["task_id"],
            "family_id": row["family_id"],
            "domain": row["domain"],
            "category": row["category"],
            "difficulty_band": row["difficulty_band"],
            "difficulty_score": row["difficulty_score"],
            "selection_strategy": row["teacher_strategy"],
            "data_gate": gate,
        }
        tasks.append(metadata)
        full_rows.append({"probe_metadata": metadata, "task": full_products[row["task_id"]]})

    calibration_quotas = {band: quotas[band] // 4 for band in BANDS}
    calibration_counts = Counter()
    calibration_tasks = []
    remaining_tasks = []
    for task in tasks:
        band = task["difficulty_band"]
        if calibration_counts[band] < calibration_quotas[band]:
            calibration_tasks.append(task)
            calibration_counts[band] += 1
        else:
            remaining_tasks.append(task)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = {
        "candidates.jsonl": tasks,
        "candidates-full.jsonl": full_rows,
        "calibration-200.jsonl": calibration_tasks,
        "remaining-600.jsonl": remaining_tasks,
        "data-gate-rejected.jsonl": data_gate_rejected,
    }
    outputs = {}
    for name, rows in output_rows.items():
        path = args.output_dir / name
        _write_atomic(path, _jsonl_bytes(rows))
        outputs[name] = {"rows": len(rows), "sha256": _sha256_file(path)}

    selected_ids = set(selected_by_id)
    selected_families = {row["family_id"] for row in selected}
    manifest = {
        "schema": "shopping-grpo-training-probe-preparation-v1",
        "gate_contract": {
            "gate_1": "offline_task_data_validity",
            "gate_2": "post_rollout_grpo_admission",
            "rollout_n": 4,
            "max_rounds": 3,
            "reward_tolerance": 0.025,
            "retry_semantics": "initial attempt plus two retries; reject after failed third attempt",
            "low_variation_rule": "terminal_utility_range <= reward_tolerance",
            "post_calibration_target_mix": {
                "regression_guard": 0.20,
                "frontier": 0.60,
                "hard_exploration": 0.20,
            },
            "target_mix_policy": "soft targets applied after calibration; never override either hard gate",
            "first_stage": "calibration-200.jsonl",
            "second_stage": "remaining-600.jsonl after threshold review",
        },
        "source": {"path": str(args.source.relative_to(ROOT)), "sha256": _sha256_file(args.source), "rows": source_rows, "train_rows": train_rows},
        "seed": args.seed,
        "quotas": quotas,
        "max_domain_share": args.max_domain_share,
        "selected_count": len(selected),
        "calibration_count": len(calibration_tasks),
        "calibration_difficulty_distribution": dict(calibration_counts),
        "difficulty_distribution": dict(Counter(row["difficulty_band"] for row in selected)),
        "domain_distribution": dict(Counter(str(row["domain"]) for row in selected)),
        "strategy_distribution": dict(Counter(row["teacher_strategy"] for row in selected)),
        "data_gate": {"eligible_before_selection": len(gate_by_id), "rejected": len(data_gate_rejected), "rejection_reasons": dict(Counter(reason for row in data_gate_rejected for reason in row["reasons"]))},
        "exclusions": exclusion_details,
        "excluded_unique_task_ids": len(excluded_ids),
        "excluded_unique_families": len(excluded_families),
        "semantic_final_rejections": semantic_rejections,
        "validation": {
            "task_overlap_with_exclusions": len(selected_ids & excluded_ids),
            "family_overlap_with_exclusions": len(selected_families & excluded_families),
            "internal_duplicate_task_ids": len(selected) - len(selected_ids),
            "internal_duplicate_families": len(selected) - len(selected_families),
            "all_selected_passed_data_gate": all(gate_by_id[row["task_id"]]["accepted"] for row in selected),
        },
        "outputs": outputs,
    }
    _write_atomic(
        args.output_dir / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
