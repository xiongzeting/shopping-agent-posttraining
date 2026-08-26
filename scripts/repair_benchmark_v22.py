#!/usr/bin/env python3
"""Surgically replace hard-invalid Final-240 and dev-probe tasks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

try:
    from build_evaluation_benchmark import (
        _challenge_eligible,
        _difficulty_bucket,
        _extreme_factor_count,
        _feature_row,
        _semantic_training_overlap,
        _query_reward_contract_eligible,
        _sha256_file,
        _stable_key,
        _with_retrieval,
        _write_atomic,
        _searcher,
    )
    from build_difficulty_stratified_teacher_tasks import _band, _eligible_strategy
    from build_grpo_dev_probe import (
        ROOT,
        DEFAULT_SOURCE,
        _jsonl_bytes,
        _jsonl_task_ids,
        _parquet_task_ids,
        _stream_json_array,
    )
except ModuleNotFoundError:
    from scripts.build_evaluation_benchmark import (
        _challenge_eligible,
        _difficulty_bucket,
        _extreme_factor_count,
        _feature_row,
        _semantic_training_overlap,
        _query_reward_contract_eligible,
        _sha256_file,
        _stable_key,
        _with_retrieval,
        _write_atomic,
        _searcher,
    )
    from scripts.build_difficulty_stratified_teacher_tasks import _band, _eligible_strategy
    from scripts.build_grpo_dev_probe import (
        ROOT,
        DEFAULT_SOURCE,
        _jsonl_bytes,
        _jsonl_task_ids,
        _parquet_task_ids,
        _stream_json_array,
    )


SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
for path in (ROOT / "src", SHOP_ENV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shopping_grpo.training.grpo.probe_gates import validate_probe_task_data
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.variant_price import resolve_variant_price


FINAL_HARD_IDS = {
    26,
    51,
    123,
    196,
    251,
    362,
    440,
    536,
    704,
    706,
    779,
    791,
    802,
    837,
    845,
    849,
    1103,
    1159,
    1177,
    1254,
    1336,
    1366,
    1438,
}
DEV_HARD_IDS = {
    10254,
    10515,
    12759,
    17717,
    17818,
    19082,
    19232,
    19694,
    19929,
    20826,
    21656,
    22659,
    22704,
    23339,
}
SEARCH_INDEX = SHOP_ENV / "search_engine/products.sqlite3"
EVALUATION_DIR = ROOT / "data/evaluation"
DEV_V1_DIR = ROOT / "data/grpo/dev-probe-v1"
DEV_V11_DIR = ROOT / "data/grpo/dev-probe-v1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--search-index", type=Path, default=SEARCH_INDEX)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument(
        "--baseline-evaluation-dir",
        type=Path,
        help="Optional pristine v2.1 input directory; outputs still go to --evaluation-dir.",
    )
    parser.add_argument("--dev-v1-dir", type=Path, default=DEV_V1_DIR)
    parser.add_argument("--dev-v11-dir", type=Path, default=DEV_V11_DIR)
    parser.add_argument("--seed", default="benchmark-v2.2-hard-data-repair-20260810")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _task_ids_from_existing_training() -> set[int]:
    ids = set()
    for path in sorted((ROOT / "data").glob("sft*/all.jsonl")):
        ids.update(_jsonl_task_ids(path))
    for path in (
        ROOT / "data/grpo/candidates-smoke-v1/tasks.jsonl",
        ROOT / "data/grpo/training-probe-v1/candidates.jsonl",
    ):
        if path.is_file():
            ids.update(_jsonl_task_ids(path))
    for path in (ROOT / "data/grpo/train.parquet", ROOT / "data/grpo/validation.parquet"):
        if path.is_file():
            ids.update(_parquet_task_ids(path))
    return ids


def _task_ids_for_semantic_eval_guard() -> set[int]:
    """Match the canonical benchmark builder's frozen SFT semantic guard."""
    path = ROOT / "data/sft/all.jsonl"
    return _jsonl_task_ids(path) if path.is_file() else set()


def _feature_tags(row: dict) -> list[str]:
    return sorted(
        name
        for name in (
            "hard_budget" if row["hard_budget"] else None,
            "approximate_price" if row["approximate_price"] else None,
            "negation" if row["negation"] else None,
            "compatibility" if row["compatibility"] else None,
            "multi_option" if row["option_axes"] >= 2 else None,
        )
        if name
    )


def _slice_row(feature: dict, old: dict) -> dict:
    return {
        "task_id": feature["task_id"],
        "suite": old["suite"],
        "domain": feature["domain"],
        "challenge_slice": old["challenge_slice"],
        "difficulty_score": feature["difficulty_score"],
        "difficulty_bucket": feature["difficulty_bucket"],
        "retrieval_rank": feature["retrieval_rank"],
        "retrieval_bucket": feature["retrieval_bucket"],
        "family_id": feature["family_id"],
        "feature_tags": _feature_tags(feature),
    }


def _choose_final_replacements(
    old_hard_rows: list[dict],
    candidates: list[dict],
    *,
    seed: str,
    blocked_families: set[str],
) -> dict[int, dict]:
    replacements = {}
    used_ids = set()
    used_families = set(blocked_families)
    for old in old_hard_rows:
        eligible = []
        for row in candidates:
            if row["task_id"] in used_ids or row["family_id"] in used_families:
                continue
            if row["domain"] != old["domain"]:
                continue
            if row["difficulty_bucket"] != old["difficulty_bucket"]:
                continue
            if row["retrieval_bucket"] != old["retrieval_bucket"]:
                continue
            challenge_slice = old["challenge_slice"]
            if challenge_slice and not _challenge_eligible(challenge_slice, row):
                continue
            eligible.append(row)
        if not eligible:
            diagnostics = {
                "total": len(candidates),
                "domain": sum(row["domain"] == old["domain"] for row in candidates),
                "domain+difficulty": sum(
                    row["domain"] == old["domain"]
                    and row["difficulty_bucket"] == old["difficulty_bucket"]
                    for row in candidates
                ),
                "domain+difficulty+retrieval": sum(
                    row["domain"] == old["domain"]
                    and row["difficulty_bucket"] == old["difficulty_bucket"]
                    and row["retrieval_bucket"] == old["retrieval_bucket"]
                    for row in candidates
                ),
                "challenge": sum(
                    not old["challenge_slice"]
                    or _challenge_eligible(old["challenge_slice"], row)
                    for row in candidates
                ),
            }
            raise SystemExit(
                f"no exact benchmark replacement for task {old['task_id']}: "
                f"{json.dumps(diagnostics, sort_keys=True)}"
            )
        old_tags = set(old.get("feature_tags") or [])
        chosen = min(
            eligible,
            key=lambda row: (
                len(old_tags ^ set(_feature_tags(row))),
                abs(float(row["difficulty_score"]) - float(old["difficulty_score"])),
                abs((row["retrieval_rank"] or 999) - (old["retrieval_rank"] or 999)),
                _stable_key(seed, "final", old["task_id"], row["task_id"]),
            ),
        )
        replacements[old["task_id"]] = chosen
        used_ids.add(chosen["task_id"])
        used_families.add(chosen["family_id"])
    return replacements


def _choose_dev_replacements(
    old_hard_rows: list[dict],
    candidates: list[dict],
    *,
    seed: str,
    blocked_families: set[str],
) -> dict[int, dict]:
    replacements = {}
    used_families = set(blocked_families)
    for old in old_hard_rows:
        eligible = [
            row
            for row in candidates
            if row["family_id"] not in used_families
            and row["domain"] == old["domain"]
            and _band(row["difficulty_score"]) == old["difficulty_band"]
            and _eligible_strategy(old["selection_strategy"], row)
        ]
        if not eligible:
            raise SystemExit(f"no exact dev-probe replacement for task {old['task_id']}")
        chosen = min(
            eligible,
            key=lambda row: (
                row["category"] != old["category"],
                abs(float(row["difficulty_score"]) - float(old["difficulty_score"])),
                _stable_key(seed, "dev", old["task_id"], row["task_id"]),
            ),
        )
        replacements[old["task_id"]] = chosen
        used_families.add(chosen["family_id"])
    return replacements


def _write_json(path: Path, value: dict) -> None:
    _write_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _recalculate_evaluation_metadata(
    old_metadata: dict,
    slices: list[dict],
    tasks_payload: bytes,
    slices_payload: bytes,
    replacements: list[dict],
    *,
    source: Path,
    search_index: Path,
    seed: str,
) -> dict:
    difficulty_counts = Counter(row["difficulty_bucket"] for row in slices)
    retrieval_counts = Counter(row["retrieval_bucket"] for row in slices)
    metadata = dict(old_metadata)
    metadata.update(
        {
            "schema_version": "shopping-evaluation-dataset-v2.2",
            "asset": "shopbench_longhorizon_final_240_v2_2",
            "contract": "environment-v2.4/reward-v4/benchmark-v2.2",
            "evaluated": False,
            "selection_seed": seed,
            "task_sha256": hashlib.sha256(tasks_payload).hexdigest(),
            "slice_sha256": hashlib.sha256(slices_payload).hexdigest(),
            "source_sha256": _sha256_file(source),
            "search_index_sha256": _sha256_file(search_index),
            "previous_benchmark_overlap": len(slices) - len(replacements),
            "retained_from_v2_1": len(slices) - len(replacements),
            "replacement_count": len(replacements),
            "replacement_task_ids": [
                {
                    "removed_task_id": row["removed_task_id"],
                    "added_task_id": row["added_task_id"],
                }
                for row in replacements
            ],
            "selected_family_duplicates": len(slices) - len({row["family_id"] for row in slices}),
            "domain_counts": dict(sorted(Counter(row["domain"] for row in slices).items())),
            "core_domain_counts": dict(sorted(Counter(row["domain"] for row in slices if row["suite"] == "core").items())),
            "suite_counts": dict(sorted(Counter(row["suite"] for row in slices).items())),
            "challenge_slice_counts": dict(sorted(Counter(row["challenge_slice"] for row in slices if row["challenge_slice"]).items())),
            "difficulty_bucket_counts": dict(difficulty_counts),
            "retrieval_bucket_counts": dict(retrieval_counts),
            "difficulty_mean": round(sum(row["difficulty_score"] for row in slices) / len(slices), 6),
            "difficulty_max": max(row["difficulty_score"] for row in slices),
        }
    )
    metadata.pop("extreme_factor_counts", None)
    return metadata


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_evaluation_dir or args.evaluation_dir
    tasks_path = args.evaluation_dir / "tasks.jsonl"
    slices_path = args.evaluation_dir / "slices.jsonl"
    metadata_path = args.evaluation_dir / "metadata.json"
    baseline_tasks_path = baseline_dir / "tasks.jsonl"
    baseline_slices_path = baseline_dir / "slices.jsonl"
    baseline_metadata_path = baseline_dir / "metadata.json"
    old_tasks = _read_jsonl(baseline_tasks_path)
    old_slices = _read_jsonl(baseline_slices_path)
    old_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    old_dev = _read_jsonl(args.dev_v1_dir / "tasks.jsonl")
    old_dev_full = _read_jsonl(args.dev_v1_dir / "tasks-full.jsonl")
    old_dev_full_by_id = {int(row["probe_metadata"]["task_id"]): row for row in old_dev_full}
    old_task_ids = {int(row["task_id"]) for row in old_tasks}
    old_dev_ids = {int(row["task_id"]) for row in old_dev}
    if not FINAL_HARD_IDS <= old_task_ids or not DEV_HARD_IDS <= old_dev_ids:
        raise SystemExit("expected hard-invalid tasks are missing from current assets")

    training_ids = _task_ids_from_existing_training()
    semantic_training_ids = _task_ids_for_semantic_eval_guard()
    eval_frequency = Counter()
    train_frequency = Counter()
    for product in _stream_json_array(args.source):
        leaf = str(product.get("category") or "").split("›")[-1]
        if product.get("tag") == "eval":
            eval_frequency[leaf] += 1
        elif product.get("tag") == "train":
            train_frequency[leaf] += 1

    retained_final_slices = [row for row in old_slices if int(row["task_id"]) not in FINAL_HARD_IDS]
    retained_dev = [row for row in old_dev if int(row["task_id"]) not in DEV_HARD_IDS]
    blocked_family_ids = {row["family_id"] for row in retained_final_slices + retained_dev}
    training_rows_by_category = defaultdict(list)
    retained_final_by_category = defaultdict(list)
    retained_dev_by_category = defaultdict(list)
    retained_final_ids = {int(row["task_id"]) for row in retained_final_slices}
    retained_dev_ids = {int(row["task_id"]) for row in retained_dev}
    for task_id, product in enumerate(_stream_json_array(args.source)):
        tag = product.get("tag")
        if tag not in {"eval", "train"}:
            continue
        frequency = eval_frequency if tag == "eval" else train_frequency
        feature = _feature_row(task_id, product, frequency)
        if task_id in semantic_training_ids:
            training_rows_by_category[feature["category_leaf"]].append(feature)
        if task_id in retained_final_ids:
            retained_final_by_category[feature["category_leaf"]].append(feature)
        if task_id in retained_dev_ids:
            retained_dev_by_category[feature["category_leaf"]].append(feature)

    eval_candidates = []
    train_candidates = []
    searcher = _searcher(args.search_index)
    for task_id, product in enumerate(_stream_json_array(args.source)):
        tag = product.get("tag")
        frequency = eval_frequency if tag == "eval" else train_frequency
        if tag not in {"eval", "train"}:
            continue
        feature = _feature_row(task_id, product, frequency)
        if task_id in old_task_ids or task_id in old_dev_ids or task_id in training_ids:
            continue
        gate = validate_probe_task_data(
            task_id,
            product,
            compile_reward_features=compile_reward_features,
            resolve_variant_price=resolve_variant_price,
            allowed_source_tags=(tag,),
        )
        if not gate["accepted"]:
            continue
        if not _query_reward_contract_eligible(product):
            continue
        if tag == "eval":
            if _semantic_training_overlap(feature, training_rows_by_category):
                continue
            eval_candidates.append(_with_retrieval(feature, product, searcher))
        else:
            if _semantic_training_overlap(feature, training_rows_by_category):
                continue
            train_candidates.append(feature)
    searcher.close()

    old_hard_slices = [row for row in old_slices if int(row["task_id"]) in FINAL_HARD_IDS]
    final_replacements = _choose_final_replacements(
        old_hard_slices,
        eval_candidates,
        seed=args.seed,
        blocked_families=blocked_family_ids,
    )
    final_added_ids = {row["task_id"] for row in final_replacements.values()}
    final_added_families = {row["family_id"] for row in final_replacements.values()}
    blocked_family_ids.update(final_added_families)

    repaired_final_by_category = defaultdict(list)
    for category, rows in retained_final_by_category.items():
        repaired_final_by_category[category].extend(rows)
    for row in final_replacements.values():
        repaired_final_by_category[row["category_leaf"]].append(row)

    hard_dev_rows = [row for row in old_dev if int(row["task_id"]) in DEV_HARD_IDS]
    train_candidates = [
        row
        for row in train_candidates
        if row["task_id"] not in final_added_ids
        and row["family_id"] not in final_added_families
    ]
    dev_replacements = _choose_dev_replacements(
        hard_dev_rows,
        train_candidates,
        seed=args.seed,
        blocked_families=blocked_family_ids,
    )

    new_tasks = []
    new_slices = []
    final_changes = []
    old_slice_by_id = {int(row["task_id"]): row for row in old_slices}
    for task in old_tasks:
        old_id = int(task["task_id"])
        old_slice = old_slice_by_id[old_id]
        replacement = final_replacements.get(old_id)
        if replacement is None:
            new_tasks.append({"task_id": old_id})
            new_slices.append(old_slice)
        else:
            new_tasks.append({"task_id": replacement["task_id"]})
            replacement_slice = _slice_row(replacement, old_slice)
            new_slices.append(replacement_slice)
            final_changes.append({"removed_task_id": old_id, "added_task_id": replacement["task_id"], "old_slice": old_slice, "new_slice": replacement_slice})

    tasks_payload = _jsonl_bytes(new_tasks)
    slices_payload = _jsonl_bytes(new_slices)
    metadata = _recalculate_evaluation_metadata(
        old_metadata,
        new_slices,
        tasks_payload,
        slices_payload,
        final_changes,
        source=args.source,
        search_index=args.search_index,
        seed=args.seed,
    )
    _write_atomic(tasks_path, tasks_payload)
    _write_atomic(slices_path, slices_payload)
    _write_json(metadata_path, metadata)
    _write_json(args.evaluation_dir / "replacement-manifest-v2.2.json", {
        "schema": "shopping-benchmark-replacement-v2.2",
        "retained_count": len(new_tasks) - len(final_changes),
        "removed_task_ids": sorted(FINAL_HARD_IDS),
        "added_task_ids": [row["added_task_id"] for row in final_changes],
        "replacements": final_changes,
        "validation": {
            "task_count": len(new_tasks),
            "unique_task_ids": len({row["task_id"] for row in new_tasks}),
            "family_duplicates": len(new_slices) - len({row["family_id"] for row in new_slices}),
            "hard_data_gate_failures": 0,
            "training_task_overlap": len({row["task_id"] for row in new_tasks} & training_ids),
        },
    })
    retest_dir = args.evaluation_dir / "retest-v2.2"
    retest_tasks = [{"task_id": row["added_task_id"]} for row in final_changes]
    _write_atomic(retest_dir / "tasks.jsonl", _jsonl_bytes(retest_tasks))
    _write_atomic(retest_dir / "slices.jsonl", _jsonl_bytes([row["new_slice"] for row in final_changes]))

    dev_products = {}
    added_dev_ids = {row["task_id"] for row in dev_replacements.values()}
    for task_id, product in enumerate(_stream_json_array(args.source)):
        if task_id in added_dev_ids:
            dev_products[task_id] = product
    new_dev = []
    new_dev_full = []
    dev_changes = []
    for row in old_dev:
        old_id = int(row["task_id"])
        replacement = dev_replacements.get(old_id)
        if replacement is None:
            new_dev.append(row)
            new_dev_full.append(old_dev_full_by_id[old_id])
            continue
        metadata_row = {
            **row,
            "task_id": replacement["task_id"],
            "family_id": replacement["family_id"],
            "domain": replacement["domain"],
            "category": replacement["category"],
            "difficulty_score": replacement["difficulty_score"],
            "selection_reason": "dev-probe-v1.1 matched replacement for hard-invalid option mapping",
        }
        new_dev.append(metadata_row)
        new_dev_full.append({"probe_metadata": metadata_row, "task": dev_products[replacement["task_id"]]})
        dev_changes.append({"removed_task_id": old_id, "added_task_id": replacement["task_id"], "old": row, "new": metadata_row})

    args.dev_v11_dir.mkdir(parents=True, exist_ok=True)
    dev_tasks_path = args.dev_v11_dir / "tasks.jsonl"
    dev_full_path = args.dev_v11_dir / "tasks-full.jsonl"
    _write_atomic(dev_tasks_path, _jsonl_bytes(new_dev))
    _write_atomic(dev_full_path, _jsonl_bytes(new_dev_full))
    _write_atomic(args.dev_v11_dir / "replacements.jsonl", _jsonl_bytes(dev_changes))
    dev_ids = {row["task_id"] for row in new_dev}
    dev_families = {row["family_id"] for row in new_dev}
    dev_manifest = {
        "schema": "shopping-grpo-dev-probe-manifest-v1.1",
        "selection_kind": "offline_dev_probe_candidates_not_online_frontier",
        "source": {"path": str(args.source.relative_to(ROOT)), "sha256": _sha256_file(args.source)},
        "seed": args.seed,
        "selected_count": len(new_dev),
        "retained_from_v1": len(new_dev) - len(dev_changes),
        "replacement_count": len(dev_changes),
        "replacements": dev_changes,
        "difficulty_distribution": dict(Counter(row["difficulty_band"] for row in new_dev)),
        "domain_distribution": dict(Counter(str(row["domain"]) for row in new_dev)),
        "strategy_distribution": dict(Counter(row["selection_strategy"] for row in new_dev)),
        "validation": {
            "task_count": len(new_dev),
            "unique_task_ids": len(dev_ids),
            "internal_duplicate_families": len(new_dev) - len(dev_families),
            "final_v2_2_task_overlap": len(dev_ids & {row["task_id"] for row in new_tasks}),
            "training_task_overlap": len(dev_ids & training_ids),
            "hard_data_gate_failures": 0,
            "full_record_count_matches": len(new_dev_full) == len(new_dev),
        },
        "outputs": {
            "tasks.jsonl": {"rows": len(new_dev), "sha256": _sha256_file(dev_tasks_path)},
            "tasks-full.jsonl": {"rows": len(new_dev_full), "sha256": _sha256_file(dev_full_path)},
        },
    }
    _write_json(args.dev_v11_dir / "manifest.json", dev_manifest)
    print(json.dumps({"benchmark_v2_2": metadata, "dev_probe_v1_1": dev_manifest}, ensure_ascii=True))


if __name__ == "__main__":
    main()
