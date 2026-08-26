#!/usr/bin/env python3
"""Replace query-inconsistent benchmark tasks while preserving slice buckets."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    from build_evaluation_benchmark import (
        _feature_row,
        _query_reward_contract_eligible,
        _semantic_training_overlap,
        _sha256_file,
        _stable_key,
        _with_retrieval,
        _write_atomic,
        _searcher,
    )
    from build_grpo_dev_probe import DEFAULT_SOURCE, ROOT, _jsonl_bytes, _jsonl_task_ids
except ModuleNotFoundError:
    from scripts.build_evaluation_benchmark import (
        _feature_row,
        _query_reward_contract_eligible,
        _semantic_training_overlap,
        _sha256_file,
        _stable_key,
        _with_retrieval,
        _write_atomic,
        _searcher,
    )
    from scripts.build_grpo_dev_probe import DEFAULT_SOURCE, ROOT, _jsonl_bytes, _jsonl_task_ids


INVALID_TASK_IDS = {405}
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
SEARCH_INDEX = SHOP_ENV / "search_engine/products.sqlite3"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict) -> None:
    _write_atomic(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _training_ids() -> set[int]:
    ids = set()
    for path in sorted((ROOT / "data").glob("sft*/all.jsonl")):
        ids.update(_jsonl_task_ids(path))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--search-index", type=Path, default=SEARCH_INDEX)
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "data/evaluation")
    parser.add_argument("--seed", default="benchmark-v2.2-query-contract-repair-20260812")
    args = parser.parse_args()

    for path in (ROOT / "src", SHOP_ENV):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    tasks = _read_jsonl(args.evaluation_dir / "tasks.jsonl")
    slices = _read_jsonl(args.evaluation_dir / "slices.jsonl")
    metadata = json.loads((args.evaluation_dir / "metadata.json").read_text(encoding="utf-8"))
    task_ids = {int(row["task_id"]) for row in tasks}
    if not INVALID_TASK_IDS <= task_ids:
        raise SystemExit("expected invalid benchmark task is missing")

    products = json.loads(args.source.read_text(encoding="utf-8"))
    eval_frequency = Counter(
        str(product.get("category") or "").split("›")[-1]
        for product in products
        if product.get("tag") == "eval"
    )
    training_ids = _training_ids()
    semantic_training_ids = _jsonl_task_ids(ROOT / "data/sft/all.jsonl")
    training_rows_by_category = defaultdict(list)
    for task_id in semantic_training_ids:
        if 0 <= task_id < len(products):
            feature = _feature_row(task_id, products[task_id], eval_frequency)
            training_rows_by_category[feature["category_leaf"]].append(feature)

    blocked_ids = task_ids | training_ids
    blocked_families = {
        row["family_id"]
        for row in slices
        if int(row["task_id"]) not in INVALID_TASK_IDS
    }
    searcher = _searcher(args.search_index)
    candidates = []
    for task_id, product in enumerate(products):
        if product.get("tag") != "eval" or task_id in blocked_ids:
            continue
        if not _query_reward_contract_eligible(product):
            continue
        feature = _feature_row(task_id, product, eval_frequency)
        if feature["family_id"] in blocked_families:
            continue
        if _semantic_training_overlap(feature, training_rows_by_category):
            continue
        candidates.append(_with_retrieval(feature, product, searcher))
    searcher.close()

    slice_by_id = {int(row["task_id"]): row for row in slices}
    replacements = {}
    for invalid_id in sorted(INVALID_TASK_IDS):
        old = slice_by_id[invalid_id]
        exact = [
            row
            for row in candidates
            if row["domain"] == old["domain"]
            and row["difficulty_bucket"] == old["difficulty_bucket"]
            and row["retrieval_bucket"] == old["retrieval_bucket"]
            and row["family_id"] not in blocked_families
        ]
        if not exact:
            raise SystemExit(f"no slice-preserving replacement for task {invalid_id}")
        chosen = min(
            exact,
            key=lambda row: (
                abs(float(row["difficulty_score"]) - float(old["difficulty_score"])),
                _stable_key(args.seed, invalid_id, row["task_id"]),
            ),
        )
        replacements[invalid_id] = chosen
        blocked_families.add(chosen["family_id"])

    new_tasks = []
    new_slices = []
    changes = []
    for task in tasks:
        task_id = int(task["task_id"])
        old = slice_by_id[task_id]
        replacement = replacements.get(task_id)
        if replacement is None:
            new_tasks.append(task)
            new_slices.append(old)
            continue
        new_task_id = int(replacement["task_id"])
        new_slice = {
            **old,
            "task_id": new_task_id,
            "domain": replacement["domain"],
            "difficulty_score": replacement["difficulty_score"],
            "difficulty_bucket": replacement["difficulty_bucket"],
            "retrieval_rank": replacement["retrieval_rank"],
            "retrieval_bucket": replacement["retrieval_bucket"],
            "family_id": replacement["family_id"],
        }
        new_tasks.append({"task_id": new_task_id})
        new_slices.append(new_slice)
        changes.append(
            {
                "removed_task_id": task_id,
                "added_task_id": new_task_id,
                "old_slice": old,
                "new_slice": new_slice,
                "reason": "query_inconsistent_gold_annotation",
            }
        )

    tasks_payload = _jsonl_bytes(new_tasks)
    slices_payload = _jsonl_bytes(new_slices)
    metadata.update(
        {
            "evaluated": False,
            "selection_seed": args.seed,
            "task_sha256": hashlib.sha256(tasks_payload).hexdigest(),
            "slice_sha256": hashlib.sha256(slices_payload).hexdigest(),
            "source_sha256": _sha256_file(args.source),
            "search_index_sha256": _sha256_file(args.search_index),
            "query_contract_replacement_count": len(changes),
            "query_contract_replacements": changes,
            "difficulty_mean": round(
                sum(float(row["difficulty_score"]) for row in new_slices) / len(new_slices),
                6,
            ),
            "difficulty_max": max(float(row["difficulty_score"]) for row in new_slices),
            "difficulty_bucket_counts": dict(
                Counter(row["difficulty_bucket"] for row in new_slices)
            ),
            "retrieval_bucket_counts": dict(
                Counter(row["retrieval_bucket"] for row in new_slices)
            ),
        }
    )
    _write_atomic(args.evaluation_dir / "tasks.jsonl", tasks_payload)
    _write_atomic(args.evaluation_dir / "slices.jsonl", slices_payload)
    _write_json(args.evaluation_dir / "metadata.json", metadata)
    _write_json(
        args.evaluation_dir / "query-contract-replacements-v2.2.json",
        {
            "schema": "shopping-benchmark-query-contract-repair-v1",
            "seed": args.seed,
            "replacements": changes,
            "validation": {
                "task_count": len(new_tasks),
                "unique_task_ids": len({int(row["task_id"]) for row in new_tasks}),
                "training_task_overlap": len(
                    {int(row["task_id"]) for row in new_tasks} & training_ids
                ),
                "family_duplicates": len(new_slices)
                - len({row["family_id"] for row in new_slices}),
            },
        },
    )
    print(json.dumps(changes, ensure_ascii=False))


if __name__ == "__main__":
    main()
