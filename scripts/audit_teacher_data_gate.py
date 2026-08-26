#!/usr/bin/env python3
"""Audit a final successful Teacher corpus against the base-friendly data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from shopping_grpo.collection.data_gate import DEFAULT_POLICY, audit_data_gate
from shopping_grpo.collection.sft import (
    COLLECTION_SCHEMA_VERSION,
    TEACHER_SELECTION_VERSION,
    acceptance_reasons,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
DEFAULT_PRODUCTS = SHOP_ENV / "data" / "items_eval_train.json"
DEFAULT_INDEX = SHOP_ENV / "search_engine" / "products.sqlite3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--search-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--target-rows", type=int, default=1000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _searcher(index_path: Path):
    shop_env = str(SHOP_ENV)
    if shop_env not in sys.path:
        sys.path.insert(0, shop_env)
    from web_agent_site.engine.search import MultiFieldBM25Searcher

    return MultiFieldBM25Searcher(index_path)


def _instruction(product: dict) -> str:
    instruction = (product.get("instructions") or [{}])[0]
    return str(
        instruction.get("instruction")
        or instruction.get("instruction_text")
        or instruction.get("instruction_simple")
        or ""
    )


def retrieval_ranks(task_ids: set[int], *, products_path: Path, index_path: Path) -> dict:
    products = json.loads(products_path.read_text(encoding="utf-8"))
    searcher = _searcher(index_path)
    ranks = {}
    for task_id in sorted(task_ids):
        if not 0 <= task_id < len(products):
            continue
        product = products[task_id]
        target_asin = str(product.get("asin") or "")
        hits = searcher.search(_instruction(product), 150)
        ranks[task_id] = next(
            (hit.rank for hit in hits if hit.asin == target_asin),
            None,
        )
    return ranks


def main() -> int:
    args = parse_args()
    if args.target_rows < 1:
        raise SystemExit("--target-rows must be positive")
    source_rows = list(read_jsonl(args.input))
    rows = [row for row in source_rows if acceptance_reasons(row)[0]]
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    policy["target_rows"] = int(args.target_rows)
    ranks = retrieval_ranks(
        {int(row["task_id"]) for row in rows},
        products_path=args.products,
        index_path=args.search_index,
    )
    report = audit_data_gate(rows, retrieval_ranks=ranks, policy=policy)
    report["audit"] = {
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "teacher_selection": TEACHER_SELECTION_VERSION,
        "search_contract": "shopsimulator-multifield-bm25-v2.1",
        "source_rows": len(source_rows),
        "quality_gate_passed_rows": len(rows),
        "audited_rows": len(rows),
        "input_sha256": _sha256(args.input),
        "products_sha256": _sha256(args.products),
        "search_index_sha256": _sha256(args.search_index),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
