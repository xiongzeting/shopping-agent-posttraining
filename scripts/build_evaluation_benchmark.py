#!/usr/bin/env python3
"""Build the frozen Final-240 benchmark from the official eval-tag pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "data"
    / "items_eval_train.json"
)
DEFAULT_SEARCH_INDEX = (
    ROOT
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "search_engine"
    / "products.sqlite3"
)
DOMAINS = (
    "家居家装",
    "服饰鞋包饰品",
    "休闲娱乐文教",
    "美妆个护健康",
    "生产材料农用品",
    "家用电器数码",
    "运动户外交通",
    "食品饮品",
    "母婴儿童",
)
CHALLENGE_SLICES = (
    "search_reformulation",
    "candidate_comparison",
    "price_semantics",
    "multi_option",
    "evidence_verification",
    "long_horizon",
)
DIFFICULTY_BUCKETS = ("under_10", "10_15", "15_18", "18_plus")
RETRIEVAL_BUCKETS = ("rank1", "rank2_5", "rank6_20", "rank21_150", "missing")
TARGET_DIFFICULTY_COUNTS = {
    "under_10": 48,
    "10_15": 138,
    "15_18": 44,
    "18_plus": 10,
}
TARGET_RETRIEVAL_COUNTS = {
    "rank1": 144,
    "rank2_5": 60,
    "rank6_20": 24,
    "rank21_150": 10,
    "missing": 2,
}
CHALLENGE_TARGET_DIFFICULTY_COUNTS = {
    "under_10": 2,
    "10_15": 32,
    "15_18": 22,
    "18_plus": 4,
}
CHALLENGE_TARGET_RETRIEVAL_COUNTS = {
    "rank1": 36,
    "rank2_5": 15,
    "rank6_20": 6,
    "rank21_150": 2,
    "missing": 1,
}
CHALLENGE_SEVERITY_QUOTAS = (1, 2, 3, 3, 1)
CHALLENGE_PERCENTILE_WINDOWS = (
    (0.30, 0.45),
    (0.45, 0.60),
    (0.60, 0.75),
    (0.75, 0.90),
    (0.90, 0.98),
)
CORE_QUANTILE_QUOTAS = (4, 10, 6)
CHALLENGE_SELECTION_ORDER = CHALLENGE_SLICES
_HARD_BUDGET = re.compile(r"以内|以下|不超过|不得超过|至多|最多|上限|低于|小于")
_APPROX_PRICE = re.compile(r"左右|上下|出头|来块|多元|多块|大约|大概|约")
_NEGATION = re.compile(r"不要|不能|不含|无需|避免|非")
_COMPATIBILITY = re.compile(r"适用|适配|兼容")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--search-index", type=Path, default=DEFAULT_SEARCH_INDEX)
    parser.add_argument(
        "--training-jsonl",
        type=Path,
        action="append",
        default=[ROOT / "data" / "sft" / "all.jsonl"],
    )
    parser.add_argument(
        "--exclude-task-file",
        type=Path,
        action="append",
        default=[],
        help="Existing benchmark files whose task IDs must not be reused.",
    )
    parser.add_argument(
        "--tasks-output",
        type=Path,
        default=ROOT / "data" / "evaluation" / "tasks.jsonl",
    )
    parser.add_argument(
        "--slices-output",
        type=Path,
        default=ROOT / "data" / "evaluation" / "slices.jsonl",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=ROOT / "data" / "evaluation" / "metadata.json",
    )
    parser.add_argument("--core-per-domain", type=int, default=20)
    parser.add_argument("--challenge-per-slice", type=int, default=10)
    parser.add_argument("--seed", default="shopbench-final240-v2.1-20260809")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: str, *values: object) -> str:
    payload = "|".join([seed, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _trigrams(value: object) -> frozenset[str]:
    text = _normalize(value)
    if len(text) < 3:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[index : index + 3] for index in range(len(text) - 2))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _searcher(index_path: Path):
    shop_env = str(ROOT / "environments" / "ShopSimulator" / "shop_env")
    if shop_env not in sys.path:
        sys.path.insert(0, shop_env)
    from web_agent_site.engine.search import MultiFieldBM25Searcher

    return MultiFieldBM25Searcher(index_path)


def _difficulty_bucket(score: float) -> str:
    if score < 10:
        return "under_10"
    if score < 15:
        return "10_15"
    if score < 18:
        return "15_18"
    return "18_plus"


def _retrieval_bucket(rank: int | None) -> str:
    if rank is None or rank > 150:
        return "missing"
    if rank <= 1:
        return "rank1"
    if rank <= 5:
        return "rank2_5"
    if rank <= 20:
        return "rank6_20"
    return "rank21_150"


def _jsonl_task_ids(paths: list[Path]) -> set[int]:
    task_ids = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row.get("task_id")
                if task_id is not None:
                    task_ids.add(int(task_id))
    return task_ids


def _instruction(product: dict) -> dict:
    rows = product.get("instructions") or []
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Final-240 source requires exactly one instruction per product")
    return rows[0]


def _family_id(product: dict) -> str:
    category_leaf = str(product.get("category") or "").split("›")[-1]
    shop = re.sub(
        r"官方旗舰店|旗舰店|专卖店|专营店|企业店|店$",
        "",
        str(product.get("shop_name") or ""),
    )
    title = re.sub(r"\d+(?:\.\d+)?", "#", str(product.get("title") or ""))
    signature = "|".join(
        (_normalize(category_leaf), _normalize(shop), _normalize(title)[:48])
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]


def _feature_row(task_id: int, product: dict, category_frequency: Counter) -> dict:
    instruction = _instruction(product)
    query = str(instruction.get("instruction") or "")
    title = str(product.get("title") or "")
    option_axes = product.get("customization_options") or {}
    variant_count = sum(len(values or []) for values in option_axes.values())
    attributes = instruction.get("attributes") or []
    category_leaf = str(product.get("category") or "").split("›")[-1]
    lexical_overlap = _jaccard(_trigrams(query), _trigrams(title))
    difficulty = (
        len(attributes)
        + 1.5 * len(option_axes)
        + min(variant_count, 12) / 4
        + len(query) / 45
        + 2.5 * (1 - lexical_overlap)
    )
    return {
        "task_id": task_id,
        "source_tag": product.get("tag"),
        "domain": product.get("domain_zh"),
        "category": product.get("category"),
        "category_leaf": category_leaf,
        "category_pool_size": category_frequency[category_leaf],
        "family_id": _family_id(product),
        "query": query,
        "title": title,
        "query_chars": len(query),
        "attribute_count": len(attributes),
        "option_axes": len(option_axes),
        "variant_count": variant_count,
        "hard_budget": bool(_HARD_BUDGET.search(query)),
        "approximate_price": bool(_APPROX_PRICE.search(query)),
        "negation": bool(_NEGATION.search(query)),
        "compatibility": bool(_COMPATIBILITY.search(query)),
        "lexical_overlap": lexical_overlap,
        "difficulty_score": round(difficulty, 6),
        "difficulty_bucket": _difficulty_bucket(difficulty),
        "retrieval_rank": None,
        "retrieval_bucket": "missing",
    }


def _with_retrieval(row: dict, product: dict, searcher) -> dict:
    target_asin = str(product.get("asin") or "")
    hits = searcher.search(row["query"], 150)
    rank = next((hit.rank for hit in hits if hit.asin == target_asin), None)
    return {
        **row,
        "retrieval_rank": rank,
        "retrieval_bucket": _retrieval_bucket(rank),
    }


def _semantic_training_overlap(
    candidate: dict,
    training_rows_by_category: dict[str, list[dict]],
) -> bool:
    for training in training_rows_by_category.get(candidate["category_leaf"], []):
        if candidate["family_id"] == training["family_id"]:
            return True
        if _normalize(candidate["title"]) == _normalize(training["title"]):
            return True
        if _normalize(candidate["query"]) == _normalize(training["query"]):
            return True
        if _jaccard(_trigrams(candidate["title"]), _trigrams(training["title"])) >= 0.88:
            return True
        if _jaccard(_trigrams(candidate["query"]), _trigrams(training["query"])) >= 0.85:
            return True
    return False


def _challenge_severity(slice_name: str, row: dict) -> float:
    if slice_name == "search_reformulation":
        severity = (1 - row["lexical_overlap"]) + row["query_chars"] / 150
    elif slice_name == "candidate_comparison":
        severity = row["category_pool_size"] / 50 + row["attribute_count"] / 10
    elif slice_name == "price_semantics":
        severity = 2 * row["approximate_price"] + row["hard_budget"] + row["query_chars"] / 200
    elif slice_name == "multi_option":
        severity = 2 * row["option_axes"] + row["variant_count"] / 5
    elif slice_name == "evidence_verification":
        severity = row["attribute_count"] + row["query_chars"] / 40
    elif slice_name == "long_horizon":
        severity = row["difficulty_score"]
    else:  # pragma: no cover
        raise ValueError(f"unknown challenge slice: {slice_name}")
    return severity


def _challenge_rank(slice_name: str, row: dict) -> tuple:
    """Rank hardest-first for the current Teacher candidate sampler."""

    return (-_challenge_severity(slice_name, row), row["task_id"])


def _challenge_eligible(slice_name: str, row: dict) -> bool:
    if slice_name == "search_reformulation":
        return row["lexical_overlap"] <= 0.30 and row["query_chars"] >= 55
    if slice_name == "candidate_comparison":
        return row["category_pool_size"] >= 8 and row["attribute_count"] >= 4
    if slice_name == "price_semantics":
        return row["hard_budget"] or row["approximate_price"]
    if slice_name == "multi_option":
        return row["option_axes"] >= 2 or row["variant_count"] >= 6
    if slice_name == "evidence_verification":
        return row["attribute_count"] >= 6 and row["query_chars"] >= 65
    if slice_name == "long_horizon":
        return row["difficulty_score"] >= 9
    return False


def _percentile_bands(
    rows: list[dict],
    windows: tuple[tuple[float, float], ...],
) -> list[list[dict]]:
    return [
        rows[int(len(rows) * start) : max(int(len(rows) * end), int(len(rows) * start) + 1)]
        for start, end in windows
    ]


def _extreme_factor_count(row: dict) -> int:
    return sum(
        (
            row["difficulty_score"] >= 18,
            row["retrieval_bucket"] in {"rank21_150", "missing"},
            row["option_axes"] >= 3,
            row["variant_count"] >= 10,
            row["lexical_overlap"] <= 0.10 and row["attribute_count"] >= 6,
        )
    )


def _distribution_cost(
    row: dict,
    *,
    difficulty_counts: Counter,
    retrieval_counts: Counter,
) -> float:
    difficulty_bucket = row["difficulty_bucket"]
    retrieval_bucket = row["retrieval_bucket"]
    difficulty_target = TARGET_DIFFICULTY_COUNTS[difficulty_bucket]
    retrieval_target = TARGET_RETRIEVAL_COUNTS[retrieval_bucket]
    difficulty_fill = difficulty_counts[difficulty_bucket] / difficulty_target
    retrieval_fill = retrieval_counts[retrieval_bucket] / retrieval_target
    overflow = 0.0
    if difficulty_counts[difficulty_bucket] >= difficulty_target:
        overflow += 100.0
    if retrieval_counts[retrieval_bucket] >= retrieval_target:
        overflow += 80.0
    return overflow + 12.0 * difficulty_fill + 8.0 * retrieval_fill


def _challenge_distribution_cost(
    row: dict,
    *,
    difficulty_counts: Counter,
    retrieval_counts: Counter,
) -> float:
    difficulty_bucket = row["difficulty_bucket"]
    retrieval_bucket = row["retrieval_bucket"]
    difficulty_target = CHALLENGE_TARGET_DIFFICULTY_COUNTS[difficulty_bucket]
    retrieval_target = CHALLENGE_TARGET_RETRIEVAL_COUNTS[retrieval_bucket]
    cost = (
        10.0 * difficulty_counts[difficulty_bucket] / difficulty_target
        + 8.0 * retrieval_counts[retrieval_bucket] / retrieval_target
    )
    if difficulty_counts[difficulty_bucket] >= difficulty_target:
        cost += 40.0
    if retrieval_counts[retrieval_bucket] >= retrieval_target:
        cost += 32.0
    return cost


def _challenge_moderation_cost(
    slice_name: str,
    row: dict,
    *,
    difficulty_counts: Counter,
    retrieval_counts: Counter,
    challenge_difficulty_counts: Counter,
    challenge_retrieval_counts: Counter,
    seed: str,
) -> tuple:
    target_difficulty = {
        "search_reformulation": 14.5,
        "candidate_comparison": 16.0,
        "price_semantics": 14.0,
        "multi_option": 14.5,
        "evidence_verification": 16.0,
        "long_horizon": 17.0,
    }[slice_name]
    extreme_penalty = 5.0 * _extreme_factor_count(row)
    if row["difficulty_score"] >= 21:
        extreme_penalty += 40.0
    if row["retrieval_bucket"] == "missing":
        extreme_penalty += 12.0
    elif row["retrieval_bucket"] == "rank21_150":
        extreme_penalty += 5.0
    cost = (
        _distribution_cost(
            row,
            difficulty_counts=difficulty_counts,
            retrieval_counts=retrieval_counts,
        )
        + _challenge_distribution_cost(
            row,
            difficulty_counts=challenge_difficulty_counts,
            retrieval_counts=challenge_retrieval_counts,
        )
        + 1.5 * abs(row["difficulty_score"] - target_difficulty)
        + extreme_penalty
    )
    return (cost, _stable_key(seed, "challenge-moderate", slice_name, row["task_id"]))


def _select_challenge_slice(
    rows: list[dict],
    *,
    slice_name: str,
    count: int,
    seed: str,
    selected_ids: set[int],
    selected_families: set[str],
    difficulty_counts: Counter,
    retrieval_counts: Counter,
    challenge_difficulty_counts: Counter,
    challenge_retrieval_counts: Counter,
) -> list[dict]:
    eligible = sorted(
        (row for row in rows if _challenge_eligible(slice_name, row)),
        key=lambda row: (_challenge_severity(slice_name, row), row["task_id"]),
    )
    if count != sum(CHALLENGE_SEVERITY_QUOTAS):
        raise ValueError("v2.1 challenge selection requires 10 tasks per slice")
    chosen = []
    bands = _percentile_bands(eligible, CHALLENGE_PERCENTILE_WINDOWS)
    for band_index, (band, quota) in enumerate(
        zip(bands, CHALLENGE_SEVERITY_QUOTAS, strict=True)
    ):
        available = [
            row
            for row in band
            if row["task_id"] not in selected_ids
            and row["family_id"] not in selected_families
        ]
        ranked = sorted(
            available,
            key=lambda row: _challenge_moderation_cost(
                slice_name,
                row,
                difficulty_counts=difficulty_counts,
                retrieval_counts=retrieval_counts,
                challenge_difficulty_counts=challenge_difficulty_counts,
                challenge_retrieval_counts=challenge_retrieval_counts,
                seed=f"{seed}|band={band_index}",
            ),
        )
        for row in ranked[:quota]:
            chosen.append(row)
            selected_ids.add(row["task_id"])
            selected_families.add(row["family_id"])
            difficulty_counts[row["difficulty_bucket"]] += 1
            retrieval_counts[row["retrieval_bucket"]] += 1
            challenge_difficulty_counts[row["difficulty_bucket"]] += 1
            challenge_retrieval_counts[row["retrieval_bucket"]] += 1
        if len(ranked) < quota:
            raise ValueError(
                f"could select only {len(ranked)} of {quota} tasks for "
                f"{slice_name} severity band {band_index}"
            )
    return chosen


def _select_core_domain(
    rows: list[dict],
    *,
    count: int,
    seed: str,
    selected_ids: set[int],
    selected_families: set[str],
    difficulty_counts: Counter,
    retrieval_counts: Counter,
) -> list[dict]:
    ordered = sorted(
        (
            row
            for row in rows
            if row["task_id"] not in selected_ids
            and row["family_id"] not in selected_families
        ),
        key=lambda row: (row["difficulty_score"], row["task_id"]),
    )
    bands = _percentile_bands(
        ordered,
        ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)),
    )
    if count != sum(CORE_QUANTILE_QUOTAS):
        raise ValueError("v2.1 core selection requires 20 tasks per domain")
    chosen = []
    slot = 0
    for band_index, (band, quota) in enumerate(
        zip(bands, CORE_QUANTILE_QUOTAS, strict=True)
    ):
        available = [
            row
            for row in band
            if row["task_id"] not in selected_ids
            and row["family_id"] not in selected_families
        ]
        if len(available) < quota:
            raise ValueError(
                f"could select only {len(available)} of {quota} core tasks from "
                f"difficulty band {band_index}"
            )

        def core_cost(row: dict) -> tuple:
            band_target = (9.5, 12.5, 15.5)[band_index]
            moderation = abs(row["difficulty_score"] - band_target)
            extreme_penalty = 5.0 * _extreme_factor_count(row)
            if row["difficulty_score"] >= 21:
                extreme_penalty += 50.0
            if row["retrieval_bucket"] == "missing":
                extreme_penalty += 20.0
            elif row["retrieval_bucket"] == "rank21_150":
                extreme_penalty += 6.0
            cost = (
                _distribution_cost(
                    row,
                    difficulty_counts=difficulty_counts,
                    retrieval_counts=retrieval_counts,
                )
                + moderation
                + extreme_penalty
            )
            return (
                cost,
                _stable_key(seed, "core", band_index, slot, row["task_id"]),
            )

        for _ in range(quota):
            row = min(available, key=core_cost)
            available.remove(row)
            chosen.append(row)
            selected_ids.add(row["task_id"])
            selected_families.add(row["family_id"])
            difficulty_counts[row["difficulty_bucket"]] += 1
            retrieval_counts[row["retrieval_bucket"]] += 1
            slot += 1
    return chosen


def _serialize_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.replace(path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    products = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(products, list):
        raise SystemExit("source product archive must be a JSON array")
    searcher = _searcher(args.search_index)
    training_ids = _jsonl_task_ids(args.training_jsonl)
    previous_ids = _jsonl_task_ids(args.exclude_task_file)
    category_frequency = Counter(
        str(product.get("category") or "").split("›")[-1]
        for product in products
        if product.get("tag") == "eval"
    )
    training_rows_by_category: dict[str, list[dict]] = defaultdict(list)
    for task_id in sorted(training_ids):
        if 0 <= task_id < len(products):
            row = _feature_row(task_id, products[task_id], category_frequency)
            training_rows_by_category[row["category_leaf"]].append(row)

    excluded_semantic = []
    candidates = []
    for task_id, product in enumerate(products):
        if product.get("tag") != "eval":
            continue
        if task_id in training_ids or task_id in previous_ids:
            continue
        row = _feature_row(task_id, product, category_frequency)
        if row["domain"] not in DOMAINS:
            continue
        if _semantic_training_overlap(row, training_rows_by_category):
            excluded_semantic.append(task_id)
            continue
        candidates.append(_with_retrieval(row, product, searcher))
    search_manifest = dict(searcher.manifest)
    searcher.close()

    selected_ids: set[int] = set()
    selected_families: set[str] = set()
    difficulty_counts: Counter = Counter()
    retrieval_counts: Counter = Counter()
    challenge_difficulty_counts: Counter = Counter()
    challenge_retrieval_counts: Counter = Counter()
    challenge_by_slice = {}
    for slice_name in CHALLENGE_SELECTION_ORDER:
        chosen = _select_challenge_slice(
            candidates,
            slice_name=slice_name,
            count=args.challenge_per_slice,
            seed=args.seed,
            selected_ids=selected_ids,
            selected_families=selected_families,
            difficulty_counts=difficulty_counts,
            retrieval_counts=retrieval_counts,
            challenge_difficulty_counts=challenge_difficulty_counts,
            challenge_retrieval_counts=challenge_retrieval_counts,
        )
        challenge_by_slice[slice_name] = [
            {**row, "challenge_slice": slice_name} for row in chosen
        ]
    challenge = [
        row
        for slice_name in CHALLENGE_SLICES
        for row in challenge_by_slice[slice_name]
    ]

    core = []
    for domain in DOMAINS:
        domain_rows = [row for row in candidates if row["domain"] == domain]
        chosen = _select_core_domain(
            domain_rows,
            count=args.core_per_domain,
            seed=args.seed,
            selected_ids=selected_ids,
            selected_families=selected_families,
            difficulty_counts=difficulty_counts,
            retrieval_counts=retrieval_counts,
        )
        core.extend({**row, "challenge_slice": None} for row in chosen)

    selected = [
        *({**row, "suite": "core"} for row in core),
        *({**row, "suite": "challenge"} for row in challenge),
    ]
    expected_count = len(DOMAINS) * args.core_per_domain + (
        len(CHALLENGE_SLICES) * args.challenge_per_slice
    )
    if len(selected) != expected_count or len(selected_ids) != expected_count:
        raise SystemExit("Final-240 selection count or uniqueness check failed")
    if selected_ids & training_ids or selected_ids & previous_ids:
        raise SystemExit("Final-240 overlaps training or previous benchmark task IDs")

    tasks_rows = [{"task_id": row["task_id"]} for row in selected]
    slices_rows = [
        {
            "task_id": row["task_id"],
            "suite": row["suite"],
            "domain": row["domain"],
            "challenge_slice": row["challenge_slice"],
            "difficulty_score": row["difficulty_score"],
            "difficulty_bucket": row["difficulty_bucket"],
            "retrieval_rank": row["retrieval_rank"],
            "retrieval_bucket": row["retrieval_bucket"],
            "family_id": row["family_id"],
            "feature_tags": sorted(
                name
                for name in (
                    "hard_budget" if row["hard_budget"] else None,
                    "approximate_price" if row["approximate_price"] else None,
                    "negation" if row["negation"] else None,
                    "compatibility" if row["compatibility"] else None,
                    "multi_option" if row["option_axes"] >= 2 else None,
                )
                if name
            ),
        }
        for row in selected
    ]
    tasks_payload = _serialize_jsonl(tasks_rows)
    slices_payload = _serialize_jsonl(slices_rows)
    metadata = {
        "schema_version": "shopping-evaluation-dataset-v2.1",
        "asset": "shopbench_longhorizon_final_240_v2_1",
        "contract": "environment-v2.4/reward-v4/benchmark-v2.1",
        "evaluated": False,
        "environment": "shopsimulator-environment-v2.4",
        "reward": "shopsimulator-reward-v4",
        "termination": "shopping-termination-v3.1",
        "observation": "shopping-observation-v2",
        "tool_schema": "shopping-tools-v2",
        "path": "data/evaluation/tasks.jsonl",
        "slice_path": "data/evaluation/slices.jsonl",
        "tasks": expected_count,
        "core_tasks": len(core),
        "challenge_tasks": len(challenge),
        "source_pool": "items_eval_train.json:tag=eval",
        "source_sha256": _sha256_file(args.source),
        "search_contract": search_manifest.get("search_version"),
        "search_index_schema_version": search_manifest.get("index_schema_version"),
        "search_index_sha256": _sha256_file(args.search_index),
        "selection_seed": args.seed,
        "selection_policy": {
            "core": "difficulty thirds with low/mid/high quotas 4/10/6 per domain",
            "challenge": "severity percentiles 30-98 with quotas 1/2/3/3/1",
            "difficulty_targets": TARGET_DIFFICULTY_COUNTS,
            "retrieval_targets": TARGET_RETRIEVAL_COUNTS,
            "challenge_difficulty_targets": CHALLENGE_TARGET_DIFFICULTY_COUNTS,
            "challenge_retrieval_targets": CHALLENGE_TARGET_RETRIEVAL_COUNTS,
            "targets_are_soft": True,
            "extreme_factor_penalty": True,
        },
        "task_sha256": _sha256_bytes(tasks_payload),
        "slice_sha256": _sha256_bytes(slices_payload),
        "training_task_overlap": 0,
        "previous_benchmark_overlap": 0,
        "selected_family_duplicates": len(selected) - len(selected_families),
        "semantic_training_candidates_excluded": len(excluded_semantic),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in selected).items())),
        "core_domain_counts": dict(
            sorted(Counter(row["domain"] for row in core).items())
        ),
        "suite_counts": dict(sorted(Counter(row["suite"] for row in selected).items())),
        "challenge_slice_counts": dict(
            sorted(
                Counter(
                    row["challenge_slice"]
                    for row in selected
                    if row["challenge_slice"]
                ).items()
            )
        ),
        "difficulty_bucket_counts": {
            bucket: difficulty_counts[bucket] for bucket in DIFFICULTY_BUCKETS
        },
        "retrieval_bucket_counts": {
            bucket: retrieval_counts[bucket] for bucket in RETRIEVAL_BUCKETS
        },
        "difficulty_mean": round(
            sum(row["difficulty_score"] for row in selected) / len(selected), 6
        ),
        "difficulty_max": max(row["difficulty_score"] for row in selected),
        "extreme_factor_counts": dict(
            sorted(Counter(_extreme_factor_count(row) for row in selected).items())
        ),
    }
    metadata_payload = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(args.tasks_output, tasks_payload)
    _write_atomic(args.slices_output, slices_payload)
    _write_atomic(args.metadata_output, metadata_payload)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
