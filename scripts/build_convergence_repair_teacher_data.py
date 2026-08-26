#!/usr/bin/env python3
"""Build a new 500-row Teacher corpus from stable and corrective trajectories.

The builder never mutates either source corpus.  It keeps every clean corrective
trajectory, then selects stable rows to recover the retrieval, length, coverage,
and action-diversity distribution required by the current data gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

from shopping_grpo.collection.data_gate import (
    DEFAULT_POLICY,
    RETRIEVAL_BUCKETS,
    action_sequence,
    audit_data_gate,
    retrieval_bucket,
    trajectory_coverage,
)
from shopping_grpo.collection.sft import (
    COLLECTION_SCHEMA_VERSION,
    TEACHER_SELECTION_VERSION,
    acceptance_reasons,
    build_collection_artifacts,
    read_jsonl,
    task_ids_from_jsonl,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
DEFAULT_PRODUCTS = SHOP_ENV / "data" / "items_eval_train.json"
DEFAULT_INDEX = SHOP_ENV / "search_engine" / "products.sqlite3"
LENGTH_BUCKETS = ("short", "medium", "long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", type=Path, required=True)
    parser.add_argument("--corrective-raw", type=Path, required=True)
    parser.add_argument("--corrective-tasks", type=Path, required=True)
    parser.add_argument("--held-out-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--search-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, row: dict) -> str:
    value = f"{seed}:{int(row['task_id'])}:{row.get('trajectory_id', '')}"
    return hashlib.sha256(value.encode()).hexdigest()


def length_bucket(row: dict) -> str:
    length = len(action_sequence(row))
    if length <= 10:
        return "short"
    if length <= 20:
        return "medium"
    return "long"


def quality_score(row: dict) -> int:
    coverage = trajectory_coverage(row)
    weights = {
        "loop_recovery": 10,
        "candidate_comparison": 8,
        "multiple_options": 7,
        "search_reformulation": 6,
        "evidence_verification": 4,
        "guard_recovery": 2,
        "variant_selection": 2,
    }
    return sum(weights[name] for name, present in coverage.items() if present and name in weights)


def select_corrective(
    candidates: list[dict],
    profiles: dict[int, dict],
    ranks: dict[int, int | None],
    seed: int,
) -> list[dict]:
    quotas = {
        "loop_recovery": 52,
        "near_miss_rejection": 64,
        "terminal_tool_commit": 32,
        "option_grounding": 12,
    }
    by_strategy = defaultdict(list)
    for row in candidates:
        strategy = profiles[int(row["task_id"])]["teacher_strategy"]
        row["_selection_strategy"] = strategy
        by_strategy[strategy].append(row)
    for strategy, quota in quotas.items():
        if len(by_strategy[strategy]) < quota:
            raise RuntimeError(
                f"not enough clean corrective rows for {strategy}: "
                f"{len(by_strategy[strategy])} < {quota}"
            )

    selected = []
    counts = Counter()
    sequence_counts = Counter()
    exact8 = 0
    while sum(counts.values()) < sum(quotas.values()):
        progress = False
        for strategy, quota in quotas.items():
            if counts[strategy] >= quota:
                continue
            eligible = [
                row
                for row in by_strategy[strategy]
                if row not in selected
                and sequence_counts[action_sequence(row)] < 40
                and not (len(action_sequence(row)) == 8 and exact8 >= 100)
            ]
            if not eligible:
                raise RuntimeError(f"corrective diversity caps block {strategy}")
            row = min(
                eligible,
                key=lambda item: (
                    sequence_counts[action_sequence(item)],
                    len(action_sequence(item)) == 8,
                    retrieval_bucket(ranks[int(item["task_id"])]) == "rank1",
                    -quality_score(item),
                    stable_key(seed, item),
                ),
            )
            selected.append(row)
            counts[strategy] += 1
            sequence_counts[action_sequence(row)] += 1
            exact8 += len(action_sequence(row)) == 8
            progress = True
        if not progress:
            raise RuntimeError("corrective selection made no progress")
    return selected


def retrieval_ranks(task_ids: set[int], products_path: Path, index_path: Path) -> dict[int, int | None]:
    shop_env = str(SHOP_ENV)
    if shop_env not in sys.path:
        sys.path.insert(0, shop_env)
    from web_agent_site.engine.search import MultiFieldBM25Searcher

    products = json.loads(products_path.read_text(encoding="utf-8"))
    searcher = MultiFieldBM25Searcher(index_path)
    ranks = {}
    for task_id in sorted(task_ids):
        product = products[task_id]
        instruction = (product.get("instructions") or [{}])[0]
        query = str(
            instruction.get("instruction")
            or instruction.get("instruction_text")
            or instruction.get("instruction_simple")
            or ""
        )
        asin = str(product.get("asin") or "")
        hits = searcher.search(query, 150)
        ranks[task_id] = next((hit.rank for hit in hits if hit.asin == asin), None)
    return ranks


def clean_rows(rows: list[dict], held_out: set[int]) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    seen = set()
    for row in rows:
        task_id = int(row["task_id"])
        ok, reasons = acceptance_reasons(row)
        coverage = trajectory_coverage(row)
        if ok and not coverage["clean_critical_actions"]:
            reasons = ["data_gate_unclean_critical_actions"]
            ok = False
        if ok and task_id in held_out:
            reasons = ["held_out_task"]
            ok = False
        if ok and task_id in seen:
            reasons = ["duplicate_task"]
            ok = False
        if ok:
            seen.add(task_id)
            accepted.append(row)
        else:
            rejected.append(
                {
                    "task_id": task_id,
                    "trajectory_id": row.get("trajectory_id"),
                    "reasons": reasons,
                }
            )
    return accepted, rejected


def target_rank_counts(corrective_counts: Counter) -> dict[str, int]:
    # Preserve the evaluation-shaped 50/30/10 distribution.  Corrective rows
    # contain more missing targets, so the residual tail absorbs that excess.
    targets = {
        "rank1": 260,
        "rank2_5": 140,
        "rank6_20": 50,
        "rank21_150": 35,
        "missing": 15,
    }
    for bucket in RETRIEVAL_BUCKETS:
        if corrective_counts[bucket] > targets[bucket]:
            raise RuntimeError(f"corrective {bucket} rows exceed final target")
    return targets


def max_flow_allocation(
    availability: dict[tuple[str, str], int],
    rank_demands: dict[str, int],
    length_demands: dict[str, int],
) -> dict[tuple[str, str], int]:
    source, sink = "source", "sink"
    ranks = [f"rank:{name}" for name in RETRIEVAL_BUCKETS]
    lengths = [f"length:{name}" for name in LENGTH_BUCKETS]
    capacity: dict[tuple[str, str], int] = {}
    adjacency: dict[str, list[str]] = defaultdict(list)

    def edge(left: str, right: str, value: int) -> None:
        capacity[left, right] = int(value)
        capacity.setdefault((right, left), 0)
        adjacency[left].append(right)
        adjacency[right].append(left)

    for bucket, node in zip(RETRIEVAL_BUCKETS, ranks):
        edge(source, node, rank_demands[bucket])
        for length, length_node in zip(LENGTH_BUCKETS, lengths):
            edge(node, length_node, availability[bucket, length])
    for length, node in zip(LENGTH_BUCKETS, lengths):
        edge(node, sink, length_demands[length])

    flow = Counter()
    total = 0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            left = queue.popleft()
            for right in adjacency[left]:
                if right not in parent and capacity[left, right] - flow[left, right] > 0:
                    parent[right] = left
                    queue.append(right)
        if sink not in parent:
            break
        amount = 10**9
        node = sink
        while parent[node] is not None:
            amount = min(amount, capacity[parent[node], node] - flow[parent[node], node])
            node = parent[node]
        node = sink
        while parent[node] is not None:
            flow[parent[node], node] += amount
            flow[node, parent[node]] -= amount
            node = parent[node]
        total += amount

    required = sum(rank_demands.values())
    if total != required:
        raise RuntimeError(
            f"cannot satisfy rank/length allocation: flow={total}, required={required}"
        )
    return {
        (bucket, length): flow[f"rank:{bucket}", f"length:{length}"]
        for bucket in RETRIEVAL_BUCKETS
        for length in LENGTH_BUCKETS
    }


def select_stable(
    candidates: list[dict],
    corrective: list[dict],
    ranks: dict[int, int | None],
    rank_demands: dict[str, int],
    length_demands: dict[str, int],
    seed: int,
) -> tuple[list[dict], dict]:
    cells = defaultdict(list)
    for row in candidates:
        key = (retrieval_bucket(ranks[int(row["task_id"])]), length_bucket(row))
        cells[key].append(row)
    availability = {key: len(cells[key]) for key in cells}
    for bucket in RETRIEVAL_BUCKETS:
        for length in LENGTH_BUCKETS:
            availability.setdefault((bucket, length), 0)
    allocation = max_flow_allocation(availability, rank_demands, length_demands)

    sequence_counts = Counter(action_sequence(row) for row in corrective)
    exact8 = sum(len(action_sequence(row)) == 8 for row in corrective)
    selected = []
    for key in sorted(
        allocation,
        key=lambda item: (
            -(allocation[item] / availability[item]) if availability[item] else 0,
            availability[item],
            item,
        ),
    ):
        count = allocation[key]
        remaining = list(cells[key])
        for _ in range(count):
            eligible = [
                row
                for row in remaining
                if sequence_counts[action_sequence(row)] < 60
                and not (len(action_sequence(row)) == 8 and exact8 >= 150)
            ]
            if not eligible:
                raise RuntimeError(f"diversity caps block stable allocation cell {key}")
            row = min(
                eligible,
                key=lambda item: (
                    sequence_counts[action_sequence(item)],
                    -quality_score(item),
                    stable_key(seed, item),
                ),
            )
            remaining.remove(row)
            selected.append(row)
            sequence_counts[action_sequence(row)] += 1
            exact8 += len(action_sequence(row)) == 8
    return selected, {
        "availability": {f"{a}/{b}": availability[a, b] for a, b in availability},
        "allocation": {f"{a}/{b}": allocation[a, b] for a, b in allocation},
    }


def choose_length_targets(
    availability: dict[tuple[str, str], int],
    rank_demands: dict[str, int],
    corrective_lengths: Counter,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    preferred = {"short": 250, "medium": 190, "long": 60}
    feasible = []
    for final_long in range(50, 101):
        for final_short in range(200, 301):
            final_medium = 500 - final_short - final_long
            targets = {
                "short": final_short,
                "medium": final_medium,
                "long": final_long,
            }
            demands = {
                name: targets[name] - corrective_lengths[name]
                for name in LENGTH_BUCKETS
            }
            if min(demands.values()) < 0:
                continue
            try:
                allocation = max_flow_allocation(availability, rank_demands, demands)
            except RuntimeError:
                continue
            score = sum(abs(targets[name] - preferred[name]) for name in LENGTH_BUCKETS)
            feasible.append((score, abs(final_long - 60), final_short, targets, allocation))
    if not feasible:
        raise RuntimeError("no feasible final length distribution")
    _, _, _, targets, allocation = min(feasible, key=lambda item: item[:3])
    return targets, allocation


def summary(rows: list[dict], ranks: dict[int, int | None]) -> dict:
    rank_counts = Counter()
    lengths = Counter()
    strategies = Counter()
    coverage = Counter()
    sequences = Counter()
    for row in rows:
        rank_counts[retrieval_bucket(ranks[int(row["task_id"])])] += 1
        lengths[length_bucket(row)] += 1
        strategies[str(row.get("_selection_strategy") or "stable")] += 1
        coverage.update(name for name, present in trajectory_coverage(row).items() if present)
        sequences[action_sequence(row)] += 1
    return {
        "rows": len(rows),
        "retrieval_counts": dict(sorted(rank_counts.items())),
        "length_counts": dict(sorted(lengths.items())),
        "strategy_counts": dict(sorted(strategies.items())),
        "coverage_counts": dict(sorted(coverage.items())),
        "unique_action_sequences": len(sequences),
        "top_sequence_count": sequences.most_common(1)[0][1] if sequences else 0,
        "top_sequence_share": round(sequences.most_common(1)[0][1] / len(rows), 6)
        if rows
        else 0,
        "top5_sequence_share": round(
            sum(count for _, count in sequences.most_common(5)) / len(rows), 6
        )
        if rows
        else 0,
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if (output / "accepted.jsonl").exists():
        raise SystemExit(f"refusing to overwrite existing Teacher version: {output}")
    output.mkdir(parents=True, exist_ok=True)

    held_out = task_ids_from_jsonl(args.held_out_tasks)
    stable_source = list(read_jsonl(args.stable))
    corrective_source = list(read_jsonl(args.corrective_raw))
    stable, stable_rejected = clean_rows(stable_source, held_out)
    corrective_clean, corrective_rejected = clean_rows(corrective_source, held_out)
    stable_ids = {int(row["task_id"]) for row in stable}
    corrective_clean = [
        row for row in corrective_clean if int(row["task_id"]) not in stable_ids
    ]

    profiles = {
        int(row["task_id"]): row for row in read_jsonl(args.corrective_tasks)
    }
    all_ids = {int(row["task_id"]) for row in stable + corrective_clean}
    ranks = retrieval_ranks(all_ids, args.products, args.search_index)
    corrective = select_corrective(corrective_clean, profiles, ranks, args.seed)
    corrective_rank_counts = Counter(
        retrieval_bucket(ranks[int(row["task_id"])]) for row in corrective
    )
    rank_targets = target_rank_counts(corrective_rank_counts)
    rank_demands = {
        bucket: rank_targets[bucket] - corrective_rank_counts[bucket]
        for bucket in RETRIEVAL_BUCKETS
    }

    corrective_length_counts = Counter(length_bucket(row) for row in corrective)
    availability = Counter(
        (retrieval_bucket(ranks[int(row["task_id"])]), length_bucket(row))
        for row in stable
    )
    length_targets, _ = choose_length_targets(
        availability, rank_demands, corrective_length_counts
    )
    length_demands = {
        bucket: length_targets[bucket] - corrective_length_counts[bucket]
        for bucket in LENGTH_BUCKETS
    }

    selected_stable, allocation = select_stable(
        stable,
        corrective,
        ranks,
        rank_demands,
        length_demands,
        args.seed,
    )
    selected = corrective + selected_stable
    if len(selected) != 500 or len({int(row["task_id"]) for row in selected}) != 500:
        raise RuntimeError("selection did not produce 500 unique tasks")

    raw_path = output / "raw.jsonl"
    write_jsonl(raw_path, selected)
    collection_summary = build_collection_artifacts(
        raw_path=raw_path,
        output_dir=output,
        held_out_task_ids=held_out,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        collection_config={
            "builder": "build_convergence_repair_teacher_data.py",
            "stable_source": str(args.stable.resolve()),
            "corrective_source": str(args.corrective_raw.resolve()),
            "corrective_prompt": "shopping-teacher-prompt-v4-convergence-repair",
        },
    )

    policy = json.loads(json.dumps(DEFAULT_POLICY))
    report = audit_data_gate(selected, retrieval_ranks=ranks, policy=policy)
    report["audit"] = {
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "teacher_selection": TEACHER_SELECTION_VERSION,
        "search_contract": "shopsimulator-multifield-bm25-v2.1",
        "source_rows": len(selected),
        "quality_gate_passed_rows": len(selected),
        "audited_rows": len(selected),
        "input_sha256": sha256(raw_path),
        "products_sha256": sha256(args.products),
        "search_index_sha256": sha256(args.search_index),
    }
    gate_path = output / "data_gate_v1.json"
    gate_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "shopping-teacher-convergence-repair-materialization-v2",
        "sources": {
            "stable": {
                "path": str(args.stable.resolve()),
                "rows": len(stable_source),
                "strict_clean_rows": len(stable),
                "sha256": sha256(args.stable),
            },
            "corrective": {
                "path": str(args.corrective_raw.resolve()),
                "raw_rows": len(corrective_source),
                "strict_clean_rows": len(corrective_clean),
                "sha256": sha256(args.corrective_raw),
            },
        },
        "selected": {
            "stable": len(selected_stable),
            "corrective": len(corrective),
            "total": len(selected),
        },
        "corrective_rejected": corrective_rejected,
        "corrective_clean_unselected": len(corrective_clean) - len(corrective),
        "stable_rejected_count": len(stable_rejected),
        "rank_targets": rank_targets,
        "length_targets": length_targets,
        "allocation": allocation,
        "collection": collection_summary,
        "data_gate_status": report["status"],
        "credentials_recorded": False,
    }
    (output / "materialization.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "distribution_summary.json").write_text(
        json.dumps(summary(selected, ranks), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_gate"] = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "path": gate_path.name,
        "sha256": sha256(gate_path),
        "rows": report["rows"],
    }
    metadata["materialization"] = {
        "schema_version": manifest["schema_version"],
        "path": "materialization.json",
        "selected_stable": len(selected_stable),
        "selected_corrective": len(corrective),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"manifest": manifest, "distribution": summary(selected, ranks), "gate": report}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
