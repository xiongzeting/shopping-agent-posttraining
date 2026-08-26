#!/usr/bin/env python3
"""Build a leak-free, full-record GRPO development probe without a GPU."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

try:
    from build_difficulty_stratified_teacher_tasks import (
        BANDS,
        DEFAULT_STRATEGY_CYCLE,
        _band,
        _select_strategy_balanced,
    )
    from build_evaluation_benchmark import (
        _family_id,
        _feature_row,
        _semantic_training_overlap,
        _sha256_file,
        _write_atomic,
    )
except ModuleNotFoundError:  # Imported as scripts.* by unit tests.
    from scripts.build_difficulty_stratified_teacher_tasks import (
        BANDS,
        DEFAULT_STRATEGY_CYCLE,
        _band,
        _select_strategy_balanced,
    )
    from scripts.build_evaluation_benchmark import (
        _family_id,
        _feature_row,
        _semantic_training_overlap,
        _sha256_file,
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
DEFAULT_OUTPUT = ROOT / "data" / "grpo" / "dev-probe-v1"
DEFAULT_QUOTAS = {"lt10": 15, "10to15": 45, "15to18": 30, "ge18": 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default="grpo-dev-probe-v1-20260810")
    parser.add_argument(
        "--quotas",
        default="15,45,30,10",
        help="Comma-separated quotas for lt10,10to15,15to18,ge18.",
    )
    parser.add_argument(
        "--exclude-jsonl", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--exclude-parquet", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--held-out", type=Path, default=ROOT / "data/evaluation/tasks.jsonl"
    )
    return parser.parse_args()


def _stream_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream objects from a top-level JSON array with bounded input buffering."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    with path.open(encoding="utf-8") as stream:
        while True:
            chunk = stream.read(chunk_size)
            eof = not chunk
            buffer = buffer[position:] + chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    finished = True
                    position += 1
                    break
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object array item")
                yield value
                position = end
            if finished:
                if buffer[position:].strip():
                    raise ValueError(f"{path} has data after the top-level array")
                break
            if eof:
                raise ValueError(f"{path} ended before the top-level array closed")


def _row_task_id(row: dict) -> int | None:
    task_id = row.get("task_id")
    if task_id is None:
        extra_info = row.get("extra_info")
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = None
        if isinstance(extra_info, dict):
            task_id = extra_info.get("task_id")
    return int(task_id) if task_id is not None else None


def _jsonl_task_ids(path: Path) -> set[int]:
    task_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            task_id = _row_task_id(json.loads(line))
            if task_id is not None:
                task_ids.add(task_id)
    return task_ids


def _parquet_task_ids(path: Path) -> set[int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("pyarrow is required to read GRPO parquet exclusions") from exc

    task_ids = set()
    parquet_file = parquet.ParquetFile(path)
    columns = [
        name
        for name in ("task_id", "extra_info")
        if name in parquet_file.schema_arrow.names
    ]
    if not columns:
        raise ValueError(f"{path} has neither task_id nor extra_info")
    for batch in parquet_file.iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            task_id = _row_task_id(row)
            if task_id is not None:
                task_ids.add(task_id)
    return task_ids


def _existing(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()})


def _default_jsonl_exclusions() -> list[Path]:
    paths = list((ROOT / "data").glob("sft*/all.jsonl"))
    paths.extend(
        (
            ROOT / "data/evaluation/tasks.jsonl",
            ROOT / "data/grpo/candidates-smoke-v1/tasks.jsonl",
        )
    )
    return _existing(paths)


def _default_parquet_exclusions() -> list[Path]:
    return _existing(
        (
            ROOT / "data/grpo/train.parquet",
            ROOT / "data/grpo/validation.parquet",
        )
    )


def _jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _parse_quotas(value: str) -> dict[str, int]:
    numbers = [int(part.strip()) for part in value.split(",")]
    if len(numbers) != len(BANDS) or any(number < 0 for number in numbers):
        raise ValueError("--quotas must contain four non-negative integers")
    if sum(numbers) < 1:
        raise ValueError("--quotas must select at least one task")
    return dict(zip(BANDS, numbers))


def main() -> None:
    args = parse_args()
    quotas = _parse_quotas(args.quotas)
    jsonl_exclusions = _existing([*_default_jsonl_exclusions(), *args.exclude_jsonl])
    parquet_exclusions = _existing(
        [*_default_parquet_exclusions(), *args.exclude_parquet]
    )
    if not args.held_out.is_file():
        raise SystemExit(f"held-out task file does not exist: {args.held_out}")

    category_frequency = Counter()
    source_rows = 0
    train_rows = 0
    for product in _stream_json_array(args.source):
        task_id = source_rows
        source_rows += 1
        if product.get("tag") != "train":
            continue
        train_rows += 1
        category_leaf = str(product.get("category") or "").split("窶ｺ")[-1]
        category_frequency[category_leaf] += 1

    exclusion_details = []
    excluded_ids: set[int] = set()
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
    excluded_families: set[str] = set()
    features: list[dict] = []
    held_out_by_category: dict[str, list[dict]] = defaultdict(list)
    for task_id, product in enumerate(_stream_json_array(args.source)):
        if product.get("tag") != "train":
            continue
        feature = _feature_row(task_id, product, category_frequency)
        if task_id in excluded_ids:
            excluded_families.add(feature["family_id"])
        if task_id in held_out_ids:
            held_out_by_category[feature["category_leaf"]].append(feature)
        features.append(feature)

    candidates: dict[str, list[dict]] = defaultdict(list)
    semantic_final_rejections = 0
    for feature in features:
        if feature["task_id"] in excluded_ids:
            continue
        if feature["family_id"] in excluded_families:
            continue
        if _semantic_training_overlap(feature, held_out_by_category):
            semantic_final_rejections += 1
            continue
        candidates[_band(feature["difficulty_score"])].append(feature)

    selected_families = set(excluded_families)
    selected: list[dict] = []
    for band in BANDS:
        rows = _select_strategy_balanced(
            candidates[band],
            limit=quotas[band],
            seed=args.seed,
            band=band,
            selected_families=selected_families,
            strategy_cycle=DEFAULT_STRATEGY_CYCLE,
        )
        if len(rows) != quotas[band]:
            raise SystemExit(f"insufficient {band} candidates: {len(rows)}/{quotas[band]}")
        selected.extend({**row, "difficulty_band": band} for row in rows)

    selected_by_id = {row["task_id"]: row for row in selected}
    full_products = {}
    for task_id, product in enumerate(_stream_json_array(args.source)):
        if task_id in selected_by_id:
            full_products[task_id] = product
    if len(full_products) != len(selected):
        raise SystemExit("failed to recover every selected full task record")

    index_rows = []
    full_rows = []
    for feature in selected:
        metadata = {
            "task_id": feature["task_id"],
            "family_id": feature["family_id"],
            "domain": feature["domain"],
            "category": feature["category"],
            "difficulty_band": feature["difficulty_band"],
            "difficulty_score": feature["difficulty_score"],
            "selection_strategy": feature["teacher_strategy"],
            "selection_reason": "offline difficulty/strategy diversity candidate; requires online rollout probe before frontier promotion",
        }
        index_rows.append(metadata)
        full_rows.append({"probe_metadata": metadata, "task": full_products[feature["task_id"]]})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.jsonl"
    full_path = args.output_dir / "tasks-full.jsonl"
    _write_atomic(tasks_path, _jsonl_bytes(index_rows))
    _write_atomic(full_path, _jsonl_bytes(full_rows))

    selected_ids = set(selected_by_id)
    selected_family_ids = {row["family_id"] for row in selected}
    manifest = {
        "schema": "shopping-grpo-dev-probe-manifest-v1",
        "selection_kind": "offline_dev_probe_candidates_not_online_frontier",
        "source": {
            "path": str(args.source.resolve().relative_to(ROOT)),
            "sha256": _sha256_file(args.source),
            "rows": source_rows,
            "train_rows": train_rows,
        },
        "seed": args.seed,
        "selected_count": len(selected),
        "quotas": quotas,
        "difficulty_distribution": dict(Counter(row["difficulty_band"] for row in selected)),
        "domain_distribution": dict(Counter(str(row["domain"]) for row in selected)),
        "strategy_distribution": dict(Counter(row["teacher_strategy"] for row in selected)),
        "exclusions": exclusion_details,
        "excluded_unique_task_ids": len(excluded_ids),
        "excluded_unique_families": len(excluded_families),
        "semantic_final_rejections": semantic_final_rejections,
        "validation": {
            "selected_task_id_overlap_with_exclusions": len(selected_ids & excluded_ids),
            "selected_family_overlap_with_exclusions": len(selected_family_ids & excluded_families),
            "selected_internal_duplicate_task_ids": len(selected) - len(selected_ids),
            "selected_internal_duplicate_families": len(selected) - len(selected_family_ids),
            "full_record_count_matches": len(full_rows) == len(selected),
        },
        "outputs": {
            "tasks.jsonl": {"sha256": _sha256_file(tasks_path), "rows": len(index_rows)},
            "tasks-full.jsonl": {"sha256": _sha256_file(full_path), "rows": len(full_rows)},
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_atomic(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
