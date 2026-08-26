#!/usr/bin/env python3
"""Build audited veRL train/validation parquet files from admitted Probe tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.artifacts import write_json_atomic
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.training.grpo.prompt import GRPO_SYSTEM_PROMPT

from scripts.build_grpo_dev_probe import _stream_json_array


DEFAULT_SOURCE = (
    ROOT
    / "environments/ShopSimulator/shop_env/data/items_eval_train.json"
)
DEFAULT_ACCEPTED = (
    ROOT / "data/grpo/training-probe-v1/admission-calibration200/accepted.jsonl",
    ROOT / "data/grpo/training-probe-v1/admission-remaining600/accepted.jsonl",
)
DEFAULT_VALIDATION = ROOT / "data/grpo/dev-probe-v1.1/tasks.jsonl"
DEFAULT_TRAIN = ROOT / "data/grpo/train.parquet"
DEFAULT_VAL = ROOT / "data/grpo/validation.parquet"
DEFAULT_MANIFEST = ROOT / "data/grpo/training-data-manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--accepted", type=Path, action="append")
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--seed", default="grpo-train-v1-20260811")
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _stable_key(seed: str, row: dict) -> str:
    material = f"{seed}:{int(row['task_id'])}:{row.get('family_id', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _proportional_quotas(counts: Counter, total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"requested {total} rows from only {available} accepted tasks")
    exact = {key: total * value / available for key, value in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - quotas[key], counts[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_training_tasks(rows: list[dict], *, size: int, seed: str) -> list[dict]:
    by_id = {int(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("accepted Probe task IDs must be unique")
    eligible = []
    for row in rows:
        gate = row.get("grpo_gate") or {}
        if gate.get("decision") != "accept" or gate.get("reason") != "valid_reward_variation":
            raise ValueError(f"task {row.get('task_id')} did not pass the online admission gate")
        accepted_round = gate.get("accepted_round")
        rounds = gate.get("rounds") or []
        if not isinstance(accepted_round, int) or not 0 <= accepted_round < len(rounds):
            raise ValueError(f"task {row.get('task_id')} has no auditable accepted Probe round")
        if int(rounds[accepted_round].get("purchase_successes") or 0) == 0:
            continue
        eligible.append(row)
    role_counts = Counter(str(row["grpo_gate"]["probe_role"]) for row in eligible)
    quotas = _proportional_quotas(role_counts, size)
    selected = []
    for role in sorted(quotas):
        candidates = [row for row in eligible if row["grpo_gate"]["probe_role"] == role]
        candidates.sort(key=lambda row: _stable_key(f"{seed}:{role}", row))
        selected.extend(candidates[: quotas[role]])
    selected.sort(key=lambda row: _stable_key(seed, row))
    if len(selected) != size:
        raise ValueError(f"selected {len(selected)} rows, expected {size}")
    return selected


def _task_ids(path: Path) -> set[int]:
    return {int(row["task_id"]) for row in _read_jsonl(path)}


def _recover_products(source: Path, task_ids: set[int]) -> dict[int, dict]:
    products = {}
    for task_id, product in enumerate(_stream_json_array(source)):
        if task_id in task_ids:
            products[task_id] = product
    missing = sorted(task_ids - products.keys())
    if missing:
        raise ValueError(f"source products missing task IDs: {missing[:20]}")
    return products


def _instruction(product: dict, task_id: int) -> str:
    instructions = product.get("instructions") or []
    if len(instructions) != 1 or not isinstance(instructions[0], dict):
        raise ValueError(f"task {task_id} has invalid instruction contract")
    text = str(instructions[0].get("instruction") or "").strip()
    if not text:
        raise ValueError(f"task {task_id} has an empty instruction")
    return text


def _parquet_row(metadata: dict, product: dict, *, split: str, index: int) -> dict:
    task_id = int(metadata["task_id"])
    gate = metadata.get("grpo_gate") or {}
    accepted_round = gate.get("accepted_round")
    rounds = gate.get("rounds") or []
    accepted_purchase_successes = None
    if split == "train":
        if not isinstance(accepted_round, int) or not 0 <= accepted_round < len(rounds):
            raise ValueError(f"task {task_id} has no auditable accepted Probe round")
        accepted_purchase_successes = int(
            rounds[accepted_round].get("purchase_successes") or 0
        )
        if accepted_purchase_successes < 1:
            raise ValueError(
                f"task {task_id} has no Gold or Valid purchase in its accepted Probe group"
            )
    return {
        "data_source": "shopping_grpo",
        "prompt": [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": f"Instruction: {_instruction(product, task_id)}"},
        ],
        "ability": "shopping_tool_use",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "task_id": task_id,
            "split": split,
            "index": index,
            "difficulty_band": str(metadata.get("difficulty_band") or ""),
            "difficulty_score": float(metadata.get("difficulty_score") or 0.0),
            "teacher_strategy": str(metadata.get("selection_strategy") or ""),
            "family_id": str(metadata.get("family_id") or ""),
            "probe_role": str((metadata.get("grpo_gate") or {}).get("probe_role") or "validation"),
            "accepted_probe_purchase_successes": accepted_purchase_successes,
        },
    }


def _write_parquet(path: Path, rows: list[dict], *, force: bool) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to build GRPO parquet files") from exc
    if path.exists() and not force:
        raise FileExistsError(f"output exists; pass --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def _distribution(rows: list[dict]) -> dict:
    return {
        "probe_role": dict(Counter((row.get("grpo_gate") or {}).get("probe_role", "validation") for row in rows)),
        "difficulty_band": dict(Counter(str(row.get("difficulty_band") or "") for row in rows)),
        "selection_strategy": dict(Counter(str(row.get("selection_strategy") or "") for row in rows)),
    }


def main() -> None:
    args = parse_args()
    accepted_paths = tuple(args.accepted or DEFAULT_ACCEPTED)
    required = (args.source, args.validation, *accepted_paths)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    accepted = [row for path in accepted_paths for row in _read_jsonl(path)]
    selected = select_training_tasks(accepted, size=args.train_size, seed=args.seed)
    validation = _read_jsonl(args.validation)
    train_ids = {int(row["task_id"]) for row in selected}
    validation_ids = {int(row["task_id"]) for row in validation}
    if len(validation_ids) != len(validation):
        raise SystemExit("validation task IDs must be unique")
    overlap = train_ids & validation_ids
    if overlap:
        raise SystemExit(f"train/validation task overlap: {sorted(overlap)}")
    products = _recover_products(args.source, train_ids | validation_ids)
    train_rows = [
        _parquet_row(row, products[int(row["task_id"])], split="train", index=index)
        for index, row in enumerate(selected)
    ]
    validation_rows = [
        _parquet_row(row, products[int(row["task_id"])], split="validation", index=index)
        for index, row in enumerate(validation)
    ]
    _write_parquet(args.train_output, train_rows, force=args.force)
    _write_parquet(args.validation_output, validation_rows, force=args.force)

    exclusion_paths = [
        ROOT / "data/evaluation/tasks.jsonl",
        ROOT / "data/sft/all.jsonl",
    ]
    exclusions = {}
    for path in exclusion_paths:
        if path.is_file():
            ids = _task_ids(path)
            overlap_ids = sorted(train_ids & ids)
            if overlap_ids:
                raise SystemExit(f"training data overlaps {path}: {overlap_ids[:20]}")
            exclusions[str(path.relative_to(ROOT))] = {
                "rows": len(ids),
                "sha256": sha256_file(path),
                "train_overlap": 0,
            }
    manifest = {
        "schema": "shopping-grpo-training-parquet-v1",
        "seed": args.seed,
        "source": {"path": str(args.source.relative_to(ROOT)), "sha256": sha256_file(args.source)},
        "accepted_inputs": [
            {"path": str(path.relative_to(ROOT)), "rows": len(_read_jsonl(path)), "sha256": sha256_file(path)}
            for path in accepted_paths
        ],
        "accepted_available": len(accepted),
        "train": {
            "path": str(args.train_output.relative_to(ROOT)),
            "rows": len(selected),
            "sha256": sha256_file(args.train_output),
            "distribution": _distribution(selected),
        },
        "validation": {
            "path": str(args.validation_output.relative_to(ROOT)),
            "rows": len(validation),
            "sha256": sha256_file(args.validation_output),
            "source": str(args.validation.relative_to(ROOT)),
            "source_sha256": sha256_file(args.validation),
            "distribution": _distribution(validation),
        },
        "validation_checks": {
            "unique_train_task_ids": len(train_ids) == len(selected),
            "unique_validation_task_ids": len(validation_ids) == len(validation),
            "train_validation_task_overlap": 0,
            "all_train_rows_passed_online_gate": True,
            "all_train_groups_have_purchase_success": all(
                int(
                    row["grpo_gate"]["rounds"][row["grpo_gate"]["accepted_round"]].get(
                        "purchase_successes"
                    )
                    or 0
                )
                >= 1
                for row in selected
            ),
        },
        "exclusions": exclusions,
    }
    write_json_atomic(args.manifest, manifest, force=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
