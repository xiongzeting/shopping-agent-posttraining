#!/usr/bin/env python3
"""Promote a strict Teacher materialization into a versioned SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--evaluation-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relative_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def copy_text_lf(source: Path, destination: Path) -> None:
    payload = source.read_bytes().replace(b"\r\n", b"\n")
    destination.write_bytes(payload)


def main() -> int:
    args = parse_args()
    teacher = args.teacher_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite SFT dataset: {output}")

    teacher_metadata = json.loads(
        (teacher / "metadata.json").read_text(encoding="utf-8")
    )
    gate_source = teacher / "data_gate_independent.json"
    if not gate_source.is_file():
        gate_source = teacher / "data_gate_v1.json"
    gate = json.loads(gate_source.read_text(encoding="utf-8"))
    distribution = json.loads(
        (teacher / "distribution_summary.json").read_text(encoding="utf-8")
    )
    if teacher_metadata.get("accepted") != 500 or gate.get("status") != "passed":
        raise SystemExit("Teacher source is not a passed 500-row materialization")

    sources = {
        "all": teacher / "sft.jsonl",
        "train": teacher / "train.jsonl",
        "validation": teacher / "validation.jsonl",
    }
    rows = {name: read_jsonl(path) for name, path in sources.items()}
    expected = {"all": 500, "train": 450, "validation": 50}
    for name, count in expected.items():
        if len(rows[name]) != count:
            raise SystemExit(f"unexpected {name} rows: {len(rows[name])} != {count}")

    ids = {name: {int(row["task_id"]) for row in split} for name, split in rows.items()}
    evaluation_ids = {
        int(row["task_id"]) for row in read_jsonl(args.evaluation_tasks.resolve())
    }
    if ids["train"] & ids["validation"]:
        raise SystemExit("train/validation overlap")
    if ids["all"] != ids["train"] | ids["validation"]:
        raise SystemExit("all split does not equal train + validation")
    if ids["all"] & evaluation_ids:
        raise SystemExit("Final-240 task overlap")

    output.mkdir(parents=True)
    destinations = {
        name: output / ("all.jsonl" if name == "all" else f"{name}.jsonl")
        for name in sources
    }
    for name, source in sources.items():
        copy_text_lf(source, destinations[name])
    copy_text_lf(gate_source, output / "data_gate.json")

    metadata = {
        "schema_version": "shopping-sft-dataset-v3",
        "status": "current",
        "contract": "environment-v2.4/reward-v4/sft-v3",
        "environment": "shopsimulator-environment-v2.4",
        "reward": "shopsimulator-reward-v4",
        "termination": "shopping-termination-v3.1",
        "observation": "shopping-observation-v2",
        "tool_schema": "shopping-tools-v2",
        "teacher_selection": "shopping-teacher-recoverable-process-v4",
        "canonical_eligibility": "pending_30k_token_audit",
        "split_method": "task_id",
        "split_seed": int(teacher_metadata["split_seed"]),
        "validation_ratio": float(teacher_metadata["validation_ratio"]),
        "train_validation_overlap": 0,
        "final_240_overlap": 0,
        "final_240_asin_overlap": 0,
        "final_240_family_overlap": 0,
        "final_240_semantic_overlap": 0,
        **{
            name: {
                "path": relative_label(path),
                "rows": len(rows[name]),
                "sha256": sha256(path),
            }
            for name, path in destinations.items()
        },
        "data_gate": {
            "schema_version": gate["schema_version"],
            "status": gate["status"],
            "path": "data_gate.json",
            "sha256": sha256(output / "data_gate.json"),
        },
        "token_audit": {
            "path": "token_audit.json",
            "status": "pending",
            "max_length": 30000,
        },
        "distribution": {
            "steps": distribution["length_counts"],
            "retrieval_rank": distribution["retrieval_counts"],
            "teacher_strategy": distribution["strategy_counts"],
            "coverage": distribution["coverage_counts"],
            "unique_action_sequences": distribution["unique_action_sequences"],
            "top_sequence_share": distribution["top_sequence_share"],
            "top5_sequence_share": distribution["top5_sequence_share"],
        },
        "source_collection": {
            "schema_version": teacher_metadata["schema_version"],
            "source": relative_label(teacher),
            "raw_rows": teacher_metadata["total"],
            "accepted_rows": teacher_metadata["accepted"],
            "rejected_rows": teacher_metadata["rejected"],
            "accepted_sha256": teacher_metadata["files"]["accepted"]["sha256"],
            "sft_sha256": teacher_metadata["files"]["sft"]["sha256"],
            "teacher_model": "deepseek-v4-flash",
            "teacher_prompt": "shopping-teacher-prompt-v4-convergence-repair",
            "max_steps": 45,
            "selected_stable": teacher_metadata["materialization"]["selected_stable"],
            "selected_corrective": teacher_metadata["materialization"][
                "selected_corrective"
            ],
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
