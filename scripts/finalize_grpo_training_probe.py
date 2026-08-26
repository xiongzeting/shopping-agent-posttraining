#!/usr/bin/env python3
"""Apply the post-rollout GRPO admission gate to training-probe trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.grpo.probe_gates import decide_grpo_admission
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic


DEFAULT_PROBE = ROOT / "data/grpo/training-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_PROBE / "candidates.jsonl")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROBE / "admission")
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--reward-tolerance", type=float, default=0.025)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finalize(
    candidates: list[dict],
    trajectories: list[dict],
    *,
    rollout_n: int,
    max_rounds: int,
    reward_tolerance: float,
) -> dict[str, list[dict]]:
    candidate_by_id = {int(row["task_id"]): row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("candidate task IDs must be unique")
    grouped = defaultdict(list)
    unexpected = set()
    for row in trajectories:
        try:
            task_id = int(row.get("task_id"))
        except (TypeError, ValueError):
            unexpected.add("missing")
            continue
        if task_id not in candidate_by_id:
            unexpected.add(task_id)
            continue
        grouped[task_id].append(row)
    if unexpected:
        raise ValueError(f"unexpected trajectory task IDs: {sorted(unexpected, key=str)}")

    outputs = {"accepted": [], "reprobe": [], "rejected": []}
    for task_id, candidate in candidate_by_id.items():
        rows = grouped.get(task_id, [])
        if not rows:
            decision = {
                "task_id": task_id,
                "decision": "reprobe",
                "reason": "no_trajectories",
                "probe_role": "unresolved",
                "accepted_round": None,
                "attempts_observed": 0,
                "max_attempts": max_rounds,
                "eligible_for_more_sampling": True,
                "quarantine": False,
                "reward_tolerance": reward_tolerance,
                "rounds": [],
            }
        else:
            decision = decide_grpo_admission(
                rows,
                rollout_n=rollout_n,
                max_rounds=max_rounds,
                reward_tolerance=reward_tolerance,
            )
        outputs[{"accept": "accepted", "reprobe": "reprobe", "reject": "rejected"}[decision["decision"]]].append(
            {**candidate, "grpo_gate": decision}
        )
    return outputs


def main() -> None:
    args = parse_args()
    for path in (args.candidates, args.trajectories):
        if not path.is_file():
            raise SystemExit(f"required file does not exist: {path}")
    candidates = _read_jsonl(args.candidates)
    trajectories = _read_jsonl(args.trajectories)
    outputs = finalize(
        candidates,
        trajectories,
        rollout_n=args.rollout_n,
        max_rounds=args.max_rounds,
        reward_tolerance=args.reward_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = {}
    for name, rows in outputs.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl_atomic(path, rows, force=True)
        output_manifest[path.name] = {"rows": len(rows), "sha256": sha256_file(path)}
    reason_counts = Counter(
        row["grpo_gate"]["reason"]
        for rows in outputs.values()
        for row in rows
    )
    signal_class_counts = Counter(
        report["signal_class"]
        for rows in outputs.values()
        for row in rows
        for report in row["grpo_gate"].get("rounds", [])
    )
    probe_role_counts = Counter(
        row["grpo_gate"]["probe_role"]
        for rows in outputs.values()
        for row in rows
    )
    manifest = {
        "schema": "shopping-grpo-training-probe-admission-v1",
        "candidates": {"path": str(args.candidates), "rows": len(candidates), "sha256": sha256_file(args.candidates)},
        "trajectories": {"path": str(args.trajectories), "rows": len(trajectories), "sha256": sha256_file(args.trajectories)},
        "contract": {"rollout_n": args.rollout_n, "max_rounds": args.max_rounds, "reward_tolerance": args.reward_tolerance},
        "decisions": {name: len(rows) for name, rows in outputs.items()},
        "reason_counts": dict(reason_counts),
        "signal_class_counts": dict(signal_class_counts),
        "probe_role_counts": dict(probe_role_counts),
        "outputs": output_manifest,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest, force=True)
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
