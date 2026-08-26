#!/usr/bin/env python3
"""Freeze exactly 60 successful Probe tasks for a 30-step GRPO run.

The script intentionally refuses to treat offline candidates as successful
training data.  It consumes the online admission outputs produced by the
Probe gate, selects 60 deterministic rows (30 steps × train batch 2), and
writes a self-contained snapshot without changing the canonical 800-task pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "grpo" / "training-probe-v1"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(seed: str, task_id: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "step30")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--expected-admitted-total", type=int, default=251)
    parser.add_argument("--expected-zero-success", type=int, default=70)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="do not fail if the supplied historical admission files differ from 251/70",
    )
    parser.add_argument("--seed", default="grpo-step30-admitted-v1-20260813")
    parser.add_argument(
        "--frontier-only",
        action="store_true",
        help="prefer frontier tasks and rank by accepted Probe success count",
    )
    parser.add_argument(
        "--frontier-stratified",
        action="store_true",
        help="select frontier tasks across 1/4, 2/4, and 3/4 Probe success strata",
    )
    args = parser.parse_args()
    accepted_paths = tuple(args.accepted or (
        DEFAULT_DIR / "admission-calibration200" / "accepted.jsonl",
        DEFAULT_DIR / "admission-remaining600" / "accepted.jsonl",
    ))
    missing = [str(path) for path in accepted_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "online admission outputs are required; refusing to promote offline "
            f"candidates: missing {missing}"
        )
    rows = [row for path in accepted_paths for row in _read(path)]
    unique = {}
    zero_success = set()
    for row in rows:
        task_id = int(row["task_id"])
        gate = row.get("grpo_gate") or {}
        round_index = gate.get("accepted_round")
        rounds = gate.get("rounds") or []
        if gate.get("decision") != "accept" or not isinstance(round_index, int):
            continue
        if not 0 <= round_index < len(rounds):
            continue
        successes = int(rounds[round_index].get("purchase_successes") or 0)
        if successes < 1:
            zero_success.add(task_id)
            continue
        unique[task_id] = row
    observed_total = len({int(row["task_id"]) for row in rows})
    if not args.allow_count_mismatch and (
        observed_total != args.expected_admitted_total
        or len(zero_success) != args.expected_zero_success
    ):
        raise SystemExit(
            "historical Probe count mismatch: expected "
            f"{args.expected_admitted_total} admitted / {args.expected_zero_success} zero-success, "
            f"observed {observed_total} / {len(zero_success)}"
        )
    if len(unique) < args.count:
        raise SystemExit(
            f"only {len(unique)} admitted tasks available; cannot freeze {args.count}"
        )
    if args.frontier_only or args.frontier_stratified:
        frontier = [
            (int(row["task_id"]), row)
            for row in unique.values()
            if str((row.get("grpo_gate") or {}).get("probe_role") or "") == "frontier"
        ]
        frontier.sort(
            key=lambda item: (
                -int((item[1].get("grpo_gate") or {}).get("rounds", [])[((item[1].get("grpo_gate") or {}).get("accepted_round"))].get("purchase_successes") or 0),
                _key(args.seed, item[0]),
            )
        )
        if len(frontier) < args.count:
            raise SystemExit(f"only {len(frontier)} successful frontier tasks available; need {args.count}")
        if args.frontier_stratified:
            by_success = {n: [] for n in (1, 2, 3)}
            for item in frontier:
                gate = item[1].get("grpo_gate") or {}
                rounds = gate.get("rounds") or []
                accepted = gate.get("accepted_round")
                n = int(rounds[accepted].get("purchase_successes") or 0)
                if n in by_success:
                    by_success[n].append(item)
            quota = args.count // 3
            remainder = args.count - quota * 3
            selected_items = []
            for n in (1, 2, 3):
                selected_items.extend(by_success[n][: quota + (1 if n <= remainder else 0)])
            if len(selected_items) < args.count:
                used = {tid for tid, _ in selected_items}
                selected_items.extend(item for item in frontier if item[0] not in used)
                selected_items = selected_items[: args.count]
            selected = [row for _, row in selected_items]
        else:
            selected = [row for _, row in frontier[: args.count]]
        selected.sort(key=lambda row: _key(args.seed, int(row["task_id"])))
        role_quotas = {"frontier": args.count}
    # The online gate's historical accepted pool has only frontier and
    # regression-guard roles: hard-exploration rows with 0/4 success were
    # quarantined. Preserve that observed role ratio when freezing 60 rows.
    by_role = {}
    for task_id, row in unique.items():
        role = str((row.get("grpo_gate") or {}).get("probe_role") or "")
        by_role.setdefault(role, []).append((task_id, row))
    available_roles = set(by_role)
    if not args.frontier_only and available_roles == {"frontier", "regression_guard"}:
        regression_quota = round(args.count * 70 / 251)
        role_quotas = {
            "regression_guard": regression_quota,
            "frontier": args.count - regression_quota,
        }
    elif not args.frontier_only:
        role_quotas = {
            "regression_guard": round(args.count * 0.20),
            "frontier": round(args.count * 0.60),
            "hard_exploration": args.count - round(args.count * 0.20) - round(args.count * 0.60),
        }
    if not args.frontier_only and not args.frontier_stratified:
        by_role = {role: by_role.get(role, []) for role in role_quotas}
        selected = []
        for role, quota in role_quotas.items():
            candidates = sorted(by_role[role], key=lambda item: _key(f"{args.seed}:{role}", item[0]))
            if len(candidates) < quota:
                raise SystemExit(
                    f"role {role!r} has only {len(candidates)} successful tasks; need {quota}"
                )
            selected.extend(row for _, row in candidates[:quota])
        selected.sort(key=lambda row: _key(args.seed, int(row["task_id"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "accepted.jsonl"
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "schema": "shopping-grpo-step30-admitted-probe-v1",
        "count": len(selected),
        "historical_admission": {
            "admitted_total": observed_total,
            "zero_success_excluded": len(zero_success),
            "successful_pool": len(unique),
        },
        "target_training_steps": 30,
        "train_batch_size": 2,
        "online_gate_required": True,
        "seed": args.seed,
        "role_quotas": role_quotas,
        "selection_policy": "frontier_stratified_1_2_3_successes" if args.frontier_stratified else ("frontier_only_ranked_by_probe_successes" if args.frontier_only else "historical_role_mix"),
        "source_inputs": [{"path": str(path.resolve().relative_to(ROOT)), "sha256": _sha(path)} for path in accepted_paths],
        "output": {"path": str(out.resolve().relative_to(ROOT)), "sha256": _sha(out)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
