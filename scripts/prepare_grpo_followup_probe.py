#!/usr/bin/env python3
"""Prepare the low-memory follow-up probe needed for a 100-step GRPO run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_DIR = ROOT / "data" / "grpo" / "training-probe-v1"
ROLE_BAND_QUOTAS = {
    "regression_guard_candidate": {"lt10": 60, "10to15": 28},
    "frontier_candidate": {"10to15": 176, "15to18": 88},
    "hard_exploration_candidate": {"15to18": 28, "ge18": 60},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROBE_DIR / "remaining-600.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROBE_DIR,
    )
    parser.add_argument("--existing-accepted", type=int, default=65)
    parser.add_argument("--target-steps", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--observed-acceptance-rate", type=float, default=0.325)
    parser.add_argument("--candidate-buffer", type=int, default=24)
    parser.add_argument("--seed", default="grpo-followup-100steps-v1-20260811")
    parser.add_argument("--max-domain-share", type=float, default=0.25)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: str, role: str, band: str, task_id: int) -> str:
    return hashlib.sha256(f"{seed}|{role}|{band}|{task_id}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"follow-up source does not exist: {args.input}")
    if args.existing_accepted < 0 or args.target_steps < 1 or args.train_batch_size < 1:
        raise SystemExit("accepted count, target steps, and batch size must be valid")
    if not 0 < args.observed_acceptance_rate <= 1:
        raise SystemExit("--observed-acceptance-rate must be in (0, 1]")
    if args.candidate_buffer < 0 or args.candidate_buffer % 4:
        raise SystemExit("--candidate-buffer must be a non-negative multiple of four")
    if not 0 < args.max_domain_share <= 1:
        raise SystemExit("--max-domain-share must be in (0, 1]")

    required_groups = args.target_steps * args.train_batch_size
    additional_groups = max(0, required_groups - args.existing_accepted)
    estimated_candidates = math.ceil(additional_groups / args.observed_acceptance_rate)
    estimated_candidates = math.ceil(estimated_candidates / 4) * 4
    candidate_count = estimated_candidates + args.candidate_buffer
    configured_count = sum(
        count for quotas in ROLE_BAND_QUOTAS.values() for count in quotas.values()
    )
    if candidate_count != configured_count or candidate_count != 440:
        raise SystemExit(
            "the canonical follow-up plan requires 440 candidates; got "
            f"{candidate_count} from the requested step budget"
        )

    rows = _read_jsonl(args.input)
    if len(rows) != 600:
        raise SystemExit(f"expected 600 remaining candidates, got {len(rows)}")
    task_ids = [int(row["task_id"]) for row in rows]
    families = [str(row["family_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)) or len(families) != len(set(families)):
        raise SystemExit("remaining candidate IDs and families must be unique")
    if not all((row.get("data_gate") or {}).get("accepted") is True for row in rows):
        raise SystemExit("every follow-up candidate must pass the offline data gate")

    by_band: dict[str, list[dict]] = {}
    for row in rows:
        by_band.setdefault(str(row["difficulty_band"]), []).append(row)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    domain_counts: Counter[str] = Counter()
    role_strategy_counts: dict[str, Counter[str]] = {
        role: Counter() for role in ROLE_BAND_QUOTAS
    }
    domain_cap = max(1, math.floor(candidate_count * args.max_domain_share))
    for role, band_quotas in ROLE_BAND_QUOTAS.items():
        for band, quota in band_quotas.items():
            chosen = []
            candidates = [
                row for row in by_band.get(band, []) if int(row["task_id"]) not in selected_ids
            ]
            while len(chosen) < quota:
                eligible = [
                    row
                    for row in candidates
                    if domain_counts[str(row.get("domain") or "")] < domain_cap
                ]
                if not eligible:
                    raise SystemExit(f"insufficient {role}/{band} candidates under domain cap")
                row = min(
                    eligible,
                    key=lambda item: (
                        role_strategy_counts[role][str(item.get("selection_strategy") or "")],
                        domain_counts[str(item.get("domain") or "")],
                        _stable_key(args.seed, role, band, int(item["task_id"])),
                    ),
                )
                candidates.remove(row)
                selected_ids.add(int(row["task_id"]))
                domain = str(row.get("domain") or "")
                strategy = str(row.get("selection_strategy") or "")
                domain_counts[domain] += 1
                role_strategy_counts[role][strategy] += 1
                planned = {
                    **row,
                    "probe_plan": {
                        "planned_role": role,
                        "role_is_soft_prior": True,
                        "accepted_only_after_online_reward_gate": True,
                    },
                }
                selected.append(planned)
                chosen.append(planned)

    deferred = [row for row in rows if int(row["task_id"]) not in selected_ids]
    selected_payload = _jsonl_bytes(selected)
    deferred_payload = _jsonl_bytes(deferred)
    selected_path = args.output_dir / "followup-440.jsonl"
    deferred_path = args.output_dir / "deferred-160.jsonl"
    _write_atomic(selected_path, selected_payload)
    _write_atomic(deferred_path, deferred_payload)

    expected_additional = math.floor(candidate_count * args.observed_acceptance_rate)
    expected_total = args.existing_accepted + expected_additional
    manifest = {
        "schema": "shopping-grpo-followup-probe-plan-v1",
        "source": {
            "path": str(args.input.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": _sha256_file(args.input),
        },
        "seed": args.seed,
        "memory_contract": {
            "gpu_required": False,
            "source_rows_loaded": len(rows),
            "safe_for_2gb_host_memory": True,
        },
        "step_budget": {
            "existing_accepted_groups": args.existing_accepted,
            "target_training_steps": args.target_steps,
            "train_batch_size": args.train_batch_size,
            "required_effective_groups": required_groups,
            "additional_effective_groups_needed": additional_groups,
            "observed_first_round_acceptance_rate": args.observed_acceptance_rate,
            "estimated_candidates_without_buffer": estimated_candidates,
            "candidate_buffer": args.candidate_buffer,
            "followup_candidates": candidate_count,
            "expected_additional_accepted_groups": expected_additional,
            "expected_total_accepted_groups": expected_total,
            "expected_training_steps": expected_total // args.train_batch_size,
        },
        "soft_target_mix": {
            "regression_guard_candidate": 0.20,
            "frontier_candidate": 0.60,
            "hard_exploration_candidate": 0.20,
        },
        "role_band_quotas": ROLE_BAND_QUOTAS,
        "selected_count": len(selected),
        "deferred_count": len(deferred),
        "difficulty_distribution": dict(Counter(row["difficulty_band"] for row in selected)),
        "role_distribution": dict(
            Counter(row["probe_plan"]["planned_role"] for row in selected)
        ),
        "strategy_distribution": dict(
            Counter(str(row.get("selection_strategy") or "") for row in selected)
        ),
        "domain_distribution": dict(domain_counts),
        "validation": {
            "unique_task_ids": len(selected_ids),
            "unique_families": len({row["family_id"] for row in selected}),
            "all_passed_offline_data_gate": all(
                row["data_gate"]["accepted"] is True for row in selected
            ),
            "selected_and_deferred_cover_source": len(selected) + len(deferred) == len(rows),
            "max_domain_count": max(domain_counts.values(), default=0),
            "domain_cap": domain_cap,
        },
        "outputs": {
            selected_path.name: {
                "rows": len(selected),
                "sha256": _sha256_bytes(selected_payload),
            },
            deferred_path.name: {
                "rows": len(deferred),
                "sha256": _sha256_bytes(deferred_payload),
            },
        },
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(args.output_dir / "followup-440-manifest.json", manifest_payload)
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
