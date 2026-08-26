#!/usr/bin/env python3
"""Collect one resumable four-trajectory round for the GRPO training probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.environment.manifest import validate_manifest
from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks


DEFAULT_PROBE = ROOT / "data/grpo/training-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_PROBE / "candidates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROBE / "online")
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--api-key")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-window", type=int, default=25000)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tasks.is_file():
        raise SystemExit(f"probe task file does not exist: {args.tasks}")
    if args.rollout_n != 4:
        raise SystemExit("training probe contract requires --rollout-n=4")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.context_window != 25000:
        raise SystemExit("training probe contract requires --context-window=25000")
    manifest = validate_manifest(
        json.loads((ROOT / "data/environment.json").read_text(encoding="utf-8"))
    )
    tasks = load_tasks(args.tasks)
    attempt_start = (args.round - 1) * args.rollout_n
    audit = {
        "schema": "shopping-grpo-training-probe-run-v1",
        "tasks": len(tasks),
        "task_file": str(args.tasks.resolve()),
        "trajectory_file": str((args.output_dir / "trajectories.jsonl").resolve()),
        "round": args.round,
        "attempt_indices": [attempt_start, attempt_start + args.rollout_n - 1],
        "rollout_n": args.rollout_n,
        "workers": args.workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "environment": manifest["environment_version"],
        "reward": manifest["reward"]["version"],
    }
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def client_factory(task, attempt_index):
        trajectory_seed = args.seed + int(task["task_id"]) * 1009 + attempt_index
        return OpenAIChatClient(
            model=args.model,
            base_url=args.llm_base_url,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            context_window=args.context_window,
            context_safety_margin=args.context_safety_margin,
            context_compaction_enable=True,
            observation_token_budget=2560,
            observation_detail_token_budget=3072,
            observation_generic_token_budget=512,
            observation_candidate_memory_token_budget=1024,
            observation_search_top_k=20,
            seed=trajectory_seed,
        )
    collect_tasks(
        tasks,
        client=None,
        client_factory=client_factory,
        output_path=args.output_dir / "trajectories.jsonl",
        base_url=args.env_url,
        max_steps=args.max_steps,
        attempts_per_task=args.rollout_n,
        attempt_start=attempt_start,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
