"""Replay frozen Final-240 terminal purchases through the current Reward v4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.reward_features import compile_reward_features

ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _old_reward_detail(trajectory: dict) -> dict:
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    if isinstance(detail, dict):
        return detail
    return {}


def replay(trajectories_path: Path) -> dict:
    target_products = _load_jsonl(
        ROOT / "data/shopsimulator_official/fine_items_eval_standard.jsonl"
    )
    products = _load_jsonl(
        ROOT / "data/shopsimulator_official/fine_items_eval_train_all.jsonl"
    )
    products_by_asin = {
        str(product.get("asin")): product
        for product in products
        if product.get("asin") is not None
    }
    trajectories = _load_jsonl(trajectories_path)
    old_counts = Counter()
    new_counts = Counter()
    transition_counts = Counter()
    changes = []
    missing_products = []
    purchase_count = 0

    for trajectory in trajectories:
        task_id = int(trajectory["task_id"])
        old_detail = _old_reward_detail(trajectory)
        old_type = str(old_detail.get("reward_type") or "unknown")
        old_counts[old_type] += 1
        terminal = trajectory.get("terminal_result") or {}
        purchase = terminal.get("purchase") or {}
        purchased_asin = str(purchase.get("asin") or "")
        if not purchased_asin:
            new_counts[old_type] += 1
            transition_counts[(old_type, old_type)] += 1
            continue

        purchase_count += 1
        purchased_product = products_by_asin.get(purchased_asin)
        if purchased_product is None:
            missing_products.append(
                {"task_id": task_id, "purchased_asin": purchased_asin}
            )
            new_counts["replay_product_missing"] += 1
            transition_counts[(old_type, "replay_product_missing")] += 1
            continue

        target_product = target_products[task_id]
        instruction = target_product["instructions"][0]
        features = compile_reward_features(instruction, target_product)
        goal = {
            "asin": target_product["asin"],
            "category": target_product["category"],
            "instruction_text": instruction["instruction"],
            **features,
        }
        selected_options = purchase.get("options") or {}
        result = evaluate_purchase(
            purchased_product,
            goal,
            selected_options=selected_options,
            price=purchase.get("price"),
            step_count=len(trajectory.get("steps") or []),
        )
        payload = result.to_dict()
        new_counts[result.reward_type] += 1
        transition_counts[(old_type, result.reward_type)] += 1
        if old_type != result.reward_type or float(
            old_detail.get("reward", trajectory.get("final_reward") or 0.0)
        ) != float(result.reward):
            changes.append(
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "query": instruction["instruction"],
                    "purchased_asin": purchased_asin,
                    "target_asin": str(target_product.get("asin")),
                    "old_reward_type": old_type,
                    "new_reward_type": result.reward_type,
                    "old_reward": old_detail.get(
                        "reward", trajectory.get("final_reward")
                    ),
                    "new_reward": result.reward,
                    "hard_failures": [
                        row
                        for row in payload["constraint_results"]
                        if row.get("strength") == "hard"
                        and row.get("status") != "pass"
                    ],
                    "soft_failures": [
                        row
                        for row in payload["constraint_results"]
                        if row.get("strength") == "soft"
                        and row.get("status") != "pass"
                    ],
                }
            )

    return {
        "schema_version": "final240-reward-semantics-replay-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trajectories_path": str(trajectories_path),
        "trajectory_count": len(trajectories),
        "purchase_count": purchase_count,
        "old_reward_type_counts": dict(old_counts),
        "new_reward_type_counts": dict(new_counts),
        "transition_counts": {
            f"{old}->{new}": count
            for (old, new), count in sorted(transition_counts.items())
        },
        "changed_count": len(changes),
        "missing_products": missing_products,
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.trajectories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "trajectory_count": report["trajectory_count"],
        "purchase_count": report["purchase_count"],
        "old_reward_type_counts": report["old_reward_type_counts"],
        "new_reward_type_counts": report["new_reward_type_counts"],
        "transition_counts": report["transition_counts"],
        "changed_count": report["changed_count"],
        "missing_product_count": len(report["missing_products"]),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
