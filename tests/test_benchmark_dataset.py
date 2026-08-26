"""Validate the frozen Final-240 dataset contract without running models."""

import json
from collections import Counter
from pathlib import Path
import unittest

from shopping_grpo.evaluation.blind_guard import (
    guard_blind_final,
    validate_canonical_benchmark_files,
)

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class BenchmarkDatasetTest(unittest.TestCase):
    def test_formal_evaluation_preflight_accepts_checked_in_final240(self):
        metadata, slices = validate_canonical_benchmark_files(
            tasks_path=ROOT / "data/evaluation/tasks.jsonl",
            metadata_path=ROOT / "data/evaluation/metadata.json",
            slices_path=ROOT / "data/evaluation/slices.jsonl",
        )

        self.assertEqual(metadata["tasks"], 240)
        self.assertEqual(len(slices), 240)
        guard_blind_final(
            [
                ROOT / "data/sft/all.jsonl",
                ROOT / "data/sft/train.jsonl",
                ROOT / "data/sft/validation.jsonl",
            ],
            allowed=False,
        )

    def test_final240_has_frozen_core_and_challenge_structure(self):
        tasks = read_jsonl(ROOT / "data" / "evaluation" / "tasks.jsonl")
        slices = read_jsonl(ROOT / "data" / "evaluation" / "slices.jsonl")
        metadata = json.loads(
            (ROOT / "data" / "evaluation" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(len(tasks), 240)
        self.assertEqual(len(slices), 240)
        self.assertTrue(all(set(row) == {"task_id"} for row in tasks))
        self.assertEqual(len({row["task_id"] for row in tasks}), 240)
        self.assertEqual(
            {row["task_id"] for row in tasks},
            {row["task_id"] for row in slices},
        )
        self.assertEqual(Counter(row["suite"] for row in slices), {"core": 180, "challenge": 60})
        self.assertEqual(
            Counter(
                row["challenge_slice"]
                for row in slices
                if row["suite"] == "challenge"
            ),
            {
                "search_reformulation": 10,
                "candidate_comparison": 10,
                "price_semantics": 10,
                "multi_option": 10,
                "evidence_verification": 10,
                "long_horizon": 10,
            },
        )
        self.assertEqual(
            Counter(row["domain"] for row in slices if row["suite"] == "core"),
            {
                "家居家装": 20,
                "服饰鞋包饰品": 20,
                "休闲娱乐文教": 20,
                "美妆个护健康": 20,
                "生产材料农用品": 20,
                "家用电器数码": 20,
                "运动户外交通": 20,
                "食品饮品": 20,
                "母婴儿童": 20,
            },
        )
        self.assertEqual(metadata["schema_version"], "shopping-evaluation-dataset-v2.2")
        self.assertEqual(metadata["asset"], "shopbench_longhorizon_final_240_v2_2")
        self.assertEqual(
            metadata["contract"],
            "environment-v2.4/reward-v4/benchmark-v2.2",
        )
        self.assertEqual(metadata["environment"], "shopsimulator-environment-v2.4")
        self.assertEqual(metadata["reward"], "shopsimulator-reward-v4")
        self.assertEqual(metadata["training_task_overlap"], 0)
        self.assertEqual(metadata["previous_benchmark_overlap"], 0)
        self.assertEqual(metadata["selected_family_duplicates"], 0)

        difficulty_counts = Counter(row["difficulty_bucket"] for row in slices)
        retrieval_counts = Counter(row["retrieval_bucket"] for row in slices)
        self.assertEqual(dict(difficulty_counts), metadata["difficulty_bucket_counts"])
        self.assertEqual(dict(retrieval_counts), metadata["retrieval_bucket_counts"])
        self.assertGreaterEqual(difficulty_counts["under_10"], 30)
        self.assertGreaterEqual(difficulty_counts["10_15"], 130)
        self.assertGreaterEqual(difficulty_counts["15_18"], 35)
        self.assertLessEqual(difficulty_counts["18_plus"], 10)
        self.assertGreaterEqual(retrieval_counts["rank1"], 135)
        self.assertLessEqual(retrieval_counts["rank1"], 150)
        self.assertGreaterEqual(retrieval_counts["rank2_5"], 55)
        self.assertGreaterEqual(retrieval_counts["rank6_20"], 18)
        self.assertLessEqual(retrieval_counts["rank21_150"], 12)
        self.assertLessEqual(retrieval_counts["missing"], 3)

        core_difficulty = [
            row["difficulty_score"] for row in slices if row["suite"] == "core"
        ]
        challenge_difficulty = [
            row["difficulty_score"]
            for row in slices
            if row["suite"] == "challenge"
        ]
        self.assertGreater(
            sum(challenge_difficulty) / len(challenge_difficulty),
            sum(core_difficulty) / len(core_difficulty),
        )

    def test_final240_task_ids_do_not_overlap_current_sft_data(self):
        final_ids = {
            row["task_id"]
            for row in read_jsonl(ROOT / "data" / "evaluation" / "tasks.jsonl")
        }
        sft_ids = {
            row["task_id"]
            for row in read_jsonl(ROOT / "data" / "sft" / "all.jsonl")
        }

        self.assertFalse(final_ids & sft_ids)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
