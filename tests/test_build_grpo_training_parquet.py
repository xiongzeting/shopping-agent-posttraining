import unittest

from scripts.build_grpo_training_parquet import (
    _parquet_row,
    _proportional_quotas,
    select_training_tasks,
)
from shopping_grpo.training.grpo.prompt import GRPO_TOOL_PROTOCOL


class BuildGrpoTrainingParquetTests(unittest.TestCase):
    def test_training_prompt_contains_grpo_tool_protocol(self):
        metadata = {
            "task_id": 0,
            "grpo_gate": {
                "accepted_round": 0,
                "rounds": [{"purchase_successes": 1}],
            },
        }
        product = {"instructions": [{"instruction": "测试购物需求"}]}
        row = _parquet_row(metadata, product, split="train", index=0)
        self.assertIn(GRPO_TOOL_PROTOCOL.strip(), row["prompt"][0]["content"])

    def test_proportional_quotas_sum_to_requested_total(self):
        quotas = _proportional_quotas({"frontier": 181, "hard_exploration": 70}, 200)
        self.assertEqual(quotas, {"frontier": 144, "hard_exploration": 56})

    def test_selection_is_deterministic_and_gate_strict(self):
        rows = [
            {
                "task_id": task_id,
                "family_id": f"family-{task_id}",
                "grpo_gate": {
                    "decision": "accept",
                    "reason": "valid_reward_variation",
                    "probe_role": "frontier" if task_id < 7 else "hard_exploration",
                    "accepted_round": 0,
                    "rounds": [{"purchase_successes": 1}],
                },
            }
            for task_id in range(10)
        ]
        first = select_training_tasks(rows, size=8, seed="seed")
        second = select_training_tasks(list(reversed(rows)), size=8, seed="seed")
        self.assertEqual([row["task_id"] for row in first], [row["task_id"] for row in second])
        self.assertEqual(len(first), 8)
        invalid = {**rows[0], "grpo_gate": {**rows[0]["grpo_gate"], "decision": "reprobe"}}
        with self.assertRaisesRegex(ValueError, "did not pass"):
            select_training_tasks([invalid, *rows[1:]], size=8, seed="seed")


if __name__ == "__main__":
    unittest.main()
